"""Multi-video corpus: ingest transcripts, build a metadata-rich FAISS store.

The `Corpus` object is the shared state the agent's tools operate on. It holds:
  * per-video transcripts (for timestamp lookups), and
  * one unified FAISS store whose chunks carry {video_id, title, timestamp,
    formatted_time, youtube_link} metadata so every retrieved passage is
    citation-ready.

Corpora are cached to disk keyed by their set of video ids, so eval runs are
cheap and deterministic and captionless videos are only fetched once.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config, utils
from .supadata import get_json as get_supadata_json


# --- Data model --------------------------------------------------------------
@dataclass
class VideoRecord:
    video_id: str
    title: str
    transcript: List[dict]  # [{"text": str, "start": float, "duration": float}, ...]
    source_url: str = ""
    source_kind: str = "manifest"
    playlist_id: Optional[str] = None

    @property
    def full_text(self) -> str:
        return " ".join(seg["text"] for seg in self.transcript)

    @property
    def duration(self) -> float:
        return self.transcript[-1]["start"] if self.transcript else 0.0


@dataclass
class Corpus:
    name: str
    videos: Dict[str, VideoRecord]
    vector_store: FAISS
    skipped: List[dict] = field(default_factory=list)  # [{"video_id", "reason"}]
    sources: List[dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # -- lookups -------------------------------------------------------------
    def list_videos(self) -> List[dict]:
        return [
            {
                "video_id": v.video_id,
                "title": v.title,
                "duration": utils.format_timestamp(v.duration),
                "youtube_link": utils.youtube_link(v.video_id, 0),
                "source_url": v.source_url or utils.youtube_link(v.video_id, 0),
                "source_kind": v.source_kind,
                "playlist_id": v.playlist_id,
                "status": "ready",
            }
            for v in self.videos.values()
        ]

    def get(self, video_id: str) -> Optional[VideoRecord]:
        return self.videos.get(self.resolve_video_id(video_id))

    def resolve_video_id(self, ref: Optional[str]) -> Optional[str]:
        """Map a reference to a real video id.

        The planner LLM sometimes passes a *title* where a video_id is expected;
        tolerate that (and case/substring title matches) instead of silently
        returning zero results.
        """
        if not ref:
            return ref
        if ref in self.videos:
            return ref
        rl = ref.strip().lower()
        for vid, rec in self.videos.items():
            if rec.title.strip().lower() == rl:
                return vid
        for vid, rec in self.videos.items():
            if rl and (rl in rec.title.lower() or rec.title.lower() in rl):
                return vid
        return ref  # unknown — let the filter yield nothing

    def retriever(self, video_id: Optional[str] = None, k: int = 4):
        """A retriever over the whole corpus, or filtered to one video."""
        search_kwargs: dict = {"k": k}
        if video_id:
            search_kwargs["filter"] = {"video_id": self.resolve_video_id(video_id)}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)

    def search(self, query: str, video_id: Optional[str] = None, k: int = 4) -> List[Document]:
        flt = {"video_id": self.resolve_video_id(video_id)} if video_id else None
        return self.vector_store.similarity_search(query, k=k, filter=flt)


# --- Transcript fetching -----------------------------------------------------
def _fetch_direct_transcript(video_id: str) -> List[dict]:
    """Fetch captions directly from YouTube for local/self-hosted use."""
    from youtube_transcript_api import YouTubeTranscriptApi

    api = YouTubeTranscriptApi()
    fetched = api.fetch(video_id)
    # v1.x FetchedTranscript -> list of {text, start, duration}
    return fetched.to_raw_data()


def _fetch_supadata_transcript(video_id: str) -> List[dict]:
    """Fetch timestamped captions through a cloud-safe managed provider."""
    payload = get_supadata_json(
        "/v1/transcript", {"url": f"https://www.youtube.com/watch?v={video_id}"}
    )

    if payload.get("error") == "transcript-unavailable":
        raise RuntimeError("Captions are unavailable for this video.")
    content = payload.get("content")
    if not isinstance(content, list):
        raise RuntimeError("The managed transcript service returned an unexpected response.")
    segments: List[dict] = []
    for item in content:
        if not isinstance(item, dict) or not str(item.get("text", "")).strip():
            continue
        segments.append(
            {
                "text": str(item["text"]).strip(),
                "start": max(0.0, float(item.get("offset") or 0) / 1000.0),
                "duration": max(0.0, float(item.get("duration") or 0) / 1000.0),
            }
        )
    return segments


def _fetch_transcript(video_id: str) -> List[dict]:
    """Return timestamped transcript segments using the configured source.

    ``auto`` deliberately prefers the managed provider when its key is present.
    That avoids a slow, known-to-fail YouTube request from cloud data centers,
    while retaining the zero-cost direct path for local development.
    """
    provider = config.TRANSCRIPT_PROVIDER
    if provider not in {"auto", "direct", "supadata"}:
        raise RuntimeError("VIDSENSE_TRANSCRIPT_PROVIDER must be auto, direct, or supadata.")
    if provider == "supadata" or (provider == "auto" and config.has_supadata()):
        return _fetch_supadata_transcript(video_id)
    return _fetch_direct_transcript(video_id)


def _public_transcript_error(exc: Exception) -> str:
    """Turn provider failures into concise customer-safe ingestion messages."""
    message = str(exc)
    lowered = message.lower()
    if "requestblocked" in lowered or "ipblocked" in lowered or "not a bot" in lowered:
        return "YouTube blocked caption access from this hosting network. The site owner must configure a cloud transcript provider."
    if "managed transcript service" in lowered or "managed video service" in lowered:
        return message
    if (
        "notranscript" in lowered
        or "transcriptsdisabled" in lowered
        or "empty transcript" in lowered
        or "captions are unavailable" in lowered
    ):
        return "Captions are unavailable for this video."
    return "The transcript could not be retrieved."


# --- Chunking ----------------------------------------------------------------
def _chunk_video(record: VideoRecord, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """Split one video's transcript into timestamped, citation-ready Documents.

    Reuses the position->timestamp mapping idea from the original
    create_chunks_with_timestamps (previously dead code).
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_chunks = splitter.create_documents([record.full_text])

    docs: List[Document] = []
    cursor = 0
    for chunk in text_chunks:
        text = chunk.page_content
        timestamp = utils.find_timestamp_for_position(record.transcript, cursor)
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "video_id": record.video_id,
                    "title": record.title,
                    "timestamp": float(timestamp),
                    "formatted_time": utils.format_timestamp(timestamp),
                    "youtube_link": utils.youtube_link(record.video_id, timestamp),
                },
            )
        )
        cursor += len(text)
    return docs


# --- Manifest handling -------------------------------------------------------
def _read_manifest(manifest_path: str | Path) -> tuple[str, List[dict]]:
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    name = data.get("name", Path(manifest_path).stem)
    specs = []
    for item in data["videos"]:
        vid = item.get("id") or utils.extract_video_id(item.get("url", ""))
        if not vid:
            continue
        specs.append({"video_id": vid, "title": item.get("title", vid)})
    return name, specs


def _cache_key(name: str, specs: List[dict]) -> str:
    """Stable cache identity for a collection and its source-aware metadata."""
    fingerprint = "|".join(
        "::".join(
            str(spec.get(key, ""))
            for key in ("video_id", "title", "source_url", "source_kind", "playlist_id")
        )
        for spec in specs
    )
    digest = hashlib.sha1(fingerprint.encode()).hexdigest()[:12]
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in name).strip("-") or "corpus"
    return f"{safe_name[:48]}-{digest}"


def _legacy_cache_key(name: str, video_ids: List[str]) -> str:
    """Locate pre-v2 cache directories bundled into existing Docker images."""
    digest = hashlib.sha1(("|".join(sorted(video_ids))).encode()).hexdigest()[:10]
    return f"{name}-{digest}"


# --- Build / load ------------------------------------------------------------
def build_corpus(
    name: str,
    specs: List[dict],
    rebuild: bool = False,
    verbose: bool = True,
    sources: Optional[List[dict]] = None,
    warnings: Optional[List[str]] = None,
    progress=None,
    transcript_loader=None,
    video_saved=None,
    cache_namespace: Optional[str] = None,
    persist_cache: bool = True,
) -> Corpus:
    """Fetch transcripts for `specs` and build (or load) the cached corpus.

    specs: list of {"video_id": str, "title": str}
    """
    # The resolver already deduplicates, but manifests and programmatic callers
    # can still contain repeated ids. Preserve the first occurrence and title.
    unique_specs: List[dict] = []
    seen_ids: set[str] = set()
    for spec in specs:
        video_id = spec.get("video_id")
        if video_id and video_id not in seen_ids:
            seen_ids.add(video_id)
            unique_specs.append(spec)
    if not unique_specs:
        raise RuntimeError(f"No valid YouTube videos were supplied for corpus '{name}'.")

    video_ids = [s["video_id"] for s in unique_specs]
    cache_dir = config.CACHE_DIR / _cache_key(cache_namespace or name, unique_specs)
    legacy_cache_dir = config.CACHE_DIR / _legacy_cache_key(name, video_ids)
    if persist_cache and not cache_namespace and not rebuild and not (cache_dir / "records.json").exists() and (legacy_cache_dir / "records.json").exists():
        # Keep the pre-built demo corpus usable in deployment images. New
        # customer collections always receive the richer source-aware key.
        cache_dir = legacy_cache_dir
    meta_path = cache_dir / "records.json"
    faiss_path = cache_dir / "faiss"
    embeddings = config.get_embeddings()

    # --- fast path: load from cache ---
    if persist_cache and not rebuild and meta_path.exists() and (faiss_path / "index.faiss").exists():
        if verbose:
            print(f"📦 Loading cached corpus '{name}' from {cache_dir}")
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        videos = {
            vid: VideoRecord(
                vid,
                rec["title"],
                rec["transcript"],
                rec.get("source_url", ""),
                rec.get("source_kind", "manifest"),
                rec.get("playlist_id"),
            )
            for vid, rec in payload["videos"].items()
        }
        store = FAISS.load_local(str(faiss_path), embeddings, allow_dangerous_deserialization=True)
        return Corpus(
            name=name,
            videos=videos,
            vector_store=store,
            skipped=payload.get("skipped", []),
            sources=payload.get("sources", sources or []),
            warnings=payload.get("warnings", warnings or []),
        )

    # --- slow path: fetch + build ---
    specs_by_id = {s["video_id"]: s for s in unique_specs}
    titles = {s["video_id"]: s.get("title", s["video_id"]) for s in unique_specs}
    videos: Dict[str, VideoRecord] = {}
    skipped: List[dict] = []
    all_docs: List[Document] = []

    for index, vid in enumerate(video_ids, 1):
        try:
            if progress:
                progress({"phase": "fetching_transcript", "current": index, "total": len(video_ids), "video_id": vid})
            if verbose:
                print(f"📥 Fetching transcript: {vid} ({titles[vid]})")
            transcript = transcript_loader(vid) if transcript_loader else None
            if transcript is None:
                transcript = _fetch_transcript(vid)
            if not transcript:
                raise ValueError("empty transcript")
            if sum(len(seg["text"]) for seg in transcript) > config.MAX_TRANSCRIPT_CHARS:
                raise ValueError("Transcript exceeds the collection size limit.")
            spec = specs_by_id[vid]
            record = VideoRecord(
                vid,
                titles[vid],
                transcript,
                spec.get("source_url", utils.youtube_link(vid, 0)),
                spec.get("source_kind", "manifest"),
                spec.get("playlist_id"),
            )
            chunks = _chunk_video(record)
            if len(all_docs) + len(chunks) > config.MAX_CORPUS_CHUNKS:
                raise ValueError("Transcript exceeds the collection size limit.")
            if video_saved:
                video_saved(record)
            videos[vid] = record
            all_docs.extend(chunks)
        except Exception as e:  # captions disabled / unavailable / network
            diagnostic = f"{type(e).__name__}: {e}"
            reason = _public_transcript_error(e)
            if verbose:
                print(f"  ⚠️  skipped {vid}: {diagnostic}")
            skipped.append(
                {
                    "video_id": vid,
                    "title": titles[vid],
                    "source_url": specs_by_id[vid].get("source_url", ""),
                    "status": "skipped",
                    "reason": reason,
                }
            )

    if not all_docs:
        explanation = skipped[0]["reason"] if skipped else "No usable captions were found."
        raise RuntimeError(f"No usable transcripts were found for collection '{name}'. {explanation}")

    if verbose:
        print(f"🔨 Building FAISS store from {len(all_docs)} chunks across {len(videos)} videos...")
    if progress:
        progress({"phase": "building_index", "current": len(video_ids), "total": len(video_ids)})
    store = FAISS.from_documents(all_docs, embeddings)

    # --- persist cache ---
    if persist_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        store.save_local(str(faiss_path))
        meta_path.write_text(
            json.dumps(
                {
                    "name": name,
                    "videos": {
                        vid: {
                            "title": r.title,
                            "transcript": r.transcript,
                            "source_url": r.source_url,
                            "source_kind": r.source_kind,
                            "playlist_id": r.playlist_id,
                        }
                        for vid, r in videos.items()
                    },
                    "skipped": skipped,
                    "sources": sources or [],
                    "warnings": warnings or [],
                },
            ),
            encoding="utf-8",
        )
    if verbose:
        print(f"✅ Corpus '{name}' ready: {len(videos)} videos, {len(skipped)} skipped")

    if progress:
        progress({"phase": "ready", "current": len(video_ids), "total": len(video_ids)})
    return Corpus(name=name, videos=videos, vector_store=store, skipped=skipped, sources=sources or [], warnings=warnings or [])


def load_corpus(manifest_path: str | Path, rebuild: bool = False, verbose: bool = True) -> Corpus:
    """Build/load a corpus from a manifest JSON file."""
    name, specs = _read_manifest(manifest_path)
    return build_corpus(name, specs, rebuild=rebuild, verbose=verbose)


def restore_corpus(name: str, records: List[dict], skipped=None, sources=None, warnings=None) -> Corpus:
    """Rebuild a local index from durable transcript data, without provider calls.

    MongoDB stores plain records, never pickled FAISS objects supplied by users.
    A process restart costs local embedding work but no transcript credits.
    """
    videos = {}
    docs = []
    if len(records) > config.MAX_PLAYLIST_VIDEOS:
        raise ValueError("The saved collection exceeds the configured video limit.")
    for data in records:
        record = VideoRecord(
            data["video_id"], data["title"], data["transcript"],
            data.get("source_url", ""), data.get("source_kind", "video"), data.get("playlist_id"),
        )
        if sum(len(segment["text"]) for segment in record.transcript) > config.MAX_TRANSCRIPT_CHARS:
            raise ValueError("The saved transcript exceeds the configured size limit.")
        videos[record.video_id] = record
        docs.extend(_chunk_video(record))
        if len(docs) > config.MAX_CORPUS_CHUNKS:
            raise ValueError("The saved collection exceeds the configured index size limit.")
    if not docs:
        raise ValueError("No saved transcripts are available for this collection.")
    return Corpus(name, videos, FAISS.from_documents(docs, config.get_embeddings()),
                  skipped or [], sources or [], warnings or [])


def corpus_from_urls(name: str, urls: List[str], rebuild: bool = False, verbose: bool = True) -> Corpus:
    """Build an ad-hoc corpus from video links and/or playlist links."""
    from .sources import resolve_sources

    resolved = resolve_sources(urls)
    return build_corpus(
        name,
        resolved.specs,
        rebuild=rebuild,
        verbose=verbose,
        sources=resolved.sources,
        warnings=resolved.warnings,
    )
