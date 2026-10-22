"""Resolve customer-provided YouTube links into corpus-ready video specs.

The transcript API fetches one video at a time and cannot enumerate a playlist.
This module owns the public-source boundary: validate YouTube URLs, enumerate
playlists with yt-dlp, preserve titles/provenance, and apply a predictable
collection-size guard before the corpus layer starts expensive transcript work.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Iterable, List

from . import config, utils
from .supadata import get_json as get_supadata_json


@dataclass
class ResolvedSources:
    specs: List[dict] = field(default_factory=list)
    sources: List[dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def split_inputs(inputs: Iterable[str]) -> List[str]:
    """Flatten API inputs and pasted multiline text into non-empty links."""
    out: List[str] = []
    for value in inputs:
        if not isinstance(value, str):
            continue
        out.extend(line.strip() for line in value.replace(",", "\n").splitlines() if line.strip())
    return out


def _video_metadata(url: str, video_id: str) -> dict:
    """Best-effort title lookup; transcript ingestion can still proceed on failure."""
    try:
        from yt_dlp import YoutubeDL

        with YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True}) as ydl:
            info = ydl.extract_info(url, download=False)
        return {"title": str(info.get("title") or video_id), "url": str(info.get("webpage_url") or url)}
    except Exception:
        return _oembed_metadata(url, video_id)


def _oembed_metadata(url: str, video_id: str) -> dict:
    """Use YouTube's lightweight metadata endpoint, which works on cloud IPs."""
    try:
        query = urllib.parse.urlencode({"url": url, "format": "json"})
        request = urllib.request.Request(
            f"https://www.youtube.com/oembed?{query}", headers={"user-agent": "VidSense/2.0"}
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return {"title": str(payload.get("title") or video_id), "url": url}
    except Exception:
        return {"title": video_id, "url": url}


def _playlist_entries(url: str, max_entries: int) -> tuple[str, List[dict], bool]:
    """Return playlist title, flat entries, and whether the collection was clipped."""
    from yt_dlp import YoutubeDL

    # Request one item beyond the limit to distinguish an exact-length list from
    # a truncated one without downloading media or full video metadata.
    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": "in_playlist",
        "playlistend": max_entries + 1,
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    entries = [entry for entry in (info.get("entries") or []) if entry and entry.get("id")]
    clipped = len(entries) > max_entries
    return str(info.get("title") or "YouTube playlist"), entries[:max_entries], clipped


def _supadata_json(path: str, params: dict) -> dict:
    return get_supadata_json(path, params)


def _supadata_playlist_entries(url: str, max_entries: int) -> tuple[str, List[dict], bool]:
    """Resolve playlists without sending YouTube requests from the cloud host."""
    metadata = _supadata_json("/v1/youtube/playlist", {"id": url})
    listing = _supadata_json(
        "/v1/youtube/playlist/videos", {"id": url, "limit": max_entries + 1}
    )
    ordered_ids: List[str] = []
    seen: set[str] = set()
    for key in ("videoIds", "shortIds", "liveIds"):
        for raw_id in listing.get(key, []) or []:
            video_id = str(raw_id)
            if video_id and video_id not in seen:
                seen.add(video_id)
                ordered_ids.append(video_id)
    clipped = len(ordered_ids) > max_entries
    selected = ordered_ids[:max_entries]

    def entry(video_id: str) -> dict:
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        info = _oembed_metadata(video_url, video_id)
        return {"id": video_id, "title": info["title"], "webpage_url": video_url}

    # oEmbed is lightweight, but bounded concurrency keeps a 50-video playlist
    # responsive without producing a burst of 50 simultaneous requests.
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(selected)))) as pool:
        entries = list(pool.map(entry, selected))
    return str(metadata.get("title") or "YouTube playlist"), entries, clipped


def _uses_managed_youtube_access() -> bool:
    return config.TRANSCRIPT_PROVIDER == "supadata" or (
        config.TRANSCRIPT_PROVIDER == "auto" and config.has_supadata()
    )


def resolve_sources(
    inputs: Iterable[str],
    max_videos: int | None = None,
    metadata_lookup: Callable[[str, str], dict] | None = None,
    playlist_lookup: Callable[[str, int], tuple[str, List[dict], bool]] | None = None,
) -> ResolvedSources:
    """Resolve video and playlist inputs to unique specs in first-seen order.

    Invalid and inaccessible sources do not abort a collection that still has
    usable videos. Their status is retained for the UI to explain the outcome.
    """
    limit = max_videos or config.MAX_PLAYLIST_VIDEOS
    metadata_lookup = metadata_lookup or (_oembed_metadata if _uses_managed_youtube_access() else _video_metadata)
    playlist_lookup = playlist_lookup or (
        _supadata_playlist_entries if _uses_managed_youtube_access() else _playlist_entries
    )
    result = ResolvedSources()
    seen: set[str] = set()

    def add_spec(video_id: str, title: str, source_url: str, source_kind: str, playlist_id: str | None = None) -> bool:
        if video_id in seen:
            return False
        seen.add(video_id)
        result.specs.append(
            {
                "video_id": video_id,
                "title": title or video_id,
                "source_url": source_url,
                "source_kind": source_kind,
                "playlist_id": playlist_id,
            }
        )
        return True

    for raw_url in split_inputs(inputs):
        if not utils.is_youtube_url(raw_url):
            result.sources.append({"input": raw_url, "kind": "unknown", "status": "invalid", "error": "Enter a valid YouTube video or playlist link."})
            continue

        playlist_id = utils.extract_playlist_id(raw_url)
        if playlist_id:
            remaining = limit - len(result.specs)
            if remaining <= 0:
                message = f"Collection limit of {limit} videos reached; this playlist was not imported."
                result.sources.append({"input": raw_url, "kind": "playlist", "playlist_id": playlist_id, "status": "limited", "resolved_count": 0, "excluded_count": 0, "error": message})
                result.warnings.append(message)
                continue
            try:
                title, entries, clipped = playlist_lookup(raw_url, remaining)
                # Third-party adapters and test doubles may return more entries
                # than requested; the corpus guard must still be authoritative.
                entries = entries[:remaining]
                added = 0
                duplicates = 0
                for entry in entries:
                    video_id = str(entry.get("id"))
                    video_url = str(entry.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}")
                    if add_spec(video_id, str(entry.get("title") or video_id), video_url, "playlist", playlist_id):
                        added += 1
                    else:
                        duplicates += 1
                excluded = 1 if clipped else 0
                status = "limited" if clipped else "resolved"
                source = {
                    "input": raw_url,
                    "kind": "playlist",
                    "playlist_id": playlist_id,
                    "title": title,
                    "status": status,
                    "resolved_count": added,
                    "duplicate_count": duplicates,
                    "excluded_count": excluded,
                }
                result.sources.append(source)
                if clipped:
                    result.warnings.append(f"{title}: imported the first {remaining} videos because collections are limited to {limit} videos.")
            except Exception as exc:
                result.sources.append({"input": raw_url, "kind": "playlist", "playlist_id": playlist_id, "status": "error", "error": f"Could not read this playlist: {exc}"})
            continue

        video_id = utils.extract_video_id(raw_url)
        if not video_id:
            result.sources.append({"input": raw_url, "kind": "video", "status": "invalid", "error": "This is not a supported YouTube video link."})
            continue
        if len(result.specs) >= limit:
            message = f"Collection limit of {limit} videos reached; this video was not imported."
            result.sources.append({"input": raw_url, "kind": "video", "status": "limited", "resolved_count": 0, "error": message})
            result.warnings.append(message)
            continue
        try:
            meta = metadata_lookup(raw_url, video_id)
        except Exception as exc:
            result.sources.append({"input": raw_url, "kind": "video", "video_id": video_id, "status": "error", "error": f"Could not inspect this video: {exc}"})
            continue
        added = add_spec(video_id, str(meta.get("title") or video_id), str(meta.get("url") or raw_url), "video")
        result.sources.append({"input": raw_url, "kind": "video", "status": "resolved" if added else "duplicate", "resolved_count": int(added), "duplicate_count": int(not added), "title": str(meta.get("title") or video_id)})

    return result
