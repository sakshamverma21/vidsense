"""VidSense customer API and single-page application host.

The API keeps user collections process-local: a browser receives an unguessable
corpus id after ingestion and passes it to question, summary, and highlight
requests. Cached transcript/index data remains reusable on disk, but no account
or cross-restart collection persistence is implied.
"""
from __future__ import annotations

import asyncio
import glob
import json
import threading
import uuid
from collections import defaultdict, deque
from pathlib import Path
from time import monotonic
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from vidsense import config
from vidsense.actions import action_question
from vidsense.agent import ResearchAgent
from vidsense.baseline import SingleShotRAG
from vidsense.corpus import Corpus, build_corpus, load_corpus
from vidsense.eval import metrics
from vidsense.sources import resolve_sources, split_inputs

app = FastAPI(title="VidSense API", version="2.0")

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
DEFAULT_MANIFEST = str(config.EVAL_DIR / "corpus.json")
DEFAULT_CORPUS_ID = "demo"
MODEL = config.DEFAULT_MODEL[config.DEFAULT_PROVIDER]

# Corpus/index creation involves blocking network and embedding work. The small
# registry is protected by one lock; the heavy work deliberately happens after
# the session entry exists so clients can immediately poll its state.
_state: dict = {"sessions": {}, "lock": threading.RLock()}
_rate_lock = threading.Lock()
_rate_buckets: dict[tuple[str, str], deque] = defaultdict(deque)
_research_slots = threading.BoundedSemaphore(config.MAX_CONCURRENT_RESEARCH)
JobThread = threading.Thread


class CorpusCreateRequest(BaseModel):
    inputs: list[str] = Field(min_length=1, max_length=100)
    name: str = Field(default="My video collection", min_length=1, max_length=80)


class ResearchRequest(BaseModel):
    question: str = Field(default="", max_length=config.MAX_QUESTION_CHARS)
    corpus_id: Optional[str] = None
    mode: Literal["question", "summary", "highlights"] = "question"
    baseline: bool = False


def _client_id(request: Request) -> str:
    # Azure's ingress supplies X-Forwarded-For. Only the first hop is relevant
    # for a lightweight portfolio-demo limit; no identity is inferred from it.
    forwarded = request.headers.get("x-forwarded-for", "").split(",", 1)[0].strip()
    return forwarded or (request.client.host if request.client else "unknown")


def _check_rate_limit(request: Request, scope: str, limit: int) -> None:
    now = monotonic()
    key = (scope, _client_id(request))
    with _rate_lock:
        bucket = _rate_buckets[key]
        cutoff = now - config.RATE_LIMIT_WINDOW_SECONDS
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()
        if len(bucket) >= limit:
            retry_after = max(1, int(bucket[0] + config.RATE_LIMIT_WINDOW_SECONDS - now))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="This public demo has reached its short-term usage limit. Please try again shortly.",
                headers={"Retry-After": str(retry_after)},
            )
        bucket.append(now)


def _prune_sessions_locked() -> None:
    now = monotonic()
    expired = [
        corpus_id
        for corpus_id, session in _state["sessions"].items()
        if corpus_id != DEFAULT_CORPUS_ID
        and session.get("status") != "processing"
        and session.get("_expires_at", now + 1) <= now
    ]
    for corpus_id in expired:
        _state["sessions"].pop(corpus_id, None)


def _public_model_error(exc: Exception) -> str:
    message = str(exc).lower()
    if "429" in message or "rate limit" in message:
        return "The AI service is busy or its usage limit was reached. Please try again in a moment."
    if "timeout" in message or "timed out" in message:
        return "The AI service took too long to respond. Please try again."
    if "503" in message or "unavailable" in message:
        return "The AI service is temporarily unavailable. Please try again shortly."
    return "We could not complete this request. Please try again."


def _provider_configured() -> bool:
    if config.DEFAULT_PROVIDER == "groq":
        return config.has_groq()
    if config.DEFAULT_PROVIDER == "gemini":
        return config.has_google()
    return True


def _session_payload(session: dict) -> dict:
    corpus: Corpus | None = session.get("corpus")
    videos = corpus.list_videos() if corpus else []
    skipped = corpus.skipped if corpus else session.get("skipped", [])
    return {
        "corpus_id": session["id"],
        "name": session["name"],
        "status": session["status"],
        "inputs": session.get("inputs", []),
        "sources": session.get("sources", []),
        "warnings": session.get("warnings", []),
        "progress": session.get("progress", {}),
        "videos": videos,
        "video_statuses": videos + skipped,
        "video_count": len(corpus.videos) if corpus else 0,
        "skipped": skipped,
        "error": session.get("error"),
    }


def _load_demo_session() -> dict:
    """Return the starter collection session without blocking an HTTP request."""
    with _state["lock"]:
        existing = _state["sessions"].get(DEFAULT_CORPUS_ID)
        if existing:
            return existing
        existing = {
            "id": DEFAULT_CORPUS_ID,
            "name": "Deep learning starter collection",
            "inputs": [],
            "status": "processing",
            "sources": [],
            "warnings": [],
            "progress": {"phase": "loading_demo"},
        }
        _state["sessions"][DEFAULT_CORPUS_ID] = existing
        JobThread(target=_run_demo_job, daemon=True).start()
        return existing


def _run_demo_job() -> None:
    session = _state["sessions"][DEFAULT_CORPUS_ID]
    try:
        corpus = load_corpus(DEFAULT_MANIFEST)
        with _state["lock"]:
            session.update(
                {
                    "corpus": corpus,
                    "name": corpus.name,
                    "status": "partial" if corpus.skipped else "ready",
                    "sources": corpus.sources,
                    "warnings": corpus.warnings,
                    "progress": {"phase": "ready", "current": len(corpus.videos), "total": len(corpus.videos)},
                }
            )
    except Exception as exc:
        with _state["lock"]:
            session.update({"status": "error", "error": str(exc), "progress": {"phase": "error"}})


def _get_session(corpus_id: Optional[str]) -> dict:
    requested_id = corpus_id or DEFAULT_CORPUS_ID
    if requested_id == DEFAULT_CORPUS_ID:
        session = _load_demo_session()
    else:
        with _state["lock"]:
            _prune_sessions_locked()
            session = _state["sessions"].get(requested_id)
            if session:
                session["_expires_at"] = monotonic() + config.SESSION_TTL_SECONDS
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="This collection is no longer available. Add the sources again to continue.")
    return session


def _ready_corpus(corpus_id: Optional[str]) -> tuple[dict, Corpus]:
    session = _get_session(corpus_id)
    if session["status"] not in {"ready", "partial"} or not session.get("corpus"):
        detail = session.get("error") or "This collection is still being prepared."
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)
    return session, session["corpus"]


def _agent_for(session: dict) -> ResearchAgent:
    with _state["lock"]:
        if "agent" not in session:
            session["agent"] = ResearchAgent(session["corpus"], max_steps=6, verify=True)
        return session["agent"]


def _baseline_for(session: dict) -> SingleShotRAG:
    with _state["lock"]:
        if "baseline" not in session:
            # Customer actions should feel immediate. They still use the same
            # corpus retrieval and citation pipeline, but avoid the planner /
            # verifier loop used by open-ended research questions.
            session["baseline"] = SingleShotRAG(session["corpus"], k=10)
        return session["baseline"]


def _run_corpus_job(corpus_id: str) -> None:
    session = _get_session(corpus_id)
    try:
        with _state["lock"]:
            session["progress"] = {"phase": "resolving_sources"}
        resolved = resolve_sources(session["inputs"])
        with _state["lock"]:
            session["sources"] = resolved.sources
            session["warnings"] = resolved.warnings
            session["progress"] = {"phase": "sources_resolved", "current": len(resolved.specs), "total": len(resolved.specs)}
        if not resolved.specs:
            raise RuntimeError("No usable YouTube videos were found. Check the links and try again.")

        def progress(data: dict) -> None:
            with _state["lock"]:
                session["progress"] = data

        corpus = build_corpus(
            session["name"],
            resolved.specs,
            sources=resolved.sources,
            warnings=resolved.warnings,
            progress=progress,
        )
        with _state["lock"]:
            session.update(
                {
                    "corpus": corpus,
                    "status": "partial" if corpus.skipped or any(source.get("status") in {"invalid", "error", "limited"} for source in resolved.sources) else "ready",
                    "skipped": corpus.skipped,
                    "progress": {"phase": "ready", "current": len(corpus.videos), "total": len(corpus.videos)},
                }
            )
    except Exception as exc:
        with _state["lock"]:
            session.update({"status": "error", "error": str(exc), "progress": {"phase": "error"}})


def _validated_question(mode: str, question: str) -> str:
    if mode == "question" and not question.strip():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Enter a question before asking the collection.")
    return action_question(mode, question)


# --- Corpus lifecycle --------------------------------------------------------
@app.post("/api/corpora", status_code=status.HTTP_202_ACCEPTED)
def api_create_corpus(req: CorpusCreateRequest, request: Request):
    _check_rate_limit(request, "ingest", config.INGEST_RATE_LIMIT)
    inputs = split_inputs(req.inputs)
    if not inputs:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Paste at least one YouTube video or playlist link.")
    if len(inputs) > config.MAX_INPUT_SOURCES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Use at most {config.MAX_INPUT_SOURCES} source links at a time.",
        )
    if any(len(item) > 2048 for item in inputs):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="One of the source links is too long.")
    corpus_id = uuid.uuid4().hex
    session = {
        "id": corpus_id,
        "name": req.name.strip(),
        "inputs": inputs,
        "status": "processing",
        "sources": [],
        "warnings": [],
        "progress": {"phase": "queued"},
        "_expires_at": monotonic() + config.SESSION_TTL_SECONDS,
    }
    with _state["lock"]:
        _prune_sessions_locked()
        custom_sessions = [item for key, item in _state["sessions"].items() if key != DEFAULT_CORPUS_ID]
        if sum(item.get("status") == "processing" for item in custom_sessions) >= config.MAX_CONCURRENT_INGESTIONS:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="The public demo is already preparing other collections. Please try again shortly.",
            )
        if len(custom_sessions) >= config.MAX_COLLECTIONS:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The public demo has reached its temporary collection capacity. Please try again later.",
            )
        _state["sessions"][corpus_id] = session
    JobThread(target=_run_corpus_job, args=(corpus_id,), daemon=True).start()
    return _session_payload(session)


@app.get("/api/corpora/{corpus_id}")
def api_corpus_status(corpus_id: str):
    return _session_payload(_get_session(corpus_id))


# Backwards-compatible endpoint for scripts and the original frontend.
@app.get("/api/corpus")
def api_corpus():
    return _session_payload(_get_session(DEFAULT_CORPUS_ID))


# --- Internal developer API --------------------------------------------------
@app.get("/api/eval")
def api_eval():
    runs = sorted(glob.glob(str(config.RUNS_DIR / "run_*.json")))
    if not runs:
        return {"error": "No eval runs found. Run: python -m vidsense.eval.runner"}
    return json.loads(Path(runs[-1]).read_text(encoding="utf-8"))


# --- Research ---------------------------------------------------------------
@app.post("/api/research")
def api_research(req: ResearchRequest, request: Request):
    _check_rate_limit(request, "research", config.RESEARCH_RATE_LIMIT)
    session, _corpus = _ready_corpus(req.corpus_id)
    question = _validated_question(req.mode, req.question)
    if not _research_slots.acquire(blocking=False):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="VidSense is handling other requests. Please try again shortly.")
    try:
        out = (_baseline_for(session).run(question) if req.mode != "question" else _agent_for(session).run(question))
        out["mode"] = req.mode
        out["cost_usd"] = metrics.cost_usd(out["usage"], MODEL)
        if req.baseline:
            baseline = _baseline_for(session).run(question)
            baseline["cost_usd"] = metrics.cost_usd(baseline["usage"], MODEL)
            out["baseline"] = baseline
        return out
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=_public_model_error(exc)) from exc
    finally:
        _research_slots.release()


@app.get("/api/research/stream")
async def api_research_stream(
    request: Request,
    question: str = Query(default="", max_length=config.MAX_QUESTION_CHARS),
    corpus_id: Optional[str] = None,
    mode: Literal["question", "summary", "highlights"] = "question",
):
    """Stream the grounded agent while keeping the chosen corpus isolated."""
    _check_rate_limit(request, "research", config.RESEARCH_RATE_LIMIT)
    session, _corpus = _ready_corpus(corpus_id)
    prepared_question = _validated_question(mode, question)
    if not _research_slots.acquire(blocking=False):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="VidSense is handling other requests. Please try again shortly.")
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def worker():
        try:
            # Summaries and highlights are explicit product actions, so keep
            # them bounded to one retrieval + one grounded generation call.
            # Open-ended questions retain the richer research-agent stream.
            events = (
                _baseline_for(session).run(prepared_question)
                if mode != "question"
                else _agent_for(session).run_stream(prepared_question)
            )
            if mode != "question":
                asyncio.run_coroutine_threadsafe(
                    queue.put({"event": "answer", "data": {"answer": events["answer"]}}), loop
                )
                events = [{"event": "done", "data": events}]
            for event in events:
                if event["event"] == "done":
                    event["data"]["cost_usd"] = metrics.cost_usd(event["data"]["usage"], MODEL)
                    event["data"]["mode"] = mode
                asyncio.run_coroutine_threadsafe(queue.put(event), loop)
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(queue.put({"event": "error", "data": {"message": _public_model_error(exc)}}), loop)
        finally:
            _research_slots.release()
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    threading.Thread(target=worker, daemon=True).start()

    async def event_gen():
        while True:
            event = await queue.get()
            if event is None:
                break
            yield {"event": event["event"], "data": json.dumps(event["data"], default=str)}

    return EventSourceResponse(event_gen())


@app.get("/api/live")
def live():
    return {"ok": True}


@app.get("/api/ready")
def ready():
    if not FRONTEND_DIR.exists() or not Path(DEFAULT_MANIFEST).exists() or not _provider_configured():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="VidSense is not fully configured.")
    return {"ok": True, "provider": config.DEFAULT_PROVIDER, "model": MODEL}


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "provider": config.DEFAULT_PROVIDER,
        "model": MODEL,
        "max_playlist_videos": config.MAX_PLAYLIST_VIDEOS,
        "temporary_session_minutes": config.SESSION_TTL_SECONDS // 60,
    }


# Mounted last so /api/* remains reachable.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
