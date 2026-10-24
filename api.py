"""Public starter demo and authenticated, MongoDB-backed video collections.

FAISS indexes remain process-local and can be rebuilt from saved transcripts.
Keep one replica until ingestion is moved to a durable queue-backed worker.
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
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from vidsense import config
from vidsense.auth import optional_user, require_user, require_verified_user, router as auth_router
from vidsense.database import enforce_rate_limit, get_database
from vidsense.storage import collection_store
from vidsense.actions import action_question
from vidsense.agent import ResearchAgent
from vidsense.baseline import SingleShotRAG
from vidsense.corpus import Corpus, build_corpus, load_corpus, restore_corpus
from pymongo.errors import PyMongoError
from vidsense.eval import metrics
from vidsense.sources import resolve_sources, split_inputs

app = FastAPI(title="VidSense API", version="2.0")
app.include_router(auth_router)


@app.middleware("http")
async def browser_security(request: Request, call_next):
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        # JSON content types alone are insufficient if permissive CORS is
        # introduced later. The SPA always supplies an explicit same-origin
        # request marker; cross-origin browser requests are also rejected.
        origin = request.headers.get("origin")
        configured = urlsplit(config.APP_URL)
        expected_origin = f"{configured.scheme}://{configured.netloc}"
        if request.headers.get("x-vidsense-request") != "1" or (origin and origin != expected_origin):
            return JSONResponse({"detail": "Refresh VidSense and try again from this site."}, status_code=403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self' https://cdn.jsdelivr.net; "
        "style-src 'self' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' https://img.youtube.com; connect-src 'self'; "
        "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
    )
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    if config.AUTH_COOKIE_SECURE:
        response.headers["Strict-Transport-Security"] = "max-age=31536000"
    return response


@app.exception_handler(PyMongoError)
async def database_error(_request, _exc):
    return JSONResponse({"detail": "The account service is temporarily unavailable."}, status_code=503)

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
_job_slots = threading.BoundedSemaphore(config.MAX_CONCURRENT_INGESTIONS)
_active_jobs: dict[str, str] = {}
JobThread = threading.Thread


def _reserve_job(corpus_id: str) -> str:
    if not _job_slots.acquire(blocking=False):
        raise HTTPException(429, "Other collections are being prepared. Please try again shortly.")
    token = uuid.uuid4().hex
    _active_jobs[token] = corpus_id
    return token


def _release_job(token: str) -> None:
    with _state["lock"]:
        if _active_jobs.pop(token, None) is not None:
            _job_slots.release()


def _dispatch_job(corpus_id: str, target, token: str) -> None:
    snapshot = _state["sessions"][corpus_id]
    def work():
        try:
            target(corpus_id, snapshot)
        finally:
            _release_job(token)
    try:
        JobThread(target=work, daemon=True).start()
    except Exception:
        _release_job(token)
        raise


def _admit_session(session: dict) -> None:
    """Bound local index cache independently of durable collection capacity."""
    _prune_sessions_locked()
    if session["id"] not in _state["sessions"]:
        custom = [item for key, item in _state["sessions"].items() if key != DEFAULT_CORPUS_ID]
        if len(custom) >= config.MAX_COLLECTIONS:
            evictable = [item for item in custom if item["status"] != "processing" and item["id"] not in _active_jobs.values()]
            if not evictable:
                raise HTTPException(429, "Collections are being prepared. Please try again shortly.")
            oldest = min(evictable, key=lambda item: item.get("_expires_at", 0))
            _state["sessions"].pop(oldest["id"], None)
    session["_expires_at"] = monotonic() + config.SESSION_TTL_SECONDS
    _state["sessions"][session["id"]] = session


class CorpusCreateRequest(BaseModel):
    inputs: list[str] = Field(min_length=1, max_length=100)
    name: str = Field(default="My video collection", min_length=1, max_length=80)


class ResearchRequest(BaseModel):
    question: str = Field(default="", max_length=config.MAX_QUESTION_CHARS)
    corpus_id: Optional[str] = None
    mode: Literal["question", "summary", "highlights"] = "question"
    baseline: bool = False


def _client_id(request: Request) -> str:
    # Configure trusted proxies in the server/ingress, never interpret an
    # arbitrary client-supplied forwarding header as an identity here.
    return request.client.host if request.client else "unknown"


def _check_rate_limit(request: Request, scope: str, limit: int) -> None:
    if config.MONGODB_URI:
        user = optional_user(request)
        subject = user["_id"] if user else _client_id(request)
        enforce_rate_limit(scope, subject, limit, config.RATE_LIMIT_WINDOW_SECONDS)
        return
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
        and corpus_id not in _active_jobs.values()
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
    videos = corpus.list_videos() if corpus else session.get("videos", [])
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
        "video_count": len(corpus.videos) if corpus else session.get("video_count", len(videos)),
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


def _get_session(corpus_id: Optional[str], user: Optional[dict] = None) -> dict:
    requested_id = corpus_id or DEFAULT_CORPUS_ID
    if requested_id == DEFAULT_CORPUS_ID:
        session = _load_demo_session()
    else:
        if not user:
            raise HTTPException(401, "Sign in to open your saved collections.")
        with _state["lock"]:
            doc = collection_store.get(requested_id, user["_id"])
            _prune_sessions_locked()
            session = _state["sessions"].get(requested_id)
            if not session:
                session = {**doc, "id": doc["_id"]}
                if session["status"] in {"ready", "partial"}:
                    token = _reserve_job(requested_id)
                    session["_saved_status"] = session["status"]
                    session["status"] = "processing"
                    session["progress"] = {"phase": "restoring_index"}
                    try:
                        _admit_session(session)
                        _dispatch_job(requested_id, _run_restore_job, token)
                    except Exception:
                        _release_job(token)
                        _state["sessions"].pop(requested_id, None)
                        raise
                else:
                    _admit_session(session)
            elif session.get("owner_id") != user["_id"]:
                raise HTTPException(404, "This collection is not available.")
            elif doc["status"] == "error":
                session.update({"status": "error", "error": doc.get("error"), "progress": doc.get("progress", {})})
            if session:
                session["_expires_at"] = monotonic() + config.SESSION_TTL_SECONDS
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="This collection is no longer available. Add the sources again to continue.")
    return session


def _ready_corpus(corpus_id: Optional[str], user: Optional[dict] = None) -> tuple[dict, Corpus]:
    session = _get_session(corpus_id, user)
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


def _save_session(session: dict) -> None:
    if session.get("owner_id"):
        if session.get("corpus"):
            session["videos"] = session["corpus"].list_videos()
            session["video_count"] = len(session["corpus"].videos)
        if not collection_store.update(session):
            raise RuntimeError("Collection preparation was replaced or cancelled.")


def _run_restore_job(corpus_id: str, session: Optional[dict] = None) -> None:
    session = session if session is not None else _state["sessions"][corpus_id]
    try:
        records = collection_store.videos(corpus_id, session["owner_id"])
        corpus = restore_corpus(session["name"], records, session.get("skipped"), session.get("sources"), session.get("warnings"))
        with _state["lock"]:
            session.update({"corpus": corpus, "status": session.pop("_saved_status", "ready"), "progress": {"phase": "ready"}})
            session.pop("_restore_failed", None)
            session.pop("error", None)
    except Exception:
        with _state["lock"]:
            session.update({"status": "error", "_restore_failed": True,
                            "error": "Could not restore this collection. Retry to reopen your saved videos."})


def _run_corpus_job(corpus_id: str, session: Optional[dict] = None) -> None:
    session = session if session is not None else _state["sessions"][corpus_id]
    heartbeat_stopped = threading.Event()

    def heartbeat():
        while not heartbeat_stopped.wait(20):
            try:
                if not collection_store.heartbeat(session):
                    return
            except Exception:
                pass  # A later progress write still enforces the generation.

    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        with _state["lock"]:
            session["progress"] = {"phase": "resolving_sources"}
        _save_session(session)
        resolved = resolve_sources(session["inputs"])
        with _state["lock"]:
            session["sources"] = resolved.sources
            session["warnings"] = resolved.warnings
            session["progress"] = {"phase": "sources_resolved", "current": len(resolved.specs), "total": len(resolved.specs)}
        _save_session(session)
        if not resolved.specs:
            raise RuntimeError("No usable YouTube videos were found. Check the links and try again.")

        def progress(data: dict) -> None:
            with _state["lock"]:
                session["progress"] = data
            _save_session(session)

        saved = {record["video_id"]: record["transcript"] for record in collection_store.videos(corpus_id, session["owner_id"])}

        corpus = build_corpus(
            session["name"],
            resolved.specs,
            sources=resolved.sources,
            warnings=resolved.warnings,
            progress=progress,
            transcript_loader=lambda video_id: saved.get(video_id),
            video_saved=lambda record: collection_store.save_video(session, record),
            cache_namespace=f"private-{corpus_id}",
            persist_cache=False,
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
        _save_session(session)
        try:
            collection_store.prune_videos(session)
        except Exception:
            pass  # The successful manifest still prevents indexing old records.
    except Exception as exc:
        # A terminal write can commit even when its acknowledgement is lost.
        try:
            completed = collection_store.get(corpus_id, session["owner_id"])
            if completed["job_id"] == session["job_id"] and completed["status"] in {"ready", "partial"} and session.get("corpus"):
                with _state["lock"]:
                    session.update({"status": completed["status"], "progress": completed.get("progress", {}), "error": None})
                return
        except Exception:
            pass
        with _state["lock"]:
            message = "No usable captions were found. Check your source links or retry later." if isinstance(exc, RuntimeError) else "Collection preparation failed. Please retry later."
            session.update({"status": "error", "error": message, "progress": {"phase": "error"}})
        try:
            _save_session(session)
        except Exception:
            pass  # Mongo outage: the durable lease eventually marks it interrupted.
    finally:
        heartbeat_stopped.set()


def _validated_question(mode: str, question: str) -> str:
    if mode == "question" and not question.strip():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Enter a question before asking the collection.")
    return action_question(mode, question)


# --- Corpus lifecycle --------------------------------------------------------
@app.post("/api/corpora", status_code=status.HTTP_202_ACCEPTED)
def api_create_corpus(req: CorpusCreateRequest, request: Request):
    user = require_verified_user(request)
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
        "owner_id": user["_id"],
        "name": req.name.strip(),
        "inputs": inputs,
        "status": "processing",
        "sources": [],
        "warnings": [],
        "progress": {"phase": "queued"},
        "_expires_at": monotonic() + config.SESSION_TTL_SECONDS,
    }
    with _state["lock"]:
        token = _reserve_job(corpus_id)
        try:
            _admit_session(session)
            collection_store.create(session)
        except Exception:
            _state["sessions"].pop(corpus_id, None)
            _release_job(token)
            raise
    _dispatch_job(corpus_id, _run_corpus_job, token)
    return _session_payload(session)


@app.get("/api/corpora/{corpus_id}")
def api_corpus_status(corpus_id: str, request: Request):
    return _session_payload(_get_session(corpus_id, optional_user(request)))


@app.get("/api/corpora")
def api_list_corpora(request: Request):
    user = require_user(request)
    docs = collection_store.list(user["_id"])
    return {"collections": [_session_payload({**doc, "id": doc["_id"]}) for doc in docs]}


@app.delete("/api/corpora/{corpus_id}")
def api_delete_corpus(corpus_id: str, request: Request):
    user = require_user(request)
    with _state["lock"]:
        collection_store.get(corpus_id, user["_id"], for_delete=True)
        if corpus_id in _active_jobs.values():
            raise HTTPException(409, "Wait for collection preparation to finish before deleting it.")
        collection_store.delete(corpus_id, user["_id"])
        _state["sessions"].pop(corpus_id, None)
    return {"ok": True}


@app.post("/api/corpora/{corpus_id}/retry", status_code=202)
def api_retry_corpus(corpus_id: str, request: Request):
    user = require_verified_user(request)
    _check_rate_limit(request, "ingest", config.INGEST_RATE_LIMIT)
    with _state["lock"]:
        doc = collection_store.get(corpus_id, user["_id"])
        if corpus_id in _active_jobs.values():
            raise HTTPException(409, "This collection is already being prepared.")
        existing = _state["sessions"].get(corpus_id)
        token = _reserve_job(corpus_id)
        try:
            if doc["status"] in {"ready", "partial"} and existing and existing.get("_restore_failed"):
                session = existing
                session.update({"_saved_status": doc["status"], "status": "processing",
                                "error": None, "progress": {"phase": "restoring_index"}})
                _admit_session(session)
                target = _run_restore_job
            else:
                session = {**doc, "id": doc["_id"]}
                _admit_session(session)
                doc = collection_store.claim_retry(corpus_id, user["_id"])
                session.update(doc)
                target = _run_corpus_job
        except Exception:
            _release_job(token)
            if existing:
                _state["sessions"][corpus_id] = existing
            else:
                _state["sessions"].pop(corpus_id, None)
            raise
    _dispatch_job(corpus_id, target, token)
    return _session_payload(session)


# Backwards-compatible endpoint for scripts and the original frontend.
@app.get("/api/corpus")
def api_corpus():
    return _session_payload(_get_session(DEFAULT_CORPUS_ID))


# --- Internal developer API --------------------------------------------------
@app.get("/api/eval")
def api_eval():
    if not config.ENABLE_DEVELOPER_API:
        raise HTTPException(404, "Not found")
    runs = sorted(glob.glob(str(config.RUNS_DIR / "run_*.json")))
    if not runs:
        return {"error": "No eval runs found. Run: python -m vidsense.eval.runner"}
    return json.loads(Path(runs[-1]).read_text(encoding="utf-8"))


# --- Research ---------------------------------------------------------------
@app.post("/api/research")
def api_research(req: ResearchRequest, request: Request):
    _check_rate_limit(request, "research", config.RESEARCH_RATE_LIMIT)
    session, _corpus = _ready_corpus(req.corpus_id, optional_user(request))
    if req.baseline and not config.ENABLE_DEVELOPER_API:
        raise HTTPException(403, "Comparison mode is available only in developer installations.")
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


async def _research_stream(
    request: Request,
    question: str = Query(default="", max_length=config.MAX_QUESTION_CHARS),
    corpus_id: Optional[str] = None,
    mode: Literal["question", "summary", "highlights"] = "question",
):
    """Stream the grounded agent while keeping the chosen corpus isolated."""
    # MongoDB lookups and index setup are blocking. Keep them off the event
    # loop that serves the SSE connection and the rest of the site.
    await asyncio.to_thread(_check_rate_limit, request, "research", config.RESEARCH_RATE_LIMIT)
    user = await asyncio.to_thread(optional_user, request)
    session, _corpus = await asyncio.to_thread(_ready_corpus, corpus_id, user)
    prepared_question = _validated_question(mode, question)
    if not _research_slots.acquire(blocking=False):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="VidSense is handling other requests. Please try again shortly.")
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=64)
    stopped = threading.Event()

    def emit(event):
        if not stopped.is_set() and not loop.is_closed():
            future = asyncio.run_coroutine_threadsafe(queue.put(event), loop)
            try:
                future.result(timeout=5)
            except Exception:
                future.cancel()
                stopped.set()

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
                emit({"event": "answer", "data": {"answer": events["answer"]}})
                events = [{"event": "done", "data": events}]
            for event in events:
                if stopped.is_set():
                    break
                if event["event"] == "done":
                    event["data"]["cost_usd"] = metrics.cost_usd(event["data"]["usage"], MODEL)
                    event["data"]["mode"] = mode
                emit(event)
        except Exception as exc:
            emit({"event": "error", "data": {"message": _public_model_error(exc)}})
        finally:
            _research_slots.release()
            emit(None)

    threading.Thread(target=worker, daemon=True).start()

    async def event_gen():
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield {"event": event["event"], "data": json.dumps(event["data"], default=str)}
        finally:
            stopped.set()

    return EventSourceResponse(event_gen())


@app.post("/api/research/stream")
async def api_research_stream_post(req: ResearchRequest, request: Request):
    if req.baseline:
        raise HTTPException(422, "Streaming comparison is not supported.")
    return await _research_stream(request, req.question, req.corpus_id, req.mode)


@app.get("/api/research/stream")
async def api_research_stream_legacy(
    request: Request, question: str = Query(default="", max_length=config.MAX_QUESTION_CHARS),
    corpus_id: Optional[str] = None,
    mode: Literal["question", "summary", "highlights"] = "question",
):
    if not config.ENABLE_DEVELOPER_API:
        raise HTTPException(405, "Use POST /api/research/stream.")
    return await _research_stream(request, question, corpus_id, mode)


@app.get("/api/live")
def live():
    return {"ok": True}


@app.get("/api/ready")
def ready():
    if not FRONTEND_DIR.exists() or not Path(DEFAULT_MANIFEST).exists() or not _provider_configured():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="VidSense is not fully configured.")
    if config.MONGODB_URI:
        get_database().command("ping")
    return {"ok": True, "provider": config.DEFAULT_PROVIDER, "model": MODEL}


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "provider": config.DEFAULT_PROVIDER,
        "model": MODEL,
        "max_playlist_videos": config.MAX_PLAYLIST_VIDEOS,
        "temporary_session_minutes": config.SESSION_TTL_SECONDS // 60,
        "collection_storage": "mongodb" if config.MONGODB_URI else "demo_only",
    }


# Mounted last so /api/* remains reachable.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
