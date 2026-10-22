"""Central configuration for VidSense.

This module is the single place that loads the environment and constructs the
LLM / embedding objects, so the rest of the package never touches os.environ
directly. Importing it calls load_dotenv() exactly once.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Windows consoles default to cp1252, which raises UnicodeEncodeError on the
# emoji we log. Make stdout/stderr UTF-8 tolerant so logging never crashes.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# --- Paths -------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
CACHE_DIR = PROJECT_ROOT / ".vidsense_cache"          # FAISS stores, transcripts
EVAL_DIR = PACKAGE_DIR / "eval"                        # corpus.json, tasks.json, runs
RUNS_DIR = EVAL_DIR / "runs"                           # persisted eval trajectories

for _d in (CACHE_DIR, RUNS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Load .env from the project root (idempotent).
load_dotenv(PROJECT_ROOT / ".env")

# --- Model defaults ----------------------------------------------------------
DEFAULT_PROVIDER = os.getenv("VIDSENSE_PROVIDER", "gemini").strip().lower()
DEFAULT_GROQ_MODEL = os.getenv("VIDSENSE_GROQ_MODEL", "openai/gpt-oss-20b")
# Gemini's current API model for new projects. Older 2.x/3.5 identifiers may
# remain configured in existing environments, but can hang while the provider
# retries an unavailable model instead of returning a useful response.
DEFAULT_GEMINI_MODEL = os.getenv("VIDSENSE_GEMINI_MODEL", "gemini-3.6-flash")
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b-instruct-q4_0"
DEFAULT_TEMPERATURE = 0.2


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _env_float(name: str, default: float, minimum: float = 1.0) -> float:
    try:
        return max(minimum, float(os.getenv(name, str(default))))
    except ValueError:
        return default

# Customer-facing ingestion safeguard. A playlist can otherwise turn a single
# click into hours of transcript fetching and a very large local index. It is
# intentionally configurable for self-hosted installations.
MAX_PLAYLIST_VIDEOS = _env_int("VIDSENSE_MAX_PLAYLIST_VIDEOS", 50)
MAX_INPUT_SOURCES = _env_int("VIDSENSE_MAX_INPUT_SOURCES", 100)
MAX_QUESTION_CHARS = _env_int("VIDSENSE_MAX_QUESTION_CHARS", 2000)
MAX_COLLECTIONS = _env_int("VIDSENSE_MAX_COLLECTIONS", 20)
MAX_CONCURRENT_INGESTIONS = _env_int("VIDSENSE_MAX_CONCURRENT_INGESTIONS", 2)
MAX_CONCURRENT_RESEARCH = _env_int("VIDSENSE_MAX_CONCURRENT_RESEARCH", 2)
SESSION_TTL_SECONDS = _env_int("VIDSENSE_SESSION_TTL_SECONDS", 7200, minimum=300)
RATE_LIMIT_WINDOW_SECONDS = _env_int("VIDSENSE_RATE_LIMIT_WINDOW_SECONDS", 300, minimum=10)
RESEARCH_RATE_LIMIT = _env_int("VIDSENSE_RESEARCH_RATE_LIMIT", 20)
INGEST_RATE_LIMIT = _env_int("VIDSENSE_INGEST_RATE_LIMIT", 4)
LLM_TIMEOUT_SECONDS = _env_float("VIDSENSE_LLM_TIMEOUT_SECONDS", 45.0, minimum=5.0)
LLM_MAX_RETRIES = _env_int("VIDSENSE_LLM_MAX_RETRIES", 1, minimum=0)
TRANSCRIPT_PROVIDER = os.getenv("VIDSENSE_TRANSCRIPT_PROVIDER", "auto").strip().lower()
TRANSCRIPT_TIMEOUT_SECONDS = _env_float("VIDSENSE_TRANSCRIPT_TIMEOUT_SECONDS", 90.0, minimum=10.0)
SUPADATA_MIN_INTERVAL_SECONDS = _env_float(
    "VIDSENSE_SUPADATA_MIN_INTERVAL_SECONDS", 1.1, minimum=0.05
)
# FastEmbed's upstream default is 256 documents per batch. That can exceed a
# small container's memory limit for multi-video collections because attention
# activations grow with both batch size and transcript length.
EMBEDDING_BATCH_SIZE = _env_int("VIDSENSE_EMBEDDING_BATCH_SIZE", 16)

# The default model id for a given provider.
DEFAULT_MODEL = {
    "groq": DEFAULT_GROQ_MODEL,
    "gemini": DEFAULT_GEMINI_MODEL,
    "ollama": DEFAULT_OLLAMA_MODEL,
}


def get_groq_api_key() -> str:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Put it in a .env file at the project root "
            "(see .env.example)."
        )
    return key


def get_google_api_key() -> str:
    key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey "
            "and add it to your .env file (see .env.example)."
        )
    return key


def get_tavily_api_key() -> str:
    key = os.getenv("TAVILY_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Sign up at https://tavily.com and add it "
            "to your .env file (see .env.example)."
        )
    return key


def get_supadata_api_key() -> str:
    key = os.getenv("SUPADATA_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "SUPADATA_API_KEY is not set. Cloud-hosted YouTube ingestion needs "
            "a managed transcript provider because YouTube commonly blocks data-center IPs."
        )
    return key


def has_google() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY", "").strip())


def has_groq() -> bool:
    return bool(os.getenv("GROQ_API_KEY", "").strip())


def has_tavily() -> bool:
    return bool(os.getenv("TAVILY_API_KEY", "").strip())


def has_supadata() -> bool:
    return bool(os.getenv("SUPADATA_API_KEY", "").strip())


# --- Factories ---------------------------------------------------------------
def make_rate_limiter(requests_per_minute: float):
    """A shared client-side rate limiter to respect free-tier RPM caps.

    Pass ONE instance to every LLM in a run (agent + tools + judge) so they share
    the budget — Gemini/Groq free tiers rate-limit per project+model, not per
    client object.
    """
    from langchain_core.rate_limiters import InMemoryRateLimiter

    return InMemoryRateLimiter(
        requests_per_second=requests_per_minute / 60.0,
        check_every_n_seconds=0.2,
        max_bucket_size=2,
    )


def make_llm(
    model_type: str = DEFAULT_PROVIDER,
    model_name: str | None = None,
    temperature: float | None = None,
    rate_limiter=None,
):
    """Build a chat LLM. Supports 'gemini' (default), 'groq', and 'ollama'."""
    model_type = (model_type or DEFAULT_PROVIDER).lower()
    temperature = DEFAULT_TEMPERATURE if temperature is None else temperature

    if model_type == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            google_api_key=get_google_api_key(),
            model=model_name or DEFAULT_GEMINI_MODEL,
            temperature=temperature,
            retries=LLM_MAX_RETRIES,
            request_timeout=LLM_TIMEOUT_SECONDS,
            rate_limiter=rate_limiter,
        )

    if model_type == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            api_key=get_groq_api_key(),
            model=model_name or DEFAULT_GROQ_MODEL,
            temperature=temperature,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=LLM_MAX_RETRIES,
            rate_limiter=rate_limiter,
        )

    from langchain_community.llms import Ollama

    return Ollama(model=model_name or DEFAULT_OLLAMA_MODEL, temperature=temperature)


@lru_cache(maxsize=1)
def get_embeddings():
    """Shared embedding model (cached so we load it once per process).

    The model is cached under a fixed dir so it can be pre-baked into a Docker
    image (warm cold starts) and reused deterministically.
    """
    from langchain_community.embeddings import FastEmbedEmbeddings

    cache_dir = os.getenv("FASTEMBED_CACHE_DIR", str(CACHE_DIR / "fastembed"))
    return FastEmbedEmbeddings(cache_dir=cache_dir, batch_size=EMBEDDING_BATCH_SIZE)
