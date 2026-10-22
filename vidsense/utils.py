"""Small pure helpers shared across the package.

These were originally private methods on YouTubeRAGSystem; factored out so the
corpus layer, the tools, and the citation formatter share one implementation.
"""
from __future__ import annotations

from typing import List, Optional
from urllib.parse import urlparse, parse_qs


def to_text(resp) -> str:
    """Normalize an LLM response (or its .content) to a plain string.

    Newer 'thinking' models (e.g. Gemini 3.x) return message content as a LIST of
    parts rather than a str; downstream regex/parsing needs a string. Handles a
    message object, a raw string, or a list of str/dict parts.
    """
    content = getattr(resp, "content", resp)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                parts.append(p.get("text", "") or "")
            else:
                parts.append(str(getattr(p, "text", "") or ""))
        return "".join(parts)
    return "" if content is None else str(content)


def extract_video_id(url: str) -> Optional[str]:
    """Extract a YouTube video ID from the common URL shapes (or return a bare id)."""
    url = url.strip()
    # Already a bare 11-char id.
    if "/" not in url and "?" not in url and len(url) == 11:
        return url

    parsed = urlparse(url)

    if parsed.hostname in ("youtu.be", "www.youtu.be"):
        return parsed.path.lstrip("/") or None

    if parsed.hostname in ("youtube.com", "www.youtube.com", "m.youtube.com"):
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]
        if parsed.path.startswith(("/embed/", "/v/", "/shorts/")):
            parts = parsed.path.split("/")
            return parts[2] if len(parts) > 2 else None

    return None


def extract_playlist_id(url: str) -> Optional[str]:
    """Return a YouTube playlist id from canonical or playlist-context URLs.

    YouTube's Share action frequently creates ``/watch?v=...&list=...`` URLs.
    Treating the explicit ``list`` parameter as a playlist makes those links as
    useful as the canonical ``/playlist?list=...`` form in the customer UI.
    """
    parsed = urlparse(url.strip())
    if parsed.hostname not in ("youtube.com", "www.youtube.com", "m.youtube.com"):
        return None
    if parsed.path not in ("/playlist", "/watch"):
        return None
    return parse_qs(parsed.query).get("list", [None])[0]


def is_youtube_url(url: str) -> bool:
    """Whether *url* is a supported public YouTube video or playlist URL."""
    parsed = urlparse(url.strip())
    return parsed.hostname in {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "youtu.be",
        "www.youtu.be",
    }


def format_timestamp(seconds: float) -> str:
    """Seconds -> HH:MM:SS (or MM:SS when under an hour)."""
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def youtube_link(video_id: str, timestamp: float) -> str:
    """Deep link into a video at a given second."""
    if not video_id:
        return ""
    return f"https://www.youtube.com/watch?v={video_id}&t={int(timestamp)}s"


def find_timestamp_for_position(transcript_entries: List[dict], char_position: int) -> float:
    """Map a character offset in the joined transcript back to a start time.

    transcript_entries: list of {"text": str, "start": float} dicts.
    """
    if not transcript_entries:
        return 0.0
    accumulated = 0
    for entry in transcript_entries:
        accumulated += len(entry["text"]) + 1  # +1 for the joining space
        if accumulated >= char_position:
            return entry["start"]
    return transcript_entries[-1]["start"]
