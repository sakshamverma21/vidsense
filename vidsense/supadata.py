"""Small, rate-aware client for VidSense's managed YouTube access.

The free Supadata plan permits one request per second. Playlist resolution and
transcript ingestion share this client so they also share one process-wide
throttle instead of independently producing avoidable 429 responses.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from time import monotonic, sleep

from . import config

_request_lock = threading.Lock()
_last_request_at = 0.0


def get_json(path: str, params: dict) -> dict:
    """Return one authenticated JSON response with a bounded 429 retry."""
    global _last_request_at
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"https://api.supadata.ai{path}?{query}",
        headers={
            "x-api-key": config.get_supadata_api_key(),
            "accept": "application/json",
            "user-agent": "VidSense/2.0",
        },
    )

    for attempt in range(2):
        try:
            with _request_lock:
                wait_for = config.SUPADATA_MIN_INTERVAL_SECONDS - (monotonic() - _last_request_at)
                if wait_for > 0:
                    sleep(wait_for)
                try:
                    with urllib.request.urlopen(request, timeout=config.TRANSCRIPT_TIMEOUT_SECONDS) as response:
                        payload = json.loads(response.read().decode("utf-8"))
                finally:
                    _last_request_at = monotonic()
            if not isinstance(payload, dict):
                raise RuntimeError("The managed video service returned an unexpected response.")
            return payload
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt == 0:
                sleep(config.SUPADATA_MIN_INTERVAL_SECONDS)
                continue
            if exc.code in {401, 403}:
                raise RuntimeError("The managed video service rejected its API key.") from exc
            if exc.code == 429:
                raise RuntimeError("The managed video service has reached its usage limit.") from exc
            raise RuntimeError(f"The managed video service returned HTTP {exc.code}.") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError("The managed video service could not be reached.") from exc

    raise RuntimeError("The managed video service could not complete the request.")
