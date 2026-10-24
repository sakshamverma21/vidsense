"""Shared MongoDB connection, indexes, and atomic fixed-window request limits."""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from functools import lru_cache

from fastapi import HTTPException
from pymongo import MongoClient, ReturnDocument
from pymongo.errors import PyMongoError

from . import config


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def ensure_indexes(db) -> None:
    db.users.create_index("email", unique=True)
    db.auth_sessions.create_index("expires_at", expireAfterSeconds=0)
    db.auth_sessions.create_index("user_id")
    db.auth_tokens.create_index("expires_at", expireAfterSeconds=0)
    db.auth_tokens.create_index([("user_id", 1), ("purpose", 1)])
    db.rate_limits.create_index("expires_at", expireAfterSeconds=0)
    db.collections.create_index([("owner_id", 1), ("created_at", -1)])
    db.collection_videos.create_index([("collection_id", 1), ("video_id", 1)], unique=True)
    db.collection_videos.create_index("owner_id")


@lru_cache(maxsize=1)
def _connect(uri: str, name: str):
    client = MongoClient(
        uri, tz_aware=True, serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000, socketTimeoutMS=10000, maxPoolSize=20,
        appname="vidsense",
    )
    try:
        client.admin.command("ping")
        db = client[name]
        ensure_indexes(db)
        return db
    except Exception:
        client.close()
        raise


def get_database():
    if not config.MONGODB_URI:
        raise HTTPException(503, "Accounts and saved collections are not configured yet.")
    try:
        return _connect(config.MONGODB_URI, config.MONGODB_DATABASE)
    except (PyMongoError, ValueError):
        # Connection errors can include credentials/host information. Never
        # return the original exception text to the browser.
        raise HTTPException(503, "The account service is temporarily unavailable.") from None


def enforce_rate_limit(scope: str, subject: str, limit: int, window_seconds: int = 300) -> None:
    now = utcnow()
    window = int(now.timestamp()) // window_seconds
    digest = hashlib.sha256(subject.encode()).hexdigest()
    key = f"{scope}:{digest}:{window}"
    result = get_database().rate_limits.find_one_and_update(
        {"_id": key},
        {"$inc": {"count": 1}, "$setOnInsert": {
            "expires_at": datetime.fromtimestamp((window + 2) * window_seconds, timezone.utc),
        }},
        upsert=True, return_document=ReturnDocument.AFTER,
    )
    if result["count"] > limit:
        retry = max(1, int((window + 1) * window_seconds - now.timestamp()))
        raise HTTPException(429, "Too many requests. Please try again shortly.", headers={"Retry-After": str(retry)})
