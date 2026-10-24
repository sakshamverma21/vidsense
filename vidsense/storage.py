"""Owner-scoped collection metadata and per-video transcript persistence."""
from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
from functools import wraps
import threading
import uuid

from fastapi import HTTPException
from pymongo import ReturnDocument

from . import config
from .database import get_database, utcnow


_lifecycle_lock = threading.RLock()


def serialized(method):
    @wraps(method)
    def call(*args, **kwargs):
        with _lifecycle_lock:
            return method(*args, **kwargs)
    return call


class CollectionStore:
    def _recover_pending(self, db, owner: str) -> None:
        account = db.users.find_one({"_id": owner}) or {}
        for corpus_id, original in account.get("pending_collections", {}).items():
            # Recovery inserts the same predetermined ID. A delayed original
            # insert cannot create a second document or overwrite this marker.
            recovered = {**original, "status": "error", "progress": {"phase": "error"},
                         "error": "Preparation was interrupted. Retry to prepare this collection."}
            db.collections.update_one({"_id": corpus_id, "owner_id": owner}, {"$setOnInsert": recovered}, upsert=True)
            db.users.update_one({"_id": owner}, {"$unset": {f"pending_collections.{corpus_id}": ""}})

    @serialized
    def create(self, session: dict) -> None:
        db = get_database()
        owner = session["owner_id"]
        self._recover_pending(db, owner)
        # Reservation IDs, unlike an increment/decrement counter, can be
        # reconciled after a lost acknowledgement without double counting.
        account = db.users.find_one({"_id": owner})
        if not account:
            raise HTTPException(401, "Please sign in again.")
        reserved_ids = account.get("collection_ids", [])
        actual_ids = {doc["_id"] for doc in db.collections.find({"owner_id": owner}, {"_id": 1})}
        missing = list(actual_ids - set(reserved_ids))
        if missing:
            db.users.update_one({"_id": owner}, {"$addToSet": {"collection_ids": {"$each": missing}}})
        orphaned = [key for key in reserved_ids if key not in actual_ids]
        if orphaned:
            db.users.update_one({"_id": owner}, {"$pull": {"collection_ids": {"$in": orphaned}}})
        now = utcnow()
        session["job_id"] = uuid.uuid4().hex
        doc = {key: session[key] for key in ("name", "inputs", "status", "sources", "warnings", "progress", "owner_id")}
        doc.update({"_id": session["id"], "job_id": session["job_id"], "created_at": now, "updated_at": now,
                    "lease_until": now + timedelta(seconds=config.JOB_LEASE_SECONDS)})
        reserved = db.users.find_one_and_update(
            {"_id": owner, "$expr": {"$lt": [
                {"$size": {"$ifNull": ["$collection_ids", []]}}, config.MAX_COLLECTIONS_PER_USER,
            ]}},
            {"$addToSet": {"collection_ids": session["id"]},
             "$set": {f"pending_collections.{session['id']}": doc}}, return_document=ReturnDocument.AFTER,
        )
        if not reserved:
            raise HTTPException(429, "Your collection limit is reached. Delete a collection before adding another.")
        try:
            db.collections.insert_one(doc)
            db.users.update_one({"_id": owner}, {"$unset": {f"pending_collections.{session['id']}": ""}})
        except Exception:
            # A network failure may happen after the insert committed. The
            # pending descriptor retains enough metadata for safe recovery.
            committed = db.collections.find_one({"_id": session["id"], "owner_id": owner})
            if committed:
                session["job_id"] = committed["job_id"]
                return
            raise

    @serialized
    def get(self, corpus_id: str, owner_id: str, for_delete: bool = False) -> dict:
        doc = get_database().collections.find_one({"_id": corpus_id, "owner_id": owner_id})
        if not doc or (doc["status"] == "deleting" and not for_delete):
            raise HTTPException(404, "This collection is not available.")
        if doc["status"] == "processing" and doc.get("lease_until", utcnow()) < utcnow():
            get_database().collections.update_one(
                {"_id": corpus_id, "owner_id": owner_id, "status": "processing", "lease_until": {"$lt": utcnow()}},
                {"$set": {"status": "error", "error": "Preparation was interrupted. Retry to continue from saved videos.",
                          "progress": {"phase": "error"}, "updated_at": utcnow()}},
            )
            doc = get_database().collections.find_one({"_id": corpus_id, "owner_id": owner_id})
        return doc

    @serialized
    def list(self, owner_id: str) -> list[dict]:
        self._recover_pending(get_database(), owner_id)
        return list(get_database().collections.find({"owner_id": owner_id}).sort("created_at", -1).limit(100))

    @serialized
    def update(self, session: dict) -> bool:
        keys = ("name", "status", "sources", "warnings", "progress", "error", "skipped", "videos", "video_count")
        payload = {key: session[key] for key in keys if key in session}
        payload.update({"updated_at": utcnow(), "lease_until": utcnow() + timedelta(seconds=config.JOB_LEASE_SECONDS)})
        result = get_database().collections.update_one(
            {"_id": session["id"], "owner_id": session["owner_id"], "job_id": session["job_id"],
             "status": {"$in": ["processing", session["status"]]}}, {"$set": payload},
        )
        return bool(result.matched_count)

    @serialized
    def heartbeat(self, session: dict) -> bool:
        return bool(get_database().collections.update_one(
            {"_id": session["id"], "owner_id": session["owner_id"], "job_id": session["job_id"], "status": "processing"},
            {"$set": {"lease_until": utcnow() + timedelta(seconds=config.JOB_LEASE_SECONDS)}},
        ).matched_count)

    @serialized
    def save_video(self, session: dict, record) -> None:
        db = get_database()
        # Deletion during preparation must not recreate orphaned transcripts.
        if not db.collections.find_one({"_id": session["id"], "owner_id": session["owner_id"],
                                       "job_id": session["job_id"], "status": {"$in": ["processing", session["status"]]}}, {"_id": 1}):
            raise RuntimeError("Collection no longer exists.")
        doc = asdict(record)
        doc.update({"owner_id": session["owner_id"], "collection_id": session["id"]})
        db.collection_videos.replace_one(
            {"collection_id": session["id"], "video_id": record.video_id, "owner_id": session["owner_id"]},
            doc, upsert=True,
        )

    @serialized
    def videos(self, corpus_id: str, owner_id: str) -> list[dict]:
        doc = self.get(corpus_id, owner_id)
        query = {"collection_id": corpus_id, "owner_id": owner_id}
        if doc["status"] in {"ready", "partial"}:
            selected = {video["video_id"] for video in doc.get("videos", [])}
            query["video_id"] = {"$in": list(selected)}
        return list(get_database().collection_videos.find(query))

    @serialized
    def prune_videos(self, session: dict) -> None:
        db = get_database()
        doc = db.collections.find_one({"_id": session["id"], "owner_id": session["owner_id"],
                                       "job_id": session["job_id"], "status": {"$in": ["ready", "partial"]}})
        if doc:
            selected = [video["video_id"] for video in doc.get("videos", [])]
            db.collection_videos.delete_many({"collection_id": session["id"], "owner_id": session["owner_id"],
                                              "video_id": {"$nin": selected}})

    @serialized
    def delete(self, corpus_id: str, owner_id: str) -> None:
        db = get_database()
        doc = self.get(corpus_id, owner_id, for_delete=True)
        # Keep deletion separate from a running thread in this increment. A
        # durable queue with cancellation is a later step.
        if doc["status"] == "processing":
            raise HTTPException(409, "Wait for collection preparation to finish before deleting it.")
        tombstone = db.collections.find_one_and_update(
            {"_id": corpus_id, "owner_id": owner_id, "status": {"$ne": "processing"}},
            {"$set": {"status": "deleting", "updated_at": utcnow(),
                      "error": "Deletion is incomplete. Delete this collection again to finish."}},
            return_document=ReturnDocument.AFTER,
        )
        if not tombstone:
            raise HTTPException(409, "Collection preparation is in progress.")
        # Every cleanup step is idempotent. Retain the tombstone until both
        # transcript cleanup and quota release have been acknowledged.
        db.collection_videos.delete_many({"collection_id": corpus_id, "owner_id": owner_id})
        db.users.update_one({"_id": owner_id}, {"$pull": {"collection_ids": corpus_id}})
        db.collections.delete_one({"_id": corpus_id, "owner_id": owner_id, "status": "deleting"})

    @serialized
    def claim_retry(self, corpus_id: str, owner_id: str) -> dict:
        self.get(corpus_id, owner_id)
        doc = get_database().collections.find_one_and_update(
            {"_id": corpus_id, "owner_id": owner_id, "status": "error"},
            {"$set": {"status": "processing", "error": None, "progress": {"phase": "queued"},
                      "job_id": uuid.uuid4().hex,
                      "lease_until": utcnow() + timedelta(seconds=config.JOB_LEASE_SECONDS)}},
            return_document=ReturnDocument.AFTER,
        )
        if not doc:
            raise HTTPException(409, "Only failed collections can be retried.")
        return doc


collection_store = CollectionStore()
