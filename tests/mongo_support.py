import hashlib
import base64
import unittest
import threading
from datetime import timedelta
from unittest.mock import patch

import mongomock
from fastapi.testclient import TestClient

import api
from vidsense import config, database


class MongoApiTestCase(unittest.TestCase):
    """A real auth/session boundary over a fresh Mongo-compatible test DB."""
    def setUp(self):
        self.db = mongomock.MongoClient(tz_aware=True).vidsense_test
        database.ensure_indexes(self.db)
        for replacement in (
            patch.object(config, "MONGODB_URI", "mongodb://unused-test-host"),
            patch.object(database, "_connect", return_value=self.db),
            patch.object(config, "APP_URL", "http://testserver"),
            patch.object(config, "AUTH_COOKIE_SECURE", False),
        ):
            replacement.start()
            self.addCleanup(replacement.stop)
        api._state["sessions"].clear()
        api._rate_buckets.clear()
        api._active_jobs.clear()
        api._job_slots = threading.BoundedSemaphore(config.MAX_CONCURRENT_INGESTIONS)
        self.client = self.client_for("alice")

    def client_for(self, user_id, verified=True):
        token = base64.urlsafe_b64encode(hashlib.sha256(f"test-session-for-{user_id}".encode()).digest()).decode().rstrip("=")
        self.db.users.update_one({"_id": user_id}, {"$setOnInsert": {
            "email": f"{user_id}@example.com", "name": user_id.title(),
            "email_verified": verified, "auth_version": 0, "collection_count": 0,
        }}, upsert=True)
        self.db.auth_sessions.replace_one({"_id": hashlib.sha256(token.encode()).hexdigest()}, {
            "_id": hashlib.sha256(token.encode()).hexdigest(), "user_id": user_id,
            "auth_version": 0, "expires_at": database.utcnow() + timedelta(days=1),
        }, upsert=True)
        client = TestClient(api.app, headers={"X-VidSense-Request": "1"})
        client.cookies.set(config.AUTH_COOKIE_NAME, token)
        return client
