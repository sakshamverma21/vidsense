from datetime import timedelta
from unittest.mock import patch
import threading

from fastapi.testclient import TestClient

import api
from tests.mongo_support import MongoApiTestCase
from vidsense import config, database
from vidsense.corpus import Corpus, VideoRecord
from vidsense.storage import collection_store
from pymongo.errors import ConnectionFailure


class CollectionSecurityTests(MongoApiTestCase):
    def create_saved(self):
        doc = {"id": "private-collection", "owner_id": "alice", "name": "Alice collection",
               "inputs": ["https://youtu.be/abc12345678"], "sources": [], "warnings": [],
               "status": "ready", "progress": {"phase": "ready"}}
        collection_store.create(doc)
        record = VideoRecord("abc12345678", "Private video", [{"text": "Saved evidence", "start": 0, "duration": 2}])
        collection_store.save_video(doc, record)
        doc.update({"videos": [{"video_id": record.video_id, "title": record.title}], "video_count": 1})
        collection_store.update(doc)
        return doc

    def test_anonymous_import_is_rejected_before_any_ingestion(self):
        client = TestClient(api.app, headers={"X-VidSense-Request": "1"})
        with patch.object(api, "_run_corpus_job") as job:
            response = client.post("/api/corpora", json={"inputs": ["https://youtu.be/abc12345678"]})
        self.assertEqual(response.status_code, 401)
        job.assert_not_called()

    def test_unverified_user_cannot_import(self):
        client = self.client_for("unverified", verified=False)
        self.assertEqual(client.post("/api/corpora", json={"inputs": ["https://youtu.be/abc12345678"]}).status_code, 403)

    def test_other_user_cannot_read_query_stream_retry_or_delete(self):
        self.create_saved()
        bob = self.client_for("bob")
        path = "/api/corpora/private-collection"
        self.assertEqual(bob.get(path).status_code, 404)
        self.assertEqual(bob.delete(path).status_code, 404)
        self.assertEqual(bob.post(path + "/retry").status_code, 404)
        request = {"corpus_id": "private-collection", "question": "Tell me about it"}
        self.assertEqual(bob.post("/api/research", json=request).status_code, 404)
        self.assertEqual(bob.post("/api/research/stream", json=request).status_code, 404)
        self.assertEqual(bob.get("/api/corpora").json()["collections"], [])

    def test_restart_restores_transcripts_without_refetching(self):
        self.create_saved()
        api._state["sessions"].clear()
        record = VideoRecord("abc12345678", "Private video", [{"text": "Saved evidence", "start": 0, "duration": 2}])
        restored = Corpus("Alice collection", {record.video_id: record}, None)
        from tests.test_customer_api import InlineThread
        with patch.object(api, "JobThread", InlineThread), patch.object(api, "restore_corpus", return_value=restored) as rebuild, patch.object(api, "build_corpus") as ingest:
            response = self.client.get("/api/corpora/private-collection")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ready")
        self.assertEqual(rebuild.call_args.args[1][0]["transcript"][0]["text"], "Saved evidence")
        ingest.assert_not_called()

    def test_collection_limit_durable_and_released_after_delete(self):
        self.create_saved()
        with patch.object(config, "MAX_COLLECTIONS_PER_USER", 1):
            response = self.client.post("/api/corpora", json={"inputs": ["https://youtu.be/abc12345678"]})
        self.assertEqual(response.status_code, 429)
        self.assertEqual(self.client.delete("/api/corpora/private-collection").status_code, 200)
        self.assertEqual(self.db.collection_videos.count_documents({}), 0)
        self.assertEqual(self.db.users.find_one({"_id": "alice"})["collection_ids"], [])

    def test_expired_processing_lease_marks_interruption(self):
        self.create_saved()
        self.db.collections.update_one({"_id": "private-collection"}, {"$set": {
            "status": "processing", "lease_until": database.utcnow() - timedelta(seconds=1),
        }})
        response = self.client.get("/api/corpora/private-collection")
        self.assertEqual(response.json()["status"], "error")
        self.assertIn("interrupted", response.json()["error"])

    def test_csrf_marker_and_origin_are_required_for_browser_mutations(self):
        client = TestClient(api.app)
        self.assertEqual(client.post("/api/auth/logout").status_code, 403)
        response = self.client.post("/api/auth/logout", headers={"Origin": "https://attacker.example"})
        self.assertEqual(response.status_code, 403)

    def test_developer_api_and_get_stream_are_disabled_by_default(self):
        self.assertEqual(self.client.get("/api/eval").status_code, 404)
        self.assertEqual(self.client.get("/api/research/stream").status_code, 405)

    def test_persistent_rate_limit_is_not_reset_with_api_memory(self):
        with patch.object(config, "INGEST_RATE_LIMIT", 1), patch.object(api, "JobThread"):
            first = self.client.post("/api/corpora", json={"inputs": ["not-a-link"]})
            api._rate_buckets.clear()
            second = self.client.post("/api/corpora", json={"inputs": ["not-a-link"]})
        self.assertEqual(first.status_code, 202)
        self.assertEqual(second.status_code, 429)

    def test_forwarded_header_does_not_choose_rate_limit_identity(self):
        from starlette.requests import Request
        request = Request({"type": "http", "headers": [(b"x-forwarded-for", b"fake-user")], "client": ("127.0.0.2", 1234)})
        self.assertEqual(api._client_id(request), "127.0.0.2")

    def test_authenticated_post_stream_returns_answer_and_done_events(self):
        doc = self.create_saved()
        record = VideoRecord("abc12345678", "Learning", [])
        doc["corpus"] = Corpus("Alice collection", {record.video_id: record}, None)
        api._state["sessions"][doc["id"]] = doc
        from tests.test_customer_api import FakeAgent
        with patch.object(api, "_baseline_for", return_value=FakeAgent()):
            response = self.client.post("/api/research/stream", json={"corpus_id": doc["id"], "mode": "summary"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        self.assertIn("event: answer", response.text)
        self.assertIn("event: done", response.text)
        self.assertIn("Grounded answer.", response.text)

    def test_failed_index_restoration_can_retry_without_new_ingestion(self):
        self.create_saved()
        from tests.test_customer_api import InlineThread
        with patch.object(api, "JobThread", InlineThread), patch.object(api, "restore_corpus", side_effect=RuntimeError("temporary error")):
            response = self.client.get("/api/corpora/private-collection")
        self.assertEqual(response.json()["status"], "error")
        restored = Corpus("Alice collection", {"abc12345678": VideoRecord("abc12345678", "Video", [])}, None)
        with patch.object(api, "JobThread", InlineThread), patch.object(api, "restore_corpus", return_value=restored), patch.object(api, "_run_corpus_job") as ingest:
            response = self.client.post("/api/corpora/private-collection/retry")
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["status"], "ready")
        ingest.assert_not_called()

    def test_old_worker_cannot_write_after_retry_claim(self):
        original = self.create_saved()
        self.db.collections.update_one({"_id": original["id"]}, {"$set": {"status": "error"}})
        newer = collection_store.claim_retry(original["id"], "alice")
        original["status"] = "ready"
        self.assertFalse(collection_store.update(original))
        with self.assertRaises(RuntimeError):
            collection_store.save_video(original, VideoRecord("another0001", "Stale", []))
        current = self.db.collections.find_one({"_id": original["id"]})
        self.assertEqual(current["job_id"], newer["job_id"])
        self.assertEqual(current["status"], "processing")

    def test_partial_deletion_can_resume_and_releases_quota_once(self):
        self.create_saved()
        with patch.object(self.db.collection_videos, "delete_many", side_effect=ConnectionFailure("temporary")):
            failed = self.client.delete("/api/corpora/private-collection")
        self.assertEqual(failed.status_code, 503)
        self.assertEqual(self.db.collections.find_one({"_id": "private-collection"})["status"], "deleting")
        self.assertEqual(self.client.get("/api/corpora/private-collection").status_code, 404)
        self.assertEqual(self.client.post("/api/corpora/private-collection/retry").status_code, 404)
        self.assertEqual(self.client.delete("/api/corpora/private-collection").status_code, 200)
        self.assertEqual(self.db.collection_videos.count_documents({}), 0)
        self.assertEqual(self.db.users.find_one({"_id": "alice"})["collection_ids"], [])

    def test_lost_insert_acknowledgement_does_not_undercount(self):
        insert = self.db.collections.insert_one
        def committed_but_failed_ack(doc):
            insert(doc)
            raise ConnectionFailure("acknowledgement lost")
        doc = {"id": "committed", "owner_id": "alice", "name": "Saved",
               "inputs": [], "status": "processing", "sources": [], "warnings": [], "progress": {}}
        with patch.object(config, "MAX_COLLECTIONS_PER_USER", 1), patch.object(
            self.db.collections, "insert_one", side_effect=committed_but_failed_ack
        ):
            collection_store.create(doc)
            with self.assertRaises(Exception) as error:
                collection_store.create({**doc, "id": "second"})
        self.assertEqual(error.exception.status_code, 429)
        self.assertEqual(self.db.collections.count_documents({"owner_id": "alice"}), 1)
        self.assertEqual(self.db.users.find_one({"_id": "alice"})["collection_ids"], ["committed"])

    def test_deletion_during_restore_does_not_release_a_worker_slot(self):
        self.create_saved()
        api._job_slots = threading.BoundedSemaphore(1)
        with patch.object(api, "JobThread"):
            response = self.client.get("/api/corpora/private-collection")
            self.assertEqual(response.json()["status"], "processing")
            self.assertEqual(self.client.delete("/api/corpora/private-collection").status_code, 409)
            with self.assertRaises(Exception) as error:
                api._reserve_job("other")
            self.assertEqual(error.exception.status_code, 429)
        for token in list(api._active_jobs):
            api._release_job(token)

    def test_retry_uses_cache_admission_and_has_expiry(self):
        doc = self.create_saved()
        self.db.collections.update_one({"_id": doc["id"]}, {"$set": {"status": "error"}})
        api._state["sessions"]["old"] = {"id": "old", "status": "ready", "_expires_at": 999999999}
        with patch.object(config, "MAX_COLLECTIONS", 1), patch.object(api, "JobThread"):
            response = self.client.post("/api/corpora/private-collection/retry")
        self.assertEqual(response.status_code, 202)
        self.assertNotIn("old", api._state["sessions"])
        self.assertIn("_expires_at", api._state["sessions"]["private-collection"])
        for token in list(api._active_jobs):
            api._release_job(token)

    def test_completed_snapshot_excludes_transcripts_from_an_earlier_attempt(self):
        doc = self.create_saved()
        self.db.collection_videos.insert_one({"collection_id": doc["id"], "owner_id": "alice", "video_id": "old-video", "title": "Old", "transcript": []})
        self.assertEqual([record["video_id"] for record in collection_store.videos(doc["id"], "alice")], ["abc12345678"])

    def test_only_one_restore_retry_for_a_collection_is_admitted(self):
        doc = self.create_saved()
        api._state["sessions"][doc["id"]] = {**doc, "status": "error", "_restore_failed": True}
        with patch.object(api, "JobThread"):
            first = self.client.post("/api/corpora/private-collection/retry")
            second = self.client.post("/api/corpora/private-collection/retry")
        self.assertEqual(first.status_code, 202)
        self.assertEqual(second.status_code, 409)
        for token in list(api._active_jobs):
            api._release_job(token)

    def test_unacknowledged_creation_is_recoverable_without_releasing_its_slot(self):
        doc = {"id": "pending", "owner_id": "alice", "name": "Interrupted",
               "inputs": [], "status": "processing", "sources": [], "warnings": [], "progress": {}}
        with patch.object(self.db.collections, "insert_one", side_effect=ConnectionFailure("connection dropped")):
            with self.assertRaises(ConnectionFailure):
                collection_store.create(doc)
        recovered = collection_store.list("alice")
        self.assertEqual(recovered[0]["_id"], "pending")
        self.assertEqual(recovered[0]["status"], "error")
        self.assertEqual(self.db.users.find_one({"_id": "alice"})["collection_ids"], ["pending"])
        with patch.object(config, "MAX_COLLECTIONS_PER_USER", 1):
            with self.assertRaises(Exception) as error:
                collection_store.create({**doc, "id": "second"})
        self.assertEqual(error.exception.status_code, 429)

    def test_lost_final_status_acknowledgement_preserves_ready_state(self):
        from tests.test_customer_api import InlineThread
        from vidsense.sources import ResolvedSources
        record = VideoRecord("abc12345678", "Saved", [{"text": "Evidence", "start": 0, "duration": 1}])
        collection = Corpus("Collection", {record.video_id: record}, None)
        update = self.db.collections.update_one
        lost = False
        def write_then_lose_ack(query, data, **kwargs):
            nonlocal lost
            result = update(query, data, **kwargs)
            if data.get("$set", {}).get("status") == "ready" and not lost:
                lost = True
                raise ConnectionFailure("ack lost")
            return result
        def build(*args, **kwargs):
            kwargs["video_saved"](record)
            return collection
        with patch.object(api, "JobThread", InlineThread), patch.object(
            api, "resolve_sources", return_value=ResolvedSources(specs=[{"video_id": record.video_id}])
        ), patch.object(api, "build_corpus", side_effect=build), patch.object(
            self.db.collections, "update_one", side_effect=write_then_lose_ack
        ):
            response = self.client.post("/api/corpora", json={"inputs": ["https://youtu.be/abc12345678"]})
        self.assertTrue(lost)
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["status"], "ready")
