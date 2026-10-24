import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import api
from tests.mongo_support import MongoApiTestCase
from vidsense.actions import action_question
from vidsense.corpus import Corpus, VideoRecord, _cache_key


class FakeAgent:
    def run(self, question):
        return {
            "question": question,
            "answer": "Grounded answer.",
            "citations": [],
            "trajectory": [],
            "tool_log": [],
            "verification": {},
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "llm_calls": 0},
            "latency": 0.01,
            "num_tool_calls": 0,
        }


class InlineThread:
    def __init__(self, target, args=(), daemon=None):
        self.target, self.args = target, args

    def start(self):
        self.target(*self.args)


class CustomerApiTests(MongoApiTestCase):
    def setUp(self):
        super().setUp()

    def test_action_prompts_select_customer_modes(self):
        self.assertIn("overview", action_question("summary").lower())
        self.assertIn("five", action_question("highlights").lower())
        self.assertEqual(action_question("question", "Where is attention covered?"), "Where is attention covered?")

    def test_cache_key_changes_when_source_metadata_changes(self):
        first = [{"video_id": "abc12345678", "title": "One", "source_url": "https://youtu.be/abc12345678"}]
        second = [{"video_id": "abc12345678", "title": "Two", "source_url": "https://youtu.be/abc12345678"}]
        self.assertNotEqual(_cache_key("collection", first), _cache_key("collection", second))

    def test_collection_lifecycle_and_scoped_research(self):
        def complete_job(corpus_id, _snapshot=None):
            session = api._state["sessions"][corpus_id]
            session["corpus"] = Corpus(
                "Test collection",
                {"abc12345678": VideoRecord("abc12345678", "Test video", [])},
                None,
            )
            session["status"] = "ready"
            session["progress"] = {"phase": "ready", "current": 1, "total": 1}
            api._save_session(session)

        with patch.object(api, "_run_corpus_job", complete_job), patch.object(api, "JobThread", InlineThread), patch.object(api, "_agent_for", return_value=FakeAgent()), patch.object(api, "_baseline_for", return_value=FakeAgent()):
            created = self.client.post("/api/corpora", json={"name": "Test collection", "inputs": ["https://youtu.be/abc12345678"]})
            self.assertEqual(created.status_code, 202)
            corpus_id = created.json()["corpus_id"]
            status = self.client.get(f"/api/corpora/{corpus_id}")
            self.assertEqual(status.status_code, 200)
            self.assertEqual(status.json()["status"], "ready")
            response = self.client.post("/api/research", json={"corpus_id": corpus_id, "mode": "summary"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["mode"], "summary")

    def test_missing_collection_returns_404(self):
        response = self.client.get("/api/corpora/expired")
        self.assertEqual(response.status_code, 404)
        research = self.client.post("/api/research", json={"corpus_id": "expired", "question": "Question"})
        self.assertEqual(research.status_code, 404)

    def test_demo_collection_returns_processing_state_without_blocking(self):
        with patch.object(api, "JobThread", InlineThread):
            # Do not execute the real embedding load; this verifies the API can
            # immediately provide a lifecycle state while it runs in production.
            with patch.object(api, "_run_demo_job", return_value=None):
                demo = api._load_demo_session()
        self.assertEqual(demo["status"], "processing")
        self.assertEqual(demo["progress"]["phase"], "loading_demo")

    def test_status_payload_marks_ready_and_caption_skipped_videos(self):
        corpus = Corpus(
            "Status collection",
            {"abc12345678": VideoRecord("abc12345678", "Ready video", [])},
            None,
            skipped=[{"video_id": "missing0001", "title": "No captions", "status": "skipped", "reason": "captions unavailable"}],
        )
        payload = api._session_payload({"id": "status", "name": corpus.name, "status": "partial", "corpus": corpus})
        self.assertEqual([item["status"] for item in payload["video_statuses"]], ["ready", "skipped"])

    def test_liveness_and_readiness_contracts(self):
        self.assertEqual(self.client.get("/api/live").json(), {"ok": True})
        with patch.object(api, "_provider_configured", return_value=True):
            ready = self.client.get("/api/ready")
        self.assertEqual(ready.status_code, 200)
        self.assertTrue(ready.json()["ok"])

    def test_ingestion_rate_limit_returns_retry_after(self):
        with patch.object(api.config, "INGEST_RATE_LIMIT", 1):
            first = self.client.post("/api/corpora", json={"inputs": ["not-a-link"]})
            second = self.client.post("/api/corpora", json={"inputs": ["not-a-link"]})
        self.assertEqual(first.status_code, 202)
        self.assertEqual(second.status_code, 429)
        self.assertIn("Retry-After", second.headers)

    def test_research_question_length_is_bounded(self):
        response = self.client.post("/api/research", json={"question": "x" * (api.config.MAX_QUESTION_CHARS + 1)})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
