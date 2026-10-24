"""Exercise accounts against local MongoDB + Mailpit without paid API calls.

Run after `docker compose up -d mongo mailpit`. A uniquely named, disposable
database is created and removed; no Atlas or configured production URI is used.
"""
from __future__ import annotations

import secrets
import os
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from pymongo import MongoClient

import api
from vidsense import auth, config
from vidsense.corpus import Corpus, VideoRecord


class InlineThread:
    def __init__(self, target, args=(), daemon=None):
        self.target, self.args = target, args

    def start(self):
        self.target(*self.args)


def main():
    # This helper deliberately cannot be pointed at a production database.
    mongo_host = os.getenv("VIDSENSE_SMOKE_MONGO_HOST", "127.0.0.1")
    mail_host = os.getenv("VIDSENSE_SMOKE_MAIL_HOST", "127.0.0.1")
    if mongo_host not in {"127.0.0.1", "mongo"} or mail_host not in {"127.0.0.1", "mailpit"}:
        raise ValueError("Smoke tests support only local or Compose dependency hosts.")
    uri = f"mongodb://{mongo_host}:27017"
    name = "vidsense_smoke_" + uuid.uuid4().hex
    connection = MongoClient(uri, serverSelectionTimeoutMS=5000)
    connection.admin.command("ping")
    settings = {
        "MONGODB_URI": uri, "MONGODB_DATABASE": name,
        "APP_URL": "http://testserver", "AUTH_COOKIE_SECURE": False,
        "SMTP_HOST": mail_host, "SMTP_PORT": 1025,
        "SMTP_FROM": "VidSense smoke test <noreply@localhost>",
        "SMTP_STARTTLS": False, "SMTP_SSL": False,
        "SMTP_USERNAME": "", "SMTP_PASSWORD": "",
    }
    try:
        with patch.multiple(config, **settings), patch.object(auth, "_send_token_email", wraps=auth._send_token_email) as mail:
            client = TestClient(api.app, headers={"X-VidSense-Request": "1", "Origin": "http://testserver"})
            password = secrets.token_urlsafe(24)
            email = f"smoke-{uuid.uuid4().hex}@example.com"
            assert client.post("/api/auth/register", json={"email": email, "password": password, "name": "Local smoke"}).status_code == 200
            assert mail.call_count == 1
            token = mail.call_args.args[1]
            assert client.post("/api/auth/verify-email", json={"token": token}).status_code == 200
            assert client.post("/api/auth/login", json={"email": email, "password": password}).status_code == 200
            assert client.get("/api/auth/session").json()["user"]["email_verified"]
            print("PASS: sign-up, real SMTP delivery to Mailpit, verification, login")

            record = VideoRecord("abc12345678", "Smoke video", [{"text": "Locally saved evidence", "start": 0.0, "duration": 3.0}])
            corpus = Corpus("Smoke collection", {record.video_id: record}, None)

            def build(*args, **kwargs):
                kwargs["video_saved"](record)
                return corpus

            from vidsense.sources import ResolvedSources
            resolved = ResolvedSources(specs=[{"video_id": record.video_id, "title": record.title}])
            with patch.object(api, "JobThread", InlineThread), patch.object(api, "resolve_sources", return_value=resolved), patch.object(api, "build_corpus", side_effect=build):
                response = client.post("/api/corpora", json={"name": corpus.name, "inputs": ["https://youtu.be/abc12345678"]})
            assert response.status_code == 202
            corpus_id = response.json()["corpus_id"]
            assert client.get("/api/corpora").json()["collections"][0]["corpus_id"] == corpus_id
            api._state["sessions"].clear()
            with patch.object(api, "JobThread", InlineThread), patch.object(api, "restore_corpus", return_value=corpus) as restore:
                reopened = client.get(f"/api/corpora/{corpus_id}")
            assert reopened.json()["status"] == "ready"
            assert restore.call_args.args[1][0]["transcript"][0]["text"] == "Locally saved evidence"
            assert client.delete(f"/api/corpora/{corpus_id}").status_code == 200
            assert client.get("/api/corpora").json()["collections"] == []
            print("PASS: real MongoDB collection persistence, index restoration hook, deletion")

            assert client.post("/api/auth/forgot-password", json={"email": email}).status_code == 200
            reset_token = mail.call_args.args[1]
            new_password = secrets.token_urlsafe(24)
            assert client.post("/api/auth/reset-password", json={"token": reset_token, "password": new_password}).status_code == 200
            assert client.get("/api/auth/session").json()["user"] is None
            assert client.post("/api/auth/login", json={"email": email, "password": new_password}).status_code == 200
            assert client.post("/api/auth/logout").status_code == 200
            assert client.get("/api/auth/session").json()["user"] is None
            print("PASS: password recovery, existing-session revocation, logout")
    finally:
        # Only the randomly named database created above is removed.
        assert name.startswith("vidsense_smoke_") and len(name) == 47
        connection.drop_database(name)
        connection.close()
        print("Removed the disposable local smoke-test database.")


if __name__ == "__main__":
    main()
