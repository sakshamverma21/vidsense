"""Exercise real password/cookie/token boundaries with Mongo and SMTP fakes."""
import hashlib
import re
import secrets
import smtplib
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from unittest.mock import patch

import mongomock
from argon2 import Type, extract_parameters
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pymongo.errors import ConnectionFailure

from vidsense import auth, config, database


PASSWORD = "correct horse battery staple"
NEW_PASSWORD = "a completely different password"


class AuthTests(unittest.TestCase):
    def setUp(self):
        self.db = mongomock.MongoClient(tz_aware=True).auth_test
        self.db.users.create_index("email", unique=True)
        # Deliberately omit TTL indexes so expiry checks cannot rely on cleanup.
        now = database.utcnow()
        for replacement in (
            patch.object(database, "_connect", return_value=self.db),
            patch.object(database, "utcnow", return_value=now),
            patch.object(auth, "utcnow", return_value=now),
            patch.multiple(config, MONGODB_URI="mongodb://unused-test-host", APP_URL="http://testserver",
                           AUTH_COOKIE_SECURE=False, AUTH_SESSION_SECONDS=604800,
                           SMTP_HOST="smtp.example.com", SMTP_PORT=587, SMTP_FROM="auth@example.com",
                           SMTP_USERNAME="", SMTP_PASSWORD="", SMTP_STARTTLS=True, SMTP_SSL=False),
        ):
            replacement.start()
            self.addCleanup(replacement.stop)
        smtp_patch = patch.object(auth.smtplib, "SMTP")
        self.smtp_factory = smtp_patch.start()
        self.addCleanup(smtp_patch.stop)
        self.smtp = self.smtp_factory.return_value.__enter__.return_value
        self.smtp.send_message.return_value = {}
        ssl_patch = patch.object(auth.smtplib, "SMTP_SSL")
        self.ssl_factory = ssl_patch.start()
        self.addCleanup(ssl_patch.stop)
        self.ssl_factory.return_value.__enter__.return_value.send_message.return_value = {}
        self.app = FastAPI()
        self.app.include_router(auth.router)

        @self.app.get("/optional")
        def optional(request: Request):
            user = auth.optional_user(request)
            return {"id": user["_id"] if user else None}

        @self.app.get("/required")
        def required(request: Request):
            return {"id": auth.require_user(request)["_id"]}

        @self.app.get("/verified")
        def verified(request: Request):
            return {"id": auth.require_verified_user(request)["_id"]}

        self.client = self.new_client()

    def new_client(self, peer="testclient", secure=False):
        return TestClient(self.app, base_url="https://testserver" if secure else "http://testserver",
                          headers={"X-VidSense-Request": "1"}, client=(peer, 12345))

    def create_user(self, verified=False):
        user = {
            "_id": "alice", "email": "alice@example.com", "name": "Alice",
            "email_verified": verified, "auth_version": 0, "collection_count": 0,
            "password_hash": auth._password_hasher.hash(PASSWORD),
        }
        self.db.users.insert_one(user)
        return user

    def set_session(self, user_id="alice", version=0, expires_at=None):
        token = secrets.token_urlsafe(32)
        self.db.auth_sessions.insert_one({
            "_id": hashlib.sha256(token.encode()).hexdigest(), "user_id": user_id,
            "auth_version": version, "expires_at": expires_at or database.utcnow() + timedelta(days=1),
        })
        # Match the host-only cookie jar entry created by an HTTP response.
        self.client.cookies.set(config.AUTH_COOKIE_NAME, token, domain="testserver.local", path="/")
        return token

    def mail_token(self, purpose="verify"):
        message = self.smtp.send_message.call_args.args[0]
        match = re.search(rf"http://testserver/\#{purpose}=([A-Za-z0-9_-]{{43}})", message.get_content())
        self.assertIsNotNone(match)
        return match.group(1)

    def register(self, password=PASSWORD, email="alice@example.com", **extra):
        return self.client.post("/api/auth/register", json={"email": email, "password": password, **extra})

    def login(self, password=PASSWORD, email="alice@example.com", client=None):
        return (client or self.client).post("/api/auth/login", json={"email": email, "password": password})

    def forgot(self):
        response = self.client.post("/api/auth/forgot-password", json={"email": "alice@example.com"})
        self.assertEqual(response.status_code, 200)
        return self.mail_token("reset")

    def verification_token(self):
        response = self.client.post("/api/auth/resend-verification", json={"email": "alice@example.com"})
        self.assertEqual(response.status_code, 200)
        return self.mail_token("verify")

    def redeem(self, token, purpose="reset", password=NEW_PASSWORD, client=None):
        body = {"token": token}
        if purpose == "reset":
            body["password"] = password
        return (client or self.client).post(
            "/api/auth/reset-password" if purpose == "reset" else "/api/auth/verify-email", json=body,
        )

    def test_disabled_session_never_calls_database_and_clears_cookie(self):
        self.client.cookies.set(config.AUTH_COOKIE_NAME, "stale", domain="testserver.local", path="/")
        with patch.object(config, "MONGODB_URI", ""), patch.object(config, "SMTP_HOST", ""), patch.object(
            auth, "get_database", side_effect=AssertionError("unexpected database call"),
        ):
            response = self.client.get("/api/auth/session")
            self.assertEqual(response.json(), {"user": None, "enabled": False, "registration_available": False})
            self.assertIn("Max-Age=0", response.headers["set-cookie"])
            self.assertEqual(self.client.get("/optional").json(), {"id": None})

    def test_no_cookie_is_anonymous_and_required_is_401(self):
        with patch.object(auth, "get_database", side_effect=AssertionError("unexpected database call")):
            self.assertEqual(self.client.get("/optional").json(), {"id": None})
            self.assertEqual(self.client.get("/required").status_code, 401)

    def test_missing_smtp_rejects_email_requests_before_any_account_lookup(self):
        with patch.object(config, "SMTP_HOST", ""), patch.object(auth, "get_database") as get_db:
            self.assertEqual(self.register().status_code, 503)
            for path in ("forgot-password", "resend-verification"):
                response = self.client.post(f"/api/auth/{path}", json={"email": "alice@example.com"})
                self.assertEqual(response.status_code, 503)
            get_db.assert_not_called()
            self.assertFalse(self.client.get("/api/auth/session").json()["registration_available"])
        self.assertEqual(self.db.users.count_documents({}), 0)

    def test_register_normalizes_and_hashes_without_exposing_secrets(self):
        response = self.register(email=" Alice@EXAMPLE.com ", name=" Alice ")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("set-cookie", response.headers)
        user = self.db.users.find_one({"email": "alice@example.com"})
        self.assertIsInstance(user["_id"], str)
        self.assertEqual((user["name"], user["auth_version"], user["collection_count"]), ("Alice", 0, 0))
        self.assertFalse(user["email_verified"])
        self.assertTrue(auth._password_hasher.verify(user["password_hash"], PASSWORD))
        parameters = extract_parameters(user["password_hash"])
        self.assertEqual((parameters.type, parameters.memory_cost, parameters.time_cost, parameters.parallelism),
                         (Type.ID, 19456, 2, 1))
        token = self.mail_token()
        stored = self.db.auth_tokens.find_one()
        self.assertEqual(stored["_id"], hashlib.sha256(token.encode()).hexdigest())
        self.assertNotIn(token, repr(stored))
        self.assertNotIn(token, response.text)
        self.assertNotIn(PASSWORD, response.text)
        self.assertIn("no-store", response.headers["cache-control"])
        self.smtp.starttls.assert_called_once()
        self.smtp.login.assert_not_called()

    def test_duplicate_signup_never_replaces_password_and_verified_reply_is_identical(self):
        first = self.register(name="Original")
        original = self.db.users.find_one()
        second = self.register(password=NEW_PASSWORD, name="Replacement")
        self.assertEqual(second.status_code, first.status_code)
        self.assertEqual(second.json(), first.json())
        self.assertEqual(self.db.users.count_documents({}), 1)
        current = self.db.users.find_one()
        self.assertEqual(current["password_hash"], original["password_hash"])
        self.assertEqual(current["name"], "Original")
        self.db.users.update_one({"_id": current["_id"]}, {"$set": {"email_verified": True}})
        self.assertEqual(self.register(password=NEW_PASSWORD).json(), first.json())
        self.assertEqual(self.smtp.send_message.call_count, 2)

    def test_concurrent_duplicate_signups_preserve_winning_password(self):
        barrier = threading.Barrier(2)
        insert = self.db.users.insert_one
        winners = []

        def simultaneous_insert(doc):
            barrier.wait(timeout=10)
            result = insert(doc)
            winners.append(doc.copy())
            return result

        def signup(password):
            with self.new_client() as client:
                return client.post("/api/auth/register", json={"email": "alice@example.com", "password": password})

        with patch.object(self.db.users, "insert_one", side_effect=simultaneous_insert), ThreadPoolExecutor(2) as pool:
            responses = list(pool.map(signup, (PASSWORD, NEW_PASSWORD)))
        self.assertEqual([response.status_code for response in responses], [200, 200])
        self.assertEqual(responses[0].json(), responses[1].json())
        self.assertEqual(len(winners), 1)
        self.assertEqual(self.db.users.find_one()["password_hash"], winners[0]["password_hash"])

    def test_failed_smtp_is_generic_sanitized_and_registration_can_retry(self):
        def fail(message):
            raise smtplib.SMTPException(message.as_string())

        self.smtp.send_message.side_effect = fail
        with self.assertLogs(auth._logger, level="WARNING") as captured:
            failed = self.register()
        failed_token = self.mail_token()
        self.assertEqual(failed.status_code, 200)
        self.assertNotIn(failed_token, failed.text + repr(captured.output))
        self.assertNotIn(PASSWORD, failed.text + repr(captured.output))
        self.assertEqual(self.db.auth_tokens.count_documents({}), 0)
        initial_hash = self.db.users.find_one()["password_hash"]
        self.smtp.send_message.side_effect = None
        retried = self.register(password=NEW_PASSWORD)
        self.assertEqual(retried.json(), failed.json())
        self.assertEqual(self.db.users.find_one()["password_hash"], initial_hash)
        self.assertEqual(self.db.auth_tokens.count_documents({}), 1)
        self.assertNotEqual(self.mail_token(), failed_token)

    def test_smtp_recipient_refusal_discards_token(self):
        self.smtp.send_message.return_value = {"alice@example.com": (550, b"mail rejected")}
        with self.assertLogs(auth._logger, level="WARNING"):
            self.assertEqual(self.register().status_code, 200)
        self.assertEqual(self.db.auth_tokens.count_documents({}), 0)

    def test_ssl_does_not_attempt_starttls(self):
        with patch.object(config, "SMTP_SSL", True):
            self.assertEqual(self.register().status_code, 200)
        self.smtp_factory.assert_not_called()
        self.ssl_factory.assert_called_once()
        smtp = self.ssl_factory.return_value.__enter__.return_value
        smtp.starttls.assert_not_called()
        smtp.send_message.assert_called_once()

    def test_input_limits_and_validation_errors_never_echo_password_or_token(self):
        for password in ("private123", "private" * 20):
            response = self.register(password=password)
            self.assertEqual(response.status_code, 422)
            self.assertNotIn(password, response.text)
        self.assertEqual(self.register(name="x" * 81).status_code, 422)
        self.assertEqual(self.register(email="not-an-email").status_code, 422)
        response = self.client.post("/api/auth/verify-email", json={"token": "private" * 20})
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("private", response.text)
        response = self.client.post("/api/auth/login", json={"email": {"$ne": ""}, "password": PASSWORD})
        self.assertEqual(response.status_code, 422)
        self.assertNotIn(PASSWORD, response.text)
        self.assertEqual(self.db.users.count_documents({}), 0)

    def test_login_bad_credentials_are_generic(self):
        self.create_user()
        known = self.login(password="wrong password")
        unknown = self.login(password="wrong password", email="unknown@example.com")
        empty = self.login(password="")
        self.assertEqual([known.status_code, unknown.status_code, empty.status_code], [401, 401, 401])
        self.assertEqual(known.json(), unknown.json())
        self.assertNotIn("set-cookie", known.headers)
        self.assertEqual(self.db.auth_sessions.count_documents({}), 0)

    def test_unverified_login_public_allowlist_and_verified_gate(self):
        self.create_user()
        response = self.login(email="ALICE@EXAMPLE.COM")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"user": {
            "id": "alice", "email": "alice@example.com", "name": "Alice", "email_verified": False,
        }})
        self.assertEqual(self.client.get("/required").status_code, 200)
        self.assertEqual(self.client.get("/verified").status_code, 403)
        token = self.client.cookies.get(config.AUTH_COOKIE_NAME)
        stored = self.db.auth_sessions.find_one()
        self.assertEqual(stored["_id"], hashlib.sha256(token.encode()).hexdigest())
        self.assertNotIn(token, repr(stored))
        self.assertNotIn(token, response.text)
        self.assertEqual(self.client.get("/api/auth/session").json()["user"], response.json()["user"])

    def test_cookie_flags_and_session_duration_are_bounded(self):
        self.create_user()
        with patch.object(config, "AUTH_COOKIE_SECURE", True), patch.object(config, "AUTH_SESSION_SECONDS", 10**12):
            response = self.login(client=self.new_client(secure=True))
        cookie = response.headers["set-cookie"]
        for flag in ("HttpOnly", "Secure", "SameSite=lax", "Path=/", f"Max-Age={30 * 86400}"):
            self.assertIn(flag, cookie)
        stored = self.db.auth_sessions.find_one()
        self.assertEqual((stored["expires_at"] - stored["created_at"]).total_seconds(), 30 * 86400)

    def test_zero_version_fixture_and_raw_user_dependency_are_supported(self):
        self.db.users.insert_one({"_id": "alice", "email": "alice@example.com", "name": "Alice",
                                  "email_verified": True, "auth_version": 0})
        token = self.set_session()
        request = Request({"type": "http", "headers": [
            (b"cookie", f"{config.AUTH_COOKIE_NAME}={token}".encode()),
        ]})
        self.assertEqual(auth.optional_user(request)["_id"], "alice")
        self.assertEqual(self.client.get("/verified").json(), {"id": "alice"})

    def test_session_clears_expired_cookie_so_next_anonymous_flow_works(self):
        self.create_user()
        self.set_session(expires_at=database.utcnow() - timedelta(seconds=1))
        self.assertEqual(self.client.get("/optional").status_code, 401)
        response = self.client.get("/api/auth/session")
        self.assertEqual(response.json()["user"], None)
        cookie = response.headers["set-cookie"]
        for flag in ("Max-Age=0", "HttpOnly", "SameSite=lax", "Path=/"):
            self.assertIn(flag, cookie)
        self.assertEqual(self.client.get("/optional").json(), {"id": None})
        self.assertEqual(self.db.auth_sessions.count_documents({}), 1)

    def test_session_clears_malformed_and_stale_version_cookies(self):
        self.create_user()
        self.client.cookies.set(config.AUTH_COOKIE_NAME, "malformed", domain="testserver.local", path="/")
        self.assertEqual(self.client.get("/optional").status_code, 401)
        self.assertIsNone(self.client.get("/api/auth/session").json()["user"])
        self.assertEqual(self.client.get("/optional").json(), {"id": None})
        self.set_session()
        self.db.users.update_one({"_id": "alice"}, {"$inc": {"auth_version": 1}})
        self.assertEqual(self.client.get("/optional").status_code, 401)
        self.assertIn("Max-Age=0", self.client.get("/api/auth/session").headers["set-cookie"])
        self.assertEqual(self.client.get("/optional").json(), {"id": None})

    def test_session_cookie_deletion_preserves_secure_flag(self):
        client = self.new_client(secure=True)
        client.cookies.set(config.AUTH_COOKIE_NAME, "malformed", domain="testserver.local", path="/")
        with patch.object(config, "AUTH_COOKIE_SECURE", True):
            response = client.get("/api/auth/session")
        self.assertIn("Secure", response.headers["set-cookie"])
        self.assertEqual(client.get("/optional").json(), {"id": None})

    def test_deleted_user_cannot_use_a_live_session(self):
        self.create_user()
        self.set_session()
        self.db.users.delete_one({"_id": "alice"})
        self.assertEqual(self.client.get("/required").status_code, 401)

    def test_login_rotates_and_logout_revokes_and_clears_cookie(self):
        self.create_user()
        old_token = self.set_session()
        self.assertEqual(self.login().status_code, 200)
        self.assertEqual(self.db.auth_sessions.count_documents({}), 1)
        self.assertIsNone(self.db.auth_sessions.find_one({"_id": hashlib.sha256(old_token.encode()).hexdigest()}))
        response = self.client.post("/api/auth/logout")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Max-Age=0", response.headers["set-cookie"])
        self.assertEqual(self.db.auth_sessions.count_documents({}), 0)
        self.assertEqual(self.client.get("/required").status_code, 401)
        self.assertEqual(self.client.post("/api/auth/logout").status_code, 200)

    def test_verify_single_use_purpose_bound_and_updates_existing_session(self):
        self.assertEqual(self.register().status_code, 200)
        token = self.mail_token()
        self.assertEqual(self.login().status_code, 200)
        wrong = self.client.post("/api/auth/reset-password", json={"token": token, "password": NEW_PASSWORD})
        self.assertEqual(wrong.status_code, 400)
        response = self.client.post("/api/auth/verify-email", json={"token": token})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.get("/verified").status_code, 200)
        self.assertEqual(self.client.post("/api/auth/verify-email", json={"token": token}).status_code, 400)
        self.assertNotIn(token, response.text)

    def test_expired_tokens_are_rejected_even_before_ttl_cleanup(self):
        self.register()
        verify = self.mail_token()
        reset = self.forgot()
        self.db.auth_tokens.update_many({}, {"$set": {"expires_at": database.utcnow() - timedelta(seconds=1)}})
        self.assertEqual(self.client.post("/api/auth/verify-email", json={"token": verify}).status_code, 400)
        self.assertEqual(self.client.post("/api/auth/reset-password", json={
            "token": reset, "password": NEW_PASSWORD,
        }).status_code, 400)
        self.assertEqual(self.db.auth_tokens.count_documents({}), 2)
        self.assertFalse(self.db.users.find_one()["email_verified"])

    def test_email_requests_are_generic_for_missing_existing_and_verified_accounts(self):
        self.create_user()
        for path in ("resend-verification", "forgot-password"):
            missing = self.client.post(f"/api/auth/{path}", json={"email": "unknown@example.com"})
            present = self.client.post(f"/api/auth/{path}", json={"email": "alice@example.com"})
            self.assertEqual((missing.status_code, missing.json()), (present.status_code, present.json()))
        self.db.users.update_one({"_id": "alice"}, {"$set": {"email_verified": True}})
        sent = self.smtp.send_message.call_count
        response = self.client.post("/api/auth/resend-verification", json={"email": "alice@example.com"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.smtp.send_message.call_count, sent)

    def test_reset_revokes_sessions_other_tokens_and_old_password(self):
        self.register()
        self.login()
        first_token = self.forgot()
        second_token = self.forgot()
        response = self.client.post("/api/auth/reset-password", json={"token": first_token, "password": NEW_PASSWORD})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.db.users.find_one()["auth_version"], 1)
        self.assertEqual(self.db.auth_sessions.count_documents({}), 0)
        self.assertEqual(self.db.auth_tokens.count_documents({}), 0)
        self.assertEqual(self.client.get("/required").status_code, 401)
        self.assertEqual(self.login().status_code, 401)
        self.assertEqual(self.login(password=NEW_PASSWORD).status_code, 200)
        for token in (first_token, second_token):
            retry = self.client.post("/api/auth/reset-password", json={"token": token, "password": PASSWORD})
            self.assertEqual(retry.status_code, 400)
        self.assertNotIn(first_token, response.text)
        self.assertNotIn(NEW_PASSWORD, response.text)

    def test_already_issued_reset_works_during_smtp_outage(self):
        self.create_user()
        token = self.forgot()
        with patch.object(config, "SMTP_HOST", ""):
            response = self.client.post("/api/auth/reset-password", json={"token": token, "password": NEW_PASSWORD})
        self.assertEqual(response.status_code, 200)

    def _assert_concurrent_redemption(self, purpose="reset", different_tokens=False, lost_ack=False,
                                     lost_no_match_ack=False):
        self.create_user()
        issue = self.forgot if purpose == "reset" else self.verification_token
        token = issue()
        tokens = (token, issue() if different_tokens else token)
        barrier = threading.Barrier(2)
        update_user = self.db.users.update_one

        def simultaneous_update(query, update):
            barrier.wait(timeout=10)
            result = update_user(query, update)
            if lost_ack and result.matched_count:
                raise ConnectionFailure("committed but acknowledgement lost")
            if lost_no_match_ack and not result.matched_count:
                raise ConnectionFailure("losing attempt acknowledgement lost")
            return result

        def redeem(index):
            with self.new_client() as client:
                return self.redeem(tokens[index], purpose, passwords[index], client)

        passwords = (PASSWORD, NEW_PASSWORD)
        with patch.object(self.db.users, "update_one", side_effect=simultaneous_update), ThreadPoolExecutor(2) as pool:
            responses = list(pool.map(redeem, (0, 1)))
        self.assertEqual(sorted(response.status_code for response in responses), [200, 503 if lost_no_match_ack else 400])
        winner = next(password for password, response in zip(passwords, responses) if response.status_code == 200)
        user = self.db.users.find_one()
        self.assertEqual(user["auth_version"], 1 if purpose == "reset" else 0)
        self.assertTrue(auth._password_hasher.verify(user["password_hash"], winner if purpose == "reset" else PASSWORD))
        if purpose == "verify":
            self.assertTrue(user["email_verified"])

    def test_simultaneous_reset_redemption_has_one_winner(self):
        self._assert_concurrent_redemption()

    def test_different_reset_tokens_for_same_version_have_one_winner(self):
        self._assert_concurrent_redemption(different_tokens=True)

    def test_concurrent_reset_with_lost_ack_has_one_winner(self):
        self._assert_concurrent_redemption(lost_ack=True)

    def test_losing_reset_cannot_claim_another_attempts_committed_marker(self):
        self._assert_concurrent_redemption(lost_no_match_ack=True)

    def test_simultaneous_verification_has_one_winner(self):
        self._assert_concurrent_redemption(purpose="verify")

    def _assert_precommit_failure_recoverable(self, purpose):
        original = self.create_user()
        token = self.forgot() if purpose == "reset" else self.verification_token()
        with patch.object(self.db.users, "update_one", side_effect=ConnectionFailure("private server error")):
            failed = self.redeem(token, purpose)
        self.assertEqual(failed.status_code, 503)
        self.assertNotIn("private server error", failed.text)
        self.assertEqual(self.db.users.find_one(), original)
        self.assertIsNotNone(self.db.auth_tokens.find_one({"_id": hashlib.sha256(token.encode()).hexdigest()}))
        self.assertEqual(self.redeem(token, purpose).status_code, 200)
        user = self.db.users.find_one()
        self.assertEqual(user["auth_version"], 1 if purpose == "reset" else 0)
        self.assertTrue(auth._password_hasher.verify(user["password_hash"], NEW_PASSWORD if purpose == "reset" else PASSWORD))
        self.assertEqual(self.redeem(token, purpose, password=PASSWORD).status_code, 400)
        self.assertEqual(self.db.users.find_one(), user)

    def test_reset_link_survives_account_update_failure_before_commit(self):
        self._assert_precommit_failure_recoverable("reset")

    def test_verification_link_survives_account_update_failure_before_commit(self):
        self._assert_precommit_failure_recoverable("verify")

    def _assert_lost_ack_reconciled(self, purpose):
        self.create_user()
        self.set_session()
        token = self.forgot() if purpose == "reset" else self.verification_token()
        update_user = self.db.users.update_one

        def committed_then_disconnected(query, update):
            result = update_user(query, update)
            self.assertEqual(result.matched_count, 1)
            raise ConnectionFailure(f"private server error {token} {NEW_PASSWORD}")

        with patch.object(self.db.users, "update_one", side_effect=committed_then_disconnected):
            response = self.redeem(token, purpose)
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(token, response.text)
        self.assertNotIn(NEW_PASSWORD, response.text)
        user = self.db.users.find_one()
        self.assertEqual(user["auth_version"], 1 if purpose == "reset" else 0)
        self.assertTrue(auth._password_hasher.verify(user["password_hash"], NEW_PASSWORD if purpose == "reset" else PASSWORD))
        self.assertEqual(self.db.auth_tokens.count_documents({}), 0)
        self.assertEqual(self.client.get("/required").status_code, 401 if purpose == "reset" else 200)
        self.assertEqual(self.redeem(token, purpose, password=PASSWORD).status_code, 400)
        self.assertEqual(self.db.users.find_one(), user)

    def test_reset_reconciles_account_update_committed_but_acknowledgement_lost(self):
        self._assert_lost_ack_reconciled("reset")

    def test_verification_reconciles_account_update_committed_but_acknowledgement_lost(self):
        self._assert_lost_ack_reconciled("verify")

    def test_failed_reconciliation_read_preserves_unchanged_account_retry(self):
        original = self.create_user()
        token = self.forgot()
        with patch.object(self.db.users, "update_one", side_effect=ConnectionFailure("write unavailable")), patch.object(
            self.db.users, "find_one", side_effect=ConnectionFailure("read unavailable"),
        ):
            self.assertEqual(self.redeem(token).status_code, 503)
        self.assertEqual(self.db.users.find_one(), original)
        self.assertEqual(self.redeem(token).status_code, 200)
        self.assertEqual(self.db.users.find_one()["auth_version"], 1)

    def test_unresolved_committed_write_cannot_be_replayed_with_a_new_password(self):
        self.create_user()
        self.set_session()
        token = self.forgot()
        update_user = self.db.users.update_one

        def committed_then_disconnected(query, update):
            update_user(query, update)
            raise ConnectionFailure("acknowledgement lost")

        with patch.object(self.db.users, "update_one", side_effect=committed_then_disconnected), patch.object(
            self.db.users, "find_one", side_effect=ConnectionFailure("reconciliation unavailable"),
        ):
            self.assertEqual(self.redeem(token).status_code, 503)
        user = self.db.users.find_one()
        self.assertEqual(user["auth_version"], 1)
        self.assertTrue(auth._password_hasher.verify(user["password_hash"], NEW_PASSWORD))
        self.assertEqual(self.db.auth_tokens.count_documents({}), 1)
        self.assertEqual(self.redeem(token, password=PASSWORD).status_code, 400)
        self.assertEqual(self.db.users.find_one(), user)
        self.assertEqual(self.client.get("/required").status_code, 401)

    def test_late_original_reset_cannot_overwrite_successful_retry(self):
        self.create_user()
        token = self.forgot()
        delayed = []

        def delay_update(query, update):
            delayed.append((query, update))
            raise ConnectionFailure("write outcome not yet known")

        with patch.object(self.db.users, "update_one", side_effect=delay_update):
            self.assertEqual(self.redeem(token).status_code, 503)
        self.assertEqual(self.redeem(token, password=PASSWORD).status_code, 200)
        result = self.db.users.update_one(*delayed[0])
        self.assertEqual(result.matched_count, 0)
        user = self.db.users.find_one()
        self.assertEqual(user["auth_version"], 1)
        self.assertTrue(auth._password_hasher.verify(user["password_hash"], PASSWORD))

    def test_failed_reset_cleanup_cannot_reopen_token_or_restore_sessions(self):
        self.create_user()
        self.set_session()
        token = self.forgot()
        with patch.object(self.db.auth_sessions, "delete_many", side_effect=ConnectionFailure("cleanup error")), self.assertLogs(
            auth._logger, level="WARNING",
        ):
            self.assertEqual(self.redeem(token).status_code, 200)
        user = self.db.users.find_one()
        self.assertEqual(self.db.auth_tokens.count_documents({}), 1)
        self.assertEqual(self.db.auth_sessions.count_documents({}), 1)
        self.assertEqual(self.client.get("/required").status_code, 401)
        self.assertEqual(self.redeem(token, password=PASSWORD).status_code, 400)
        self.assertEqual(self.db.users.find_one(), user)

    def test_failed_verification_cleanup_cannot_reopen_token(self):
        self.create_user()
        token = self.verification_token()
        with patch.object(self.db.auth_tokens, "delete_one", side_effect=ConnectionFailure("cleanup error")), self.assertLogs(
            auth._logger, level="WARNING",
        ):
            self.assertEqual(self.redeem(token, "verify").status_code, 200)
        user = self.db.users.find_one()
        self.assertTrue(user["email_verified"])
        self.assertEqual(user["auth_version"], 0)
        self.assertEqual(self.db.auth_tokens.count_documents({}), 1)
        self.assertEqual(self.redeem(token, "verify").status_code, 400)
        self.assertEqual(self.db.users.find_one(), user)

    def test_reset_during_password_verification_prevents_session_creation(self):
        self.create_user()
        hasher = auth._password_hasher

        def verify_then_reset(encoded, password):
            result = hasher.verify(encoded, password)
            self.db.users.update_one({"_id": "alice"}, {"$inc": {"auth_version": 1},
                                     "$set": {"password_hash": hasher.hash(NEW_PASSWORD)}})
            return result

        with patch.object(auth, "_password_hasher", wraps=hasher) as wrapper:
            wrapper.verify.side_effect = verify_then_reset
            self.assertEqual(self.login().status_code, 401)
        self.assertEqual(self.db.auth_sessions.count_documents({}), 0)

    def test_reset_before_racing_login_insert_leaves_an_unusable_session(self):
        self.create_user()
        insert = self.db.auth_sessions.insert_one

        def reset_then_insert(doc):
            self.db.users.update_one({"_id": "alice"}, {"$inc": {"auth_version": 1},
                                     "$set": {"password_hash": auth._password_hasher.hash(NEW_PASSWORD)}})
            self.db.auth_sessions.delete_many({"user_id": "alice"})
            return insert(doc)

        with patch.object(self.db.auth_sessions, "insert_one", side_effect=reset_then_insert):
            self.assertEqual(self.login().status_code, 200)
        self.assertEqual(self.db.auth_sessions.find_one()["auth_version"], 0)
        self.assertEqual(self.client.get("/required").status_code, 401)
        self.assertIsNone(self.client.get("/api/auth/session").json()["user"])

    def test_reset_cleanup_preserves_new_version_login(self):
        self.create_user()
        self.set_session()
        token = self.forgot()
        delete = self.db.auth_sessions.delete_many

        def new_login_then_cleanup(query):
            self.db.auth_sessions.insert_one({"_id": "new-session", "user_id": "alice", "auth_version": 1,
                                              "expires_at": database.utcnow() + timedelta(days=1)})
            return delete(query)

        with patch.object(self.db.auth_sessions, "delete_many", side_effect=new_login_then_cleanup):
            response = self.client.post("/api/auth/reset-password", json={"token": token, "password": NEW_PASSWORD})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(list(self.db.auth_sessions.find()), [{
            "_id": "new-session", "user_id": "alice", "auth_version": 1,
            "expires_at": self.db.auth_sessions.find_one()["expires_at"],
        }])

    def test_per_account_rate_limit_survives_changing_peer_addresses(self):
        statuses = []
        for i in range(6):
            response = self.login(password="wrong", client=self.new_client(peer=f"192.0.2.{i}"))
            statuses.append(response.status_code)
        self.assertEqual(statuses, [401] * 5 + [429])
        self.assertIn("retry-after", response.headers)

    def test_per_ip_rate_limit_ignores_spoofed_forwarded_headers(self):
        statuses = []
        for i in range(21):
            response = self.client.post("/api/auth/login", json={"email": f"person{i}@example.com", "password": "wrong"},
                                        headers={"X-Forwarded-For": f"192.0.2.{i}", "X-Real-IP": f"192.0.2.{i}"})
            statuses.append(response.status_code)
        self.assertEqual(statuses, [401] * 20 + [429])
        self.assertIn("retry-after", response.headers)
        peer_digest = hashlib.sha256(b"testclient").hexdigest()
        self.assertEqual(self.db.rate_limits.count_documents({"_id": {"$regex": f"^auth:login:ip:{peer_digest}:"}}), 1)

    def test_database_errors_are_sanitized_without_clearing_potentially_valid_cookie(self):
        self.create_user()
        self.set_session()
        with patch.object(self.db.auth_sessions, "find_one", side_effect=ConnectionFailure("secret credentials")):
            response = self.client.get("/api/auth/session")
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("secret credentials", response.text)
        self.assertNotIn("set-cookie", response.headers)


if __name__ == "__main__":
    unittest.main()
