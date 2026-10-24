"""Cookie authentication backed by MongoDB and SMTP.

Mutation CSRF/origin checks belong to the application's middleware. Mount this
router behind that middleware, and call the synchronous dependencies from a
worker thread when using them in async endpoints. No raw tokens are persisted.

Configured SMTP delivery failures deliberately retain the generic public reply
to avoid an account-existence oracle. Failed tokens are discarded; registration
can be retried, or verification/reset mail requested again. An existing signup
never changes the stored password, including after a failed mail delivery.

Redemption commits in one conditional user update. Tokens remain available on
pre-commit failures; version/verification checks prevent reuse after a commit,
even if cleanup fails. Per-attempt markers reconcile lost acknowledgements.
"""
from __future__ import annotations

import hashlib
import logging
import re
import secrets
import smtplib
import ssl
import uuid
import threading
from contextlib import contextmanager
from datetime import timedelta
from email.message import EmailMessage
from urllib.parse import urlsplit

from argon2 import PasswordHasher, Type
from argon2.exceptions import InvalidHashError, VerificationError
from email_validator import EmailNotValidError, validate_email
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field, SecretStr, field_validator
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError, PyMongoError

from . import config
from .database import enforce_rate_limit, get_database, utcnow

_logger = logging.getLogger(__name__)
_password_hasher = PasswordHasher(
    type=Type.ID, memory_cost=19456, time_cost=2, parallelism=1,
)
# Unknown accounts still pay the same password-verification cost as known ones.
_dummy_hash = _password_hasher.hash(secrets.token_urlsafe(32))
_password_slots = threading.BoundedSemaphore(2)
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_-]{43}\Z")
_MAX_SESSION_SECONDS = 30 * 24 * 60 * 60
_VERIFY_SECONDS = 24 * 60 * 60
_RESET_SECONDS = 30 * 60
_VERIFICATION_REPLY = {
    "message": "If this address needs verification, check your inbox for the next steps. "
    "You can request another email if needed.",
}
_RESET_REPLY = {
    "message": "If an account exists for this address, check your inbox for password reset instructions.",
}
_MAIL_UNAVAILABLE = "Email delivery is temporarily unavailable. Please try again later."


class _AuthRoute(APIRoute):
    def get_route_handler(self):
        original = super().get_route_handler()

        async def handle(request: Request):
            try:
                response = await original(request)
            except RequestValidationError:
                # Pydantic's normal errors can echo plaintext password/token input.
                raise HTTPException(
                    422, "Invalid authentication request.", headers={"Cache-Control": "no-store"},
                ) from None
            except HTTPException as exc:
                exc.headers = {**(exc.headers or {}), "Cache-Control": "no-store"}
                raise
            response.headers["Cache-Control"] = "no-store"
            return response

        return handle


router = APIRouter(prefix="/api/auth", route_class=_AuthRoute)


@contextmanager
def _password_work():
    # Per-IP limits alone cannot bound memory under a multi-client burst.
    if not _password_slots.acquire(blocking=False):
        raise HTTPException(429, "The account service is busy. Please try again shortly.", headers={"Retry-After": "2"})
    try:
        yield
    finally:
        _password_slots.release()


class EmailRequest(BaseModel):
    email: str = Field(min_length=1, max_length=320)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        try:
            # Account email identity is case-insensitive, without provider-specific
            # dot/plus rewriting. DNS isn't needed to validate a login address.
            return validate_email(value.strip(), check_deliverability=False).normalized.lower()
        except EmailNotValidError:
            raise ValueError("Enter a valid email address.") from None


class RegisterRequest(EmailRequest):
    password: SecretStr = Field(min_length=15, max_length=128)
    name: str | None = Field(default=None, max_length=80)


class LoginRequest(EmailRequest):
    password: SecretStr = Field(max_length=128)


class TokenRequest(BaseModel):
    token: SecretStr = Field(max_length=128)


class ResetPasswordRequest(TokenRequest):
    password: SecretStr = Field(min_length=15, max_length=128)


@contextmanager
def _database():
    try:
        yield get_database()
    except PyMongoError:
        raise HTTPException(503, "The account service is temporarily unavailable.") from None


def _rate_limit(request: Request, action: str, ip_limit: int, account: str | None = None,
                account_limit: int = 5) -> None:
    # The socket peer is authoritative here; forwarded headers are untrusted.
    peer = request.client.host if request.client else "unknown"
    try:
        enforce_rate_limit(f"auth:{action}:ip", peer, ip_limit, window_seconds=300)
        if account is not None:
            _account_limit(action, account, account_limit)
    except PyMongoError:
        raise HTTPException(503, "The account service is temporarily unavailable.") from None


def _account_limit(action: str, account: str, limit: int) -> None:
    # The database helper also hashes subjects; never pass a bearer token here.
    enforce_rate_limit(f"auth:{action}:account", account, limit, window_seconds=300)


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def _invalid_session() -> HTTPException:
    return HTTPException(401, "Please sign in again to continue.")


def _public_user(user: dict) -> dict:
    return {
        "id": str(user["_id"]), "email": user["email"],
        "name": user.get("name") or "", "email_verified": bool(user.get("email_verified")),
    }


def _clear_cookie(response: Response) -> None:
    response.delete_cookie(
        config.AUTH_COOKIE_NAME, path="/", httponly=True,
        secure=config.AUTH_COOKIE_SECURE, samesite="lax",
    )


def optional_user(request: Request) -> dict | None:
    """Anonymous if no cookie; reject present but invalid/stale credentials.

    Returns the raw user document for internal ownership checks. Only the two
    session/login endpoints serialize it, through the explicit public allowlist.
    """
    cookie = request.cookies.get(config.AUTH_COOKIE_NAME)
    if cookie is None:
        return None
    if not _TOKEN_PATTERN.fullmatch(cookie):
        raise _invalid_session()
    with _database() as db:
        session = db.auth_sessions.find_one({"_id": _digest(cookie), "expires_at": {"$gt": utcnow()}})
        if not session:
            raise _invalid_session()
        version = session.get("auth_version")
        if type(version) is not int or version < 0 or not isinstance(session.get("user_id"), str):
            raise _invalid_session()
        user = db.users.find_one({"_id": session["user_id"], "auth_version": version})
        if not user:
            raise _invalid_session()
        return user


def require_user(request: Request) -> dict:
    user = optional_user(request)
    if user is None:
        raise _invalid_session()
    return user


def require_verified_user(request: Request) -> dict:
    user = require_user(request)
    if not user.get("email_verified"):
        raise HTTPException(403, "Verify your email address before continuing.")
    return user


def _require_mail_config() -> None:
    try:
        base = urlsplit(config.APP_URL)
        valid_url = (base.scheme in {"http", "https"} and base.hostname
                     and not (base.username or base.password or base.query or base.fragment))
    except ValueError:
        valid_url = False
    if not config.SMTP_HOST or not config.SMTP_FROM or not valid_url:
        raise HTTPException(503, _MAIL_UNAVAILABLE)


class _MailDeliveryError(Exception):
    """Intentionally carries no server reply, recipient, or message content."""


def _send_token_email(email: str, token: str, purpose: str) -> None:
    try:
        message = EmailMessage()
        message["From"] = config.SMTP_FROM
        message["To"] = email
        verification = purpose == "verify"
        message["Subject"] = "Verify your VidSense email" if verification else "Reset your VidSense password"
        fragment = "verify" if verification else "reset"
        link = f"{config.APP_URL.rstrip('/')}/#{fragment}={token}"
        minutes = _VERIFY_SECONDS // 60 if verification else _RESET_SECONDS // 60
        message.set_content(
            f"{'Verify your email address' if verification else 'Reset your password'}:\n\n"
            f"{link}\n\nThis link expires in {minutes} minutes and can be used once.\n"
            "If you did not request this email, you can ignore it.\n"
        )
        context = ssl.create_default_context()
        if config.SMTP_SSL:
            connection = smtplib.SMTP_SSL(
                config.SMTP_HOST, config.SMTP_PORT, timeout=15, context=context,
            )
        else:
            connection = smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=15)
        with connection as smtp:
            if config.SMTP_STARTTLS and not config.SMTP_SSL:
                smtp.ehlo()
                smtp.starttls(context=context)
                smtp.ehlo()
            if config.SMTP_USERNAME:
                smtp.login(config.SMTP_USERNAME, config.SMTP_PASSWORD)
            if smtp.send_message(message):
                raise _MailDeliveryError()
    except (OSError, ValueError, smtplib.SMTPException):
        raise _MailDeliveryError() from None


def _issue_email_token(db: Database, user: dict, purpose: str) -> None:
    token = secrets.token_urlsafe(32)
    digest = _digest(token)
    now = utcnow()
    db.auth_tokens.insert_one({
        "_id": digest, "user_id": user["_id"], "purpose": purpose,
        "auth_version": user["auth_version"], "created_at": now,
        "expires_at": now + timedelta(seconds=_VERIFY_SECONDS if purpose == "verify" else _RESET_SECONDS),
    })
    try:
        _send_token_email(user["email"], token, purpose)
    except _MailDeliveryError:
        db.auth_tokens.delete_one({"_id": digest})
        _logger.warning("Authentication email delivery failed; a new email can be requested.")


def _get_email_token(db: Database, raw: str, purpose: str) -> dict:
    if not _TOKEN_PATTERN.fullmatch(raw):
        raise HTTPException(400, "This link is invalid or has expired. Request a new email.")
    query = {"_id": _digest(raw), "purpose": purpose, "expires_at": {"$gt": utcnow()}}
    token = db.auth_tokens.find_one(query)
    if token:
        _account_limit(purpose, token["user_id"], 10 if purpose == "verify" else 5)
    if not token:
        raise HTTPException(400, "This link is invalid or has expired. Request a new email.")
    return token


def _redeem_email_token(db: Database, token: dict, password_hash: str | None = None) -> None:
    """Commit redemption in the user document; token deletion is only cleanup.

    A reset's old auth_version can win once, and verification can only change
    an unverified user. Thus a crash or failed cleanup cannot reopen a token.
    The attempt marker is committed with the mutation, so an ambiguous write
    error can be reconciled without applying the supplied password again.
    """
    if token["expires_at"] <= utcnow():
        raise HTTPException(400, "This link is invalid or has expired. Request a new email.")
    reset = token["purpose"] == "reset"
    marker_field = "last_password_reset" if reset else "last_email_verification"
    marker = {"token_id": token["_id"], "attempt_id": uuid.uuid4().hex}
    expected = {"_id": token["user_id"], "auth_version": token["auth_version"]}
    fields = {marker_field: marker, "updated_at": utcnow()}
    update = {"$set": fields}
    if reset:
        fields["password_hash"] = password_hash
        update["$inc"] = {"auth_version": 1}
    else:
        expected["email_verified"] = False
        fields["email_verified"] = True
    try:
        result = db.users.update_one(expected, update)
    except PyMongoError:
        # A timeout/disconnect doesn't tell us whether Mongo committed. A marker
        # from this exact attempt proves it did; another attempt's cannot do so.
        committed = {
            "_id": token["user_id"], marker_field: marker,
            "auth_version": token["auth_version"] + (1 if reset else 0),
        }
        if not reset:
            committed["email_verified"] = True
        if not db.users.find_one(committed, {"_id": 1}):
            # Leave the token intact. If the write arrives late, the user CAS
            # still fences a retry; an unchanged account can retry immediately.
            raise
    else:
        if not result.matched_count:
            raise HTTPException(400, "This link is invalid or has expired. Request a new email.")
    try:
        if reset:
            # Version matching already revoked access. Preserve newer logins and
            # tokens if they were created after this password change committed.
            old_version = {"user_id": token["user_id"], "auth_version": {"$lte": token["auth_version"]}}
            db.auth_sessions.delete_many(old_version)
            db.auth_tokens.delete_many(old_version)
        else:
            db.auth_tokens.delete_one({"_id": token["_id"]})
    except PyMongoError:
        # Cleanup is optional for correctness; TTL handles retained documents.
        _logger.warning("Authentication redemption cleanup failed; the token remains unusable.")


@router.get("/session")
def session(request: Request, response: Response) -> dict:
    enabled = bool(config.MONGODB_URI)
    user = None
    if enabled:
        try:
            user = optional_user(request)
        except HTTPException as exc:
            if exc.status_code != 401:
                raise
            _clear_cookie(response)
    elif config.AUTH_COOKIE_NAME in request.cookies:
        _clear_cookie(response)
    return {
        "user": _public_user(user) if user else None, "enabled": enabled,
        "registration_available": bool(config.SMTP_HOST),
    }


@router.post("/register")
def register(payload: RegisterRequest, request: Request) -> dict:
    _require_mail_config()
    _rate_limit(request, "register", 10, payload.email, 3)
    # Hash for duplicates too. The unique email index resolves concurrent signups.
    with _password_work():
        password_hash = _password_hasher.hash(payload.password.get_secret_value())
    now = utcnow()
    user = {
        "_id": str(uuid.uuid4()), "email": payload.email, "name": (payload.name or "").strip(),
        "password_hash": password_hash, "email_verified": False, "auth_version": 0, "collection_count": 0,
        "created_at": now, "updated_at": now,
    }
    with _database() as db:
        try:
            db.users.insert_one(user)
        except DuplicateKeyError:
            user = db.users.find_one({"email": payload.email})
            if user is None:
                raise HTTPException(503, "The account service is temporarily unavailable.") from None
        if not user.get("email_verified"):
            _issue_email_token(db, user, "verify")
    return _VERIFICATION_REPLY


@router.post("/login")
def login(payload: LoginRequest, request: Request, response: Response) -> dict:
    _rate_limit(request, "login", 20, payload.email, 5)
    with _database() as db:
        user = db.users.find_one({"email": payload.email})
        try:
            with _password_work():
                _password_hasher.verify(
                    user["password_hash"] if user else _dummy_hash, payload.password.get_secret_value(),
                )
        except (VerificationError, InvalidHashError):
            raise HTTPException(401, "Invalid email or password.") from None
        if user is None:
            raise HTTPException(401, "Invalid email or password.")
        # A reset during password verification must not upgrade an old password
        # into a session with the new version. Keep the authenticated snapshot.
        user = db.users.find_one({
            "_id": user["_id"], "auth_version": user["auth_version"], "password_hash": user["password_hash"],
        })
        if not user:
            raise HTTPException(401, "Invalid email or password.")
        token = secrets.token_urlsafe(32)
        seconds = max(1, min(int(config.AUTH_SESSION_SECONDS), _MAX_SESSION_SECONDS))
        now = utcnow()
        db.auth_sessions.insert_one({
            "_id": _digest(token), "user_id": user["_id"], "auth_version": user["auth_version"],
            "created_at": now, "expires_at": now + timedelta(seconds=seconds),
        })
        # Rotating a cookie also retires this browser's previous session.
        previous = request.cookies.get(config.AUTH_COOKIE_NAME, "")
        if _TOKEN_PATTERN.fullmatch(previous):
            db.auth_sessions.delete_one({"_id": _digest(previous)})
    response.set_cookie(
        config.AUTH_COOKIE_NAME, token, max_age=seconds, expires=now + timedelta(seconds=seconds),
        httponly=True, secure=config.AUTH_COOKIE_SECURE, samesite="lax", path="/",
    )
    return {"user": _public_user(user)}


@router.post("/logout")
def logout(request: Request, response: Response) -> dict:
    _rate_limit(request, "logout", 60)
    cookie = request.cookies.get(config.AUTH_COOKIE_NAME, "")
    if _TOKEN_PATTERN.fullmatch(cookie):
        with _database() as db:
            session = db.auth_sessions.find_one({"_id": _digest(cookie)})
            if session:
                _account_limit("logout", session["user_id"], 60)
                db.auth_sessions.delete_one({"_id": session["_id"]})
    _clear_cookie(response)
    return {"message": "Signed out."}


@router.post("/verify-email")
def verify_email(payload: TokenRequest, request: Request) -> dict:
    _rate_limit(request, "verify", 30)
    with _database() as db:
        token = _get_email_token(db, payload.token.get_secret_value(), "verify")
        _redeem_email_token(db, token)
    return {"message": "Email verified."}


@router.post("/resend-verification")
def resend_verification(payload: EmailRequest, request: Request) -> dict:
    _require_mail_config()
    _rate_limit(request, "verification-mail", 10, payload.email, 3)
    with _database() as db:
        user = db.users.find_one({"email": payload.email, "email_verified": False})
        if user:
            _issue_email_token(db, user, "verify")
    return _VERIFICATION_REPLY


@router.post("/forgot-password")
def forgot_password(payload: EmailRequest, request: Request) -> dict:
    _require_mail_config()
    _rate_limit(request, "reset-mail", 10, payload.email, 3)
    with _database() as db:
        user = db.users.find_one({"email": payload.email})
        if user:
            _issue_email_token(db, user, "reset")
    return _RESET_REPLY


@router.post("/reset-password")
def reset_password(payload: ResetPasswordRequest, request: Request) -> dict:
    _rate_limit(request, "reset", 20)
    with _database() as db:
        # Hashing precedes redemption; failures cannot consume the link.
        with _password_work():
            password_hash = _password_hasher.hash(payload.password.get_secret_value())
        token = _get_email_token(db, payload.token.get_secret_value(), "reset")
        _redeem_email_token(db, token, password_hash)
    return {"message": "Password reset. Sign in with your new password."}
