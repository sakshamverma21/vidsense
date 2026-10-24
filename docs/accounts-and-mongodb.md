# Email/password accounts and MongoDB

The SPA now uses its own email/password account screens. Google sign-in is not
required. Passwords use Argon2id; the database never receives plaintext
passwords. Sessions use random opaque cookies marked HttpOnly and SameSite=Lax,
with Secure enabled when VIDSENSE_APP_URL uses HTTPS. Only session-token hashes
are stored in MongoDB. Email verification is required before importing videos.

## Local development

Install runtime dependencies and start the local services:

```powershell
python -m pip install -r requirements.txt
docker compose up -d mongo mailpit
```

Add the following non-secret development settings to your local `.env`:

```dotenv
MONGODB_URI=mongodb://127.0.0.1:27017
MONGODB_DATABASE=vidsense
VIDSENSE_APP_URL=http://127.0.0.1:8000
SMTP_HOST=127.0.0.1
SMTP_PORT=1025
SMTP_FROM=VidSense <noreply@localhost>
SMTP_STARTTLS=false
SMTP_SSL=false
```

Run `uvicorn api:app --host 127.0.0.1 --port 8000`, open the site, and create an
account. Read its verification email at http://127.0.0.1:8025. Mailpit captures
messages locally and does not deliver real mail. The reset and verification
tokens are in URL fragments, which browsers do not include in HTTP access logs.
Open verification links in the same browser, select Verify, then sign in.

Use exactly the configured origin (`127.0.0.1` and `localhost` are different).
Without MongoDB the starter demo works, while custom imports and accounts remain
disabled. Without SMTP, new sign-ups and recovery emails are unavailable.

For Dockerized API testing, the API must use `mongo:27017` and `mailpit:1025`
on the Compose network, rather than host-loopback addresses. The Compose file
intentionally starts only dependencies, to avoid altering a running app.

## MongoDB Atlas setup

1. Create an Atlas Free cluster and choose a supported region near the API.
2. Create a database user restricted to read/write access on the `vidsense`
   database. This is separate from website user accounts.
3. Configure network access for the deployment's outbound addresses. Avoid
   allowing all Internet addresses. Review Azure outbound-network requirements
   before relying on a small IP allowlist.
4. Copy the driver's connection string into local `.env` as MONGODB_URI. URL
   encode special characters in the database user's password. Never paste the
   URI into source, logs, GitHub, a browser address bar, or a support message.
5. Store the URI and SMTP password as Azure Container Apps secrets and reference
   them in environment variables. Keep MONGODB_DATABASE=vidsense and set
   VIDSENSE_APP_URL=https://vidsense.sakshamverma.dev.
6. Use a real email provider and a verified sender with SPF/DKIM. Configure its
   SMTP host, port, username, password and TLS mode. Never use Mailpit publicly.

The Free tier is intended for a small beta, with about 512 MB for documents and
indexes. Check current limits and backup options before launch:
https://www.mongodb.com/docs/atlas/reference/free-shared-limitations/
Backups are not assumed by this app; configure and test an export/restore or
an Atlas backup option explicitly.

## Storage and ownership

MongoDB stores accounts, expiring sessions, single-use email token hashes,
request-limit counters, collection metadata, and one transcript record per video
per collection. Every collection operation checks the authenticated owner.
Unknown and other users' collection IDs both return 404. The demo is public.

FAISS remains a local search index. Reopening a saved collection after a restart
rebuilds its index from MongoDB records, with no transcript-provider requests.
Private collection transcripts are not written to the shared on-disk demo cache.
User collection data is deleted through DELETE /api/corpora/{id}; deletion of a
currently processing collection returns 409 until processing stops. Backups may
retain data until their own retention period expires.

Imports renew their lease while working and fence writes with a job generation.
Interrupted imports expire their processing lease, become retryable errors,
and POST /api/corpora/{id}/retry reuses saved video transcripts. This is not yet
a durable queue: continue to use one replica and one Uvicorn worker process.
Local indexes can be evicted from
memory without deleting the database collections.

If a database failure interrupts deletion, a deletion marker hides the collection
from queries while retaining enough state to finish cleanup. Repeat Delete from
the collection list to finish. Completed collection manifests exclude videos
left over from earlier failed attempts.

## API additions

```text
GET  /api/auth/session
POST /api/auth/register          {email, password, name?}
POST /api/auth/login             {email, password}
POST /api/auth/logout
POST /api/auth/verify-email      {token}
POST /api/auth/resend-verification {email}
POST /api/auth/forgot-password   {email}
POST /api/auth/reset-password    {token, password}
GET  /api/corpora
DELETE /api/corpora/{id}
POST /api/corpora/{id}/retry
POST /api/research/stream        {corpus_id, mode, question}
```

Mutations require `X-VidSense-Request: 1`. Browser Origin must match
VIDSENSE_APP_URL. The application deliberately does not enable cross-origin API
access. Send the session cookie with requests; do not put it in URL parameters.
The old GET streaming API and /api/eval are disabled unless the explicit
VIDSENSE_ENABLE_DEVELOPER_API flag is enabled in a private developer installation.

IP throttles use the connection peer, not arbitrary forwarded headers. If an
ingress proxy hides client addresses, configure Uvicorn's trusted proxy list for
that deployment and verify its forwarded-header behavior. Do not broadly trust
Internet-supplied forwarding headers. Authenticated API limits additionally use
the account ID, and password hashing has a process-wide concurrency bound.

## Validation and remaining rollout work

Automated tests use mongomock for repeatable owner-isolation, token expiry,
password reset, and recovery checks. This does not replace exercising the same
workflow on a real MongoDB instance before deploying.

Run `python scripts/smoke_accounts.py` with the local MongoDB and Mailpit services
up to exercise real persistence and SMTP without AI/transcript calls. It creates
and removes only its own randomly named local smoke-test database.

Before a public beta: test real email delivery and MongoDB connectivity, configure
backups, add global provider-credit budgets, move ingestion into a durable worker
queue, run a concurrency test, and document retention and privacy practices.
The account increment alone does not make the whole service launch-ready.
