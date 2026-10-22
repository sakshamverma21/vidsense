# VidSense — Video Collection Intelligence

**Live demo:** [vidsense.sakshamverma.dev](https://vidsense.sakshamverma.dev)

VidSense turns one or more YouTube videos into a single, grounded collection you
can ask questions about, summarize, and scan for high-value moments. Instead of
one-shot Q&A over a single transcript, the research engine **plans**:
it decides which tool to use next based on what it just found, gathers evidence
from multiple videos, and returns an answer with **timestamp-grounded citations**
plus a **claim-verification** step for hallucination control.

Crucially, it ships with an **evaluation harness** that measures the agent the way
you'd measure a production system — task success, tool-call accuracy, trajectory
quality, and cost/latency per task — and compares it head-to-head against a
single-shot RAG baseline.

> Built with **LangGraph** (agent), **FAISS** + **FastEmbed** (retrieval),
> **Gemini** / **Groq** LLMs (pluggable provider), a **FastAPI + SSE** backend with a
> custom **HTML/CSS/JS** frontend, and a from-scratch eval harness.

> **Portfolio deployment note:** the bundled starter corpus is always available.
> Collections created by visitors are intentionally temporary and may reset
> after inactivity, scale-to-zero, or a new deployment. Durable multi-user
> storage is described in the production roadmap below.

---

## Why this exists

A single-shot RAG chatbot is easy to build and impossible to trust blindly. This
project is about the two things that actually matter for an agentic system:

1. **Planning over tools** — the agent chooses *what to look at next* (search one
   video, compare across videos, verify a claim on the web) rather than running a
   fixed pipeline.
2. **Measuring it honestly** — an agent is only as good as your ability to
   evaluate it. The harness quantifies where the agent's extra cost is justified
   and where a cheaper baseline wins.

## Architecture

```
                       ┌──────────────── LangGraph StateGraph ────────────────┐
  question ──▶ planner ─▶ (executor ─▶ planner)* ─▶ synthesizer ─▶ verifier ─▶ answer
                 │            │                          │             │
        tool-bound LLM   runs tools,               grounded      checks each
        picks next tool  collects citations        answer +      claim vs
        + args           as evidence               citations     evidence
                       └──────────────────────────────────────────────────────┘

Tools:  transcript_search · timestamp_lookup · cross_video_compare · external_verify · list_videos
Corpus: multi-video transcripts → timestamped chunks → one FAISS store (metadata: video_id, timestamp, youtube_link)
```

| Module | Responsibility |
|---|---|
| `vidsense/corpus.py` | Ingest transcripts, build a metadata-rich, disk-cached FAISS store |
| `vidsense/tools.py` | The five tools the agent plans over (each returns grounded citations) |
| `vidsense/agent.py` | The LangGraph research agent + trajectory logging + token/cost tracking |
| `vidsense/citations.py` | Citation formatting + LLM claim-verification (hallucination control) |
| `vidsense/baseline.py` | Single-shot RAG baseline (the control for evaluation) |
| `vidsense/eval/` | Task set, metrics, runner, report — the evaluation harness |
| `api.py` | FastAPI backend: REST + SSE endpoints, serves the frontend |
| `frontend/` | Customer-facing HTML/CSS/JS SPA: collections, answers, summaries, highlights, citations |
| `app.py` | Alternative Streamlit UI (same features) |

## Key features

- **Video collections** — paste one video, multiple links, or a public YouTube
  playlist. VidSense deduplicates sources and builds one FAISS store across the
  resulting transcripts.
- **Customer workspace** — add sources, monitor preparation, ask questions,
  create a collection overview, or find five timestamped highlights.
- **Timestamp-grounded citations** — every claim links back to the exact second
  in the source video (`youtube.com/watch?v=…&t=…s`).
- **Hallucination control** — a verifier splits the answer into claims and checks
  each against the retrieved evidence; reports a `grounded_ratio`.
- **Full trajectory logging** — every plan/tool/synth/verify step is recorded
  (thought, tool, args, latency, tokens) for display and evaluation.
- **Evaluation harness** — task success (LLM-judge), tool-call accuracy
  (set precision/recall/F1), trajectory quality (LLM-judged 1–5), and cost/latency
  per task, with an **agent-vs-baseline** comparison and matplotlib charts.

## Setup

```bash
python -m venv venv && venv\Scripts\activate      # Windows
pip install -r requirements.txt
cp .env.example .env                               # then add your keys
```

**Providers** (select with `VIDSENSE_PROVIDER`):

| Provider | Default model | Why |
|---|---|---|
| Gemini | `gemini-3.6-flash` | Current Gemini option |
| **Groq** | `openai/gpt-oss-20b` | Fast default for the portfolio deployment |
| Ollama | `qwen2.5:7b-instruct` | Fully local, no limits (slower on CPU) |

`.env` keys: `GOOGLE_API_KEY` (default provider — free at
[aistudio.google.com/apikey](https://aistudio.google.com/apikey)),
`GROQ_API_KEY` (alternative), `TAVILY_API_KEY` (only for `external_verify`),
and optionally `SUPADATA_API_KEY` for cloud-hosted transcript ingestion.

Provider/model overrides are supported with `VIDSENSE_PROVIDER`,
`VIDSENSE_GEMINI_MODEL`, and `VIDSENSE_GROQ_MODEL`. Never commit `.env`, paste
keys into source files, or bake them into a Docker image. Run
`python scripts/check_secrets.py` before pushing; CI also runs a Gitleaks scan.

## Usage

**Run the full-stack app** (FastAPI backend + custom frontend, with live streaming trajectory):
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```
Then open http://localhost:8000 — add a collection, ask grounded questions, get
an overview, and find timestamped highlights. Evaluation remains available as a
developer API and CLI workflow rather than a customer-facing navigation item.

### Build a collection

The customer workspace accepts one YouTube link per line. Each line can be an
individual video or a public/accessibly unlisted playlist:

```text
https://www.youtube.com/watch?v=VIDEO_ID
https://www.youtube.com/playlist?list=PLAYLIST_ID
```

The app resolves playlist entries with `yt-dlp`, keeps their titles, skips
duplicate video IDs, and combines all usable captions into one collection. To
avoid unexpectedly large jobs, a collection imports at most **50 videos** by
default; set `VIDSENSE_MAX_PLAYLIST_VIDEOS` to change that limit. Videos without
captions remain visible as skipped rather than blocking the rest of the
collection.

Local development fetches public captions directly. YouTube commonly blocks
transcript and `yt-dlp` requests from Azure, AWS, and GCP IP ranges, so a cloud
deployment should set `SUPADATA_API_KEY` and leave
`VIDSENSE_TRANSCRIPT_PROVIDER=auto` (the default). The app then uses the managed
provider for timestamped transcripts while preserving the same corpus and
citation pipeline. Set `VIDSENSE_TRANSCRIPT_PROVIDER=direct` only for a network
where direct YouTube caption access is known to work.

Managed requests are serialized with a 1.1-second minimum interval by default
to respect Supadata's starter-tier request rate. Multi-video indexes use an
embedding batch size of 16 to stay within small container memory limits. Both
are configurable through `VIDSENSE_SUPADATA_MIN_INTERVAL_SECONDS` and
`VIDSENSE_EMBEDDING_BATCH_SIZE`.

Collections are process-local and expire after two hours of inactivity by
default. Their cached transcripts/indexes are reused on disk, but the active
collection is not restored after the API process restarts or an Azure instance
scales to zero. Add the sources again if that happens.

### Customer API

```text
POST /api/corpora                  # start a collection from { inputs, name? }
GET  /api/corpora/{corpus_id}      # poll preparation/status
POST /api/research                 # question/summary/highlights for corpus_id
GET  /api/research/stream          # streaming equivalent, with corpus_id + mode
GET  /api/live                     # liveness probe
GET  /api/ready                    # readiness/configuration probe
```

`mode` is `question`, `summary`, or `highlights`. The latter two power the
workspace action buttons and return the same citation-ready response shape as a
question.

**Or the built-in Streamlit UI** (same features, simpler):
```bash
streamlit run app.py
```

**Run the agent from Python:**
```python
from vidsense.corpus import load_corpus
from vidsense.agent import ResearchAgent

corpus = load_corpus("vidsense/eval/corpus.json")
agent = ResearchAgent(corpus)
out = agent.run("Compare how the videos explain what a neural network learns.")
print(out["answer"])          # grounded answer with [title @ MM:SS] citations
print(out["trajectory"])      # every step the agent took
print(out["usage"], out["latency"])
```

**Run the evaluation:**
```bash
python -m vidsense.eval.runner            # agent vs baseline over the task set
python -m vidsense.eval.report            # markdown report + charts
python -m vidsense.eval.author_tasks      # draft more candidate tasks to curate
```

## Deployment

The FastAPI app is containerized (`Dockerfile`). The image installs runtime
dependencies without `torch`/`transformers` and downloads the embedding model at
build time. A fresh Git checkout can build without generated transcript data.

Transcript caches and FAISS indexes are excluded from Git. To prepare the starter
collection before a deployment, run the following locally after configuring
`.env` and installing the runtime dependencies:

```bash
python -c "from vidsense.corpus import load_corpus; load_corpus('vidsense/eval/corpus.json')"
```

This can consume transcript-provider credits if the starter collection is not
already cached. A subsequent Docker build includes the prepared `dl-series-*`
cache, while excluding customer collections. Without a prepared cache, the app
attempts to fetch starter transcripts at runtime; that requires working YouTube
access or a configured managed transcript provider. Never commit transcript
caches, API keys, or your local `.env`.

```bash
# Build
docker build -t vidsense .

# Run with locally configured secrets; choose the provider in .env.
docker run --env-file .env -p 8000:8000 vidsense
# open http://localhost:8000
```

The container reads `$PORT` (defaulting to 8000), so it drops straight into
**Azure Container Apps** / Cloud Run. Store provider keys as platform secrets;
the portfolio deployment uses `VIDSENSE_PROVIDER=groq` with `GROQ_API_KEY`,
`SUPADATA_API_KEY` for visitor-added videos, and an optional `TAVILY_API_KEY` for
external verification.

For Azure portfolio hosting, keep `minReplicas=0` and `maxReplicas=1`. Scaling
to zero controls idle cost; a single maximum replica is required because active
visitor collections are process-local.

## Production roadmap

The portfolio deployment deliberately favors a small, inspectable architecture.
A durable multi-user version would replace process-local sessions and per-container
FAISS indexes with PostgreSQL + pgvector, associate collections with authenticated
users, and move playlist/transcript ingestion to a durable queue-backed worker.
That would allow safe horizontal scaling and preserve collections across revisions.

## Evaluation methodology

- **Corpus**: a fixed manifest (`vidsense/eval/corpus.json`) so runs are
  reproducible and cheap (transcripts + FAISS are cached on disk).
- **Tasks**: `vidsense/eval/tasks.json` — single-video, cross-video, and
  verification tasks, each with a reference answer, expected tools, and gold
  video ids. `author_tasks.py` drafts candidates from transcripts (LLM proposes,
  you curate).
- **Metrics**: task success (LLM-judge vs reference), tool-call accuracy,
  trajectory quality (LLM-judge rubric), grounded-ratio, cost (token accounting),
  latency.
- **Baseline**: identical corpus/retrieval/citation format but one retrieval +
  one LLM call — isolating what the agent's machinery buys.

## Results

<!-- RESULTS:START -->
Historical evaluation on the 16-task set over the `dl-series` corpus (6 videos),
agent and baseline both on `gemini-3.5-flash-lite`, LLM-judged. The live portfolio
deployment now uses Groq, so these results are retained as a reproducible prior
run rather than presented as a benchmark of the currently deployed model:

| Metric | Agent | Single-shot RAG |
|---|---|---|
| **Task success (overall)** | **93.8%** | 56.2% |
| — single-video tasks | **100%** | 71.4% |
| — cross-video tasks | **83.3%** | 50.0% |
| — verify (web) tasks | **100%** | 33.3% |
| Grounded ratio (verifier) | 94.3% | — |
| Trajectory quality (LLM-judge) | 68.8% | — |
| Tool coverage | 81.3% | — |
| Avg cost / task | $0.0012 | $0.0001 |
| Avg tool calls | 3.7 | 0 |

**Takeaway:** the agent lifts task success **+37 points** overall, winning on every
task type. It is strongest where the baseline structurally can't compete —
**cross-video** synthesis (one retrieval can't compare across videos) and **verify**
tasks (the baseline has no web tool at all). The trade-off is ~12× the cost, so the
honest design conclusion is to **route by task type**: cheap single-shot RAG for
simple lookups, the agent for cross-video research and fact-checking.

_An earlier run scored 0/3 on verify tasks — the harness surfaced a real bug (the
synthesizer ignored `external_verify` output because it carried no transcript
passages); routing web-verification findings into synthesis fixed it to 3/3. That
find-fix-remeasure loop is exactly what the eval is for._

_Latency is intentionally omitted from the headline: this run used a 12 req/min
client-side rate limiter to fit the Gemini free tier, which inflates per-task
wall-clock to ~37s; unthrottled it is ~10s/task. Reproduce with
`python -m vidsense.eval.runner --provider gemini --model gemini-3.5-flash-lite --rpm 12`
then `python -m vidsense.eval.report`._
<!-- RESULTS:END -->

## Notes & limitations

- Playlist enumeration uses `yt-dlp` locally and the managed YouTube adapter in
  cloud deployments. Private or login-protected playlists are outside the
  current customer workflow.
- Direct transcript fetching is suitable locally but is commonly blocked on
  cloud-provider IPs. The optional managed transcript adapter is therefore the
  supported path for visitor-created collections in the hosted demo.
- The LLM judge and the agent use the same model family — a known self-preference
  caveat; swap `--judge-model` to probe it.
- Groq prices in `eval/metrics.py` are approximate and configurable; the point is
  a transparent cost methodology.
