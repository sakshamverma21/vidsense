# VidSense interview questions and practice answers

Use these as explanations to reconstruct in your own words. For each answer, be able to name the responsible file, give an example, describe a failure mode, and justify an alternative. Personal contribution answers must reflect what you actually did.

## Project scope and architecture

### 1. What is VidSense?

VidSense is a transcript-based research application for YouTube video collections. It ingests caption text, builds a local semantic index, and supports questions, overviews, and highlights with video source links. Questions use an LLM-directed research loop; overview/highlight buttons use a cheaper single-shot path. An evaluation harness compares the agent with a RAG baseline.

### 2. What problem does it solve?

It helps users find and connect information across long videos without manually watching every minute. The engineering challenge is to retrieve relevant evidence, preserve its source, and answer across multiple videos. Citation cards make evidence inspectable, while evaluation measures when the added research machinery is useful.

### 3. Is it actually analyzing video?

It analyzes timestamped captions, not frames or audio directly. It cannot see a silent equation, animation, or slide. Supporting those would require a separate frame/OCR/multimodal ingestion path and provenance alignment. This distinction is important because the product name can imply a broader capability.

### 4. What makes it different from a basic chatbot?

A basic LLM response has no collection-specific evidence. Single-shot RAG adds one retrieval. VidSense's question agent can inspect video inventory, reformulate searches, compare videos, locate topics, and verify external claims based on earlier observations. The graph also records actions, usage, and latency. Added complexity is justified by measured capability, not by the “agent” label alone.

### 5. Explain the end-to-end pipeline in a minute.

Source links go through validation, playlist enumeration, deduplication, and caption retrieval. Caption text is chunked, embedded locally, and indexed with FAISS, retaining video metadata. For a question, the planner chooses evidence tools, the executor collects structured results, synthesis writes the answer, and verification reports claim support. FastAPI streams completed steps and the result to the browser.

### 6. What are the main modules?

`sources` resolves inputs, `supadata` manages cloud HTTP access, `corpus` fetches/chunks/caches/indexes, `tools` exposes research operations, `agent` orchestrates graph state, and `citations` formats/checks evidence. `baseline` supplies the control workflow, `eval` measures it, `api.py` serves the application, and `frontend` implements the customer workspace.

### 7. Which features are legacy and which are current?

BERT topic clustering, summary styles, and automatic chapter detection live in the legacy single-video module. The current web app provides collections, questions, summaries, highlights, a video library, and citations. The optional Streamlit interface shares the current agent but emphasizes settings, trajectories, baseline comparison, and evaluation views.

### 8. What did you train or fine-tune?

The code does not train or fine-tune a generator, BGE, or BERT. It uses pretrained embeddings and hosted/local generation interfaces. K-means in the legacy module fits cluster centroids, which is different from training the neural embedding model. The contribution is application architecture, evidence handling, orchestration, product behavior, and measurement.

### 9. What is your main technical contribution?

Choose the parts you personally implemented and connect them to outcomes. A factual project-level answer can discuss metadata-rich multi-video retrieval, typed tool outputs, bounded graph orchestration, streaming, provider adaptation, and evaluation. Do not claim you invented RAG, built FAISS, trained the hosted LLM, or implemented changes that only appear in the proposed roadmap.

### 10. What is the strongest limitation you would acknowledge?

The current timestamp mapping mishandles overlapping chunk offsets, so source links can drift forward. There are also temporary sessions and imperfect evidence coverage. A strong answer names a specific limitation, explains its cause and consequence, and gives a concrete repair/validation plan rather than saying vaguely “hallucinations can happen.”

## Ingestion and retrieval

### 11. Why use captions instead of speech recognition?

Captions are already text with timestamps and avoid a media-download/ASR compute pipeline. The tradeoff is availability and caption quality. ASR would extend support to uncaptained videos but introduces model cost, language/acoustic errors, storage, and alignment work. The managed adapter still retrieves transcripts; it is not an in-app ASR implementation.

### 12. Why are there two transcript providers?

Direct public caption fetching is suitable for local access, but cloud-provider networks can be blocked. The managed provider addresses that deployment constraint while returning the same internal segment schema. `auto` selects managed immediately if configured; it does not first try direct and then retry on failure.

### 13. How are playlists supported?

The resolver enumerates entries with flat `yt-dlp` metadata locally or managed playlist endpoints when configured. It requests one extra entry to detect clipping, keeps titles/provenance, deduplicates IDs, and limits the entire collection to fifty specs by default. It does not download videos. Links with `/watch?...&list=...` take playlist precedence.

### 14. What happens if one video has no captions?

That video becomes a skipped record and other usable videos proceed. The API returns `partial` when appropriate, and the library can show skipped sources. If no usable transcript remains, collection building fails. Partial success is intentional; the user should still understand what was excluded.

### 15. What is the transcript schema?

Each segment has `text`, `start` in seconds, and `duration` in seconds. Managed offsets/durations arrive in milliseconds and are converted. A `VideoRecord` adds title, source URL/type, and playlist ID. A `Corpus` combines those records with one vector store and ingestion diagnostics.

### 16. Why chunk the transcript?

A full transcript is too broad for localized retrieval and expensive to repeatedly send to a model. Chunking creates retrieval units with source information. Too-small chunks lose explanatory context; too-large chunks mix topics and spend context budget. Overlap protects boundary context but introduces redundancy and demands correct offset mapping.

### 17. Why 1000 and 200?

They are practical character-based defaults: enough local context with overlap around boundaries. There is no repository evidence that these values were selected by an ablation. I would tune them using passage recall, answer quality, timestamp accuracy, latency, and token cost on representative transcripts. They are not 1000/200 tokens.

### 18. How should timestamps be computed correctly?

Preserve the actual original character span of each chunk, then map its start to caption segment boundaries using prefix offsets/binary search. Caption-aware chunk construction can make this more direct. Overlap means summing chunk lengths is wrong. Even a repaired mapping gives caption/passsage localization, not exact spoken-word alignment.

### 19. What is an embedding?

It is a learned numeric representation that makes semantically related texts close in vector space. Document chunks are embedded during indexing and tool queries are embedded during research. The embedding model chooses representation; FAISS performs nearest-neighbor search; the generator uses retrieved text to write the answer.

### 20. Which embedding model and dimension are used?

The project relies on FastEmbed's LangChain wrapper default. The installed implementation uses `BAAI/bge-small-en-v1.5`, with 384-dimensional vectors and a 512-token limit. It is an English model running through ONNX, with a configured batch size of sixteen. Explicit model/version configuration would make future builds more reproducible.

### 21. Why FastEmbed instead of BERT/PyTorch in production?

FastEmbed supplies retrieval-oriented sentence embeddings through a lighter ONNX runtime, and the comments explicitly aim to avoid the heavy legacy dependency stack. Base BERT `[CLS]` clustering is a separate legacy task. Compare candidates on retrieval quality, language coverage, memory, cold starts, and indexing speed rather than equating a bigger model with a better application.

### 22. What exactly does FAISS do here?

It stores numeric chunk vectors and returns nearest neighbors. LangChain separately keeps text/metadata and maps vector positions to documents. The actual bundled index is `IndexFlatL2`, an exact exhaustive index. It is not a managed vector database and does not provide user authentication or durable job/session management.

### 23. Is this cosine similarity or L2?

The index uses squared L2 distance. BGE/FastEmbed vectors are normalized in this path, so minimizing `2 - 2 cos(x,y)` yields the cosine ranking. That equivalence depends on normalization. The legacy chapter detector explicitly calculates cosine similarity and is a separate implementation.

### 24. Is FAISS search approximate here?

No. `IndexFlatL2` exhaustively compares vectors. FAISS offers approximate indexes, but this code does not select them. At 147 demo chunks, exact search is cheap and easy to reason about. Approximate indexing would be a measured scaling decision, with a recall/speed tradeoff.

### 25. How does video filtering work, and what can go wrong?

The wrapper fetches twenty global candidates and filters them by metadata. That can miss a video's best in-scope passages if none enter the global twenty. The offline audit reproduced underfilled scoped results. Increase/adapt candidate counts or use per-video/native scoped search, then evaluate relevance—not just returned count.

### 26. Why not just use keyword search?

Semantic embeddings handle paraphrases that lack exact keywords. Keyword retrieval can be stronger for names, quotations, codes, and numbers. A hybrid approach is a reasonable improvement, but this project currently uses dense retrieval only. Retrieval mode should be chosen from measured error patterns.

### 27. Would increasing k solve missing information?

Sometimes, but it may add irrelevant passages, redundancy, and token cost. Scoped candidate `fetch_k` differs from final `k`; increasing final k alone does not fix filtering starvation. Evidence text truncation can also hide the answer even when the right chunk is retrieved. Diagnose which stage loses information first.

### 28. How does the cache work?

It persists normalized records and FAISS/docstore files under a name plus fingerprint. Matching collections can reuse caption/index artifacts. The richer fingerprint includes ordered source metadata, while an older ID-set key supports the bundled cache. A cache hit avoids document embedding work but still needs query embeddings.

### 29. How would changing the embedding model affect cached indexes?

Documents and queries need compatible vector dimensions and semantics. A different model can produce a dimension error or silently meaningless neighbors if dimensions match but representations differ. Include model/version and chunking/provenance schema in the cache identity, rebuild document vectors, and never query old embeddings with a new incompatible model.

### 30. Why RAG instead of fine-tuning or sending everything?

RAG lets collections change without retraining and exposes specific source passages. Fine-tuning is better suited to behavior/style changes than reversible per-video knowledge with citations. Full-context prompting can be competitive for small transcripts and deserves benchmarking, but repeated long-collection prompts increase context cost and irrelevant material.

## Agent behavior and evidence

### 31. Why LangGraph instead of a simple while loop?

It makes state merging, conditional edges, bounded looping, node telemetry, and streaming explicit. A while loop would also be viable for this scale with fewer dependencies, but we'd have to implement those contracts ourselves. Framework selection is a maintainability/control argument; it does not prove better answers.

### 32. What makes the workflow agentic?

The model chooses the next tool and arguments based on previous observations, and chooses when to stop. The set of tools and outer synthesis/verification workflow remain fixed by code. A deterministic recipe calling every operation in order would be a workflow, without that adaptive action selection.

### 33. What are the graph nodes?

Planner, executor, synthesizer, and verifier. Planner routes either to execution and back to planning or to synthesis. Synthesis goes to verification, then END. The verifier currently cannot route back for repair, and the graph has no persisted checkpointer or human-intervention node.

### 34. What is stored in state, and why reducers?

State stores question, conversation messages, evidence, external findings, trajectory, planner count, answer, and verification. Evidence/findings/trajectory concatenate so earlier results survive. `add_messages` handles message-aware merges. Scalars overwrite. Without the right reducer, later node updates could erase useful accumulated context.

### 35. How does tool calling work?

The chat LLM is bound to typed tool definitions. It returns structured name/arguments/call-ID records. The executor invokes the matching tool and returns a `ToolMessage` with that call ID. The model selects an operation; Python performs it. Valid JSON/argument shape alone does not establish semantic correctness.

### 36. What is content versus artifact?

Content is the readable observation used by the planner. Artifact carries structured passages/source metadata used by application code. This prevents brittle extraction of citation data from LLM prose. External tools have no transcript passages, so their readable findings require a separate evidence channel for synthesis.

### 37. Name all five tools and their use cases.

Inventory uses `list_videos`; topic evidence uses `transcript_search`; localization uses `timestamp_lookup`; balanced multi-video research uses `cross_video_compare`; outside fact checking uses `external_verify`. Search retrieves five passages, timestamp lookup two, comparison two per selected video, and web search at most four results.

### 38. Does comparison guarantee coverage?

It attempts retrieval for each selected video rather than one global top-k, improving representation. The scoped-filtering issue can still return zero results. It also retrieves only two passages per video, may include irrelevant videos, and uses truncated snippets. Coverage should be measured from retrieved and used evidence.

### 39. Does timestamp_lookup find the exact word or quote?

No. It semantically retrieves two chunks from one video and reports the first chunk's metadata time. It neither does exact substring matching nor audio/text alignment. Quote localization would need lexical matching and refined caption ranges; current offset mapping also needs repair.

### 40. What bounds the agent loop?

The model stops by replying without calls, or code forces synthesis when the planner count reaches `max_steps`. At default six, there are at most five executor rounds, though multiple calls can occur in one round. The separate recursion limit protects graph execution but there is no overall token/dollar/time budget.

### 41. Does the agent execute tools in parallel?

No. The executor iterates sequentially over calls. Playlist oEmbed title lookups do use a bounded thread pool, but that is unrelated to the agent's tool execution. Parallel model tool-call support and parallel application execution are separate capabilities.

### 42. Why a separate synthesizer?

It can write the final response from accumulated structured evidence with a focused citation prompt, instead of trusting the planner's stop message. This costs an extra call. It receives evidence and web findings, not the whole planning conversation or comparison narrative, so useful interpretation can also be lost.

### 43. Is the “thought” field the model's full reasoning?

No. It is a truncated preview of returned message content, often empty during native tool calling. The useful audit trail is which observable actions were selected and executed, their inputs/results/latency, and resulting evidence. Do not describe it as complete access to internal reasoning.

### 44. What happens when a tool fails?

Known tool exceptions become an error observation with the matching call ID; planning can continue. Missing Tavily configuration produces an unavailable finding. Unknown tool requests also get an observation but are not consistently logged as executed tools. Model failures at the node level can still terminate the run.

### 45. Can the system remember earlier questions?

No. Each run starts new question/conversation state. Reusing the agent object does not preserve conversational history. To support follow-ups, add explicitly scoped history/checkpoints with context limits and user ownership. That also changes evaluation and privacy/storage responsibilities.

## Citations, verification, and product actions

### 46. How are citations represented?

Each citation stores video ID, title, numeric timestamp, formatted time, deep link, and retrieved text. Tools produce these from document metadata, the graph collects them, and the UI renders watchable cards. The answer uses readable `[title @ time]` tags. No parser currently proves every inline tag corresponds to a returned supporting passage.

### 47. How do you prevent hallucination?

The system constrains prompts to evidence, exposes sources, gathers additional evidence through tools, and reports claim support. Those measures reduce risk but do not guarantee correctness. Retrieval omissions, caption errors, prompt injection, generation mistakes, and same-model verification errors remain possible. The current verifier does not repair the answer.

### 48. What does a grounded ratio of 1.0 mean?

All claims retained by the heuristic splitter were judged supported by the provided evidence. It is not a calibrated probability of truth, completeness, or correct timestamps. A grounded refusal can still fail the task; final saved task `c5` does exactly that. Short/compound claims and verifier output parsing affect the denominator and verdicts.

### 49. How is verification different from external_verify?

The final verifier asks whether the answer is supported by gathered evidence. `external_verify` searches outside the collection to check a factual claim against web snippets. A video can be quoted accurately while its claim is false, so source faithfulness and external factual correctness are different dimensions.

### 50. What is wrong with splitting by sentences?

One sentence may contain several independently true/false facts, while headings or short bullets may be omitted. Abbreviations and formatting can fragment claims incorrectly. Atomic structured claim extraction with source IDs is more precise, but adds complexity/calls. Evaluation must check whether this refinement improves useful detection.

### 51. What does the system do with unsupported claims?

It records them in the verification report and returns the original answer unchanged. The browser does not currently surface the ratio or detailed verdicts. A stronger design could repair claims, retrieve more evidence, or abstain under explicit bounded rules. That is proposed behavior, not the current implementation.

### 52. Does every feature use the agent?

No. Open-ended questions use the agent. Customer summary and highlight modes use a single-shot retrieval/generation baseline with k=10. They get citation metadata but no claim verification. Streamlit does not expose the same action buttons. This difference resolves several overbroad README descriptions.

### 53. Are highlights learned or ranked by an algorithm?

No dedicated importance model ranks the whole timeline. A fixed prompt asks the LLM for five useful distinct moments from retrieved passages, when evidence allows. There is no code enforcement of five, diversity, or exact alignment. A stronger implementation would define value/coverage criteria and validate structured highlight spans.

### 54. Why can a summary miss an entire video?

It uses a broad query and ten global nearest chunks; some videos may contribute none. The model cannot summarize information that never reaches its context. Per-video retrieval/summary followed by a collection synthesis would improve coverage, with additional calls and careful handling of cumulative summary errors.

### 55. Does external verification always show web sources to users?

No guarantee exists. URLs are in the tool artifact/content and gathered findings, but the final structured citations contain transcript passages only. Synthesis may omit web URLs, leaving an externally verified answer without source cards. Add explicit web-source output and claim-source validation if presenting verification as auditable fact checking.

## Backend, deployment, and system design

### 56. Why does collection creation return 202?

Caption/embedding work may take time. The API registers the session first, returns accepted status, and performs work in a thread while clients poll. 202 does not mean the corpus is ready. Research returns 409 if the selected corpus is still processing or errored.

### 57. What is SSE, and why choose it?

It is an HTTP stream of named server-to-browser events consumed with EventSource. Research progress is a one-way feed, so SSE fits with less bidirectional-session machinery. WebSockets would be suitable for live client interaction/cancellation. This implementation streams completed graph updates and whole answers, not generated tokens.

### 58. Explain the thread/async queue bridge.

A worker thread runs synchronous research. It schedules queue writes on the event loop with `run_coroutine_threadsafe`. The async response generator awaits the queue and serializes events through EventSourceResponse. This keeps blocking provider work away from the SSE handler's event loop.

### 59. Why are locks and semaphores necessary?

The registry lock protects shared session mutations, the rate lock protects timestamp buckets, and the research semaphore caps simultaneous runs. The managed client lock serializes external requests. These structures solve different problems; the GIL does not guarantee correctness of all multi-operation updates, and none of these controls are shared across processes.

### 60. What happens if the browser disconnects?

The current code does not explicitly cancel the worker. Model/search work may continue and consume resources until completion/error. Production research jobs need cancellation flags/deadlines, bounded queues, disconnect-aware cleanup, and a defined reconnect policy. Closing the UI connection is not necessarily canceling the computation.

### 61. Why do custom collections disappear?

The API registry is in process memory and expires custom sessions after two hours without access. Restart/scale-to-zero loses that registry regardless of disk cache. LocalStorage remembers an ID but cannot restore server state. A surviving matching cache can make re-adding sources faster; it does not resurrect the old session automatically.

### 62. Can you scale by adding Uvicorn workers or replicas?

Not safely with the current registry/quotas. Requests could land on a process without that corpus ID, while each process applies independent limits. Externalize collection/session state, artifact/vector storage, job coordination, and rate quotas before scaling out. Single replica settings are a portfolio constraint, not a general scalability solution.

### 63. What are liveness and readiness?

Liveness indicates the handler/process is responding. Readiness checks expected frontend/manifest files and whether the configured model key exists. It does not validate the key, query the model, or load the corpus. A deployment can pass these checks while research or ingestion still fails.

### 64. How is the public demo protected from expensive use?

There are per-client sliding-window counts, two concurrent research slots, two concurrent custom ingestion slots, twenty custom sessions, question/source limits, and fifty videos per collection. These are resource safeguards, not robust authenticated quotas. Forwarded-header trust, global disk growth, total-run budgets, and distributed limits still need work.

### 65. Where are secrets stored?

Runtime keys are read from environment variables loaded locally via `.env` or injected by the hosting platform. Docker excludes `.env` and the image runs as a non-root user. The repo has a local pattern scanner and Gitleaks workflow. Neither a secret scanner nor a non-root user eliminates every application/data-access risk.

### 66. How do you reduce container size and cold starts?

The runtime excludes the heavy legacy ML/UI/chart stack, installs dependencies before source, and warms the ONNX embedding model in a reusable Docker layer. The demo index is baked in, avoiding live fetching during demo initialization. Explain the mechanism without quoting an exact size/startup speed unless you measure the built image.

### 67. How would you support real authenticated users?

Add authentication and collection ownership checks, move metadata/jobs to shared durable storage, store versioned transcripts/chunks, and use shared scoped vector retrieval. A durable queue handles ingestion, while research runs store results/events and support budgets/cancellation. A PostgreSQL/pgvector design is plausible but still needs tenant-scoped recall and operational measurements.

### 68. Is an unguessable UUID authorization?

It reduces brute-force guessing, but in this implementation possession of an ID enables access. It does not prove a user owns the collection or enable account-based auditing/revocation. Treat it as a temporary capability-like identifier; add authenticated ownership for durable/private data.

### 69. What is a concrete frontend bug you found?

The `esc()` regex matches apostrophes but the replacement map omits them. Node reproduces `don't` becoming `donundefinedt`. Also, switching collections while research is active can associate stale output with the new view. Use a complete escaping/sanitization policy and bind responses to run/corpus identity.

### 70. Does escaping HTML make generated Markdown safe?

It helps neutralize raw HTML, but Markdown parsing creates HTML links/images afterward. Protocols and attributes still need validation/sanitization. Marked does not sanitize output itself. Prefer a maintained sanitizer and allowed URL schemes, then exercise rendering with malicious and ordinary inputs.

## Evaluation and tradeoffs

### 71. How did you evaluate the project?

There is a fixed six-video corpus and sixteen reference tasks. Agent and baseline answer each task; an LLM judge scores correctness, deterministic set metrics score tool names, another judge rates trajectories, and usage/latency are tracked. Saved runs and generated reports support inspection. It is a small controlled evaluation, not comprehensive production validation.

### 72. Explain the main result accurately.

The final saved Gemini 3.5 Flash-Lite run has fifteen correct agent answers out of sixteen versus nine baseline answers: 93.75% versus 56.25%, a 37.5-percentage-point lift. It is a historical same-model/same-corpus comparison and does not benchmark the documented current Groq deployment.

### 73. What are the costs of that improvement?

Recorded average latency was about 34.9 seconds versus 5.35, including the run's throttling. Proxy model-cost totals imply about 11.8× higher cost, and agent tokens averaged roughly 9800 versus 722. The price table lacks that model, and service/judge/hosting/retry costs are excluded, so do not present these as invoices.

### 74. Is “task success rate” a strict accuracy percentage?

The code averages 1/0.5/0 judged scores, so partial credit makes it a mean success score. The final run happens to contain only correct/incorrect labels, making fifteen/sixteen appropriate there. Judge failures and engine failures are handled differently and need explicit denominators.

### 75. What is wrong with the failure denominator?

Engine-error rows are not appended to aggregate scoring rows, so their failures vanish from the success mean. An earlier thirteen-task run reports 85.71% from seven scored tasks despite six engine errors. Operational reporting should show completion rate and success over all requested tasks, alongside conditional quality of completed answers.

### 76. How fair is the baseline comparison?

The corpus/embeddings/model/citation format are shared, but the agent receives more searches/evidence/calls and a web tool. It demonstrates benefit of the complete agent system over that simple control, not planning alone. Stronger controls would match evidence budgets and external capability and include fixed-workflow ablations.

### 77. Can single-shot RAG compare videos?

Yes, if its retrieved context contains the needed passages from multiple videos. One global retrieval does not enforce coverage, so it can fail more often, but calling cross-video reasoning impossible is inaccurate. The saved baseline gets three of six cross tasks correct.

### 78. What does tool-call F1 measure?

Overlap between sets of used/expected tool names. It does not grade arguments, frequency, order, or results. Useful inventory calls can count as false positives, while repeated bad calls may escape penalty. Use it as a diagnostic and extend the harness for acceptable strategies and evidence/argument quality.

### 79. What is a weakness of LLM judging here?

Agent/judge use the same model in the headline run, raising correlated-error/self-preference concerns. The trajectory judge sees a lossy representation; it incorrectly calls zero-passage inventory/web results failures. Use independent human samples, better observation representations, repeat evaluations, and explicit judge rubrics/versions.

### 80. Tell me about an evaluation-driven bug fix.

The saved prior run scores zero of three verification tasks despite selecting web tools. The README says web output was lost because synthesis only consumed transcript passages. Current code retains a separate findings channel and passes it to synthesis/verification; the following run gets three of three. Cite this as evidence-path debugging, with the caveat that other outcomes and model randomness changed too.

### 81. What failed even in the best run?

Task `c5`, comparing attention with a plain hidden layer. The agent repeatedly searched, did not use its expected comparison tool, and ultimately said the available evidence lacked a direct comparison. It scored zero for task success but 1.0 grounded ratio. Analyze retrieval, truncation, and synthesis before blaming only the model.

### 82. What do the existing tests prove?

All twenty-two tests pass; they cover mocked source/provider/API/client contracts. They do not prove live model quality, precise citations, full summaries, safe rendering, or horizontal scalability. The additional network-disabled audit reproduces mechanics and retrieval/provenance bugs using local data; it is not a fresh live benchmark.

### 83. What would you measure next?

Timestamp-range accuracy, gold passage/video recall, citation support/completeness, summary/highlight coverage, provider failure/timeout rate, p50/p95 latency, cost categories, and budget/cancellation behavior. Expand to held-out collections, repeat paired trials, preserve failed tasks in denominators, and save code/configuration/artifact fingerprints.

### 84. How would you improve cost without losing quality?

Fix evidence access first, route genuinely easy lookups to a simpler workflow, avoid redundant planning/inventory, and evaluate whether the comparison-generation call adds value. Cap evidence/token/time budgets and cache reusable information where appropriate. Use measured task slices to choose tradeoffs, not just a cheaper model name.

## Legacy ML and behavioral discussion

### 85. How did the legacy BERT topics work?

Caption windows were truncated, encoded with pretrained BERT `[CLS]` vectors, and grouped with K-means. An LLM labeled up to three sample texts per cluster. Importance came from cluster size, which measures segment frequency rather than value. It was unsupervised grouping plus generated labeling, not supervised topic training.

### 86. How were chapters detected?

Separate from BERT clustering: roughly thirty-second windows were FastEmbed-encoded, and low cosine similarity to the previous window and current topic anchor triggered a boundary. The “centroid” was not updated as a running mean. An LLM titled boundary windows, so these were heuristic approximate chapter markers.

### 87. Why not include BERT in the current runtime?

The current research path uses pretrained sentence embeddings and tools rather than BERT clustering. Keeping torch/transformers out of the image reduces dependencies and memory/cold-start pressure. Legacy features remain optional. Removing a heavy feature from the deployed path can be an intentional scope/resource decision.

### 88. What was the hardest challenge?

Pick your actual experience. Strong supported topics include maintaining evidence/provenance through tools, handling cloud caption blocking, avoiding quota failures, and measuring quality fairly. Explain the concrete symptom, how you localized it, what changed, and how you checked it. Do not recite a repository bug as personal experience if you did not work on it.

### 89. What would you change if starting again?

A defensible answer is caption-aware provenance and immutable chunk IDs from the start, explicit versioned cache/model configuration, evidence coverage tests, typed verification, and a more honest evaluation record. Keep the simple deployment while it meets scope, and externalize state when durability/authenticated users become requirements.

### 90. What did this project teach you?

Connect architecture to evidence: more tools can increase capability and cost together; a correct citation identity can still have bad time alignment; grounding and task success differ; and a judge can be wrong because its input representation is incomplete. Good AI application engineering includes measuring and debugging data contracts, not only prompting.

## Ten debugging drills

These drills extend the ninety questions above. Practice walking through the source without running paid calls.

1. **“The answer is reasonable but its video jumps five minutes late.”** Inspect actual chunk offsets, overlap, caption boundaries, stored metadata, dedupe, and cache version. Compare retrieved text with captions before blaming synthesis. Rebuild old caches after repair.
2. **“A video is listed, but scoped searches return no results.”** Verify resolved ID/title, available chunks, global candidate set, and post-filter count. Compare normal `fetch_k=20` with full candidates to localize starvation, then inspect relevance.
3. **“The web tool says SUPPORTED, but the final answer refuses.”** Trace tool artifact/content into `findings`, synthesis input, evidence formatting, and model output. Do not assume a zero transcript-passage count means web failure.
4. **“Every collection summary describes only the first topic.”** Inspect the action query and returned video/span coverage. The issue may be global top-k and 320-character snippet truncation. Compare a per-video coverage workflow on held-out examples.
5. **“Users can research locally but ingestion fails in Azure.”** Separate title lookup, playlist enumeration, caption provider, network blocking, keys/quota, and ONNX memory. Confirm `auto` selection; do not blindly increase all retries.
6. **“Collections randomly return 404 after scaling.”** Inspect process/replica routing, restarts and expiry. The ID exists only in one registry. Persistent vector files alone do not restore session metadata; shared state is needed.
7. **“Research says done planning, yet a logged tool never ran.”** Count planner rounds and check `steps < max_steps`. Distinguish requested tool calls from executed log entries and report the cap stop reason.
8. **“Our success percentage went up during an outage.”** Inspect requested/completed/scored counts and whether engine errors were excluded. Report all-attempt success alongside conditional completed-answer quality.
9. **“Memory spikes on two long playlists.”** Measure all-doc text/vector construction, two ONNX batches, concurrent jobs and model thread counts. Bound work/length and batch/concurrency; a fifty-video cap does not cap transcript duration.
10. **“A browser closes but our model bills continue.”** Trace worker lifetime, disconnect handling, semaphore release and cancellation. Add research run IDs/deadlines/flags rather than assuming EventSource.close cancels provider calls.

## A useful answer structure

Use **problem → implementation → tradeoff → evidence → next improvement**. For example: “We needed video-specific evidence. `Corpus.search` scopes the unified FAISS index by video metadata. That keeps the corpus simple, but its wrapper filters twenty global candidates. The offline check reproduced missed in-scope results, so I would change scoped candidate retrieval and measure recall before introducing a new database.”

If you do not know an answer, say what you can establish from the code and how you would investigate the remaining point. A well-supported limitation is better than a fabricated benchmark or claim about a library.
