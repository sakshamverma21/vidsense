# VidSense interview revision sheet

Read this after the [deep dive](<C:/Users/saksh/OneDrive/Desktop/vidsense-gpt/docs/interview/01-project-deep-dive.md>) and use the [question bank](<C:/Users/saksh/OneDrive/Desktop/vidsense-gpt/docs/interview/03-interview-questions.md>) for active recall.

## A 30-second project pitch

“VidSense helps users research collections of YouTube videos through their captions. It builds a local semantic index and uses an LLM-directed tool loop for questions, returning answers with video source links and an evidence-support report. I also built a baseline comparison and evaluation workflow to understand when the agent's additional retrieval and tools justify its cost.”

Use “I built” only for your actual contribution. Replace it with a precise account of your role if this was collaborative or assisted work.

## A two-minute technical explanation

“There are separate ingestion and research pipelines. Ingestion validates YouTube links, expands playlists, deduplicates videos, and fetches captions through direct or managed access. It creates overlapping text chunks and attaches video metadata. FastEmbed runs a pretrained ONNX model locally, and a collection-level FAISS index stores the vectors. Transcripts and indexes are cached on disk.

“For questions, LangGraph controls planner, executor, synthesizer, and verifier nodes. The planner selects tools from previous observations. Tool outputs have readable content and structured evidence artifacts, so source metadata travels through the graph. Synthesis writes from accumulated evidence, and verification reports which extracted claims are supported. The API uses SSE to stream completed steps and the full answer. Summary/highlight buttons use a faster single-retrieval workflow.

“The historical evaluation used sixteen tasks and a simple same-corpus RAG baseline. The final saved Gemini run scored fifteen correct answers versus nine, but used more tokens and time. The evaluation is small and LLM-judged, with unmatched web capability and proxy cost pricing, so I describe the result narrowly. The demo also has temporary sessions, and the current timestamp mapping needs an overlap-offset repair. For production I would improve evidence accuracy first, then add durable authenticated collections and jobs.”

## Draw this from memory

```text
Sources -> resolve/deduplicate -> captions -> chunks + metadata
        -> FastEmbed -> FAISS + records cache -> Corpus

Question -> planner -> executor -> planner ... -> synthesizer -> verifier
                         |                         ^             |
                         +-> evidence + findings --+             v
                                                     answer + source cards

Summary/highlights -> one corpus retrieval -> one LLM call

API registry + threads + locks/semaphore -> SSE -> browser
Fixed tasks -> agent + baseline -> metrics -> JSON/report
```

## Exact facts worth remembering

| Item | Fact |
|---|---|
| Data analyzed | Caption text; no frame/OCR/audio analysis |
| Demo corpus | Six deep-learning videos; 147 indexed chunks |
| Embeddings | Installed default BGE small English v1.5, 384 dimensions, ONNX |
| Chunking | 1000 characters, up to 200 overlap |
| Embedding batch | 16 |
| FAISS index | Exact IndexFlatL2; normalized vectors make L2/cosine ranking equivalent |
| Scoped candidate count | Wrapper default fetch_k=20 before metadata filtering |
| Tool retrieval sizes | Search 5; timestamp 2; comparison 2/video; web up to 4 |
| Prompt passage length | First 320 characters per chunk |
| API action retrieval | k=10; evaluation baseline k=6 |
| Planner default | Six calls maximum; at most five executor rounds |
| Tools | Inventory, transcript search, timestamp lookup, comparison, external verification |
| Session TTL | Two hours idle; process-local; demo exempt |
| Collection limit | Fifty unique video specs; twenty custom sessions |
| Concurrent work | Two custom ingestions, two research runs |
| Rate limits | Four ingestion/twenty research requests per 300 seconds per client scope |
| Model timeout | 45 seconds per configured LLM request |
| Managed transcript timeout/spacing | 90 seconds; 1.1-second interval after previous completion |
| Streaming | Completed steps and full answer, then done; no token streaming |
| Historical model | Gemini 3.5 Flash-Lite; documented portfolio provider is Groq |
| Final historical score | Agent 15/16, baseline 9/16; +37.5 percentage points |
| Task slices | Single 7/7 vs 5/7; cross 5/6 vs 3/6; verify 3/3 vs 1/3 |
| Historical latency | About 34.9 s vs 5.35 s, includes throttling |
| Historical model-cost proxy | About 11.8× using totals; price fallback, incomplete cost categories |
| Existing tests | 22 passed; mocked contracts, not live answer-quality proof |
| Largest audited timestamp drift | 317.847 seconds against caption text in bundled cache |

## Technical distinctions interviewers may probe

- **Embedding model vs generator:** vectors for retrieval versus text generation.
- **Vector library vs database:** FAISS similarity search versus ownership/durability/operations.
- **Exact search vs complete scoped retrieval:** exact global distances can still miss filtered results after candidate truncation.
- **Caption timestamp vs exact quote alignment:** passage localization does not imply word precision.
- **Groundedness vs correctness vs completeness:** all can differ independently.
- **External fact checking vs answer verification:** outside truth evidence versus support in gathered evidence.
- **Requested tool vs executed tool:** the cap can drop a planned action.
- **Graph reuse vs memory:** cached agent objects do not preserve prior conversation.
- **Disk cache vs active session:** cached artifacts do not restore process-local IDs.
- **Async I/O vs threads vs processes:** event-loop coordination, blocking-work offloading, and independent address spaces.
- **HTTP rate limit vs provider RPM limit:** client-request quotas versus model-call quotas.
- **SSE vs token streaming:** transport can carry tokens, but this implementation carries node updates.
- **Success score vs accuracy:** partial-credit mean and skipped failures require explicit denominator.
- **Percentage points vs relative percent:** +37.5 points is different from a 37.5% relative improvement.
- **Configured support vs working provider:** the old Ollama object lacks agent tool binding.

## Claims to replace with precise wording

| Avoid claiming | Say instead |
|---|---|
| “We trained a model on videos.” | “We indexed captions with pretrained embeddings and used an existing LLM.” |
| “It watches the videos.” | “It researches their timestamped captions.” |
| “Every link is the exact second of the claim.” | “It produces caption/chunk-based links; current offset drift needs repair.” |
| “The verifier eliminates hallucinations.” | “It reports judged evidence support without changing the answer.” |
| “All features run the agent and verifier.” | “Questions do; customer overview/highlights use single-shot RAG.” |
| “One retrieval cannot compare videos.” | “One global retrieval lacks guaranteed cross-video coverage.” |
| “We proved 93.8% accuracy in production.” | “A historical sixteen-task same-model evaluation scored fifteen correct agent answers.” |
| “The cost is exactly twelve times.” | “The recorded proxy estimate is roughly twelve times, with pricing/category caveats.” |
| “We can scale horizontally now.” | “Current temporary sessions require one process; shared state is a production roadmap.” |
| “BERT chapters are in the live web app.” | “BERT topics and heuristic chapters are legacy module features.” |
| “All these alternatives were benchmarked.” | “These are engineering alternatives; only the saved experiments are evidenced.” |

## Practice plan

**Pass 1: understand.** Read the deep dive, draw the architecture without looking, and trace one source from URL to a cited passage. Open `sources.py`, `corpus.py`, `tools.py`, and `agent.py` as you explain. Define every unfamiliar term in one sentence.

**Pass 2: defend.** Read the choice/audit document. For each major choice, name one viable alternative, when you would switch, and the current cost. Recompute fifteen/sixteen, nine/sixteen, the lift, and normalized trajectory score. Explain why cost and error denominators need care.

**Pass 3: debug.** Work through the ten drills. Start each answer with the observation you would inspect and the hypothesis it tests. Use the offline audit result as evidence, and distinguish a proposed fix from existing code.

**Pass 4: rehearse.** Answer questions aloud in shuffled order. Aim for 45–90 seconds, then handle a follow-up. Record your pitch and listen for exaggerated scope, vague library lists, unsupported personal claims, or benchmarks without conditions.

**Before the interview:** know your actual role, one debugging story you experienced, one deliberate scope tradeoff, the historical result with caveats, and your next improvement. Practice explaining a limitation calmly and specifically.

## Questions to personalize yourself

Write down your real answers to: Why did you choose this problem? Which modules did you personally own? What was the first working version? Which failures did you encounter directly? Which alternative did you actually try? What feedback changed the product? What would you do differently with more time?

The repository does not establish those personal facts. Technical preparation is strongest when the architecture and your own experience tell the same truthful story.
