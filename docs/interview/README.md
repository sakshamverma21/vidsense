# VidSense interview preparation

Prepared across 4–5 October 2026 from the full application source in `vidsense-gpt`, the installed dependency implementations, the bundled transcript/index cache, all seven saved evaluation runs, and the existing tests.

## Read in this order

1. [Project deep dive](<C:/Users/saksh/OneDrive/Desktop/vidsense-gpt/docs/interview/01-project-deep-dive.md>) — what the system does, every layer of the pipeline, the agent, retrieval, frontend, backend, legacy implementation, and a file-by-file map.
2. [Design tradeoffs and audit](<C:/Users/saksh/OneDrive/Desktop/vidsense-gpt/docs/interview/02-design-tradeoffs-and-audit.md>) — alternatives to each major technical choice, verified results, bugs, limitations, and a production redesign.
3. [Interview questions and answers](<C:/Users/saksh/OneDrive/Desktop/vidsense-gpt/docs/interview/03-interview-questions.md>) — practice answers, technical follow-ups, debugging scenarios, and questions designed to challenge the headline claims.
4. [Revision sheet](<C:/Users/saksh/OneDrive/Desktop/vidsense-gpt/docs/interview/04-revision-sheet.md>) — pitches, exact numbers, concepts to remember, and an active study plan.

The documents use three evidence levels:

- **Implemented:** directly established by source code or an offline reproduction.
- **Documented rationale:** stated in project comments or documentation; it is evidence of intended rationale, not a record of a measured comparison.
- **Engineering interpretation / proposed:** a defensible explanation or future option. Do not present it as an experiment you personally performed.

There is no usable Git history in the analyzed checkout. The other supplied workspace contains an empty Git repository. Consequently, personal authorship, exact development chronology, prior interviews, and rejected approaches cannot be established. Fill in your own contribution and decisions from your actual experience.

## What was checked

- Read all original project Python, JavaScript, HTML, CSS, requirements, deployment documents, workflows, test files, task definitions, and report generation code.
- Parsed all seven saved runs, including individual answers, trajectories, errors, metrics, and task types.
- Inspected all six cached transcript records and the 147-document FAISS store; compared stored timestamp metadata with the original text offsets.
- Checked installed FastEmbed and FAISS wrapper behavior rather than assuming their defaults.
- Ran the 22 existing tests: all passed. Checked frontend JavaScript syntax: passed. Ran the project's secret scanner: passed.
- Ran an additional offline audit with network connections disabled: real local embeddings, FAISS retrieval, deterministic LangChain chat responses, LangGraph invocation/streaming, nested token callbacks, and parser edge cases.
- Verified relevant concepts against primary documentation. External sources are linked near the associated explanations.

No application implementation was changed. No paid model calls, transcript requests, new benchmark runs, deployments, messages, or infrastructure changes were made. The `.env` contents were not printed or included in these materials.

## Reproduce the offline audit

From the project root in PowerShell:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
node --check frontend/app.js
.\venv\Scripts\python.exe docs/interview/audit.py
```

The audit reads the existing project-owned FAISS pickle and the already downloaded ONNX model. Never reuse that deserialization path with an untrusted index. It writes only [audit_results.json](<C:/Users/saksh/OneDrive/Desktop/vidsense-gpt/docs/interview/audit_results.json>); the implementation is [audit.py](<C:/Users/saksh/OneDrive/Desktop/vidsense-gpt/docs/interview/audit.py>).

## The central interview message

VidSense is a transcript-based research application over YouTube collections. It combines local semantic retrieval, an LLM-directed tool loop, source metadata, and an evaluation harness. Its strongest story is the relationship between architecture and measurement: the tools broaden what it can answer, the baseline exposes the cost of that extra capability, and evaluation helps identify failures.

Explain the actual scope carefully: it processes captions rather than video frames; timestamp links are currently approximate and affected by a verified offset bug; the verifier reports evidence support without correcting answers; customer summaries/highlights use a single-shot path; and active custom collections are temporary. These qualifications make your explanation technically stronger.
