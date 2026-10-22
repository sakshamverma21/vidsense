"""Single-shot RAG baseline — the control the agent is measured against.

Same corpus, same embeddings, same citation format as the agent, but ONE
retrieval + ONE LLM call: no planning, no tool selection, no cross-video logic.
This isolates what the agent's extra machinery actually buys. Its output dict
mirrors ResearchAgent.run() so the eval runner can treat both uniformly.
"""
from __future__ import annotations

from time import perf_counter
from typing import Optional

from . import config
from .agent import UsageTracker
from .citations import dedupe, doc_to_citation, format_passages
from .utils import to_text

BASELINE_PROMPT = """Answer the question using ONLY the evidence passages below, \
which were retrieved from video transcripts. Each passage has a citation tag like \
[title @ MM:SS]. Cite the supporting passage after each claim. If the evidence is \
insufficient, say so.

QUESTION:
{question}

EVIDENCE:
{evidence}

ANSWER:"""


class SingleShotRAG:
    def __init__(self, corpus, model_type: str = config.DEFAULT_PROVIDER, model_name: Optional[str] = None, k: int = 6, rate_limiter=None):
        self.corpus = corpus
        self.llm = config.make_llm(model_type, model_name, rate_limiter=rate_limiter)
        self.k = k

    def run(self, question: str) -> dict:
        tracker = UsageTracker()
        t0 = perf_counter()
        docs = self.corpus.search(question, k=self.k)
        cits = dedupe([doc_to_citation(d) for d in docs])
        prompt = BASELINE_PROMPT.format(question=question, evidence=format_passages(cits))
        resp = self.llm.invoke(prompt, config={"callbacks": [tracker]})
        latency = perf_counter() - t0
        answer = to_text(resp)
        return {
            "question": question,
            "answer": answer,
            "citations": cits,
            "trajectory": [{"type": "retrieve", "num_passages": len(cits)}],
            "tool_log": [],
            "verification": {},
            "usage": {
                "input_tokens": tracker.input_tokens,
                "output_tokens": tracker.output_tokens,
                "total_tokens": tracker.input_tokens + tracker.output_tokens,
                "llm_calls": tracker.llm_calls,
            },
            "latency": latency,
            "num_tool_calls": 0,
        }
