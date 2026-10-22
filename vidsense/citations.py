"""Grounded-citation helpers + claim verification.

A *citation* is a retrieved transcript passage carrying enough metadata to link
back to the exact moment in a video: {video_id, title, timestamp,
formatted_time, youtube_link, text}. Tools emit these; the agent collects them;
the synthesizer cites them; the verifier checks the final answer against them.
"""
from __future__ import annotations

import re
from typing import Dict, List

from langchain_core.documents import Document

from .utils import to_text


# --- Building citations ------------------------------------------------------
def doc_to_citation(doc: Document) -> Dict:
    m = doc.metadata
    return {
        "video_id": m.get("video_id", ""),
        "title": m.get("title", m.get("video_id", "")),
        "timestamp": float(m.get("timestamp", 0.0)),
        "formatted_time": m.get("formatted_time", "00:00"),
        "youtube_link": m.get("youtube_link", ""),
        "text": doc.page_content,
    }


def tag(citation: Dict) -> str:
    """Short inline citation tag, e.g. `[But what is a neural network? @ 04:06]`."""
    return f"[{citation['title']} @ {citation['formatted_time']}]"


def format_passages(citations: List[Dict], header: str | None = None) -> str:
    """Render passages as a readable, citation-tagged block for the LLM to read."""
    if not citations:
        return "(no matching passages found)"
    lines = [header] if header else []
    for i, c in enumerate(citations, 1):
        snippet = " ".join(c["text"].split())
        if len(snippet) > 320:
            snippet = snippet[:320] + "…"
        lines.append(f"{i}. {tag(c)} {snippet}")
    return "\n".join(lines)


def dedupe(citations: List[Dict]) -> List[Dict]:
    """Drop duplicate passages (same video + timestamp)."""
    seen = set()
    out = []
    for c in citations:
        key = (c["video_id"], round(c["timestamp"]))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


# --- Claim verification (hallucination control) ------------------------------
_VERIFY_PROMPT = """You are checking whether an answer is grounded in evidence \
retrieved from video transcripts.

For each numbered CLAIM, decide if it is SUPPORTED by the EVIDENCE passages.
Reply with strict JSON: a list of objects
{{"claim_index": int, "supported": true/false, "reason": "short"}}.

EVIDENCE:
{evidence}

CLAIMS:
{claims}

JSON:"""


def split_claims(answer: str) -> List[str]:
    """Naively split an answer into sentence-level claims for checking."""
    # Strip inline citation tags before splitting so they don't fragment sentences.
    text = re.sub(r"\[[^\]]*@[^\]]*\]", "", answer)
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 25]


def verify_answer(answer: str, citations: List[Dict], llm, extra_evidence: str = "") -> Dict:
    """Check each claim in `answer` against `citations` (+ optional extra evidence).

    `extra_evidence` carries non-passage grounding such as web-verification verdicts,
    so claims backed by `external_verify` count as supported rather than being
    flagged as hallucinations.

    Returns {"grounded_ratio": float, "claims": [{claim, supported, reason}]}.
    Fail-open: if parsing fails, report grounded_ratio=None rather than crash.
    """
    import json

    claims = split_claims(answer)
    if not claims:
        return {"grounded_ratio": None, "claims": []}

    evidence = format_passages(citations)
    if extra_evidence.strip():
        evidence += "\n\nExternal findings (web verification):\n" + extra_evidence
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims))
    prompt = _VERIFY_PROMPT.format(evidence=evidence, claims=numbered)

    try:
        resp = llm.invoke(prompt)
        raw = to_text(resp)
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        verdicts = json.loads(match.group()) if match else []
        by_idx = {v["claim_index"]: v for v in verdicts}
        results = []
        supported = 0
        for i, claim in enumerate(claims):
            v = by_idx.get(i, {})
            ok = bool(v.get("supported", False))
            supported += int(ok)
            results.append({"claim": claim, "supported": ok, "reason": v.get("reason", "")})
        return {"grounded_ratio": supported / len(claims), "claims": results}
    except Exception as e:
        return {"grounded_ratio": None, "claims": [], "error": str(e)}
