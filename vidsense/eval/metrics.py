"""Evaluation metrics for the research agent.

  * task_success       — LLM-judged correctness vs a reference answer (0 / .5 / 1)
  * tool_call_accuracy — set precision/recall/F1 of tools used vs expected
  * trajectory_quality — LLM-judged 1–5 rubric (efficiency, tool choice, stopping)
  * cost_usd           — dollars from token usage via a per-model price table
  * grounded_ratio     — comes straight from the agent's verifier

All LLM-judge calls fail open (return None) rather than crash a run.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from ..utils import to_text

# Approximate list prices (USD per 1M tokens). Adjust to current rates; the point
# is a transparent, reproducible cost methodology, not to-the-cent accuracy.
# (Gemini flash models have a generous FREE tier; these are paid-tier list prices
# used only so cost comparisons are apples-to-apples.)
PRICES = {
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "default": {"input": 0.10, "output": 0.40},
}


def cost_usd(usage: Dict, model: str = "gemini-2.0-flash") -> float:
    price = PRICES.get(model, PRICES["default"])
    return round(
        usage.get("input_tokens", 0) / 1e6 * price["input"]
        + usage.get("output_tokens", 0) / 1e6 * price["output"],
        6,
    )


def _extract_json(text: str) -> Optional[dict]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except Exception:
        return None


# --- Task success (LLM judge) ------------------------------------------------
_SUCCESS_PROMPT = """You are grading an answer against a reference answer.

Decide if the CANDIDATE answers the QUESTION as well as the REFERENCE:
- "correct"   = captures the key facts of the reference, no significant errors
- "partial"   = partially correct or missing important points
- "incorrect" = wrong, off-topic, or says it cannot answer

Reply with strict JSON: {{"label": "correct|partial|incorrect", "reason": "short"}}

QUESTION: {question}

REFERENCE: {reference}

CANDIDATE: {candidate}

JSON:"""

_LABEL_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


def task_success(question: str, candidate: str, reference: str, judge_llm) -> Dict:
    prompt = _SUCCESS_PROMPT.format(question=question, reference=reference, candidate=candidate)
    try:
        resp = judge_llm.invoke(prompt)
        data = _extract_json(to_text(resp)) or {}
        label = str(data.get("label", "")).lower()
        return {"score": _LABEL_SCORE.get(label, 0.0), "label": label or "unknown", "reason": data.get("reason", "")}
    except Exception as e:
        return {"score": None, "label": "error", "reason": str(e)}


# --- Tool-call accuracy (deterministic) --------------------------------------
def tool_call_accuracy(actual_tools: List[str], expected_tools: List[str]) -> Dict:
    """Set-based precision/recall/F1 of tools used vs expected, plus coverage."""
    actual, expected = set(actual_tools), set(expected_tools)
    if not expected:
        return {"precision": None, "recall": None, "f1": None, "exact_set": None, "coverage": None}
    tp = len(actual & expected)
    precision = tp / len(actual) if actual else 0.0
    recall = tp / len(expected)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "exact_set": actual == expected,
        "coverage": round(recall, 3),  # fraction of expected tools that were used
        "actual": sorted(actual),
        "expected": sorted(expected),
    }


# --- Trajectory quality (LLM judge) ------------------------------------------
_TRAJ_PROMPT = """Rate the QUALITY of a research agent's trajectory on a 1-5 scale.

Consider: did it pick appropriate tools, avoid redundant/unnecessary calls, order
steps sensibly, and stop at the right time (neither too early nor looping)?
5 = crisp and efficient; 1 = confused, wasteful, or wrong tools.

Reply with strict JSON: {{"score": 1-5, "reason": "short"}}

QUESTION: {question}

TRAJECTORY (each step the agent took):
{trajectory}

JSON:"""


def _render_trajectory(trajectory: List[Dict]) -> str:
    lines = []
    for s in trajectory:
        if s["type"] == "plan":
            calls = ", ".join(f"{t['name']}({t['args']})" for t in s.get("tool_calls", [])) or "(final answer, no tool)"
            lines.append(f"PLAN -> {calls}")
        elif s["type"] == "tool":
            lines.append(f"TOOL {s['name']}({s['args']}) -> {s.get('num_passages', 0)} passages")
        else:
            lines.append(s["type"].upper())
    return "\n".join(lines)


def trajectory_quality(question: str, trajectory: List[Dict], judge_llm) -> Dict:
    prompt = _TRAJ_PROMPT.format(question=question, trajectory=_render_trajectory(trajectory))
    try:
        resp = judge_llm.invoke(prompt)
        data = _extract_json(to_text(resp)) or {}
        score = float(data.get("score")) if data.get("score") is not None else None
        norm = (score - 1) / 4 if score is not None else None
        return {"score_1_5": score, "normalized": round(norm, 3) if norm is not None else None, "reason": data.get("reason", "")}
    except Exception as e:
        return {"score_1_5": None, "normalized": None, "reason": str(e)}
