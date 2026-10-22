"""Customer-facing research actions built on the existing grounded agent."""
from __future__ import annotations

from typing import Literal

ActionMode = Literal["question", "summary", "highlights"]


def action_question(mode: str, question: str = "") -> str:
    """Turn an explicit UI action into a clear, grounded research request."""
    question = question.strip()
    if mode == "summary":
        return (
            "Create a concise, customer-friendly overview of this entire video collection. "
            "Cover the major themes and the most useful takeaways across the selected videos. "
            "Use headings for Overview and Key takeaways, and cite every transcript-based claim."
        )
    if mode == "highlights":
        return (
            "Find exactly five distinct, high-value highlights across this video collection when the evidence allows. "
            "For each, give a short title, explain why the moment matters, and cite its exact video and timestamp. "
            "Avoid duplicate moments and write it as a scannable numbered list."
        )
    return question
