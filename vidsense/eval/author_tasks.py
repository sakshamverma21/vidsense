"""Draft candidate eval tasks from the corpus transcripts (the 'hybrid' half of
authoring: the LLM proposes, a human curates).

Writes eval/tasks_candidates.json — DRAFTS, not ground truth. Review each,
fix the reference answer against the actual video, then move keepers into
tasks.json. This is how you scale the task set to 25–40 without hand-writing
every question from scratch.

Usage:
    python -m vidsense.eval.author_tasks --single 8 --cross 6
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from .. import config
from ..corpus import load_corpus
from ..utils import to_text

EVAL_DIR = Path(__file__).resolve().parent

_SINGLE_PROMPT = """From this transcript excerpt of the video titled "{title}", \
write ONE factual question a viewer might ask, plus a short reference answer that \
is fully supported by the excerpt.

Excerpt:
{excerpt}

Reply as strict JSON: {{"question": "...", "reference_answer": "..."}}"""

_CROSS_PROMPT = """Two videos: "{title_a}" and "{title_b}". Write ONE question that \
requires COMPARING or CONNECTING how both videos treat a shared topic, plus a short \
reference answer. Only use topics plausibly covered by both.

Excerpt A: {excerpt_a}

Excerpt B: {excerpt_b}

Reply as strict JSON: {{"question": "...", "reference_answer": "..."}}"""


def _json(text: str):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group()) if m else None


def _excerpt(record, n_chars: int = 1200) -> str:
    text = record.full_text
    if len(text) <= n_chars:
        return text
    start = random.randint(0, len(text) - n_chars)
    return text[start : start + n_chars]


def draft(n_single: int = 8, n_cross: int = 6, manifest=EVAL_DIR / "corpus.json") -> Path:
    corpus = load_corpus(manifest)
    llm = config.make_llm("groq", temperature=0.4)
    records = list(corpus.videos.values())
    candidates = []

    for i in range(n_single):
        rec = random.choice(records)
        try:
            resp = llm.invoke(_SINGLE_PROMPT.format(title=rec.title, excerpt=_excerpt(rec)))
            data = _json(to_text(resp))
            if data:
                candidates.append(
                    {
                        "id": f"draft_s{i+1}",
                        "type": "single",
                        "question": data["question"],
                        "reference_answer": data["reference_answer"],
                        "expected_tools": ["transcript_search"],
                        "gold_video_ids": [rec.video_id],
                        "_source_video": rec.title,
                    }
                )
        except Exception as e:  # noqa: BLE001
            print(f"single {i}: {e}")

    for i in range(n_cross):
        if len(records) < 2:
            break
        a, b = random.sample(records, 2)
        try:
            resp = llm.invoke(
                _CROSS_PROMPT.format(
                    title_a=a.title, title_b=b.title, excerpt_a=_excerpt(a, 800), excerpt_b=_excerpt(b, 800)
                )
            )
            data = _json(to_text(resp))
            if data:
                candidates.append(
                    {
                        "id": f"draft_c{i+1}",
                        "type": "cross",
                        "question": data["question"],
                        "reference_answer": data["reference_answer"],
                        "expected_tools": ["cross_video_compare"],
                        "gold_video_ids": [a.video_id, b.video_id],
                        "_source_videos": [a.title, b.title],
                    }
                )
        except Exception as e:  # noqa: BLE001
            print(f"cross {i}: {e}")

    out = EVAL_DIR / "tasks_candidates.json"
    out.write_text(json.dumps({"note": "DRAFTS — curate before use", "tasks": candidates}, indent=2), encoding="utf-8")
    print(f"✅ Drafted {len(candidates)} candidate tasks -> {out}")
    print("   Review/fix reference answers against the videos, then move keepers into tasks.json.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", type=int, default=8)
    ap.add_argument("--cross", type=int, default=6)
    args = ap.parse_args()
    draft(args.single, args.cross)


if __name__ == "__main__":
    main()
