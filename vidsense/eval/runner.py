"""Eval runner: execute the agent (and the single-shot RAG baseline) over the
task set, score every metric, aggregate, and persist the run.

Usage:
    python -m vidsense.eval.runner                # all tasks, agent + baseline
    python -m vidsense.eval.runner --limit 4      # quick smoke run
    python -m vidsense.eval.runner --systems agent
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import List, Optional

from .. import config
from ..agent import ResearchAgent
from ..baseline import SingleShotRAG
from ..corpus import load_corpus
from . import metrics

EVAL_DIR = Path(__file__).resolve().parent


def _retry(fn, tries: int = 3, backoff: float = 4.0):
    """Run fn() with simple exponential backoff (Groq rate limits, transient nets)."""
    last = None
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < tries - 1:
                time.sleep(backoff * (attempt + 1))
    raise last


def _mean(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return round(mean(vals), 4) if vals else None


def score_result(task: dict, result: dict, judge_llm, model: str, is_agent: bool) -> dict:
    """Compute all metrics for one system's answer to one task."""
    success = metrics.task_success(task["question"], result["answer"], task["reference_answer"], judge_llm)
    row = {
        "success_score": success["score"],
        "success_label": success["label"],
        "cost_usd": metrics.cost_usd(result["usage"], model),
        "latency_s": round(result["latency"], 3),
        "total_tokens": result["usage"]["total_tokens"],
        "num_tool_calls": result["num_tool_calls"],
        "num_citations": len(result["citations"]),
    }
    if is_agent:
        actual_tools = [t["name"] for t in result["tool_log"]]
        row["tool_accuracy"] = metrics.tool_call_accuracy(actual_tools, task.get("expected_tools", []))
        row["trajectory_quality"] = metrics.trajectory_quality(task["question"], result["trajectory"], judge_llm)
        row["grounded_ratio"] = (result.get("verification") or {}).get("grounded_ratio")
    return row


def aggregate(rows: List[dict], is_agent: bool) -> dict:
    agg = {
        "task_success_rate": _mean([r["success_score"] for r in rows]),
        "avg_cost_usd": _mean([r["cost_usd"] for r in rows]),
        "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 6),
        "avg_latency_s": _mean([r["latency_s"] for r in rows]),
        "avg_tokens": _mean([r["total_tokens"] for r in rows]),
        "avg_tool_calls": _mean([r["num_tool_calls"] for r in rows]),
    }
    if is_agent:
        agg["avg_tool_f1"] = _mean([r["tool_accuracy"].get("f1") for r in rows if "tool_accuracy" in r])
        agg["avg_tool_coverage"] = _mean([r["tool_accuracy"].get("coverage") for r in rows if "tool_accuracy" in r])
        agg["tool_exact_set_rate"] = _mean(
            [1.0 if r["tool_accuracy"].get("exact_set") else 0.0 for r in rows if "tool_accuracy" in r]
        )
        agg["avg_trajectory_quality"] = _mean([r["trajectory_quality"].get("normalized") for r in rows if "trajectory_quality" in r])
        agg["avg_grounded_ratio"] = _mean([r.get("grounded_ratio") for r in rows])
    return agg


def run_eval(
    tasks_path: str | Path = EVAL_DIR / "tasks.json",
    systems: tuple = ("agent", "baseline"),
    limit: Optional[int] = None,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    judge_model: Optional[str] = None,
    rpm: Optional[float] = None,
    rebuild_corpus: bool = False,
    verbose: bool = True,
) -> dict:
    tasks_path = Path(tasks_path)
    spec = json.loads(tasks_path.read_text(encoding="utf-8"))
    tasks = spec["tasks"][: limit or len(spec["tasks"])]
    manifest = tasks_path.parent / spec.get("corpus", "corpus.json")

    provider = provider or config.DEFAULT_PROVIDER
    model = model_name or config.DEFAULT_MODEL.get(provider)
    judge_model = judge_model or model

    # One SHARED rate limiter across agent + baseline + judge so they respect the
    # provider's per-project RPM cap together (free tiers limit per project+model).
    rate_limiter = config.make_rate_limiter(rpm) if rpm else None

    corpus = load_corpus(manifest, rebuild=rebuild_corpus, verbose=verbose)
    judge_llm = config.make_llm(provider, judge_model, temperature=0.0, rate_limiter=rate_limiter)

    engines = {}
    if "agent" in systems:
        engines["agent"] = ResearchAgent(corpus, model_type=provider, model_name=model, rate_limiter=rate_limiter)
    if "baseline" in systems:
        engines["baseline"] = SingleShotRAG(corpus, model_type=provider, model_name=model, rate_limiter=rate_limiter)

    per_task = []
    rows_by_system = {name: [] for name in engines}
    tavily = config.has_tavily()

    for i, task in enumerate(tasks, 1):
        if verbose:
            print(f"\n[{i}/{len(tasks)}] {task['id']} ({task['type']}): {task['question'][:70]}")
        if task.get("requires_tavily") and not tavily and verbose:
            print("   (note: this task expects external_verify but TAVILY_API_KEY is unset)")

        entry = {"task_id": task["id"], "type": task["type"], "question": task["question"], "systems": {}}
        for name, engine in engines.items():
            try:
                result = _retry(lambda e=engine: e.run(task["question"]))
                row = score_result(task, result, judge_llm, model, is_agent=(name == "agent"))
                row["answer"] = result["answer"]
                if name == "agent":
                    row["trajectory"] = result["trajectory"]
                rows_by_system[name].append(row)
                entry["systems"][name] = row
                if verbose:
                    extra = f", traj={row['trajectory_quality'].get('normalized')}" if name == "agent" else ""
                    print(f"   {name:8s} success={row['success_score']} cost=${row['cost_usd']:.5f} {row['latency_s']:.1f}s{extra}")
            except Exception as e:  # noqa: BLE001
                entry["systems"][name] = {"error": str(e)}
                if verbose:
                    print(f"   {name:8s} ERROR: {e}")
        per_task.append(entry)

    results = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "corpus": corpus.name,
        "num_videos": len(corpus.videos),
        "num_tasks": len(tasks),
        "provider": provider,
        "model": model,
        "judge_model": judge_model,
        "tavily_enabled": tavily,
        "aggregate": {name: aggregate(rows, is_agent=(name == "agent")) for name, rows in rows_by_system.items()},
        "per_task": per_task,
    }

    out_path = config.RUNS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    results["_path"] = str(out_path)
    if verbose:
        print(f"\n✅ Saved run to {out_path}")
        _print_summary(results)
    return results


def _print_summary(results: dict) -> None:
    print("\n=== AGGREGATE ===")
    agg = results["aggregate"]
    keys = sorted({k for a in agg.values() for k in a})
    header = "metric".ljust(24) + "".join(s.ljust(14) for s in agg)
    print(header)
    for k in keys:
        line = k.ljust(24) + "".join(str(agg[s].get(k, "-")).ljust(14) for s in agg)
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the VidSense agent evaluation harness.")
    ap.add_argument("--tasks", default=str(EVAL_DIR / "tasks.json"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--systems", nargs="+", default=["agent", "baseline"], choices=["agent", "baseline"])
    ap.add_argument("--provider", default=None, choices=["gemini", "groq", "ollama"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--rpm", type=float, default=None, help="Cap total LLM requests/minute (free-tier RPM). e.g. 12 for Gemini free tier.")
    ap.add_argument("--rebuild-corpus", action="store_true")
    args = ap.parse_args()
    run_eval(
        tasks_path=args.tasks,
        systems=tuple(args.systems),
        limit=args.limit,
        provider=args.provider,
        model_name=args.model,
        judge_model=args.judge_model,
        rpm=args.rpm,
        rebuild_corpus=args.rebuild_corpus,
    )


if __name__ == "__main__":
    main()
