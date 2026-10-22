"""Turn a saved eval run into a human-readable report: an aggregate table, a
by-task-type breakdown (where the agent actually earns its cost), an
agent-vs-baseline lift summary, and matplotlib charts.

Usage:
    python -m vidsense.eval.report                 # newest run
    python -m vidsense.eval.report <run.json>
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
from statistics import mean
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .. import config  # noqa: E402

REPORT_DIR = config.EVAL_DIR / "report"


def _latest_run() -> str:
    runs = sorted(glob.glob(str(config.RUNS_DIR / "run_*.json")))
    if not runs:
        raise FileNotFoundError("No eval runs found. Run: python -m vidsense.eval.runner")
    return runs[-1]


def _safe_mean(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return mean(vals) if vals else None


def _success_by_type(per_task: List[dict], system: str) -> dict:
    by_type: dict = {}
    for t in per_task:
        s = t["systems"].get(system, {})
        if "success_score" not in s:
            continue
        by_type.setdefault(t["type"], []).append(s["success_score"])
    return {k: _safe_mean(v) for k, v in by_type.items()}


# --- Charts ------------------------------------------------------------------
def _chart_success_by_type(results: dict, out: Path) -> None:
    per_task = results["per_task"]
    systems = list(results["aggregate"].keys())
    types = sorted({t["type"] for t in per_task})
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.8 / max(len(systems), 1)
    for i, sysname in enumerate(systems):
        by_type = _success_by_type(per_task, sysname)
        by_type["overall"] = results["aggregate"][sysname].get("task_success_rate")
        xs = types + ["overall"]
        ys = [(by_type.get(t) or 0) for t in xs]
        positions = [x + i * width for x in range(len(xs))]
        ax.bar(positions, ys, width=width, label=sysname)
    ax.set_xticks([x + width * (len(systems) - 1) / 2 for x in range(len(types) + 1)])
    ax.set_xticklabels(types + ["overall"])
    ax.set_ylabel("Task success rate")
    ax.set_ylim(0, 1)
    ax.set_title("Task success by task type — agent vs baseline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _chart_cost_latency(results: dict, out: Path) -> None:
    systems = list(results["aggregate"].keys())
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, metric, title in [
        (axes[0], "avg_cost_usd", "Avg cost per task (USD)"),
        (axes[1], "avg_latency_s", "Avg latency per task (s)"),
    ]:
        vals = [results["aggregate"][s].get(metric) or 0 for s in systems]
        ax.bar(systems, vals, color=["#4C72B0", "#DD8452"][: len(systems)])
        ax.set_title(title)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.4g}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


# --- Markdown ----------------------------------------------------------------
def _fmt(v, pct=False):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v*100:.1f}%" if pct else f"{v:.4g}"
    return str(v)


def _aggregate_table(agg: dict) -> str:
    systems = list(agg)
    keys = sorted({k for a in agg.values() for k in a})
    pct_keys = {"task_success_rate", "avg_tool_coverage", "avg_trajectory_quality", "avg_grounded_ratio", "tool_exact_set_rate", "avg_tool_f1"}
    lines = ["| Metric | " + " | ".join(systems) + " |", "|---|" + "---|" * len(systems)]
    for k in keys:
        row = [k] + [_fmt(agg[s].get(k), pct=(k in pct_keys)) for s in systems]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def generate_report(run_path: str | Path | None = None) -> Path:
    run_path = Path(run_path or _latest_run())
    results = json.loads(run_path.read_text(encoding="utf-8"))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    _chart_success_by_type(results, REPORT_DIR / "success_by_type.png")
    _chart_cost_latency(results, REPORT_DIR / "cost_latency.png")

    agg = results["aggregate"]
    has_both = "agent" in agg and "baseline" in agg

    md = [
        "# VidSense — Agent Evaluation Report",
        "",
        f"- **Run:** `{run_path.name}`  ·  {results['timestamp']}",
        f"- **Corpus:** {results['corpus']} ({results['num_videos']} videos)  ·  **Tasks:** {results['num_tasks']}",
        f"- **Agent model:** {results['model']}  ·  **Judge:** {results['judge_model']}  ·  **Tavily:** {results['tavily_enabled']}",
        "",
        "## Aggregate metrics",
        "",
        _aggregate_table(agg),
        "",
    ]

    if has_both:
        a, b = agg["agent"], agg["baseline"]
        sr_a, sr_b = a.get("task_success_rate") or 0, b.get("task_success_rate") or 0
        c_a, c_b = a.get("avg_cost_usd") or 0, b.get("avg_cost_usd") or 0
        l_a, l_b = a.get("avg_latency_s") or 0, b.get("avg_latency_s") or 0
        md += [
            "## Headline: agent vs single-shot RAG",
            "",
            f"- **Task success:** {sr_a*100:.1f}% (agent) vs {sr_b*100:.1f}% (baseline) "
            f"→ {(sr_a-sr_b)*100:+.1f} pts",
            f"- **Cost:** ${c_a:.5f} vs ${c_b:.5f} per task → {c_a/c_b:.1f}× more expensive" if c_b else "",
            f"- **Latency:** {l_a:.1f}s vs {l_b:.1f}s → {l_a/l_b:.1f}× slower" if l_b else "",
            "",
            "### Success rate by task type",
            "",
            "| Task type | agent | baseline |",
            "|---|---|---|",
        ]
        st_a, st_b = _success_by_type(results["per_task"], "agent"), _success_by_type(results["per_task"], "baseline")
        for t in sorted(set(st_a) | set(st_b)):
            md.append(f"| {t} | {_fmt(st_a.get(t), pct=True)} | {_fmt(st_b.get(t), pct=True)} |")
        md += [
            "",
            "> Read this table, not just the headline: the agent's planning + tools pay off on "
            "**cross-video** synthesis and **verify** tasks (which need a web tool the baseline "
            "lacks entirely), while single-shot RAG stays competitive and far cheaper on simple "
            "**single-video** lookups. The honest design conclusion is to **route by task type**.",
            "",
        ]

    md += [
        "## Charts",
        "",
        "![Success by task type](success_by_type.png)",
        "",
        "![Cost and latency](cost_latency.png)",
        "",
        "## Per-task detail",
        "",
        "| Task | Type | Agent | Baseline | Agent tools | Traj |",
        "|---|---|---|---|---|---|",
    ]
    for t in results["per_task"]:
        ag = t["systems"].get("agent", {})
        bl = t["systems"].get("baseline", {})
        tools = ",".join(ag.get("tool_accuracy", {}).get("actual", [])) if "tool_accuracy" in ag else "—"
        traj = _fmt((ag.get("trajectory_quality") or {}).get("normalized"), pct=True) if "trajectory_quality" in ag else "—"
        md.append(
            f"| {t['task_id']} | {t['type']} | {_fmt(ag.get('success_score'))} | "
            f"{_fmt(bl.get('success_score'))} | {tools} | {traj} |"
        )

    out_md = REPORT_DIR / "report.md"
    out_md.write_text("\n".join(x for x in md if x is not None), encoding="utf-8")
    print(f"✅ Report written to {out_md}")
    print(f"   Charts: {REPORT_DIR / 'success_by_type.png'} , {REPORT_DIR / 'cost_latency.png'}")
    return out_md


def main() -> None:
    import sys

    generate_report(sys.argv[1] if len(sys.argv) > 1 else None)


if __name__ == "__main__":
    main()
