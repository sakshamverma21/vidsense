"""VidSense — multi-video research agent, Streamlit UI.

Three tabs:
  🔍 Research    ask a research question; watch the agent's trajectory; get a
                 grounded, timestamp-cited answer (optionally vs the baseline).
  📊 Evaluation  view the latest eval run: agent-vs-baseline metrics, by-type
                 breakdown, and charts.
  📁 Corpus      the videos the agent can research over.
"""
import glob
import json
from pathlib import Path

import streamlit as st

from vidsense import config
from vidsense.corpus import load_corpus, corpus_from_urls
from vidsense.agent import ResearchAgent
from vidsense.baseline import SingleShotRAG
from vidsense.eval import metrics

st.set_page_config(page_title="VidSense", layout="wide")

st.markdown(
    """
<style>
    .main-header { font-size: 2.4rem; font-weight: 800; color: #FF4B4B;
        text-align: center; margin-bottom: .2rem; }
    .sub-header { text-align:center; color:#9aa0a6; margin-bottom:1.5rem; }
    .step-plan { border-left:4px solid #7C3AED; background:rgba(124,58,237,.08);
        padding:.5rem .8rem; border-radius:8px; margin:.3rem 0; }
    .step-tool { border-left:4px solid #059669; background:rgba(5,150,105,.08);
        padding:.5rem .8rem; border-radius:8px; margin:.3rem 0; }
    .step-final { border-left:4px solid #DC2626; background:rgba(220,38,38,.08);
        padding:.5rem .8rem; border-radius:8px; margin:.3rem 0; }
    .metric-pill { display:inline-block; padding:.25rem .7rem; border-radius:14px;
        background:#2C2C2C; color:#eee; font-size:.8rem; margin:.15rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<p class="main-header">VidSense</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">A tool-using agent that researches across a corpus of YouTube videos — '
    "with timestamp-grounded citations and a built-in evaluation harness.</p>",
    unsafe_allow_html=True,
)


# --- resource loaders (cached) ----------------------------------------------
@st.cache_resource(show_spinner="Loading corpus (transcripts + FAISS)…")
def get_corpus_from_manifest(manifest_path: str):
    return load_corpus(manifest_path)


@st.cache_resource(show_spinner="Fetching transcripts + building FAISS…")
def get_corpus_from_urls(name: str, urls_tuple: tuple):
    return corpus_from_urls(name, list(urls_tuple))


def render_citations(citations):
    if not citations:
        st.caption("No sources cited.")
        return
    st.markdown("**🔗 Sources (click to jump to the moment):**")
    for c in citations:
        snippet = " ".join(c["text"].split())[:160]
        st.markdown(f"- [{c['title']} @ {c['formatted_time']}]({c['youtube_link']}) — {snippet}…")


def render_trajectory(trajectory):
    for i, s in enumerate(trajectory):
        if s["type"] == "plan":
            calls = s.get("tool_calls", [])
            if calls:
                body = "  ·  ".join(f"<code>{t['name']}({t['args']})</code>" for t in calls)
                st.markdown(f'<div class="step-plan">🧭 <b>Plan {i}</b> → {body}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="step-final">✅ <b>Plan {i}</b> → enough evidence; writing answer</div>', unsafe_allow_html=True)
        elif s["type"] == "tool":
            st.markdown(
                f'<div class="step-tool">🛠️ <b>{s["name"]}</b>({s["args"]}) → '
                f'{s["num_passages"]} passages · {s["latency"]:.2f}s</div>',
                unsafe_allow_html=True,
            )
            with st.expander("observation"):
                st.text(s.get("result_preview", ""))
        elif s["type"] == "synthesize":
            st.markdown(f'<div class="step-final">✍️ <b>Synthesize</b> grounded answer · {s.get("latency",0):.2f}s</div>', unsafe_allow_html=True)
        elif s["type"] == "verify":
            gr = s.get("grounded_ratio")
            gr_txt = f"{gr*100:.0f}% claims grounded" if gr is not None else "n/a"
            st.markdown(f'<div class="step-final">🔎 <b>Verify</b> → {gr_txt} · {s.get("latency",0):.2f}s</div>', unsafe_allow_html=True)


# --- sidebar -----------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    provider = st.selectbox("Provider", ["gemini", "groq"], index=0,
                            help="Gemini has the most generous free tier.")
    if provider == "gemini":
        model = st.selectbox("Model", ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.5-flash"],
                             help="flash-lite is the most free-tier-friendly (higher RPM).")
    else:
        model = st.selectbox("Model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"])
    max_steps = st.slider("Max agent steps", 3, 10, 6)
    verify = st.checkbox("Verify answer (hallucination check)", value=True)
    compare_baseline = st.checkbox("Also run single-shot RAG baseline", value=False)
    st.divider()
    st.caption("Gemini key: " + ("✅ set" if config.has_google() else "⚠️ not set"))
    st.caption("GROQ key: " + ("✅ set" if config.has_groq() else "⚠️ not set"))
    st.caption("Tavily (web verify): " + ("✅ set" if config.has_tavily() else "⚠️ not set"))

tab_research, tab_eval, tab_corpus = st.tabs(["🔍 Research", "📊 Evaluation", "📁 Corpus"])

# --- Corpus selection state --------------------------------------------------
DEFAULT_MANIFEST = str(config.EVAL_DIR / "corpus.json")

with tab_corpus:
    st.header("📁 Corpus")
    source = st.radio("Corpus source", ["Bundled manifest (dl-series)", "Custom YouTube sources"], horizontal=True)
    if source.startswith("Bundled"):
        st.session_state["corpus_kind"] = ("manifest", DEFAULT_MANIFEST)
        st.caption(f"Manifest: `{DEFAULT_MANIFEST}`")
    else:
        urls_text = st.text_area("YouTube video or playlist links (one per line)", height=140,
                                 placeholder="https://www.youtube.com/watch?v=...\nhttps://www.youtube.com/playlist?list=...")
        if st.button("Build collection"):
            urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
            if urls:
                st.session_state["corpus_kind"] = ("urls", tuple(urls))

    kind = st.session_state.get("corpus_kind", ("manifest", DEFAULT_MANIFEST))
    try:
        corpus = get_corpus_from_manifest(kind[1]) if kind[0] == "manifest" else get_corpus_from_urls("custom", kind[1])
        st.success(f"Corpus '{corpus.name}': {len(corpus.videos)} videos")
        st.table(corpus.list_videos())
        if corpus.skipped:
            st.warning(f"Skipped (no captions / unavailable): {corpus.skipped}")
    except Exception as e:
        corpus = None
        st.error(f"Could not load corpus: {e}")

# --- Research tab ------------------------------------------------------------
with tab_research:
    st.header("🔍 Ask a research question")
    kind = st.session_state.get("corpus_kind", ("manifest", DEFAULT_MANIFEST))
    try:
        corpus = get_corpus_from_manifest(kind[1]) if kind[0] == "manifest" else get_corpus_from_urls("custom", kind[1])
    except Exception as e:
        corpus = None
        st.error(f"Load a corpus in the 📁 Corpus tab first. ({e})")

    question = st.text_input("Question", placeholder="Compare how the videos explain what a neural network learns.")
    ask = st.button("🔬 Research", type="primary")

    if ask and question and corpus is not None:
        agent = ResearchAgent(corpus, model_type=provider, model_name=model, max_steps=max_steps, verify=verify)
        with st.spinner("🤖 Agent is researching…"):
            out = agent.run(question)

        st.subheader("💡 Answer")
        st.markdown(out["answer"])
        render_citations(out["citations"])

        cost = metrics.cost_usd(out["usage"], model)
        gr = (out.get("verification") or {}).get("grounded_ratio")
        pills = [
            f"⏱️ {out['latency']:.1f}s",
            f"🔢 {out['usage']['total_tokens']} tok",
            f"💵 ${cost:.5f}",
            f"🛠️ {out['num_tool_calls']} tool calls",
        ]
        if gr is not None:
            pills.append(f"🔎 {gr*100:.0f}% grounded")
        st.markdown(" ".join(f'<span class="metric-pill">{p}</span>' for p in pills), unsafe_allow_html=True)

        st.subheader("🧭 Agent trajectory")
        render_trajectory(out["trajectory"])

        if compare_baseline:
            st.divider()
            st.subheader("📎 Single-shot RAG baseline (for comparison)")
            with st.spinner("Running baseline…"):
                b = SingleShotRAG(corpus, model_type=provider, model_name=model).run(question)
            st.markdown(b["answer"])
            render_citations(b["citations"])
            bcost = metrics.cost_usd(b["usage"], model)
            st.markdown(
                f'<span class="metric-pill">⏱️ {b["latency"]:.1f}s</span> '
                f'<span class="metric-pill">💵 ${bcost:.5f}</span> '
                f'<span class="metric-pill">🔢 {b["usage"]["total_tokens"]} tok</span>',
                unsafe_allow_html=True,
            )

# --- Evaluation tab ----------------------------------------------------------
with tab_eval:
    st.header("📊 Evaluation")
    runs = sorted(glob.glob(str(config.RUNS_DIR / "run_*.json")))
    if not runs:
        st.info("No eval runs yet. Run:  `python -m vidsense.eval.runner`")
    else:
        run_path = st.selectbox("Run", runs[::-1], format_func=lambda p: Path(p).name)
        results = json.loads(Path(run_path).read_text(encoding="utf-8"))
        agg = results["aggregate"]

        st.caption(
            f"Corpus **{results['corpus']}** ({results['num_videos']} videos) · "
            f"{results['num_tasks']} tasks · model **{results['model']}** · judge **{results['judge_model']}**"
        )

        if "agent" in agg and "baseline" in agg:
            a, b = agg["agent"], agg["baseline"]
            c1, c2, c3 = st.columns(3)
            sr_a, sr_b = (a.get("task_success_rate") or 0), (b.get("task_success_rate") or 0)
            c1.metric("Task success — agent", f"{sr_a*100:.0f}%", f"{(sr_a-sr_b)*100:+.0f} pts vs baseline")
            if b.get("avg_cost_usd"):
                c2.metric("Cost multiple", f"{(a.get('avg_cost_usd') or 0)/b['avg_cost_usd']:.1f}×", "agent vs baseline")
            if b.get("avg_latency_s"):
                c3.metric("Latency multiple", f"{(a.get('avg_latency_s') or 0)/b['avg_latency_s']:.1f}×", "agent vs baseline")

        # aggregate table
        rows = sorted({k for x in agg.values() for k in x})
        table = {"metric": rows}
        for s in agg:
            table[s] = [agg[s].get(k, None) for k in rows]
        st.subheader("Aggregate metrics")
        st.dataframe(table, use_container_width=True)

        # success by task type
        by_type = {}
        for t in results["per_task"]:
            for s in agg:
                sc = t["systems"].get(s, {}).get("success_score")
                if sc is not None:
                    by_type.setdefault(t["type"], {}).setdefault(s, []).append(sc)
        if by_type:
            st.subheader("Success rate by task type")
            chart_rows = {"task_type": list(by_type.keys())}
            for s in agg:
                chart_rows[s] = [
                    (sum(by_type[t][s]) / len(by_type[t][s])) if by_type[t].get(s) else 0 for t in by_type
                ]
            st.bar_chart(chart_rows, x="task_type", use_container_width=True)
            st.caption(
                "The agent's planning + tools earn their extra cost on cross-video tasks; "
                "single-shot RAG is competitive and cheaper on simple single-video lookups."
            )

        st.subheader("Per-task detail")
        detail = []
        for t in results["per_task"]:
            row = {"task": t["task_id"], "type": t["type"]}
            for s in agg:
                row[f"{s}_success"] = t["systems"].get(s, {}).get("success_score")
            ag = t["systems"].get("agent", {})
            row["agent_tools"] = ",".join(ag.get("tool_accuracy", {}).get("actual", [])) if "tool_accuracy" in ag else ""
            detail.append(row)
        st.dataframe(detail, use_container_width=True)

st.divider()
st.caption("VidSense · LangGraph agent · FAISS retrieval · Gemini / Groq LLMs · built-in eval harness")
