"""The VidSense research agent — a LangGraph StateGraph.

Flow:  planner -> (executor -> planner)*  -> synthesizer -> verifier -> END

  * planner     asks the tool-bound LLM what to do next (which tool + args), or
                signals it has enough evidence by answering without tool calls.
  * executor    runs the requested tools, appends observations, and collects
                structured citations as evidence.
  * synthesizer writes the final answer grounded in the collected evidence,
                citing [title @ time] tags.
  * verifier    checks each claim in the answer against the evidence
                (hallucination control).

Every node appends to a structured `trajectory`, and a UsageTracker accumulates
token usage across all LLM calls (planner, tools, synthesizer, verifier) so the
eval harness can report cost and latency per task.
"""
from __future__ import annotations

import operator
from time import perf_counter
from typing import Annotated, List, Optional, TypedDict

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from . import config
from .citations import dedupe, format_passages, verify_answer
from .tools import make_tools
from .utils import to_text


SYSTEM_PROMPT = """You are a research agent that answers questions about a corpus \
of YouTube videos by using tools. You cannot see the transcripts directly — you \
must gather evidence with the tools before answering.

Available strategy:
- If you don't know what's in the corpus, call `list_videos` first.
- Use `transcript_search` to find what is said about a topic (optionally within one video).
- Use `timestamp_lookup` to locate exactly where one video covers something.
- Use `cross_video_compare` for compare/contrast-across-videos questions.
- Use `external_verify` only to fact-check a specific claim against the open web.

Decide each next action based on what previous tool results returned. Take only \
the steps you need. When you have gathered enough evidence to answer well, reply \
WITHOUT calling any tool — that signals you are done. Do not fabricate content \
that the tools did not return."""

SYNTH_PROMPT = """You are writing the final answer to a research question using \
the evidence gathered by tools. Two kinds may be present:
- EVIDENCE PASSAGES from the video transcripts, each tagged [title @ MM:SS].
- EXTERNAL FINDINGS from web verification (a SUPPORTED/CONTRADICTED/UNCERTAIN \
verdict with sources), used to fact-check claims not covered by the videos.

Write a clear, well-structured answer using whichever evidence is relevant. After \
each claim from a transcript, cite its exact [title @ MM:SS] tag. If the question \
asks you to VERIFY a claim, rely on the external findings and state the verdict. \
If nothing relevant was gathered, say so plainly rather than guessing.

QUESTION:
{question}

EVIDENCE PASSAGES:
{evidence}

EXTERNAL FINDINGS (web verification):
{findings}

ANSWER:"""


class UsageTracker(BaseCallbackHandler):
    """Accumulates token usage across every LLM call in a run."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.llm_calls = 0

    def on_llm_end(self, response, **kwargs) -> None:
        self.llm_calls += 1
        try:
            for gen_list in response.generations:
                for gen in gen_list:
                    um = getattr(getattr(gen, "message", None), "usage_metadata", None)
                    if um:
                        self.input_tokens += um.get("input_tokens", 0) or 0
                        self.output_tokens += um.get("output_tokens", 0) or 0
                        return
            lo = response.llm_output or {}
            tu = lo.get("token_usage") or lo.get("usage") or {}
            self.input_tokens += tu.get("prompt_tokens", 0) or 0
            self.output_tokens += tu.get("completion_tokens", 0) or 0
        except Exception:
            pass


class AgentState(TypedDict):
    question: str
    messages: Annotated[list, add_messages]
    evidence: Annotated[list, operator.add]
    findings: Annotated[list, operator.add]  # non-passage tool outputs, e.g. web-verification verdicts
    trajectory: Annotated[list, operator.add]
    steps: int
    answer: str
    verification: dict


class ResearchAgent:
    def __init__(
        self,
        corpus,
        model_type: str = config.DEFAULT_PROVIDER,
        model_name: Optional[str] = None,
        max_steps: int = 6,
        verify: bool = True,
        rate_limiter=None,
    ) -> None:
        self.corpus = corpus
        self.llm = config.make_llm(model_type, model_name, rate_limiter=rate_limiter)
        self.tools = make_tools(corpus, self.llm)
        self.tools_by_name = {t.name: t for t in self.tools}
        self.bound_llm = self.llm.bind_tools(self.tools)
        self.max_steps = max_steps
        self.verify = verify
        self.graph = self._build_graph()

    # --- nodes --------------------------------------------------------------
    def _planner(self, state: AgentState) -> dict:
        t0 = perf_counter()
        resp = self.bound_llm.invoke(state["messages"])
        dt = perf_counter() - t0
        tool_calls = [{"name": tc["name"], "args": tc["args"]} for tc in (resp.tool_calls or [])]
        um = getattr(resp, "usage_metadata", None) or {}
        rec = {
            "type": "plan",
            "tool_calls": tool_calls,
            "thought": to_text(resp)[:600],
            "latency": dt,
            "tokens": um.get("total_tokens", 0),
        }
        return {"messages": [resp], "trajectory": [rec], "steps": state["steps"] + 1}

    def _route(self, state: AgentState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None) and state["steps"] < self.max_steps:
            return "executor"
        return "synthesizer"

    def _executor(self, state: AgentState) -> dict:
        last = state["messages"][-1]
        tool_msgs: List[ToolMessage] = []
        new_evidence: list = []
        new_findings: list = []
        recs: list = []
        for tc in last.tool_calls:
            tool = self.tools_by_name.get(tc["name"])
            t0 = perf_counter()
            if tool is None:
                tool_msgs.append(ToolMessage(content=f"Unknown tool: {tc['name']}", tool_call_id=tc["id"]))
                continue
            try:
                msg = tool.invoke(tc)  # ToolMessage with .content and .artifact
            except Exception as e:
                msg = ToolMessage(content=f"Tool error in {tc['name']}: {e}", tool_call_id=tc["id"])
            dt = perf_counter() - t0
            art = getattr(msg, "artifact", None) or {}
            passages = art.get("passages", []) or []
            new_evidence.extend(passages)
            # Web-verification style tools carry a "sources" key and no passages;
            # keep their verdict so the synthesizer/verifier can actually use it.
            if "sources" in art:
                new_findings.append((msg.content or "").strip())
            tool_msgs.append(msg)
            recs.append(
                {
                    "type": "tool",
                    "name": tc["name"],
                    "args": tc["args"],
                    "latency": dt,
                    "num_passages": len(passages),
                    "result_preview": (msg.content or "")[:220],
                }
            )
        return {"messages": tool_msgs, "evidence": new_evidence, "findings": new_findings, "trajectory": recs}

    def _synthesizer(self, state: AgentState) -> dict:
        cits = dedupe(state["evidence"])
        findings = state.get("findings", [])
        prompt = SYNTH_PROMPT.format(
            question=state["question"],
            evidence=format_passages(cits),
            findings="\n\n".join(findings) if findings else "(none)",
        )
        t0 = perf_counter()
        resp = self.llm.invoke(prompt)
        dt = perf_counter() - t0
        answer = to_text(resp)
        um = getattr(resp, "usage_metadata", None) or {}
        rec = {"type": "synthesize", "latency": dt, "tokens": um.get("total_tokens", 0)}
        return {"answer": answer, "trajectory": [rec]}

    def _verifier(self, state: AgentState) -> dict:
        if not self.verify:
            return {}
        cits = dedupe(state["evidence"])
        t0 = perf_counter()
        result = verify_answer(
            state["answer"], cits, self.llm, extra_evidence="\n\n".join(state.get("findings", []))
        )
        dt = perf_counter() - t0
        rec = {"type": "verify", "latency": dt, "grounded_ratio": result.get("grounded_ratio")}
        return {"verification": result, "trajectory": [rec]}

    # --- graph --------------------------------------------------------------
    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("planner", self._planner)
        g.add_node("executor", self._executor)
        g.add_node("synthesizer", self._synthesizer)
        g.add_node("verifier", self._verifier)
        g.set_entry_point("planner")
        g.add_conditional_edges(
            "planner", self._route, {"executor": "executor", "synthesizer": "synthesizer"}
        )
        g.add_edge("executor", "planner")
        g.add_edge("synthesizer", "verifier")
        g.add_edge("verifier", END)
        return g.compile()

    # --- public API ---------------------------------------------------------
    def run(self, question: str) -> dict:
        tracker = UsageTracker()
        init: AgentState = {
            "question": question,
            "messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)],
            "evidence": [],
            "findings": [],
            "trajectory": [],
            "steps": 0,
            "answer": "",
            "verification": {},
        }
        t0 = perf_counter()
        final = self.graph.invoke(
            init,
            config={"callbacks": [tracker], "recursion_limit": 2 * self.max_steps + 10},
        )
        latency = perf_counter() - t0

        citations = dedupe(final["evidence"])
        trajectory = final["trajectory"]
        tool_log = [r for r in trajectory if r["type"] == "tool"]
        return {
            "question": question,
            "answer": final["answer"],
            "citations": citations,
            "trajectory": trajectory,
            "tool_log": tool_log,
            "verification": final.get("verification", {}),
            "usage": {
                "input_tokens": tracker.input_tokens,
                "output_tokens": tracker.output_tokens,
                "total_tokens": tracker.input_tokens + tracker.output_tokens,
                "llm_calls": tracker.llm_calls,
            },
            "latency": latency,
            "num_tool_calls": len(tool_log),
        }

    def run_stream(self, question: str):
        """Generator yielding events as the agent works, for live UIs (SSE).

        Emits dicts: {"event": "step"|"answer"|"done", "data": ...}. `step`
        carries each trajectory record as it happens; `done` carries the final
        answer, citations, verification, usage, and latency.
        """
        tracker = UsageTracker()
        init: AgentState = {
            "question": question,
            "messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)],
            "evidence": [],
            "findings": [],
            "trajectory": [],
            "steps": 0,
            "answer": "",
            "verification": {},
        }
        t0 = perf_counter()
        evidence: list = []
        answer, verification = "", {}
        for chunk in self.graph.stream(
            init,
            config={"callbacks": [tracker], "recursion_limit": 2 * self.max_steps + 10},
            stream_mode="updates",
        ):
            for _node, delta in chunk.items():
                if not isinstance(delta, dict):
                    continue
                for rec in delta.get("trajectory", []) or []:
                    yield {"event": "step", "data": rec}
                if delta.get("evidence"):
                    evidence.extend(delta["evidence"])
                if delta.get("answer"):
                    answer = delta["answer"]
                    yield {"event": "answer", "data": {"answer": answer}}
                if delta.get("verification"):
                    verification = delta["verification"]

        citations = dedupe(evidence)
        yield {
            "event": "done",
            "data": {
                "answer": answer,
                "citations": citations,
                "verification": verification,
                "usage": {
                    "input_tokens": tracker.input_tokens,
                    "output_tokens": tracker.output_tokens,
                    "total_tokens": tracker.input_tokens + tracker.output_tokens,
                    "llm_calls": tracker.llm_calls,
                },
                "latency": perf_counter() - t0,
            },
        }
