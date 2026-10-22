"""Reproduce the interview audit locally without calling any external service.

Run from the project root: venv/Scripts/python.exe docs/interview/audit.py
This reads the bundled, project-owned FAISS pickle. Never point it at an
untrusted downloaded index. Application files and existing caches are unchanged.
Only docs/interview/audit_results.json is written.
"""
from __future__ import annotations

import bisect
import importlib.metadata
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from vidsense import config, utils
from vidsense.agent import ResearchAgent
from vidsense.citations import verify_answer
from vidsense.corpus import Corpus, VideoRecord, _chunk_video


def timestamp_at(record: VideoRecord, position: int) -> float:
    starts, cursor = [], 0
    for segment in record.transcript:
        starts.append(cursor)
        cursor += len(segment["text"]) + 1
    return record.transcript[max(0, bisect.bisect_right(starts, position) - 1)]["start"]


class ScriptedChat(BaseChatModel):
    """A real LangChain chat interface with deterministic, local responses."""
    responses: list[Any]
    _cursor: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "offline-interview-audit"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        item = self.responses[self._cursor]
        self._cursor += 1
        message = item if isinstance(item, AIMessage) else AIMessage(content=item)
        message.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        return ChatResult(generations=[ChatGeneration(message=message)])


def tool_message(name: str, args: dict, ident: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": ident}])


def fake_corpus() -> Corpus:
    from langchain_core.documents import Document
    corpus = Corpus("Offline", {"offline0001": VideoRecord("offline0001", "Offline video", [])}, None)
    doc = Document(page_content="A neural network has learned weights and activation values.", metadata={
        "video_id": "offline0001", "title": "Offline video", "timestamp": 12.0,
        "formatted_time": "00:12", "youtube_link": utils.youtube_link("offline0001", 12.0),
    })
    corpus.search = lambda *args, **kwargs: [doc]
    return corpus


def graph_checks() -> dict:
    corpus = fake_corpus()
    responses = [tool_message("transcript_search", {"query": "weights"}, "t1"),
                 "Enough evidence.", "The network has learned weights [Offline video @ 00:12].",
                 '[{"claim_index":0,"supported":true,"reason":"Evidence supports weights."}]']
    with patch.object(config, "make_llm", return_value=ScriptedChat(responses=responses)):
        result = ResearchAgent(corpus).run("What is learned?")
    with patch.object(config, "make_llm", return_value=ScriptedChat(responses=responses)):
        streamed = list(ResearchAgent(corpus).run_stream("What is learned?"))
    calls = [tool_message("transcript_search", {"query": "weights"}, f"t{i}") for i in range(6)]
    calls.append("The network has learned weights [Offline video @ 00:12].")
    with patch.object(config, "make_llm", return_value=ScriptedChat(responses=calls)):
        capped = ResearchAgent(corpus, max_steps=6, verify=False).run("Keep searching.")
    cross_responses = [tool_message("cross_video_compare", {"topic": "weights"}, "cross1"),
                       "Tool comparison.", "Enough evidence.", "Final comparison."]
    with patch.object(config, "make_llm", return_value=ScriptedChat(responses=cross_responses)):
        nested = ResearchAgent(corpus, verify=False).run("Compare weights.")
    return {
        "normal_usage": result["usage"],
        "normal_steps": [s["type"] for s in result["trajectory"]],
        "stream_events": [e["event"] for e in streamed],
        "stream_done_usage": streamed[-1]["data"]["usage"],
        "stream_done_keys": sorted(streamed[-1]["data"]),
        "max_steps_6_planners": sum(s["type"] == "plan" for s in capped["trajectory"]),
        "max_steps_6_executed_tools": capped["num_tool_calls"],
        "nested_cross_compare_usage": nested["usage"],
    }


def parser_checks() -> dict:
    class Response:
        def __init__(self, content):
            self.content = content
    class FakeVerifier:
        def __init__(self, raw):
            self.raw = raw
        def invoke(self, prompt):
            return Response(self.raw)
    answer = "This is a sufficiently long factual claim for the sentence splitter."
    return {
        "string_false_supported": verify_answer(answer, [], FakeVerifier('[{"claim_index":0,"supported":"false"}]')),
        "non_json_verifier": verify_answer(answer, [], FakeVerifier("No JSON returned.")),
        "short_answer_skips_verification": verify_answer("Yes.", [], FakeVerifier("Unused.")),
        "boundary_position": {"offset": 4, "code_timestamp": utils.find_timestamp_for_position([
            {"text": "abc", "start": 0.0}, {"text": "def", "start": 10.0}], 4), "expected_timestamp": 10.0},
    }


def historical_runs() -> list[dict]:
    output = []
    for path in sorted((ROOT / "vidsense/eval/runs").glob("run_*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        row = {"file": path.name, "num_tasks": run["num_tasks"], "model": run["model"],
               "judge_model": run["judge_model"], "tavily_enabled": run["tavily_enabled"],
               "aggregate": run["aggregate"], "systems": {}}
        for name in run["aggregate"]:
            scored = [t["systems"][name] for t in run["per_task"]
                      if name in t["systems"] and "success_score" in t["systems"][name]]
            counts = Counter("engine_error" if "error" in t["systems"].get(name, {}) else
                             t["systems"].get(name, {}).get("success_label", "missing") for t in run["per_task"])
            success = [s["success_score"] for s in scored if s["success_score"] is not None]
            costs = [s["cost_usd"] for s in scored]
            by_type = {}
            for typ in {t["type"] for t in run["per_task"]}:
                values = [t["systems"].get(name, {}).get("success_score") for t in run["per_task"] if t["type"] == typ]
                values = [v for v in values if v is not None]
                by_type[typ] = mean(values) if values else None
            row["systems"][name] = {"labels": dict(counts), "scored_tasks": len(success),
                "mean_success_recomputed": mean(success) if success else None,
                "mean_cost_unrounded": mean(costs) if costs else None,
                "cost_table_has_model": run["model"] in __import__("vidsense.eval.metrics", fromlist=["PRICES"]).PRICES,
                "success_by_type": by_type}
        output.append(row)
    return output


class LocalEmbeddings(Embeddings):
    def __init__(self, model_path: Path):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", threads=2,
            specific_model_path=str(model_path), local_files_only=True)
    def embed_documents(self, texts):
        return [x.tolist() for x in self.model.embed(texts, batch_size=16)]
    def embed_query(self, text):
        return next(self.model.query_embed(text)).tolist()


def cache_checks() -> dict:
    import faiss
    from langchain_community.vectorstores import FAISS
    cache_path = ROOT / ".vidsense_cache/dl-series-40c56c7fbc"
    payload = json.loads((cache_path / "records.json").read_text(encoding="utf-8"))
    model_path = next((ROOT / ".vidsense_cache/fastembed").glob("models--qdrant--bge-small-en-v1.5-onnx-q/snapshots/*"))
    embeddings = LocalEmbeddings(model_path)
    # Only the existing project-owned index is loaded; no user-supplied pickle.
    store = FAISS.load_local(str(cache_path / "faiss"), embeddings, allow_dangerous_deserialization=True)
    videos, spans = [], []
    for vid, raw in payload["videos"].items():
        record = VideoRecord(vid, raw["title"], raw["transcript"])
        docs = [d for d in store.docstore._dict.values() if d.metadata["video_id"] == vid]
        previous = -1
        errors = []
        for doc in docs:
            at = record.full_text.find(doc.page_content, max(0, previous))
            if at < 0:
                continue
            previous = at + 1
            true_time = timestamp_at(record, at)
            error = doc.metadata["timestamp"] - true_time
            errors.append(error)
            spans.append({"video_id": vid, "assigned_s": doc.metadata["timestamp"],
                          "caption_start_s": true_time, "drift_s": error})
        videos.append({"id": vid, "title": record.title, "segments": len(record.transcript),
            "characters": len(record.full_text), "duration_code_s": record.duration,
            "last_caption_end_s": record.transcript[-1]["start"] + record.transcript[-1].get("duration", 0),
            "chunks": len(docs), "distinct_timestamps": len({d.metadata["timestamp"] for d in docs}),
            "matched_chunk_starts": len(errors), "max_drift_s": max(errors), "mean_drift_s": mean(errors)})
    queries = ["chain rule", "query key value", "hidden layer attention", "gradient descent", "input pixels"]
    filtering = []
    for query in queries:
        vec = embeddings.embed_query(query)
        for vid in payload["videos"]:
            a = store.similarity_search_with_score_by_vector(vec, k=2, filter={"video_id": vid})
            b = store.similarity_search_with_score_by_vector(vec, k=2, filter={"video_id": vid}, fetch_k=store.index.ntotal)
            if len(a) < len(b):
                filtering.append({"query": query, "video_id": vid, "default_fetch_k_20_results": len(a),
                                  "fetch_all_results": len(b)})
    import numpy as np
    return {"index_type": type(store.index).__name__, "dimensions": store.index.d,
        "vectors": store.index.ntotal, "vector_bytes": store.index.ntotal * store.index.d * 4,
        "vector_norm_sample": [float(np.linalg.norm(store.index.reconstruct(i))) for i in range(3)],
        "videos": videos, "max_drift_examples": sorted(spans, key=lambda s: s["drift_s"], reverse=True)[:5],
        "filter_starvation_examples": filtering}


def main():
    # Fail on accidental external calls. The embedding model is already local.
    with patch("socket.socket.connect", side_effect=RuntimeError("Network disabled for this audit")), patch(
        "socket.create_connection", side_effect=RuntimeError("Network disabled for this audit")):
        results = {"audit_date": "2026-10-04", "network_calls": "disabled",
            "source_root": str(ROOT), "historical_runs": historical_runs(),
            "graph_checks": graph_checks(), "parser_checks": parser_checks(), "cache_checks": cache_checks()}
    results["packages"] = {name: importlib.metadata.version(name) for name in [
        "fastapi", "langchain-core", "langchain-community", "langgraph", "langchain-google-genai",
        "langchain-groq", "fastembed", "faiss-cpu", "youtube-transcript-api", "yt-dlp", "onnxruntime"]}
    from langchain_community.llms import Ollama
    results["legacy_ollama_has_bind_tools"] = hasattr(Ollama, "bind_tools")
    path = Path(__file__).with_name("audit_results.json")
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Audit written to {path}")
    print(json.dumps({"graph_checks": results["graph_checks"], "cache_summary": {
        k: v for k, v in results["cache_checks"].items() if k not in {"videos", "max_drift_examples"}}}, indent=2))


if __name__ == "__main__":
    main()
