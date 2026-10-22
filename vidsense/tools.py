"""The tools the research agent plans over.

`make_tools(corpus, llm)` returns LangChain tools closed over the shared corpus
and LLM, so those never appear in the tool schema the planner sees. Every
retrieval tool uses response_format="content_and_artifact": the *content* string
is what the planner reads; the *artifact* carries structured citations that the
graph collects as evidence for grounded answers.
"""
from __future__ import annotations

from typing import List, Optional

from langchain_core.tools import tool

from . import config
from .citations import doc_to_citation, format_passages
from .utils import to_text


def make_tools(corpus, llm) -> list:
    """Build the tool set bound to a specific corpus + LLM."""

    @tool(response_format="content_and_artifact")
    def transcript_search(query: str, video_id: Optional[str] = None) -> tuple:
        """Semantically search the transcripts for passages relevant to `query`.
        Searches ALL videos by default; pass `video_id` (the short id from
        list_videos, e.g. 'aircAruvnKk') to restrict to one video. Use this to
        find what is said about a topic. Returns timestamped passages."""
        docs = corpus.search(query, video_id=video_id, k=5)
        cits = [doc_to_citation(d) for d in docs]
        scope = f", video={video_id}" if video_id else ", all videos"
        content = format_passages(cits, header=f"transcript_search('{query}'{scope}):")
        return content, {"passages": cits}

    @tool(response_format="content_and_artifact")
    def timestamp_lookup(video_id: str, query: str) -> tuple:
        """Find WHERE a specific topic or quote is discussed in ONE given video.
        Returns the best-matching moment as a clickable timestamped YouTube link.
        Use when the user asks 'where/when does video X cover Y'."""
        docs = corpus.search(query, video_id=video_id, k=2)
        if not docs:
            return (f"No match for '{query}' in video {video_id}.", {"passages": []})
        cits = [doc_to_citation(d) for d in docs]
        top = cits[0]
        content = (
            f"'{query}' in \"{top['title']}\" is discussed around {top['formatted_time']}: "
            f"{top['youtube_link']}"
        )
        return content, {"passages": cits}

    @tool(response_format="content_and_artifact")
    def cross_video_compare(topic: str, video_ids: Optional[List[str]] = None) -> tuple:
        """Compare how MULTIPLE videos treat `topic` (agreements, differences,
        unique points). Compares all videos by default; pass `video_ids` to limit.
        Use for 'compare/contrast across videos' style research questions."""
        targets = video_ids or list(corpus.videos.keys())
        blocks, all_cits = [], []
        for vid in targets:
            rec = corpus.get(vid)
            if not rec:
                continue
            cits = [doc_to_citation(d) for d in corpus.search(topic, video_id=vid, k=2)]
            all_cits.extend(cits)
            blocks.append(f"### {rec.title} ({vid})\n{format_passages(cits)}")
        if not blocks:
            return (f"No videos available to compare on '{topic}'.", {"passages": []})
        prompt = (
            f'Compare how these videos discuss "{topic}". Note agreements, '
            f"differences, and any unique points. Cite using the [title @ time] "
            f"tags exactly as given.\n\n" + "\n\n".join(blocks) + "\n\nComparison:"
        )
        resp = llm.invoke(prompt)
        return to_text(resp), {"passages": all_cits}

    @tool(response_format="content_and_artifact")
    def external_verify(claim: str) -> tuple:
        """Verify a factual `claim` against outside web sources (not the videos).
        Use to fact-check a specific claim a video makes. Returns a
        SUPPORTED / CONTRADICTED / UNCERTAIN verdict with source URLs."""
        if not config.has_tavily():
            return (
                "External verification unavailable: TAVILY_API_KEY is not set.",
                {"passages": [], "sources": []},
            )
        try:
            from langchain_tavily import TavilySearch

            res = TavilySearch(max_results=4).invoke({"query": claim})
        except Exception as e:  # network / auth
            return (f"External verification failed: {e}", {"passages": [], "sources": []})

        results = res.get("results", []) if isinstance(res, dict) else []
        if not results:
            return (f"No web sources found for: {claim}", {"passages": [], "sources": []})
        snippets = "\n".join(
            f"- {r.get('title', '')} ({r.get('url', '')}): {str(r.get('content', ''))[:200]}"
            for r in results
        )
        verdict = llm.invoke(
            "Given ONLY these web sources, does the claim hold? Answer with one of "
            "SUPPORTED / CONTRADICTED / UNCERTAIN, then one sentence of justification.\n"
            f"Claim: {claim}\nSources:\n{snippets}\nVerdict:"
        )
        vtext = to_text(verdict)
        content = f"Verification of '{claim}':\n{vtext}\n\nSources:\n{snippets}"
        return content, {"passages": [], "sources": [r.get("url", "") for r in results]}

    @tool(response_format="content_and_artifact")
    def list_videos() -> tuple:
        """List the videos in the corpus (id, title, duration) so you can decide
        which video(s) to inspect. Call this first if you don't know the corpus."""
        vids = corpus.list_videos()
        content = "Videos in corpus:\n" + "\n".join(
            f"- {v['video_id']}: {v['title']} ({v['duration']})" for v in vids
        )
        return content, {"passages": []}

    return [transcript_search, timestamp_lookup, cross_video_compare, external_verify, list_videos]
