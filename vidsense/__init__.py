"""VidSense: an evaluated multi-video research agent over YouTube transcripts.

Package layout:
    config     - environment / model / path configuration (loads .env)
    utils      - small pure helpers (video-id parsing, timestamps, links)
    corpus     - multi-video ingestion + metadata-rich FAISS store
    tools      - the tools the agent plans over
    agent      - the LangGraph research agent
    citations  - grounded-citation formatting + claim verification
    eval/      - the evaluation harness (tasks, metrics, runner, report)
"""

__all__ = ["config", "utils"]
