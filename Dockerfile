# VidSense — FastAPI app image (research agent + custom frontend).
# Slim: no torch/transformers (the agent path uses ONNX-based fastembed).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FASTEMBED_CACHE_DIR=/app/.vidsense_cache/fastembed

WORKDIR /app

# 1) Dependencies first for better layer caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2) Warm the embedding model in a dependency-only layer. Application edits
#    can then reuse this relatively expensive download from Docker's cache.
RUN mkdir -p /app/.vidsense_cache/fastembed \
    && python -c "from langchain_community.embeddings import FastEmbedEmbeddings; FastEmbedEmbeddings(cache_dir='/app/.vidsense_cache/fastembed', batch_size=16).embed_query('warmup')"

# 3) Application code, frontend, and an optional locally prepared starter cache.
#    A fresh checkout includes the cache directory's placeholder, so this COPY
#    also works without generated data. See README for prewarming the starter
#    collection before deployment to networks that block YouTube access.
COPY api.py ./
COPY vidsense/ ./vidsense/
COPY frontend/ ./frontend/
COPY .vidsense_cache/ ./.vidsense_cache/

# Do not run the web service as root. Deployment platforms inject API keys at
# runtime; they are never copied into the image.
RUN useradd --create-home --uid 10001 vidsense \
    && chown -R vidsense:vidsense /app
USER vidsense

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:' + __import__('os').getenv('PORT', '8000') + '/api/live', timeout=3)" || exit 1
# Azure Container Apps / Cloud Run inject $PORT; default to 8000 for local runs.
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
