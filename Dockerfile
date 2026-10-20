# HR-Chatbot: one image for the API (FastAPI) and the UI (Streamlit). docker-compose.yml runs both.
# Built and published to ghcr.io by .github/workflows/cd.yml after CI passes on main.
# No secrets are baked in: pass API keys at run time (env vars or --env-file).

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf \
    # App defaults: local Ollama on the host machine, graph tool on its NetworkX fallback
    LLM_PROVIDER=ollama \
    OLLAMA_MODEL_CHEAP=qwen2.5:7b \
    OLLAMA_MODEL_STRONG=qwen2.5:7b \
    OLLAMA_BASE_URL=http://host.docker.internal:11434 \
    NEO4J_URI= \
    LANGSMITH_TRACING=false

WORKDIR /app

# CPU-only torch first (same pin as CI), then the rest
COPY requirements.txt .
RUN pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY agent/ agent/
COPY tools/ tools/
COPY api/ api/
COPY frontend/ frontend/
COPY data/ data/

# Bake the embedding model and the search index into the image, so a container starts with no downloads
RUN python -c "from tools.rag_tool import _get_collection; _get_collection()"

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:8000/', timeout=4)" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
