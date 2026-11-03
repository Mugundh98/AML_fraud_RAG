# One container with the backend (FastAPI + agent + MCP servers) and the Streamlit UI.
# Only port 7860 (the UI) is published, so the API stays private inside the container.
# Keys come from the host's secrets as environment variables; nothing secret is in the image.
FROM python:3.13-slim

# Run as an unprivileged user.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/home/user/models \
    API_URL=http://127.0.0.1:8000 \
    LANGFUSE_TRACING_ENVIRONMENT=production
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Download the search models at build time, so a start doesn't fetch ~100 MB.
RUN python -c "from fastembed import SparseTextEmbedding, TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5'); SparseTextEmbedding('Qdrant/bm25')"

COPY --chown=user . .

EXPOSE 7860
CMD ["sh", "deploy/start.sh"]
