# AML & Fraud Assistant

A learning project for **RAG** and **MCP**, built as an enterprise-style AML and fraud assistant on free services.

## Architecture (target)

```
 Streamlit UI
      │  HTTP
      ▼
 FastAPI backend ──── SQLite (users, roles, audit log)            [phase 5]
  ├─ agent loop (Gemini → Groq fallback)
  └─ MCP client                                                   [phase 3]
       ├── aml-data-mcp (ours) ──┬── Neo4j  (customers, accounts, transactions)   [phase 2]
       │                         └── Qdrant (regulations, policies, typologies)   [phase 1 ✓]
       ├── tavily-mcp            (web search)                     [phase 4]
       ├── google-calendar-mcp   (case review meetings)           [phase 4]
       └── slack-mcp             (escalations)                    [phase 4]
```

## Roadmap

| Phase | What | Status |
|---|---|---|
| 1 | Document RAG: hybrid search in Qdrant, cited answers, Gemini→Groq fallback, Streamlit chat | **Done** |
| 2 | Synthetic transactions (≤ 50k rows) in Neo4j + our own `aml-data-mcp` server | Next |
| 3 | Agent loop that routes between documents and the graph via MCP | |
| 4 | Tavily, Google Calendar, Slack MCP servers, with confirmation before any write | |
| 5 | Hardening: audit log, roles, PII redaction, Langfuse tracing, evaluation set | |

## Phase 1: how it works

1. **Load** – `.md`, `.txt` and `.pdf` files from `data/docs/` (`app/rag/loaders.py`).
2. **Chunk** – markdown is split on headings so every chunk knows its section, then packed into ~900-character chunks with 150 characters of overlap (`app/rag/chunking.py`).
3. **Embed** – each chunk gets two vectors, computed on your CPU with `fastembed`:
   - a **dense** vector (`bge-small-en-v1.5`) for meaning ("hiding cash deposits" ≈ "structuring")
   - a **sparse BM25** vector for exact terms ("CTR", "31 U.S.C. 5324")
4. **Search** – Qdrant runs both searches and merges them with Reciprocal Rank Fusion (`app/rag/store.py`).
5. **Answer** – the top passages are numbered and sent to the LLM, which must cite them as `[1]`, `[2]` (`app/rag/pipeline.py`).
6. **Fallback** – models are tried in order: Gemini Flash → Gemini Flash-Lite → Groq (`app/llm/router.py`). Each call has a 30-second limit; on a missing key, timeout, rate limit or overload (free-tier 503s are common) the next one answers. If none works, you still get the retrieved passages.
7. **Follow-ups** – the previous question is added to the search, so "is it a crime?" still finds the right passages.

## Quick start (Windows)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Add at least one LLM key to `.env` (see `SETUP_ACCOUNTS.md`), then check it:

```bash
python -m scripts.check_llm
```

Index the documents (the first run downloads ~100 MB of embedding models):

```bash
python -m scripts.ingest_docs
```

Start the backend, then the UI in a second terminal:

```bash
uvicorn app.api.main:app --port 8000
```

```bash
streamlit run ui/streamlit_app.py
```

Without `QDRANT_URL`, Qdrant runs embedded in `storage/qdrant/`. Only one process can open that folder, so stop the API before running `scripts.ingest_docs`, or use the **Re-index documents** button in the UI.

## Adding documents

Drop PDFs or markdown files into `data/docs/` (any subfolder) and re-index. `data/SOURCES.md` lists public regulatory documents worth adding. Re-indexing replaces a file's old chunks, so it's safe to run repeatedly.

## Tests

```bash
python -m pytest
```

## Data policy

Everything in this project is public or synthetic. Free LLM tiers may use prompts for training, which is another reason to never load real customer data.
