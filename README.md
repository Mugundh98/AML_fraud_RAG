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
       ├── aml-data-mcp (ours) ──┬── Neo4j  (customers, accounts, transactions)   [phase 2 ✓]
       │                         └── Qdrant (regulations, policies, typologies)   [phase 1 ✓]
       ├── tavily-mcp            (web search)                     [phase 4]
       ├── google-calendar-mcp   (case review meetings)           [phase 4]
       └── slack-mcp             (escalations)                    [phase 4]
```

## Roadmap

| Phase | What | Status |
|---|---|---|
| 1 | Document RAG: hybrid search in Qdrant, cited answers, Gemini→Groq fallback, Streamlit chat | **Done** |
| 2 | Synthetic transactions (≤ 50k rows) in Neo4j + our own `aml-data-mcp` server | **Done** |
| 3 | Agent loop that routes between documents and the graph via MCP | Next |
| 4 | Tavily, Google Calendar, Slack MCP servers, with confirmation before any write | |
| 5 | Hardening: audit log, roles, PII redaction, Langfuse tracing, evaluation set | |

## Phase 1: how it works

1. **Load** – `.md`, `.txt` and `.pdf` files from `data/docs/` (`app/rag/loaders.py`).
2. **Chunk** – markdown is split on headings so every chunk knows its section, then packed into ~900-character chunks with 150 characters of overlap (`app/rag/chunking.py`).
3. **Embed** – each chunk gets two vectors, computed on your CPU with `fastembed`:
   - a **dense** vector (`bge-small-en-v1.5`) for meaning ("hiding cash deposits" ≈ "structuring")
   - a **sparse BM25** vector for exact terms ("CTR", "31 U.S.C. 5324")
4. **Search** – Qdrant runs both searches and merges them with distribution-based score fusion, DBSF (`app/rag/store.py`). We started with Reciprocal Rank Fusion, but it only compares ranks: for "structuring red flags" the dense search liked every chunk of a document *titled* "…Red Flags", and RRF let that outvote a much stronger keyword match. DBSF uses the actual scores and got 8/8 test queries right (RRF 7/8); a cross-encoder reranker also got 8/8 but was ~5× slower.
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

## Phase 2: the transaction graph and the MCP server

### Synthetic data

`app/graph/generator.py` builds a fictional bank from a fixed seed: about 1,200 customers, 1,500 accounts, 20 branches and 20,000 transactions over 180 days ending 2026-10-31. Most activity is normal (payroll, card spend, bills, P2P, cash). On top of that it plants 19 suspicious scenarios that match the knowledge base:

| Typology | What's planted |
|---|---|
| Structuring | 5–8 cash deposits of $9,000–$9,900 across branches in a week, then an offshore wire |
| Money mule | New account, many unrelated P2P senders, ~90% forwarded to a crypto exchange within hours |
| Circular flow | 3–5 shell consulting companies at one address passing the same money around a loop |
| Funnel account | Cash deposited in 6 states, withdrawn at a border branch |
| Pass-through | Offshore wire in, 92–98% wired to another offshore account within 2 days |
| Synthetic identity ring | 4–6 new customers sharing a phone number and device |
| PEP | Politically exposed person receiving offshore "consultancy fees" far above declared income |

Alerts exist for most scenarios, but **not all**: a few are deliberately missed, and a few alerts are false positives (a restaurant's big cash weekend, home-sale proceeds). `data/synthetic/scenarios.csv` is the answer key; it's never loaded into Neo4j, so phase 5 can use it to score the agent.

### Graph model

```
(:Customer)-[:OWNS]->(:Account)
(:Customer)-[:HAS_PHONE|USES_DEVICE|REGISTERED_AT]->(:Phone|Device|Address)
(:Account|Branch)-[:TRANSFER {amount, timestamp, channel}]->(:Account|Branch)
(:Alert)-[:ON_ACCOUNT]->(:Account)
```

Cash deposits come *from* a `Branch`, so "deposits at 4 branches" is a graph pattern, and shared phones or devices link customers directly. Full schema: `app/graph/schema.py`.

### Load it

```bash
python -m scripts.generate_data --transactions 20000
```

```bash
python -m scripts.load_graph
```

`--transactions` is capped by `MAX_TRANSACTIONS` in `.env` (50,000). Loading replaces whatever is in the database.

### The `aml-data` MCP server

`mcp_servers/aml_data/server.py` uses the official MCP Python SDK (v2, `MCPServer`). All tools are read-only fixed queries; the model never writes raw Cypher.

| Primitive | Name | Purpose |
|---|---|---|
| Tool | `search_policies` | RAG search over the knowledge base |
| Tool | `find_customers`, `get_customer_profile` | KYC profile, accounts, alerts, activity vs expected volume |
| Tool | `get_account_transactions`, `get_counterparty_summary` | Transaction history; fan-in/fan-out and pass-through ratio |
| Tool | `check_structuring`, `scan_for_structuring` | Near-threshold cash for one account, or bank-wide |
| Tool | `find_circular_flows` | Time-ordered loops of similar amounts, grouped by route (Cypher quantified path patterns) |
| Tool | `find_shared_identifiers` | Customers linked by phone, device or address |
| Tool | `list_alerts`, `get_alert` | Alert queue and alert detail with flagged transactions |
| Resource | `aml://graph-schema`, `aml://alerts/{alert_id}` | Context a client can attach |
| Prompt | `investigate_alert` | Level 1 triage template ending in close/escalate + SAR draft |

Try it:

```bash
python -m scripts.mcp_smoke
```

Check that the tools' queries find every planted scenario (currently 19/19, including the 5 with no alert):

```bash
python -m scripts.check_detection
```

Or explore it in the official MCP Inspector (needs Node.js):

```bash
npx @modelcontextprotocol/inspector .venv/Scripts/python.exe -m mcp_servers.aml_data
```

To use it from Claude Desktop or another MCP client, add a stdio server whose command is `.venv\Scripts\python.exe`, args `-m mcp_servers.aml_data`, and working directory set to this project.

With embedded Qdrant (no `QDRANT_URL`), `search_policies` can't run while the API is also running, because only one process can open the local store. Use Qdrant Cloud to run both.

## Adding documents

Drop PDFs or markdown files into `data/docs/` (any subfolder) and re-index. `data/SOURCES.md` lists public regulatory documents worth adding. Re-indexing replaces a file's old chunks, so it's safe to run repeatedly.

## Tests

```bash
python -m pytest
```

## Data policy

Everything in this project is public or synthetic. Free LLM tiers may use prompts for training, which is another reason to never load real customer data.
