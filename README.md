# AML & Fraud Assistant

A learning project for **RAG** and **MCP**, built as an enterprise-style AML and fraud assistant on free services.

## Architecture (target)

```
 Streamlit UI
      │  HTTP
      ▼
 FastAPI backend ──── SQLite (users, roles, audit log)            [phase 5]
  ├─ agent loop (Gemini → Groq fallback)                          [phase 3 ✓]
  └─ MCP client                                                   [phase 3 ✓]
       ├── aml-data-mcp (ours) ──┬── Neo4j  (customers, accounts, transactions)   [phase 2 ✓]
       │                         └── Qdrant (regulations, policies, typologies)   [phase 1 ✓]
       ├── tavily (hosted, HTTP) (web search)                     [phase 4 ✓]
       ├── google-calendar (ours) (case review meetings)          [phase 4 ✓]
       └── slack (ours)          (escalations)                    [phase 4 ✓]
```

## Roadmap

| Phase | What | Status |
|---|---|---|
| 1 | Document RAG: hybrid search in Qdrant, cited answers, Gemini→Groq fallback, Streamlit chat | **Done** |
| 2 | Synthetic transactions (≤ 50k rows) in Neo4j + our own `aml-data-mcp` server | **Done** |
| 3 | Agent loop that routes between documents and the graph via MCP | **Done** |
| 4 | Tavily, Google Calendar, Slack MCP servers, with confirmation before any write | **Done** |
| 5 | Hardening: Langfuse tracing ✓, audit log, roles, PII redaction, evaluation set | In progress |

## Phase 1: how it works

1. **Load** – `.md`, `.txt` and `.pdf` files from `data/docs/` (`app/rag/loaders.py`).
2. **Chunk** – markdown is split on headings so every chunk knows its section, then packed into ~900-character chunks with 150 characters of overlap (`app/rag/chunking.py`).
3. **Embed** – each chunk gets two vectors, computed on your CPU with `fastembed`:
   - a **dense** vector (`bge-small-en-v1.5`) for meaning ("hiding cash deposits" ≈ "structuring")
   - a **sparse BM25** vector for exact terms ("CTR", "31 U.S.C. 5324")
4. **Search** – Qdrant runs both searches and merges them with distribution-based score fusion, DBSF (`app/rag/store.py`). We started with Reciprocal Rank Fusion, but it only compares ranks: for "structuring red flags" the dense search liked every chunk of a document *titled* "…Red Flags", and RRF let that outvote a much stronger keyword match. DBSF uses the actual scores and got 8/8 test queries right (RRF 7/8); a cross-encoder reranker also got 8/8 but was ~5× slower.
5. **Answer** – the top passages are numbered and sent to the LLM, which must cite them as `[1]`, `[2]`. In phase 1 this was a fixed pipeline (search, then answer); since phase 3 the agent decides when to search, through the `search_policies` MCP tool.
6. **Fallback** – models are tried in order: Gemini Flash → Gemini Flash-Lite → Groq (`app/llm/router.py`). Each call has a 30-second limit; on a missing key, timeout, rate limit or overload (free-tier 503s are common) the next one answers. If none works, you still get the retrieved passages.
7. **Follow-ups** – the agent sees the recent conversation and writes its own search query, so "is it a crime?" still finds the right passages.

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

Without `QDRANT_URL`, Qdrant runs embedded in `storage/qdrant/`. Only one process can open that folder, and since phase 3 that's the MCP server the API starts. So with embedded Qdrant, stop the API before running `scripts.ingest_docs`; the **Re-index documents** button needs Qdrant Cloud.

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

With embedded Qdrant (no `QDRANT_URL`), only one copy of the server can search at a time, because only one process can open the local store: running the API (which starts its own copy) and the Inspector together needs Qdrant Cloud.

## Phase 3: the agent

The API no longer runs a fixed "search, then answer" pipeline. An agent loop (`app/agent/loop.py`) gives the model the MCP tools and lets it decide:

```
question ─► LLM ─► tool calls? ──no──► answer (+ cited sources, tool trace)
             ▲          │yes
             └── results ◄── MCP servers (calls in one round run in parallel)
```

- **MCP client** – `app/agent/mcp_hub.py` starts each MCP server as a subprocess when the API starts, lists its tools and keeps the connection open. Tools are offered to the model by their MCP name, description and JSON Schema. Several servers can be attached; phase 4 adds more.
- **Approval for writes** – tools that change something only run after the analyst approves (phase 4, below).
- **One tool format, two providers** – messages are stored in a neutral format and translated per provider (`app/llm/gemini.py`, `app/llm/groq_provider.py`). Gemini 3 attaches a *thought signature* to each tool call and rejects the next request if it isn't sent back, so the signature travels with the call.
- **Fallback mid-task** – if Gemini fails on step 3, Groq continues from the same conversation, including Gemini's tool calls and their results. A task only moves down the list, never back.
- **Circuit breaker** – a model that fails is skipped for 2 minutes, so an overloaded Gemini Flash doesn't cost every request a 30-second timeout (`app/llm/router.py`).
- **Citations across searches** – each `search_policies` call numbers its passages from 1, so the agent renumbers them per answer; the same passage found twice keeps its number. Only the passages the answer cites are returned as sources.
- **Bounded** – after 6 rounds of tool calls the model is told to answer with what it has. Tool results over 12,000 characters are cut, with a note telling the model to narrow its query.

Ask from the terminal and see every tool call:

```bash
python -m scripts.ask "Who owns account A001520? Is their cash activity suspicious?"
```

The Streamlit UI shows the same trace under **Tool calls** for each answer.

What live testing taught us (each one is now fixed):

| Problem | Fix |
|---|---|
| No tool mapped an account to its owner, so the model guessed and made 7 calls | `get_account_transactions` returns the owner; tool descriptions show the ID formats |
| Asked if activity was suspicious, the model compared monthly totals with the declared volume (which matched) and missed 5 deposits just under $10,000 | The system prompt tells it to run the matching detection tools |
| Told it may not call tools, Gemini Flash-Lite returned an empty reply | The last round also says, in words, to answer now |
| Gemini Flash overloaded (503/504) on most requests that day | Circuit breaker, so later requests go straight to Flash-Lite |

## Phase 4: web search, calendar, Slack, and approval before any write

Three more MCP servers, each started only when its credentials are in `.env` (see `SETUP_ACCOUNTS.md`):

| Server | How it connects | Tools | Needs approval |
|---|---|---|---|
| `tavily` | Tavily's **hosted** MCP server over HTTP; the key goes in an `Authorization` header | `tavily_search`, `tavily_extract` | no |
| `google-calendar` | ours, stdio (`mcp_servers/google_calendar/`), OAuth sign-in | `list_events`, `find_free_slots` | `create_event` |
| `slack` | ours, stdio (`mcp_servers/slack/`), bot token | `list_channels` | `post_message` |

Tavily also offers crawl, map and research tools; they're left out because they use many more of the free tier's 1,000 monthly credits.

### Approval before any write

```
model calls post_message(...) ─► not run: stored as a pending action ─► "waiting for your approval"
                                                  │
          analyst clicks Approve in the UI ───────┴─► POST /actions/{id} ─► tool runs with the stored arguments
```

- **What needs approval** – for our own servers, any tool not annotated `readOnlyHint`. For a third-party server the annotations are only that server's claim (the MCP spec says not to rely on them from untrusted servers), so `ServerConfig.auto_run` lists the tools that may run without approval and everything else needs it (`app/agent/mcp_hub.py`).
- **No LLM between approval and action** – approving runs the exact tool and arguments shown on the card. The model can't change the message after you've read it.
- **Once, and not forever** – an action is removed when it's decided, so a double click or a replayed request can't post twice. Undecided actions expire after `APPROVAL_TTL_MINUTES` (30) (`app/agent/approvals.py`).
- **The conversation remembers** – the UI adds the outcome ("approved and done", "rejected") to the chat, so a follow-up question knows what happened.
- **Dates** – the system prompt now tells the model today's date and timezone, because "tomorrow afternoon" is a real date, while time windows in the bank data still count back from the dataset's as-of date.

In the terminal, `scripts.ask` asks y/n for each proposed action:

```bash
python -m scripts.ask "Escalate alert AL-0002 to the compliance Slack channel with a short summary."
```

## Phase 5: tracing with Langfuse

Every chat turn is one Langfuse trace (Python SDK v5, built on OpenTelemetry), set up following the
[Langfuse skill](https://github.com/langfuse/skills)'s instrumentation guide and best-practices checklist:

```
answer-question (agent)            input: the question · output: the answer · session: the conversation
├── plan-next-step (generation)    level ERROR: Gemini Flash 504, so the router fell back
├── plan-next-step (generation)    gemini-3.5-flash-lite · 4,689 in / 76 out tokens · tool_calls ×3
├── get_account_transactions (tool)
├── check_structuring (tool)
├── search_policies (retriever)    the passages, with the numbers the model cited
└── plan-next-step (generation)    the answer · 4,013 tokens read from Gemini's cache
```

- **Generations** are recorded per attempt in the router (`app/llm/router.py`), so fallbacks are visible. Each has the messages in OpenAI format, the assistant message out (with `tool_calls` and, when the model shares it, `reasoning_content`), the model version that actually answered (aliases like `gemini-flash-latest` resolve to e.g. `gemini-3.5-flash-lite`, which Langfuse can price), and token usage as non-overlapping buckets (`input`, `output`, `cache_read_input_tokens`).
- **Tools** (`app/agent/loop.py`) are siblings of the generation that requested them, with the arguments, result, MCP server and tool call ID. A proposed write shows `awaiting_approval`; the decision is its own `decide-action` trace in the same session.
- **Sessions**: the UI sends one session ID per conversation, so turns and approval decisions group together. `scripts.ask` doesn't set one, since a single question isn't a conversation.
- **Secrets are masked at export** (`app/observability.py`): any string that looks like an API key or token is replaced with `[REDACTED]` before it leaves the process. (Customer data isn't masked yet; that's the PII-redaction step of phase 5. All bank data here is synthetic.)
- **Environment** is `development`, so experiments stay out of production views.
- Tracing is off unless both Langfuse keys are set; the code then uses a no-op client.

See the latest trace without opening Langfuse:

```bash
python -m scripts.show_trace
```

Two things the first traces showed:
- **Groq's free tier allows 8,000 tokens a minute**, and an agent step is about 3,900 tokens (most of it the 15 tool descriptions), so Groq-only conversations hit 429 errors after two steps.
- **Groq has no price in Langfuse**: `openai/gpt-oss-120b` isn't in its default model list, so those generations have tokens but no cost. Add a custom model definition in Langfuse (Settings > Models) if you want one.

## Adding documents

Drop PDFs or markdown files into `data/docs/` (any subfolder) and re-index. `data/SOURCES.md` lists public regulatory documents worth adding. Re-indexing replaces a file's old chunks, so it's safe to run repeatedly.

## Tests

```bash
python -m pytest
```

## Data policy

Everything in this project is public or synthetic. Free LLM tiers may use prompts for training, which is another reason to never load real customer data.
