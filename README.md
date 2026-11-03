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
| 5 | Hardening: Langfuse tracing, users and roles, audit log, PII pseudonymization, evaluation set | **Done** |

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

What the first traces showed, and what changed:
- **Groq's free tier allows 8,000 tokens a minute**, and an agent step is 2,500–4,000 tokens (most of it the 18 tool definitions), so a quick Groq-only investigation hit a 429 on its third step and failed. Two fixes:
  - **Compact tool schemas** (`compact_schema` in `app/agent/mcp_hub.py`): the labels pydantic adds to every field, `"default": null` and `anyOf [X, null]` wrappers are dropped, and Tavily's search tool exposes 5 of its 17 options. Definitions went from 16,290 to 10,579 characters, but Groq's first-step input only fell from 2,804 to 2,508 tokens (−11%): the descriptions that tell the model when to use each tool are most of the cost, and they stay.
  - **Waiting out short rate limits** (`app/llm/router.py`): providers report "try again in N seconds" as `RateLimited`. A rate-limited provider is benched for exactly that long (not the 2-minute circuit-breaker cooldown), and when no other model is left the router waits up to 20 seconds and retries. The same investigation now pauses ~4 seconds on step 3 and finishes.
- **Groq has no price in Langfuse**: `openai/gpt-oss-120b` isn't in its default model list, so those generations have tokens but no cost. Add a custom model definition in Langfuse (Settings > Models) if you want one.

## Phase 5: users, roles, audit log and PII

### Users and roles (`app/auth.py`)

Each analyst has an API key, sent as the `X-API-Key` header; the UI asks for it in the sidebar. Only SHA-256 hashes are stored, in `secrets/users.json` (git ignores `secrets/`).

```bash
python -m scripts.create_user --name "Priya Shah" --role senior_analyst
```

| Role | Read-only tools | Book meetings (`create_event`) | Escalate to Slack (`post_message`) | Read the audit log |
|---|---|---|---|---|
| `analyst` (Level 1) | ✓ | ✓ | | |
| `senior_analyst` (Level 2) | ✓ | ✓ | ✓ | |
| `lead` | ✓ | ✓ | ✓ | ✓ |

The model is only offered the tools the signed-in analyst may use, and is told which ones are withheld, so a Level 1 analyst who asks to escalate gets "only a senior analyst or lead can escalate" and a draft to hand over, not a proposal they can't approve. Approval is checked again on the server; a denied approval leaves the action waiting for someone who can approve it.

Without a users file the API runs in **open mode**: everyone is one local lead, and `/health` and the sidebar say so.

### Audit log (`app/audit.py`)

An append-only SQLite table (`storage/audit.db`) with real names, since it's the bank's own record: every answered question with the tools and arguments used, every proposed action, every approval, rejection and denial, failed sign-ins, re-indexing, and every read of the audit log itself. Entries are linked into a **hash chain**: each stores the SHA-256 of its content plus the previous entry's hash, so editing or deleting a past entry breaks verification from that point on. Leads can see the log and its verification in the UI (**Show audit log**), or via `GET /audit` and `GET /audit/verify`. The chain makes tampering visible, not impossible: someone who can rewrite the whole file can rebuild it, which is why real systems also copy the latest hash somewhere else.

Each audit entry stores the Langfuse trace ID, so a decision can be traced back to exactly what the model saw.

### PII pseudonymization (`app/agent/pii.py`)

The LLM providers and Langfuse never see customer names or identifiers:

```
graph tool result     {"name": "Jessica Martin", "phone": "+1-892-555-0194"}
what the model sees   {"name": "<NAME_1>",       "phone": "<PHONE_1>"}
model calls           check_structuring(...) / find_customers(name="<NAME_1>")  ─► our server gets "Jessica Martin"
model answers         "<NAME_1> made 5 deposits just under $10,000..."
analyst reads         "Jessica Martin made 5 deposits just under $10,000..."
```

- **What's PII**: customer, account-holder and counterparty names (branches excluded: "Miami branch" is a place), the phone, device and address values behind shared identifiers, and emails and phone numbers anywhere in text.
- **Per conversation**: `<NAME_1>` means the same person in every turn, and earlier answers in the history are re-pseudonymized before they go back to the model.
- **Outside services get nothing**: each MCP server declares `returns_pii` / `receives_pii`. Placeholders are swapped back for real values only for our own servers; a web search containing a placeholder is refused, so customer data can't leave through it.
- **Last line of defence**: the Langfuse exporter masks every real value the app has pseudonymized as `[PII]`. In live testing that caught a leak the first layer missed (a name inside a transaction description, scanned before its own field was), which is now fixed: redaction is two-pass, and a regression test covers it. A clean trace now has zero `[PII]` masks.
- **Limits**: a name the analyst types before any tool has returned it goes to the model as typed, and free text isn't scanned for names it hasn't seen. Production systems add a named-entity detector for both. Turn it all off with `PII_PSEUDONYMIZATION=false`.

### Evaluation (`app/evaluation.py`, `scripts/evaluate.py`)

A 12-item Langfuse dataset, `aml-investigations-v0`, built from the generator's answer key (`scenarios.csv`), so the expected outputs are ground truth by construction:

| Items | What they test |
|---|---|
| 7 positive, one per typology (S04, S07, S10, S11, S14, S17, S18; 5 of them have no alert) | finds the planted activity and names the typology |
| 2 ordinary customers | doesn't accuse people who did nothing |
| 1 false positive: the business the structuring scan flags (near-threshold deposits that match its profile) | judgement, not just pattern matching |
| 2 policy facts (SAR deadline, CTR threshold) | answers from the knowledge base |

Each review question ends with "Finish with one line: Verdict: suspicious / not suspicious…", so scoring is plain code (free, fast, reproducible, no LLM judge):

| Score | Meaning |
|---|---|
| `verdict_correct` | the final verdict line matches the answer key |
| `typology_named` | the answer names the planted typology |
| `right_tool_used` | the matching detection tool was called (e.g. `find_circular_flows` for a cycle) |
| `facts_mentioned` | required IDs or facts appear (account ID, "30" days, "$10,000") |
| `tool_calls` | how many tool calls it took |

Scores land on each item's trace, and the run gets averages, so runs can be compared side by side in Langfuse (**Datasets > aml-investigations-v0**). Each item runs the full agent against the `aml-data` server only, as an analyst, one item at a time (~40-60 model calls per run).

```bash
python -m scripts.evaluate show                      # review the items
python -m scripts.evaluate upload                    # create or update the dataset (idempotent)
python -m scripts.evaluate run --name my-change      # run an experiment
```

**Baseline (`agent-baseline`, 1 Nov 2026, answered by Gemini Flash-Lite after Flash hit its free daily quota):** 12/12. Verdict correct on all 10 reviews (7 planted cases, 2 ordinary customers, the false-positive business), typology named and the right detection tool used on all 7 positives, both policy facts right; 4.2 tool calls per item on average.

A perfect score mostly says v0 is too easy to tell versions apart: every question names the customer and account, and the agent runs the same battery of checks each time (profile, transactions, counterparties, structuring). The next dataset version should add harder items, such as a case found only through a linked customer, a review starting from just a name, or a borderline business, and track `tool_calls` to catch an agent that gets right answers by calling everything.

## Adding documents

Drop PDFs or markdown files into `data/docs/` (any subfolder) and re-index. `data/SOURCES.md` lists public regulatory documents worth adding. Re-indexing replaces a file's old chunks, so it's safe to run repeatedly.

## Running in a container

The app also runs as one Docker container (`Dockerfile`): the API listens only inside it, and Streamlit is the one published port (7860). Neo4j, Qdrant and the LLMs are already cloud services, so only the backend and UI are in it. It's ready for hosting, but isn't hosted.

```bash
docker build -t aml-assistant .
```

```bash
docker run --rm -p 127.0.0.1:7860:7860 --env-file .env -e USERS_JSON="$(cat secrets/users.json)" aml-assistant
```

Then open http://localhost:7860. Add `-e GOOGLE_CLIENT_SECRET_JSON="$(cat secrets/google_client_secret.json)" -e GOOGLE_TOKEN_JSON="$(cat secrets/google_token.json)"` for Google Calendar.

How settings reach the app: hosts pass secrets as environment variables, not files. `deploy/start.sh` runs `deploy/bootstrap.py`, which writes the three that are files (`USERS_JSON`, `GOOGLE_CLIENT_SECRET_JSON`, `GOOGLE_TOKEN_JSON`) where the app expects them, and `settings_env()` in `app/agent/mcp_hub.py` forwards the settings to the MCP server subprocesses (the MCP SDK passes them almost no environment by default, so without it the aml-data server would start without its database password).

A hosted copy refuses to start without users (`deploy/bootstrap.py`), so a stranger with the link can't act as a lead, post to Slack or use up the AI quota. `-e ALLOW_OPEN_MODE=true` overrides that for a local test.

### Hosting options we checked (November 2026)

Measured memory: about 700 MB in the container (Linux), 1,030 MB on Windows (aml-data server 531 MB with the search models, API 392 MB, Slack and Calendar servers ~50 MB each).

| Host | Result |
|---|---|
| Hugging Face Spaces | Docker Spaces on free CPU now need a PRO subscription ($9/month); creating one on a free account returns 402 |
| Render free | 512 MB, too small as is. Would need the MCP servers in-process (one copy of the search models) and possibly hosted embeddings |
| Google Cloud Run | Should run this container as is with 1 GB of memory, within the free monthly allowance, but needs a billing card |
| A tunnel (ngrok, Cloudflare) | Free; works only while the laptop runs the app |

On any of them: rotate keys that have been pasted anywhere first; free hosts have no permanent disk, so the audit log resets on restart (a disk mounted at `/data` is used automatically); and Google Calendar's sign-in expires every 7 days while the Google app is in Testing mode.

## Tests

```bash
python -m pytest
```

## Data policy

Everything in this project is public or synthetic. Free LLM tiers may use prompts for training, which is another reason to never load real customer data.
