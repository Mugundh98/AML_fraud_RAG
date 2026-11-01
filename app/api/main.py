import logging
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException

from app.agent.approvals import ApprovalQueue
from app.agent.loop import Agent
from app.agent.mcp_hub import MCPHub, default_servers
from app.api.schemas import (
    ActionDecision, ActionOutcomeOut, ChatRequest, ChatResponse, HealthResponse, IngestResponse,
)
from app.config import PROJECT_ROOT, get_settings
from app.llm import Message, build_router
from app.rag.ingest import ingest_directory
from app.rag.store import DocStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

state: dict = {}
# Re-indexing mid-search gives odd results, so health checks and ingestion take turns.
store_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    router = build_router(settings)
    # Embedded Qdrant can only be opened by one process, and the aml-data MCP server needs it for
    # search_policies. So the API only opens the store itself when it's Qdrant Cloud.
    store = DocStore(settings) if settings.qdrant_url else None
    approvals = ApprovalQueue(ttl_seconds=settings.approval_ttl_minutes * 60)
    # The MCP servers stay connected for as long as the API is up.
    async with MCPHub(default_servers(settings, PROJECT_ROOT)) as hub:
        state.update(settings=settings, router=router, store=store, hub=hub, agent=Agent(hub, router, approvals))
        yield
    if store:
        store.client.close()


app = FastAPI(title="AML & Fraud Assistant", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    store: DocStore | None = state["store"]
    count = None
    if store:
        with store_lock:
            count = store.count()
    return HealthResponse(
        status="ok",
        llm_providers=state["router"].configured(),
        vector_store="qdrant-cloud" if store else "qdrant-local",
        indexed_chunks=count,
        mcp_servers={name: asdict(status) for name, status in state["hub"].status.items()},
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    history = [Message(turn.role, turn.content) for turn in request.history]
    result = await state["agent"].answer(request.message, history)
    return ChatResponse(**asdict(result))


@app.post("/actions/{action_id}", response_model=ActionOutcomeOut)
async def decide(action_id: str, decision: ActionDecision):
    """Approve or reject an action the agent proposed. Approving runs it exactly as proposed, once."""
    outcome = await state["agent"].execute(action_id, decision.approve)
    return ActionOutcomeOut(**asdict(outcome))


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    store: DocStore | None = state["store"]
    if store is None:
        raise HTTPException(409, "Embedded Qdrant is held by the MCP server. Stop the API and run "
                                 "`python -m scripts.ingest_docs`, or set QDRANT_URL to use Qdrant Cloud.")
    with store_lock:
        stats = ingest_directory(store, state["settings"])
    return IngestResponse(**asdict(stats))
