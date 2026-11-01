import logging
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import Depends, FastAPI, Header, HTTPException, Request

from app.agent.approvals import ApprovalQueue
from app.agent.loop import Agent
from app.agent.mcp_hub import MCPHub, default_servers
from app.api.schemas import (
    ActionDecision, ActionOutcomeOut, AuditEventOut, AuditVerifyOut, ChatRequest, ChatResponse, HealthResponse,
    IngestResponse, UserOut,
)
from app.audit import AuditLog
from app.auth import User, UserDirectory
from app.config import PROJECT_ROOT, get_settings
from app.llm import Message, build_router
from app.observability import init_tracing, langfuse
from app.rag.ingest import ingest_directory
from app.rag.store import DocStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

state: dict = {}
# Re-indexing mid-search gives odd results, so health checks and ingestion take turns.
store_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    tracing = init_tracing(settings)
    router = build_router(settings)
    # Embedded Qdrant can only be opened by one process, and the aml-data MCP server needs it for
    # search_policies. So the API only opens the store itself when it's Qdrant Cloud.
    store = DocStore(settings) if settings.qdrant_url else None
    approvals = ApprovalQueue(ttl_seconds=settings.approval_ttl_minutes * 60)
    users = UserDirectory(settings.users_file)
    if users.open_mode:
        log.warning("No users file (%s): open mode, everyone is a local lead. "
                    "Create users with `python -m scripts.create_user`.", settings.users_file)
    audit = AuditLog(settings.audit_db)
    # The MCP servers stay connected for as long as the API is up.
    async with MCPHub(default_servers(settings, PROJECT_ROOT)) as hub:
        agent = Agent(hub, router, approvals, audit=audit, pseudonymize=settings.pii_pseudonymization)
        state.update(settings=settings, router=router, store=store, hub=hub, agent=agent, tracing=tracing,
                     users=users, audit=audit)
        yield
    if store:
        store.client.close()
    tracing.shutdown()  # send any buffered spans before the process exits


app = FastAPI(title="AML & Fraud Assistant", lifespan=lifespan)


def current_user(request: Request, x_api_key: str | None = Header(default=None)) -> User:
    user = state["users"].authenticate(x_api_key)
    if user is None:
        state["audit"].record(actor_id="anonymous", actor_role="none", event="auth.failed",
                              details={"path": request.url.path, "key_given": bool(x_api_key)})
        raise HTTPException(401, "Missing or invalid API key (X-API-Key header)")
    return user


def audit_reader(user: User = Depends(current_user)) -> User:
    if not user.can_read_audit:
        raise HTTPException(403, "Only a lead can read the audit log")
    return user


@app.get("/health", response_model=HealthResponse)
def health():
    store: DocStore | None = state["store"]
    count = None
    if store:
        with store_lock:
            count = store.count()
    settings = state["settings"]
    return HealthResponse(
        status="ok",
        llm_providers=state["router"].configured(),
        vector_store="qdrant-cloud" if store else "qdrant-local",
        indexed_chunks=count,
        mcp_servers={name: asdict(status) for name, status in state["hub"].status.items()},
        tracing=bool(settings.langfuse_public_key and settings.langfuse_secret_key),
        auth="open" if state["users"].open_mode else "api_key",
        pii_pseudonymization=settings.pii_pseudonymization,
    )


@app.get("/me", response_model=UserOut)
def me(user: User = Depends(current_user)):
    return UserOut(**asdict(user), can_read_audit=user.can_read_audit,
                   approvable_tools=[t for t in state["hub"].approval_tools() if user.can_use(t, True)])


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, user: User = Depends(current_user)):
    history = [Message(turn.role, turn.content) for turn in request.history]
    result = await state["agent"].answer(request.message, history, request.session_id, user)
    trace_url = langfuse().get_trace_url(trace_id=result.trace_id) if result.trace_id else None
    return ChatResponse(**asdict(result), trace_url=trace_url)


@app.post("/actions/{action_id}", response_model=ActionOutcomeOut)
async def decide(action_id: str, decision: ActionDecision, user: User = Depends(current_user)):
    """Approve or reject an action the agent proposed. Approving runs it exactly as proposed, once."""
    outcome = await state["agent"].execute(action_id, decision.approve, decision.session_id, user)
    return ActionOutcomeOut(**asdict(outcome))


@app.get("/audit", response_model=list[AuditEventOut])
def audit_events(limit: int = 50, event: str | None = None, user: User = Depends(audit_reader)):
    # Reading the audit log is itself recorded.
    state["audit"].record(actor_id=user.user_id, actor_role=user.role, event="audit.viewed",
                          details={"limit": limit, "event": event})
    return state["audit"].recent(min(max(limit, 1), 500), event)


@app.get("/audit/verify", response_model=AuditVerifyOut)
def audit_verify(user: User = Depends(audit_reader)):
    return state["audit"].verify()


@app.post("/ingest", response_model=IngestResponse)
def ingest(user: User = Depends(current_user)):
    store: DocStore | None = state["store"]
    if store is None:
        raise HTTPException(409, "Embedded Qdrant is held by the MCP server. Stop the API and run "
                                 "`python -m scripts.ingest_docs`, or set QDRANT_URL to use Qdrant Cloud.")
    with store_lock:
        stats = ingest_directory(store, state["settings"])
    state["audit"].record(actor_id=user.user_id, actor_role=user.role, event="docs.ingested", details=asdict(stats))
    return IngestResponse(**asdict(stats))
