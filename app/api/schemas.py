from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[ChatTurn] = []
    # Groups a conversation's turns (and approval decisions) into one Langfuse session.
    session_id: str | None = Field(default=None, max_length=200)


class SourceOut(BaseModel):
    number: int
    title: str
    source: str
    section: str
    page: int | None
    score: float
    text: str


class ToolStepOut(BaseModel):
    tool: str
    arguments: dict[str, Any]
    ok: bool
    preview: str
    duration_ms: int
    round: int
    needs_approval: bool


class PendingActionOut(BaseModel):
    id: str
    tool: str
    server: str
    arguments: dict[str, Any]
    created_at: float
    proposed_by: str


class ChatResponse(BaseModel):
    answer: str
    provider: str
    model: str
    sources: list[SourceOut]
    tool_calls: list[ToolStepOut]
    fallback_reasons: list[str]
    pending_actions: list[PendingActionOut]
    trace_id: str | None = None
    trace_url: str | None = None


class ActionDecision(BaseModel):
    approve: bool
    session_id: str | None = Field(default=None, max_length=200)


class ActionOutcomeOut(BaseModel):
    action_id: str
    status: Literal["done", "failed", "rejected", "expired", "denied"]
    tool: str
    result: str


class MCPServerOut(BaseModel):
    tools: list[str]
    approval_tools: list[str]
    error: str | None


class HealthResponse(BaseModel):
    status: str
    llm_providers: list[str]
    vector_store: str
    indexed_chunks: int | None  # None with embedded Qdrant, which the MCP server process holds open
    mcp_servers: dict[str, MCPServerOut]
    tracing: bool
    auth: Literal["open", "api_key"]
    pii_pseudonymization: bool


class UserOut(BaseModel):
    user_id: str
    name: str
    role: str
    can_read_audit: bool
    approvable_tools: list[str]


class AuditEventOut(BaseModel):
    id: int
    ts: str
    actor_id: str
    actor_role: str
    event: str
    session_id: str | None
    trace_id: str | None
    details: dict[str, Any]
    prev_hash: str
    hash: str


class AuditVerifyOut(BaseModel):
    ok: bool
    entries: int
    latest_hash: str | None = None
    broken_at: int | None = None
    reason: str | None = None


class IngestResponse(BaseModel):
    files: int
    chunks: int
    total_points: int
