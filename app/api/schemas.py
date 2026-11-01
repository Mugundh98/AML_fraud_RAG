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


class ChatResponse(BaseModel):
    answer: str
    provider: str
    model: str
    sources: list[SourceOut]
    tool_calls: list[ToolStepOut]
    fallback_reasons: list[str]
    pending_actions: list[PendingActionOut]


class ActionDecision(BaseModel):
    approve: bool
    session_id: str | None = Field(default=None, max_length=200)


class ActionOutcomeOut(BaseModel):
    action_id: str
    status: Literal["done", "failed", "rejected", "expired"]
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


class IngestResponse(BaseModel):
    files: int
    chunks: int
    total_points: int
