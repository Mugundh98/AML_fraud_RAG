from typing import Literal

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[ChatTurn] = []


class SourceOut(BaseModel):
    number: int
    title: str
    source: str
    section: str
    page: int | None
    score: float
    text: str


class ChatResponse(BaseModel):
    answer: str
    provider: str
    model: str
    sources: list[SourceOut]
    fallback_reasons: list[str]


class HealthResponse(BaseModel):
    status: str
    llm_providers: list[str]
    vector_store: str
    indexed_chunks: int


class IngestResponse(BaseModel):
    files: int
    chunks: int
    total_points: int
