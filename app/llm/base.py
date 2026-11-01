from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    # Gemini 3 returns an opaque "thought signature" with tool calls and rejects the next request if it isn't
    # sent back. It's only valid for the model that produced it, so we remember which one that was.
    signature: bytes | None = None
    model: str = ""


@dataclass
class Message:
    role: Literal["user", "assistant", "tool"]
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)  # assistant messages that request tools
    tool_call_id: str = ""  # tool messages: which call this is the result of
    tool_name: str = ""


@dataclass
class ToolSpec:
    """A tool the model may call, described with JSON Schema (MCP's inputSchema)."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class AssistantTurn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    # Why earlier providers were skipped, e.g. ["gemini: 429 RESOURCE_EXHAUSTED"].
    fallback_reasons: list[str] = field(default_factory=list)


class LLMProvider(Protocol):
    name: str
    model: str

    def is_configured(self) -> bool: ...

    def generate(self, system: str, messages: list[Message]) -> str: ...

    def chat(self, system: str, messages: list[Message], tools: list[ToolSpec],
             force_answer: bool = False) -> AssistantTurn: ...


class NoProviderAvailable(RuntimeError):
    def __init__(self, reasons: list[str]):
        self.reasons = reasons
        super().__init__("No LLM provider could answer: " + "; ".join(reasons))
