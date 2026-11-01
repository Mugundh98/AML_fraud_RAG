import json
import re
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
    # For tracing: the exact model that answered (aliases like gemini-flash-latest resolve to a version),
    # token counts as non-overlapping Langfuse usage buckets, and the model's thinking if it shares it.
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    reasoning: str = ""

    def as_message(self) -> dict:
        """OpenAI-style assistant message, the format Langfuse renders as a conversation."""
        message: dict = {"role": "assistant", "content": self.text}
        if self.tool_calls:
            message["tool_calls"] = [_tool_call_dict(c) for c in self.tool_calls]
        if self.reasoning:
            message["reasoning_content"] = self.reasoning
        return message


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


def chat_format(system: str, messages: list[Message]) -> list[dict]:
    """Our messages in the OpenAI chat format (Groq's API, and Langfuse's conversation view)."""
    chat = [{"role": "system", "content": system}]
    for m in messages:
        if m.role == "tool":
            chat.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
        elif m.tool_calls:
            chat.append({"role": "assistant", "content": m.content or None,
                         "tool_calls": [_tool_call_dict(c) for c in m.tool_calls]})
        else:
            chat.append({"role": m.role, "content": m.content})
    return chat


def _tool_call_dict(call: ToolCall) -> dict:
    return {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": json.dumps(call.arguments)}}


class RateLimited(RuntimeError):
    """The provider refused the request for now; `retry_after` says how long to wait, if it told us."""

    def __init__(self, message: str, retry_after: float | None):
        super().__init__(message)
        self.retry_after = retry_after


# "Please try again in 8.4s" / "in 1m2.5s" (Groq), "'retryDelay': '40s'" (Gemini)
_WAIT = re.compile(r"(?:try again in|retryDelay'?:\s*'?)\s*(?:(\d+)m)?\s*(\d+(?:\.\d+)?)s", re.IGNORECASE)


def retry_after_from(text: str) -> float | None:
    match = _WAIT.search(text)
    if not match:
        return None
    return int(match.group(1) or 0) * 60 + float(match.group(2))


class NoProviderAvailable(RuntimeError):
    def __init__(self, reasons: list[str]):
        self.reasons = reasons
        super().__init__("No LLM provider could answer: " + "; ".join(reasons))
