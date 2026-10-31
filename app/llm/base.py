from dataclasses import dataclass, field
from typing import Literal, Protocol


@dataclass
class Message:
    role: Literal["user", "assistant"]
    content: str


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


class NoProviderAvailable(RuntimeError):
    def __init__(self, reasons: list[str]):
        self.reasons = reasons
        super().__init__("No LLM provider could answer: " + "; ".join(reasons))
