import logging
import time
from dataclasses import dataclass, field

from app.config import Settings
from app.llm.base import AssistantTurn, LLMProvider, LLMResult, Message, NoProviderAvailable, ToolSpec
from app.llm.gemini import GeminiProvider
from app.llm.groq_provider import GroqProvider

log = logging.getLogger(__name__)


@dataclass
class ChatStep:
    turn: AssistantTurn
    provider_index: int
    label: str  # "provider/model"
    fallback_reasons: list[str] = field(default_factory=list)


class LLMRouter:
    """Tries providers in order; the first one that answers wins.

    A provider is skipped when it has no API key, and falls through to the next
    one on any error (rate limit, outage, retired model, empty response).

    Circuit breaker: a provider that fails is skipped for `cooldown_seconds`, so an overloaded model
    doesn't cost every request a 30-second timeout. If every remaining provider is cooling down, they're
    tried anyway: a slow answer beats none.
    """

    def __init__(self, providers: list[LLMProvider], cooldown_seconds: float = 120, clock=time.monotonic):
        self.providers = providers
        self.cooldown_seconds = cooldown_seconds
        self.clock = clock
        self._failed_at: dict[int, float] = {}

    def configured(self) -> list[str]:
        return [f"{p.name}/{p.model}" for p in self.providers if p.is_configured()]

    def generate(self, system: str, messages: list[Message]) -> LLMResult:
        text, index, reasons = self._first_answer(lambda p: p.generate(system, messages))
        provider = self.providers[index]
        return LLMResult(text=text, provider=provider.name, model=provider.model, fallback_reasons=reasons)

    def chat(self, system: str, messages: list[Message], tools: list[ToolSpec], start: int = 0,
             force_answer: bool = False) -> ChatStep:
        """One step of a tool-using conversation, starting from provider `start`.

        The agent passes the index of the provider that answered its previous step, so a conversation only ever
        moves down the list: once Gemini has failed mid-task we don't go back to it for the next step.
        `force_answer` keeps the tools visible (the history contains tool calls) but forbids calling them.
        """
        turn, index, reasons = self._first_answer(lambda p: p.chat(system, messages, tools, force_answer), start)
        return ChatStep(turn=turn, provider_index=index, label=self._label(self.providers[index]), fallback_reasons=reasons)

    def _first_answer(self, call, start: int = 0):
        reasons: list[str] = []
        cooling: list[int] = []
        for index in range(start, len(self.providers)):
            label = self._label(self.providers[index])
            if not self.providers[index].is_configured():
                reasons.append(f"{label}: no API key")
            elif self._cooling_down(index):
                reasons.append(f"{label}: skipped, failed {self.clock() - self._failed_at[index]:.0f}s ago")
                cooling.append(index)
            else:
                answer = self._try(call, index, reasons)
                if answer is not None:
                    return answer, index, reasons
        for index in cooling:  # last resort
            answer = self._try(call, index, reasons)
            if answer is not None:
                return answer, index, reasons
        raise NoProviderAvailable(reasons)

    def _try(self, call, index: int, reasons: list[str]):
        provider = self.providers[index]
        label = self._label(provider)
        try:
            answer = call(provider)
        except Exception as exc:
            log.warning("LLM %s failed, trying next: %s", label, exc)
            reasons.append(f"{label}: {exc}")
            self._failed_at[index] = self.clock()
            return None
        self._failed_at.pop(index, None)
        return answer

    def _cooling_down(self, index: int) -> bool:
        failed_at = self._failed_at.get(index)
        return failed_at is not None and self.clock() - failed_at < self.cooldown_seconds

    @staticmethod
    def _label(provider: LLMProvider) -> str:
        return f"{provider.name}/{provider.model}"


def build_router(settings: Settings) -> LLMRouter:
    gemini_models = [settings.gemini_model]
    if settings.gemini_fallback_model and settings.gemini_fallback_model != settings.gemini_model:
        gemini_models.append(settings.gemini_fallback_model)

    common = dict(temperature=settings.llm_temperature, timeout_seconds=settings.llm_timeout_seconds)
    providers: list[LLMProvider] = [GeminiProvider(settings.gemini_api_key, m, **common) for m in gemini_models]
    providers.append(GroqProvider(settings.groq_api_key, settings.groq_model, **common))
    return LLMRouter(providers)
