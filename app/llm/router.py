import logging
import time
from dataclasses import dataclass, field

from app.config import Settings
from app.llm.base import (
    AssistantTurn, LLMProvider, LLMResult, Message, NoProviderAvailable, RateLimited, ToolSpec, chat_format,
)
from app.llm.gemini import GeminiProvider
from app.llm.groq_provider import GroqProvider
from app.observability import langfuse

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
    doesn't cost every request a 30-second timeout. A rate-limited provider is skipped for as long as it
    asked us to wait instead.

    Last resort, when nothing else answered: providers that are cooling down are tried anyway (a slow answer
    beats none), and a rate-limited one is waited for if that's at most `max_wait_seconds`. Groq's free tier
    allows 8,000 tokens a minute, so a quick multi-step task often needs a few seconds' pause.
    """

    def __init__(self, providers: list[LLMProvider], cooldown_seconds: float = 120, max_wait_seconds: float = 20,
                 clock=time.monotonic, sleep=time.sleep):
        self.providers = providers
        self.cooldown_seconds = cooldown_seconds
        self.max_wait_seconds = max_wait_seconds
        self.clock = clock
        self.sleep = sleep  # the agent runs the router in a worker thread, so sleeping blocks nothing else
        self._down_until: dict[int, float] = {}
        self._rate_limited: set[int] = set()

    def configured(self) -> list[str]:
        return [f"{p.name}/{p.model}" for p in self.providers if p.is_configured()]

    def generate(self, system: str, messages: list[Message]) -> LLMResult:
        trace = _Generation("generate-text", chat_format(system, messages))
        text, index, reasons = self._first_answer(lambda p: p.generate(system, messages), trace)
        provider = self.providers[index]
        return LLMResult(text=text, provider=provider.name, model=provider.model, fallback_reasons=reasons)

    def chat(self, system: str, messages: list[Message], tools: list[ToolSpec], start: int = 0,
             force_answer: bool = False) -> ChatStep:
        """One step of a tool-using conversation, starting from provider `start`.

        The agent passes the index of the provider that answered its previous step, so a conversation only ever
        moves down the list: once Gemini has failed mid-task we don't go back to it for the next step.
        `force_answer` keeps the tools visible (the history contains tool calls) but forbids calling them.
        """
        trace = _Generation("plan-next-step", chat_format(system, messages),
                            {"tool_choice": "none" if force_answer else "auto"},
                            {"tools_offered": [t.name for t in tools]})
        turn, index, reasons = self._first_answer(
            lambda p: p.chat(system, messages, tools, force_answer), trace, start
        )
        return ChatStep(turn=turn, provider_index=index, label=self._label(self.providers[index]), fallback_reasons=reasons)

    def _first_answer(self, call, trace: "_Generation", start: int = 0):
        reasons: list[str] = []
        second_chance: list[int] = []
        for index in range(start, len(self.providers)):
            label = self._label(self.providers[index])
            if not self.providers[index].is_configured():
                reasons.append(f"{label}: no API key")
            elif self._remaining(index) > 0:
                reasons.append(f"{label}: skipped for {self._remaining(index):.0f}s more (failed recently)")
                second_chance.append(index)
            else:
                answer = self._try(call, index, reasons, trace)
                if answer is not None:
                    return answer, index, reasons
                if index in self._rate_limited:
                    second_chance.append(index)
        # Last resort: nothing has answered.
        for index in sorted(second_chance, key=self._remaining):
            wait = self._remaining(index)
            if index in self._rate_limited:
                if wait > self.max_wait_seconds:
                    continue  # asking again now would only be refused again
                if wait > 0:
                    reasons.append(f"{self._label(self.providers[index])}: waited {wait:.0f}s for its rate limit")
                    log.info("Waiting %.1fs for %s's rate limit", wait, self._label(self.providers[index]))
                    self.sleep(wait)
            answer = self._try(call, index, reasons, trace)
            if answer is not None:
                return answer, index, reasons
        raise NoProviderAvailable(reasons)

    def _try(self, call, index: int, reasons: list[str], trace: "_Generation"):
        """One attempt, traced as a generation. Failed attempts are traced too (level ERROR), so a fallback
        shows up in Langfuse as the attempt that failed, followed by the one that answered."""
        provider = self.providers[index]
        label = self._label(provider)
        with langfuse().start_as_current_observation(
            as_type="generation", name=trace.name, model=provider.model, input=trace.messages,
            model_parameters={"temperature": getattr(provider, "temperature", None), **trace.parameters},
            metadata={"provider": provider.name, "configured_model": provider.model, "fallback_position": index,
                      **trace.metadata},
        ) as generation:
            try:
                answer = call(provider)
            except Exception as exc:
                log.warning("LLM %s failed, trying next: %s", label, exc)
                generation.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}"[:1000])
                reasons.append(f"{label}: {exc}")
                retry_after = exc.retry_after if isinstance(exc, RateLimited) else None
                if retry_after is not None:
                    self._rate_limited.add(index)
                else:
                    self._rate_limited.discard(index)
                self._down_until[index] = self.clock() + (
                    retry_after if retry_after is not None else self.cooldown_seconds
                )
                return None
            if isinstance(answer, AssistantTurn):
                generation.update(output=answer.as_message(), model=answer.model or provider.model,
                                  usage_details=answer.usage or None)
            else:
                generation.update(output={"role": "assistant", "content": answer})
        self._down_until.pop(index, None)
        self._rate_limited.discard(index)
        return answer

    def _remaining(self, index: int) -> float:
        """Seconds until a provider that failed should be tried again (0 if it's available)."""
        return max(0.0, self._down_until.get(index, 0.0) - self.clock())

    @staticmethod
    def _label(provider: LLMProvider) -> str:
        return f"{provider.name}/{provider.model}"


@dataclass
class _Generation:
    """What to record for each attempt at one LLM step."""

    name: str  # stable, verb-first, no model name: dashboards and evaluators filter on it
    messages: list[dict]
    parameters: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


def build_router(settings: Settings) -> LLMRouter:
    gemini_models = [settings.gemini_model]
    if settings.gemini_fallback_model and settings.gemini_fallback_model != settings.gemini_model:
        gemini_models.append(settings.gemini_fallback_model)

    common = dict(temperature=settings.llm_temperature, timeout_seconds=settings.llm_timeout_seconds)
    providers: list[LLMProvider] = [GeminiProvider(settings.gemini_api_key, m, **common) for m in gemini_models]
    providers.append(GroqProvider(settings.groq_api_key, settings.groq_model, **common))
    return LLMRouter(providers)
