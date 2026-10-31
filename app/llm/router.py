import logging

from app.config import Settings
from app.llm.base import LLMProvider, LLMResult, Message, NoProviderAvailable
from app.llm.gemini import GeminiProvider
from app.llm.groq_provider import GroqProvider

log = logging.getLogger(__name__)


class LLMRouter:
    """Tries providers in order; the first one that answers wins.

    A provider is skipped when it has no API key, and falls through to the next
    one on any error (rate limit, outage, retired model, empty response).
    """

    def __init__(self, providers: list[LLMProvider]):
        self.providers = providers

    def configured(self) -> list[str]:
        return [f"{p.name}/{p.model}" for p in self.providers if p.is_configured()]

    def generate(self, system: str, messages: list[Message]) -> LLMResult:
        reasons: list[str] = []
        for provider in self.providers:
            label = f"{provider.name}/{provider.model}"
            if not provider.is_configured():
                reasons.append(f"{label}: no API key")
                continue
            try:
                text = provider.generate(system, messages)
            except Exception as exc:
                log.warning("LLM %s failed, trying next: %s", label, exc)
                reasons.append(f"{label}: {exc}")
                continue
            return LLMResult(text=text, provider=provider.name, model=provider.model, fallback_reasons=reasons)
        raise NoProviderAvailable(reasons)


def build_router(settings: Settings) -> LLMRouter:
    gemini_models = [settings.gemini_model]
    if settings.gemini_fallback_model and settings.gemini_fallback_model != settings.gemini_model:
        gemini_models.append(settings.gemini_fallback_model)

    common = dict(temperature=settings.llm_temperature, timeout_seconds=settings.llm_timeout_seconds)
    providers: list[LLMProvider] = [GeminiProvider(settings.gemini_api_key, m, **common) for m in gemini_models]
    providers.append(GroqProvider(settings.groq_api_key, settings.groq_model, **common))
    return LLMRouter(providers)
