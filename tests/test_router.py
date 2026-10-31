import pytest

from app.llm import LLMRouter, Message, NoProviderAvailable


class FakeProvider:
    def __init__(self, name, key="key", reply=None, error=None):
        self.name = name
        self.model = f"{name}-model"
        self.key = key
        self.reply = reply
        self.error = error
        self.calls = 0

    def is_configured(self):
        return bool(self.key)

    def generate(self, system, messages):
        self.calls += 1
        if self.error:
            raise self.error
        return self.reply


MESSAGES = [Message("user", "hi")]


def test_primary_answers_when_healthy():
    gemini = FakeProvider("gemini", reply="from gemini")
    groq = FakeProvider("groq", reply="from groq")

    result = LLMRouter([gemini, groq]).generate("sys", MESSAGES)

    assert (result.provider, result.text, result.fallback_reasons) == ("gemini", "from gemini", [])
    assert groq.calls == 0


def test_falls_back_on_error():
    gemini = FakeProvider("gemini", error=RuntimeError("429 RESOURCE_EXHAUSTED"))
    groq = FakeProvider("groq", reply="from groq")

    result = LLMRouter([gemini, groq]).generate("sys", MESSAGES)

    assert result.provider == "groq"
    assert result.fallback_reasons == ["gemini/gemini-model: 429 RESOURCE_EXHAUSTED"]


def test_skips_provider_without_key():
    gemini = FakeProvider("gemini", key="", reply="unused")
    groq = FakeProvider("groq", reply="from groq")

    result = LLMRouter([gemini, groq]).generate("sys", MESSAGES)

    assert result.provider == "groq"
    assert gemini.calls == 0


def test_raises_when_all_fail():
    router = LLMRouter([FakeProvider("gemini", key=""), FakeProvider("groq", error=RuntimeError("down"))])

    with pytest.raises(NoProviderAvailable) as exc:
        router.generate("sys", MESSAGES)
    assert exc.value.reasons == ["gemini/gemini-model: no API key", "groq/groq-model: down"]


def test_build_router_orders_gemini_models_before_groq():
    from app.config import Settings
    from app.llm import build_router

    settings = Settings(_env_file=None, gemini_api_key="g", groq_api_key="q")
    router = build_router(settings)

    assert router.configured() == [
        "gemini/gemini-flash-latest",
        "gemini/gemini-flash-lite-latest",
        "groq/openai/gpt-oss-120b",
    ]
