import pytest

from app.llm import LLMRouter, Message, NoProviderAvailable
from app.llm.base import RateLimited


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


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_failed_provider_is_skipped_during_cooldown():
    clock = Clock()
    gemini = FakeProvider("gemini", error=RuntimeError("504 DEADLINE_EXCEEDED"))
    groq = FakeProvider("groq", reply="from groq")
    router = LLMRouter([gemini, groq], cooldown_seconds=120, clock=clock)

    router.generate("sys", MESSAGES)
    clock.now += 60
    result = router.generate("sys", MESSAGES)
    assert gemini.calls == 1
    assert result.fallback_reasons == ["gemini/gemini-model: skipped for 60s more (failed recently)"]

    clock.now += 61  # cooldown over: try it again
    router.generate("sys", MESSAGES)
    assert gemini.calls == 2


def test_cooling_providers_are_still_tried_as_last_resort():
    clock = Clock()
    gemini = FakeProvider("gemini", error=RuntimeError("503"))
    groq = FakeProvider("groq", error=RuntimeError("429"))
    router = LLMRouter([gemini, groq], clock=clock)
    with pytest.raises(NoProviderAvailable):
        router.generate("sys", MESSAGES)

    gemini.error = None
    gemini.reply = "back"
    clock.now += 10
    assert router.generate("sys", MESSAGES).text == "back"


class FlakyProvider(FakeProvider):
    """Fails with the given errors in order, then answers."""

    def __init__(self, name, errors, reply="answer"):
        super().__init__(name, reply=reply)
        self.errors = list(errors)

    def generate(self, system, messages):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.reply


def test_last_provider_rate_limited_is_waited_for():
    clock, slept = Clock(), []

    def sleep(seconds):
        slept.append(seconds)
        clock.now += seconds

    gemini = FakeProvider("gemini", error=RuntimeError("503"))
    groq = FlakyProvider("groq", [RateLimited("429 try again in 8.4s", 8.4)])
    result = LLMRouter([gemini, groq], clock=clock, sleep=sleep).generate("sys", MESSAGES)

    assert result.text == "answer" and groq.calls == 2
    assert slept == [pytest.approx(8.4)]
    assert result.fallback_reasons[-1] == "groq/groq-model: waited 8s for its rate limit"


def test_long_rate_limit_is_not_waited_for():
    slept = []
    groq = FlakyProvider("groq", [RateLimited("429 try again in 1m0s", 60)])
    with pytest.raises(NoProviderAvailable):
        LLMRouter([groq], clock=Clock(), sleep=slept.append).generate("sys", MESSAGES)
    assert slept == [] and groq.calls == 1


def test_rate_limited_provider_is_benched_only_as_long_as_asked():
    clock = Clock()
    gemini = FlakyProvider("gemini", [RateLimited("429", 5)])
    groq = FakeProvider("groq", reply="from groq")
    router = LLMRouter([gemini, groq], cooldown_seconds=120, clock=clock)

    router.generate("sys", MESSAGES)  # gemini rate-limited, groq answers
    clock.now += 6
    assert router.generate("sys", MESSAGES).text == "answer"  # gemini is back after 5s, not 120s


def test_retry_after_is_read_from_provider_messages():
    from app.llm.base import retry_after_from

    assert retry_after_from("Rate limit reached ... Please try again in 8.4s. Need more tokens?") == 8.4
    assert retry_after_from("try again in 1m2.5s") == 62.5
    assert retry_after_from("429 RESOURCE_EXHAUSTED ... 'retryDelay': '40s'") == 40
    assert retry_after_from("503 UNAVAILABLE") is None
