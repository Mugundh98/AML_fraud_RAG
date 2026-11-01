"""How our provider-neutral messages are translated into each provider's tool-calling format (no network)."""

import json

from app.llm import Message, ToolCall
from app.llm.gemini import SKIP_SIGNATURE, GeminiProvider
from app.llm.groq_provider import GroqProvider

CALLS = [
    ToolCall("c1", "list_alerts", {"priority": "High"}, signature=b"sig-from-flash", model="flash"),
    ToolCall("c2", "search_policies", {"query": "SAR"}, model="flash"),
]
HISTORY = [
    Message("user", "Which alerts are urgent?"),
    Message("assistant", "", tool_calls=CALLS),
    Message("tool", '{"count": 0}', tool_call_id="c1", tool_name="list_alerts"),
    Message("tool", '{"passages": []}', tool_call_id="c2", tool_name="search_policies"),
]


def gemini(model):
    return GeminiProvider("key", model, temperature=0.1, timeout_seconds=30)


def test_gemini_returns_signature_to_the_model_that_made_it():
    contents = gemini("flash").to_contents(HISTORY)

    assert [c.role for c in contents] == ["user", "model", "user"]
    call_parts = contents[1].parts
    assert call_parts[0].function_call.name == "list_alerts"
    assert call_parts[0].thought_signature == b"sig-from-flash"
    assert call_parts[1].thought_signature is None  # parallel calls: only the first one carries it
    # Both results go back in one turn.
    assert [p.function_response.name for p in contents[2].parts] == ["list_alerts", "search_policies"]


def test_gemini_uses_placeholder_signature_for_another_models_calls():
    call_parts = gemini("flash-lite").to_contents(HISTORY)[1].parts
    assert call_parts[0].thought_signature == SKIP_SIGNATURE


def test_groq_messages_use_openai_tool_format():
    chat = GroqProvider.to_messages("system prompt", HISTORY)

    assert [m["role"] for m in chat] == ["system", "user", "assistant", "tool", "tool"]
    assert chat[2]["tool_calls"][0]["function"] == {"name": "list_alerts", "arguments": json.dumps({"priority": "High"})}
    assert chat[3] == {"role": "tool", "tool_call_id": "c1", "content": '{"count": 0}'}
