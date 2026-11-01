"""Langfuse tracing: run the real agent loop and inspect the exported OpenTelemetry spans (kept in memory)."""

import asyncio
import json
import uuid

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app import observability
from app.agent.approvals import ApprovalQueue
from app.agent.loop import Agent
from app.agent.mcp_hub import MCPHub
from app.config import Settings
from app.llm import AssistantTurn, LLMRouter
from mcp_servers.aml_data.server import create_server
from tests.test_agent import ScriptedLLM, TwoDocs, call, mixed_server
from tests.test_mcp_server import FakeGraph

TYPE, INPUT, OUTPUT, LEVEL = ("langfuse.observation.type", "langfuse.observation.input",
                              "langfuse.observation.output", "langfuse.observation.level")


@pytest.fixture
def spans():
    """Tracing on, with spans exported to memory instead of Langfuse."""
    exporter = InMemorySpanExporter()
    # The SDK keeps one client per public key for the whole process, so each test gets its own key.
    settings = Settings(_env_file=None, langfuse_public_key=f"pk-lf-{uuid.uuid4()}", langfuse_secret_key="sk-lf-test")
    client = observability.init_tracing(settings, span_exporter=exporter)

    def finished():
        client.flush()
        return {s.name: s for s in exporter.get_finished_spans()}, exporter.get_finished_spans()

    yield finished
    client.shutdown()
    observability._client = None


def value(span, key):
    """Span attribute as Python: the SDK stores strings raw and everything else as JSON."""
    raw = span.attributes[key]
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def ask(llms, question="q", session_id="session-1"):
    async def main():
        async with MCPHub({"aml-data": create_server(lambda: FakeGraph(), lambda: TwoDocs())}) as hub:
            return await Agent(hub, LLMRouter(llms)).answer(question, [], session_id=session_id)

    return asyncio.run(main())


def test_turn_is_one_trace_with_generations_and_tools_under_the_agent(spans):
    llm = ScriptedLLM("gemini", [
        AssistantTurn("", [call("find_customers", name="test"), call("search_policies", query="SAR")],
                      model="gemini-3-flash", usage={"input": 120, "output": 15}),
        AssistantTurn("Test Person is C00001 [1].", model="gemini-3-flash", usage={"input": 300, "output": 20}),
    ])
    ask([llm], question="Who is Test Person?")
    by_name, all_spans = spans()

    root = by_name["answer-question"]
    assert root.attributes[TYPE] == "agent"
    assert value(root, INPUT) == "Who is Test Person?"
    assert value(root, OUTPUT) == "Test Person is C00001 [1]."
    assert root.attributes["session.id"] == "session-1"
    assert len({s.context.trace_id for s in all_spans}) == 1

    generations = [s for s in all_spans if s.name == "plan-next-step"]
    assert len(generations) == 2 and all(g.attributes[TYPE] == "generation" for g in generations)
    assert all(g.parent.span_id == root.context.span_id for g in generations)
    first = generations[0] if json.loads(generations[0].attributes[OUTPUT]).get("tool_calls") else generations[1]
    tool_calls = json.loads(first.attributes[OUTPUT])["tool_calls"]
    assert [c["function"]["name"] for c in tool_calls] == ["find_customers", "search_policies"]
    assert first.attributes["langfuse.observation.model.name"] == "gemini-3-flash"
    assert json.loads(first.attributes["langfuse.observation.usage_details"]) == {"input": 120, "output": 15}

    # Tools are siblings of the generation that asked for them, typed, and carry the tool call ID.
    assert by_name["find_customers"].attributes[TYPE] == "tool"
    assert by_name["search_policies"].attributes[TYPE] == "retriever"
    assert by_name["find_customers"].parent.span_id == root.context.span_id
    metadata = json.loads(by_name["find_customers"].attributes["langfuse.observation.metadata.tool_call_id"])
    assert metadata == "id-find_customers"
    # The retriever's output shows the passage numbers the model saw.
    assert json.loads(by_name["search_policies"].attributes[OUTPUT])["passages"][0]["number"] == 1


def test_failed_attempt_is_an_error_generation_before_the_fallback(spans):
    gemini = ScriptedLLM("gemini", [RuntimeError("503 UNAVAILABLE")])
    groq = ScriptedLLM("groq", [AssistantTurn("Done.")])
    ask([gemini, groq])
    _, all_spans = spans()

    attempts = [s for s in all_spans if s.name == "plan-next-step"]
    assert [s.attributes.get(LEVEL) for s in sorted(attempts, key=lambda s: s.start_time)] == ["ERROR", None]
    failed = next(s for s in attempts if s.attributes.get(LEVEL) == "ERROR")
    assert "503 UNAVAILABLE" in failed.attributes["langfuse.observation.status_message"]


def test_secrets_are_masked_before_export(spans):
    llm = ScriptedLLM("gemini", [AssistantTurn("Use token xoxb-1234567890-abcdefghij please.")])
    ask([llm], question="my key is tvly-dev-ABCDEFGHIJKLMNOP")
    exported = " ".join(str(v) for s in spans()[1] for v in s.attributes.values())

    assert "xoxb-1234567890" not in exported and "tvly-dev-ABCD" not in exported
    assert "[REDACTED]" in exported


def test_approval_decision_is_its_own_trace_in_the_same_session(spans):
    sent = []

    async def main():
        async with MCPHub({"mixed": mixed_server(sent)}) as hub:
            agent = Agent(hub, LLMRouter([ScriptedLLM("gemini", [
                AssistantTurn("", [call("send_message", text="escalate")]),
                AssistantTurn("Waiting for your approval."),
            ])]), ApprovalQueue(ttl_seconds=600))
            answer = await agent.answer("escalate it", [], session_id="s-9")
            await agent.execute(answer.pending_actions[0].id, True, session_id="s-9")

    asyncio.run(main())
    _, all_spans = spans()

    decide = next(s for s in all_spans if s.name == "decide-action")
    proposal = next(s for s in all_spans if s.name == "send_message" and s.parent.span_id != decide.context.span_id)
    executed = next(s for s in all_spans if s.name == "send_message" and s.parent.span_id == decide.context.span_id)
    assert json.loads(proposal.attributes[OUTPUT])["status"] == "awaiting_approval"
    assert decide.attributes["session.id"] == "s-9"
    assert decide.context.trace_id != proposal.context.trace_id
    assert executed.attributes[TYPE] == "tool" and sent == ["escalate"]


def test_tracing_is_off_without_keys():
    client = observability.init_tracing(Settings(_env_file=None))
    try:
        with client.start_as_current_observation(as_type="agent", name="x") as span:
            span.update(output="fine")  # a no-op, not an error
    finally:
        client.shutdown()
        observability._client = None
