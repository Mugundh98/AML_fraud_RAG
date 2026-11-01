"""The agent loop end to end: a scripted fake LLM, our real MCP server (in-process, fake databases), real MCP client."""

import asyncio
from types import SimpleNamespace

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from app.agent.loop import Agent
from app.agent.mcp_hub import MCPHub
from app.llm import AssistantTurn, LLMRouter, Message, ToolCall
from mcp_servers.aml_data.server import create_server
from tests.test_mcp_server import FakeGraph


class TwoDocs:
    def search(self, query, limit):
        hits = [
            SimpleNamespace(title="US Reporting", section="SAR", source="seed/us.md", page=None, score=0.9,
                            text="File a SAR within 30 days."),
            SimpleNamespace(title="Typologies", section="Structuring", source="seed/typ.md", page=None, score=0.8,
                            text="Structuring is splitting cash deposits."),
        ]
        return hits[:limit] if "SAR" in query else hits[1:limit + 1]


class ScriptedLLM:
    """Plays back a list of turns (or exceptions) and records what it was sent."""

    def __init__(self, name, script):
        self.name = name
        self.model = f"{name}-model"
        self.script = list(script)
        self.seen: list[list[Message]] = []
        self.forced = []

    def is_configured(self):
        return True

    def chat(self, system, messages, tools, force_answer=False):
        self.seen.append(list(messages))
        self.forced.append(force_answer)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def call(tool, **args):
    return ToolCall(id=f"id-{tool}", name=tool, arguments=args)


def ask(providers, question="q", server=None, **agent_kwargs):
    async def main():
        target = server or create_server(lambda: FakeGraph(), lambda: TwoDocs())
        async with MCPHub({"aml-data": target}) as hub:
            return await Agent(hub, LLMRouter(providers), **agent_kwargs).answer(question, [])

    return asyncio.run(main())


def test_tool_result_goes_back_to_the_model():
    llm = ScriptedLLM("gemini", [
        AssistantTurn("", [call("find_customers", name="test")]),
        AssistantTurn("Test Person is C00001."),
    ])
    result = ask([llm])

    assert result.answer == "Test Person is C00001."
    assert [(s.tool, s.ok, s.round) for s in result.tool_calls] == [("find_customers", True, 1)]
    tool_message = llm.seen[1][-1]
    assert tool_message.role == "tool" and tool_message.tool_name == "find_customers"
    assert '"customer_id": "C00001"' in tool_message.content


def test_parallel_calls_and_citations_numbered_across_searches():
    llm = ScriptedLLM("gemini", [
        AssistantTurn("", [call("search_policies", query="SAR deadline"), call("search_policies", query="structuring")]),
        AssistantTurn("Deadline is 30 days【1】; structuring【2】."),
    ])
    result = ask([llm])

    # The first search returns passages 1-2, the second finds the structuring passage again: it keeps number 2.
    assert [(s.number, s.source) for s in result.sources] == [(1, "seed/us.md"), (2, "seed/typ.md")]
    second_search = llm.seen[1][-1].content
    assert '"number": 2' in second_search and '"number": 1' not in second_search
    assert result.answer == "Deadline is 30 days[1]; structuring[2]."


def test_tool_errors_are_shown_to_the_model():
    llm = ScriptedLLM("gemini", [
        AssistantTurn("", [call("get_alert", alert_id="AL-9999")]),
        AssistantTurn("That alert doesn't exist."),
    ])
    result = ask([llm])

    assert result.tool_calls[0].ok is False
    assert llm.seen[1][-1].content.startswith("ERROR: ") and "AL-9999" in llm.seen[1][-1].content


def test_fallback_mid_task_continues_with_next_provider_and_never_goes_back():
    gemini = ScriptedLLM("gemini", [
        AssistantTurn("", [call("list_alerts", priority="High")]),
        RuntimeError("503 UNAVAILABLE"),
    ])
    groq = ScriptedLLM("groq", [
        AssistantTurn("", [call("find_customers", name="test")]),
        AssistantTurn("Done."),
    ])
    result = ask([gemini, groq])

    assert (result.provider, result.answer) == ("groq", "Done.")
    assert result.fallback_reasons == ["gemini/gemini-model: 503 UNAVAILABLE"]
    assert len(gemini.seen) == 2  # not asked again after failing
    # Groq picks up the conversation including Gemini's tool call and its result.
    assert [m.role for m in groq.seen[0]] == ["user", "assistant", "tool"]


def test_model_must_answer_after_max_rounds():
    llm = ScriptedLLM("gemini", [
        AssistantTurn("", [call("list_alerts")]),
        AssistantTurn("", [call("list_alerts")]),
        AssistantTurn("Best answer with what I have."),
    ])
    result = ask([llm], max_rounds=2)

    assert llm.forced == [False, False, True]
    assert llm.seen[2][-1].content.startswith("You have used all your tool calls")
    assert result.answer == "Best answer with what I have."


def test_model_that_ignores_the_limit_gets_no_more_tools():
    llm = ScriptedLLM("gemini", [AssistantTurn("", [call("list_alerts")])] * 2)
    result = ask([llm], max_rounds=1)

    assert len(result.tool_calls) == 1
    assert "couldn't reach an answer within 1 rounds" in result.answer


def test_no_llm_falls_back_to_passages():
    llm = ScriptedLLM("gemini", [RuntimeError("429 RESOURCE_EXHAUSTED")])
    result = ask([llm], question="SAR deadline")

    assert result.provider == "none"
    assert [s.number for s in result.sources] == [1, 2]
    assert "passages" in result.answer


def test_tools_that_are_not_read_only_are_not_offered():
    server = MCPServer("mixed")

    @server.tool(annotations=ToolAnnotations(read_only_hint=True))
    def lookup(x: str) -> str:
        return x

    @server.tool()
    def send_message(text: str) -> str:
        return "sent"

    async def main():
        async with MCPHub({"mixed": server}) as hub:
            return [t.name for t in hub.specs()], hub.status["mixed"], await hub.call("send_message", {"text": "hi"})

    offered, status, refused = asyncio.run(main())
    assert offered == ["lookup"]
    assert status.blocked_tools == ["send_message"]
    assert not refused.ok
