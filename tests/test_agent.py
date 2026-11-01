"""The agent loop end to end: a scripted fake LLM, our real MCP server (in-process, fake databases), real MCP client."""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from app.agent.approvals import ApprovalQueue
from app.agent.loop import Agent
from app.agent.mcp_hub import MCPHub, ServerConfig
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
        self.offered: list[list[str]] = []

    def is_configured(self):
        return True

    def chat(self, system, messages, tools, force_answer=False):
        self.seen.append(list(messages))
        self.forced.append(force_answer)
        self.offered.append(sorted(t.name for t in tools))
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


def mixed_server(sent: list):
    """A server with one read-only tool and one that changes something."""
    server = MCPServer("mixed")

    @server.tool(annotations=ToolAnnotations(read_only_hint=True))
    def lookup(x: str) -> str:
        return x

    @server.tool()
    def send_message(text: str) -> str:
        sent.append(text)
        return "sent"

    return server


def test_tools_without_read_only_hint_need_approval():
    async def main():
        async with MCPHub({"mixed": mixed_server([])}) as hub:
            return hub, await hub.call("send_message", {"text": "hi"})

    hub, refused = asyncio.run(main())
    assert hub.status["mixed"].tools == ["lookup"]
    assert hub.status["mixed"].approval_tools == ["send_message"]
    assert not refused.ok and "approval" in refused.text


def test_third_party_policy_overrides_annotations_and_filters_tools():
    async def main():
        config = ServerConfig(mixed_server([]), tools={"send_message"}, auto_run=set())
        async with MCPHub({"web": config}) as hub:
            return [t.name for t in hub.specs()], hub.needs_approval("send_message")

    offered, needs_approval = asyncio.run(main())
    assert offered == ["send_message"]  # lookup filtered out
    assert needs_approval


def run_with_actions(script, decide):
    """Ask once with the mixed server, then call decide(agent, answer) while the hub is still open."""
    sent = []

    async def main():
        async with MCPHub({"mixed": mixed_server(sent)}) as hub:
            agent = Agent(hub, LLMRouter([ScriptedLLM("gemini", script)]), ApprovalQueue(ttl_seconds=600))
            answer = await agent.answer("escalate it", [])
            return answer, await decide(agent, answer)

    answer, decided = asyncio.run(main())
    return answer, decided, sent


PROPOSE = [
    AssistantTurn("", [call("send_message", text="AL-0002 needs Level 2 review")]),
    AssistantTurn("I've drafted the Slack message; it's waiting for your approval."),
]


def test_write_tool_becomes_a_pending_action_and_does_not_run():
    async def nothing(agent, answer):
        return None

    answer, _, sent = run_with_actions(PROPOSE, nothing)

    assert sent == []
    assert [(a.tool, a.arguments) for a in answer.pending_actions] == [
        ("send_message", {"text": "AL-0002 needs Level 2 review"})
    ]
    assert answer.tool_calls[0].needs_approval


def test_approval_runs_the_stored_arguments_exactly_once():
    async def approve_twice(agent, answer):
        action_id = answer.pending_actions[0].id
        return await agent.execute(action_id, True), await agent.execute(action_id, True)

    _, (first, second), sent = run_with_actions(PROPOSE, approve_twice)

    assert sent == ["AL-0002 needs Level 2 review"]
    assert first.status == "done" and second.status == "expired"


def test_rejection_runs_nothing():
    async def reject(agent, answer):
        return await agent.execute(answer.pending_actions[0].id, False)

    _, outcome, sent = run_with_actions(PROPOSE, reject)
    assert outcome.status == "rejected" and sent == []


def test_repeated_proposal_is_one_action():
    script = [
        AssistantTurn("", [call("send_message", text="same"), call("send_message", text="same")]),
        AssistantTurn("Waiting for approval."),
    ]

    async def nothing(agent, answer):
        return None

    answer, _, _ = run_with_actions(script, nothing)
    assert len(answer.pending_actions) == 1


def test_actions_expire():
    clock = [1000.0]
    queue = ApprovalQueue(ttl_seconds=60, clock=lambda: clock[0])
    action = queue.add("send_message", "slack", {"text": "hi"})
    clock[0] += 61
    assert queue.take(action.id) is None


def test_system_prompt_states_today_and_timezone():
    now = datetime(2026, 11, 1, 9, 30, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    agent = Agent(hub=None, router=None, now=lambda: now)
    assert "Today is Sunday 01 November 2026. The analyst's timezone is UTC+05:30." in agent.system_prompt()


def test_schemas_are_compacted_without_losing_constraints():
    from app.agent.mcp_hub import compact_schema

    schema = {
        "type": "object", "title": "createArguments",
        "properties": {
            "title": {"title": "Title", "type": "string"},
            "channel": {"anyOf": [{"enum": ["a", "b"], "type": "string"}, {"type": "null"}], "default": None,
                        "title": "Channel", "description": "Which channel"},
            "days": {"default": 90, "title": "Days", "type": "integer"},
            "noise": {"type": "boolean"},
        },
        "required": ["title", "noise"],
    }
    assert compact_schema(schema, keep={"title", "channel", "days"}) == {
        "type": "object",
        "properties": {
            "title": {"type": "string"},  # a parameter called "title" survives
            "channel": {"enum": ["a", "b"], "type": "string", "description": "Which channel"},
            "days": {"default": 90, "type": "integer"},
        },
        "required": ["title"],
    }
