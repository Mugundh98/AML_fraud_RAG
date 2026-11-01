"""Phase 5 hardening: users and roles, the tamper-evident audit log, and PII pseudonymization."""

import asyncio
import json
import sqlite3
import uuid

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from app import observability
from app.agent.approvals import ApprovalQueue
from app.agent.loop import Agent
from app.agent.mcp_hub import MCPHub, ServerConfig
from app.agent.pii import Pseudonymizer, mask_known
from app.audit import AuditLog
from app.auth import LOCAL_USER, User, UserDirectory
from app.config import Settings
from app.llm import AssistantTurn, LLMRouter, Message
from mcp_servers.aml_data.server import create_server
from tests.test_agent import ScriptedLLM, TwoDocs, call, mixed_server
from tests.test_mcp_server import FakeGraph

ANALYST = User("u001", "Ana Lyst", "analyst")
SENIOR = User("u002", "Sen Ior", "senior_analyst")
LEAD = User("u003", "Lee Dur", "lead")


# ---------- users and roles ----------

def test_open_mode_without_users_file(tmp_path):
    directory = UserDirectory(tmp_path / "users.json")
    assert directory.open_mode and directory.authenticate(None) == LOCAL_USER


def test_api_keys_are_checked_and_only_hashes_stored(tmp_path):
    path = tmp_path / "users.json"
    user, key = UserDirectory(path).add("Priya Shah", "senior_analyst")

    directory = UserDirectory(path)  # reloaded from disk
    assert directory.authenticate(key) == user
    assert directory.authenticate("aml_wrong") is None and directory.authenticate(None) is None
    assert key not in path.read_text() and "key_sha256" in path.read_text()


def test_escalation_needs_a_senior_analyst():
    assert ANALYST.can_use("search_policies", needs_approval=False)
    assert ANALYST.can_use("create_event", needs_approval=True)
    assert not ANALYST.can_use("post_message", needs_approval=True)
    assert SENIOR.can_use("post_message", needs_approval=True)
    assert not SENIOR.can_use("unknown_write_tool", needs_approval=True) and LEAD.can_use("unknown_write_tool", True)
    assert LEAD.can_read_audit and not SENIOR.can_read_audit


def test_tools_a_role_cannot_approve_are_not_offered_to_the_model():
    def offered(user):
        llm = ScriptedLLM("gemini", [AssistantTurn("ok")])

        async def main():
            async with MCPHub({"mixed": mixed_server([])}) as hub:
                await Agent(hub, LLMRouter([llm])).answer("q", [], user=user)

        asyncio.run(main())
        return llm.offered[0]

    assert offered(ANALYST) == ["lookup"]  # send_message isn't a listed write tool, so it needs a lead
    assert offered(LEAD) == ["lookup", "send_message"]


def test_approval_needs_the_right_role_and_waits_for_it(tmp_path):
    sent = []
    audit = AuditLog(tmp_path / "audit.db")

    async def main():
        async with MCPHub({"mixed": mixed_server(sent)}) as hub:
            agent = Agent(hub, LLMRouter([ScriptedLLM("gemini", [
                AssistantTurn("", [call("send_message", text="escalate AL-0002")]),
                AssistantTurn("Waiting for approval."),
            ])]), ApprovalQueue(ttl_seconds=600), audit=audit)
            answer = await agent.answer("escalate it", [], user=LEAD)
            action_id = answer.pending_actions[0].id
            return (await agent.execute(action_id, True, user=ANALYST),
                    await agent.execute(action_id, True, user=LEAD))

    denied, done = asyncio.run(main())
    assert denied.status == "denied" and done.status == "done" and sent == ["escalate AL-0002"]
    events = [e["event"] for e in reversed(audit.recent())]
    assert events == ["chat.answer", "action.proposed", "action.denied", "action.decided"]


# ---------- audit log ----------

def test_audit_chain_detects_edits_and_deletions(tmp_path):
    path = tmp_path / "audit.db"
    log = AuditLog(path)
    for i in range(4):
        log.record(actor_id="u001", actor_role="analyst", event="chat.answer", details={"n": i})
    assert log.verify() == {"ok": True, "entries": 4, "latest_hash": log.recent(1)[0]["hash"]}

    with sqlite3.connect(path) as db:
        db.execute("DELETE FROM audit_events WHERE id = 2")
    assert log.verify()["broken_at"] == 3  # entry 3 no longer links to its predecessor

    path2 = tmp_path / "audit2.db"
    log2 = AuditLog(path2)
    for i in range(3):
        log2.record(actor_id="u001", actor_role="analyst", event="chat.answer", details={"n": i})
    with sqlite3.connect(path2) as db:
        db.execute("UPDATE audit_events SET actor_id = 'someone-else' WHERE id = 1")
    assert log2.verify() == {"ok": False, "entries": 0, "broken_at": 1, "reason": "content changed"}


# ---------- PII ----------

def test_pseudonymizer_round_trip():
    vault = Pseudonymizer()
    data = {"customer_id": "C00001", "name": "Jessica Martin",
            "shared": [{"type": "HAS_PHONE", "value": "+1-892-555-0194"}],
            "transactions": [{"counterparty_id": "BR-09", "counterparty_name": "Miami branch"},
                             {"counterparty_id": "X001496", "counterparty_name": "Keystone Capital Ltd"}],
            "note": "email jm@example.com"}
    redacted = vault.redact(data)

    assert redacted["name"] == "<NAME_1>" and redacted["customer_id"] == "C00001"
    assert redacted["shared"][0]["value"] == "<PHONE_1>"
    assert redacted["transactions"][0]["counterparty_name"] == "Miami branch"  # places aren't people
    assert redacted["transactions"][1]["counterparty_name"] == "<NAME_2>"
    assert redacted["note"] == "email <EMAIL_1>"
    assert vault.restore(redacted) == data
    assert vault.redact_text("Is Jessica Martin a PEP?") == "Is <NAME_1> a PEP?"  # known values in typed text
    assert mask_known("payment to Jessica Martin") == "payment to [PII]"


def test_names_in_free_text_are_caught_even_before_their_field():
    vault = Pseudonymizer()
    row = {"description": "P2P from Rahul Verma", "counterparty_id": "A000777", "counterparty_name": "Rahul Verma"}
    assert vault.redact(row) == {"description": "P2P from <NAME_1>", "counterparty_id": "A000777",
                                 "counterparty_name": "<NAME_1>"}


def pii_hub(web_calls: list):
    web = MCPServer("web")

    @web.tool(annotations=ToolAnnotations(read_only_hint=True))
    def lookup(query: str) -> str:
        web_calls.append(query)
        return "results"

    graph = FakeGraph()
    servers = {
        "aml-data": ServerConfig(create_server(lambda: graph, lambda: TwoDocs()), returns_pii=True, receives_pii=True),
        "web": ServerConfig(web, auto_run={"lookup"}),  # third-party: receives_pii=False by default
    }
    return MCPHub(servers), graph


def test_model_sees_placeholders_tools_and_analyst_see_real_values():
    web_calls = []
    llm = ScriptedLLM("gemini", [
        AssistantTurn("", [call("find_customers", name="test")]),
        AssistantTurn("", [call("find_customers", name="<NAME_1>"), call("lookup", query="<NAME_1> fraud news")]),
        AssistantTurn("<NAME_1> is customer C00001."),
    ])

    async def main():
        hub, graph = pii_hub(web_calls)
        async with hub:
            answer = await Agent(hub, LLMRouter([llm])).answer("Who is test?", [], session_id="s1")
        return answer, graph

    answer, graph = asyncio.run(main())
    first_result = llm.seen[1][-1].content
    assert "Test Person" not in first_result and '"name": "<NAME_1>"' in first_result
    assert ("find_customers", "Test Person", 10) in graph.calls  # the placeholder was restored for our server
    assert web_calls == []  # ...but never sent to the third-party one
    assert "external service" in llm.seen[2][-1].content
    assert answer.answer == "Test Person is customer C00001."
    assert answer.tool_calls[1].arguments == {"name": "Test Person"}  # the analyst's view


def test_history_is_pseudonymized_consistently_across_turns():
    llm = ScriptedLLM("gemini", [AssistantTurn("", [call("find_customers", name="test")]), AssistantTurn("done"),
                                 AssistantTurn("ok")])

    async def main():
        hub, _ = pii_hub([])
        async with hub:
            agent = Agent(hub, LLMRouter([llm]))
            await agent.answer("Who is test?", [], session_id="s2")
            history = [Message("user", "Who is test?"), Message("assistant", "It's Test Person.")]
            await agent.answer("Tell me more about Test Person", history, session_id="s2")

    asyncio.run(main())
    second_turn = llm.seen[2]
    assert [m.content for m in second_turn] == ["Who is test?", "It's <NAME_1>.", "Tell me more about <NAME_1>"]


def test_traces_never_contain_customer_names():
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    client = observability.init_tracing(
        Settings(_env_file=None, langfuse_public_key=f"pk-lf-{uuid.uuid4()}", langfuse_secret_key="sk-lf-test"),
        span_exporter=exporter)
    llm = ScriptedLLM("gemini", [AssistantTurn("", [call("find_customers", name="test")]),
                                 AssistantTurn("<NAME_1> is customer C00001.")])
    try:
        async def main():
            hub, _ = pii_hub([])
            async with hub:
                return await Agent(hub, LLMRouter([llm])).answer("Who is test?", [], session_id="s3")

        answer = asyncio.run(main())
        client.flush()
        exported = json.dumps([dict(s.attributes) for s in exporter.get_finished_spans()])
    finally:
        client.shutdown()
        observability._client = None

    assert answer.answer == "Test Person is customer C00001."
    assert "Test Person" not in exported and "<NAME_1>" in exported
