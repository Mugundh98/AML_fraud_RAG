"""Talks to the MCP server through a real MCP client, in-process, with fakes instead of Neo4j and Qdrant."""

import asyncio
from types import SimpleNamespace

from mcp import Client

from mcp_servers.aml_data.server import create_server

ALERT = {
    "alert": {"alert_id": "AL-0001", "rule": "STRUCTURING_CASH", "txn_ids": ["T0000001"]},
    "account": {"account_id": "A000001"},
    "customer": {"customer_id": "C00001", "name": "Test Person"},
    "flagged_transactions": [{"txn_id": "T0000001", "amount": 9500.0}],
}


class FakeGraph:
    def __init__(self):
        self.calls = []

    def find_customers(self, name, limit):
        self.calls.append(("find_customers", name, limit))
        return [{"customer_id": "C00001", "name": "Test Person", "account_ids": ["A000001"]}]

    def list_alerts(self, status, priority, rule, limit):
        self.calls.append(("list_alerts", status, priority, rule, limit))
        return []

    def get_alert(self, alert_id):
        return ALERT if alert_id == "AL-0001" else None


class FakeDocs:
    def search(self, query, limit):
        hit = SimpleNamespace(title="US BSA/AML Reporting Requirements", section="Filing deadlines",
                              source="seed/us.md", page=None, score=0.9, text="File within 30 days.")
        return [hit][:limit]


def run(coro_fn, graph=None):
    graph = graph or FakeGraph()

    async def main():
        async with Client(create_server(lambda: graph, lambda: FakeDocs())) as client:
            return await coro_fn(client)

    return asyncio.run(main())


def test_tools_are_listed_and_read_only():
    tools = run(lambda c: c.list_tools()).tools
    assert {t.name for t in tools} == {
        "search_policies", "find_customers", "get_customer_profile", "find_shared_identifiers",
        "get_account_transactions", "get_counterparty_summary", "check_structuring", "scan_for_structuring",
        "find_circular_flows", "list_alerts", "get_alert",
    }
    assert all(t.annotations.read_only_hint for t in tools)


def test_tool_returns_structured_result():
    result = run(lambda c: c.call_tool("find_customers", {"name": "test"}))
    assert not result.is_error
    assert result.structured_content["customers"][0]["customer_id"] == "C00001"


def test_limits_are_clamped():
    graph = FakeGraph()
    run(lambda c: c.call_tool("list_alerts", {"limit": 10_000, "priority": "High"}), graph)
    assert graph.calls == [("list_alerts", None, "High", None, 100)]


def test_invalid_enum_is_rejected():
    result = run(lambda c: c.call_tool("list_alerts", {"priority": "urgent"}))
    assert result.is_error


def test_unknown_alert_is_a_tool_error():
    result = run(lambda c: c.call_tool("get_alert", {"alert_id": "AL-9999"}))
    assert result.is_error
    assert "No alert with id AL-9999" in result.content[0].text


def test_graph_outage_is_reported_not_raised():
    def broken():
        raise RuntimeError("NEO4J_URI is not set")

    async def main():
        async with Client(create_server(broken, lambda: FakeDocs())) as client:
            return await client.call_tool("list_alerts", {})

    result = asyncio.run(main())
    assert result.is_error
    assert "Graph database unavailable: NEO4J_URI is not set" in result.content[0].text


def test_search_policies_numbers_passages():
    result = run(lambda c: c.call_tool("search_policies", {"query": "SAR deadline"}))
    passage = result.structured_content["passages"][0]
    assert passage["number"] == 1 and passage["section"] == "Filing deadlines"


def test_schema_resource_and_alert_template():
    async def read(client):
        schema = await client.read_resource("aml://graph-schema")
        alert = await client.read_resource("aml://alerts/AL-0001")
        return schema, alert

    schema, alert = run(read)
    assert "TRANSFER" in schema.contents[0].text
    assert '"alert_id": "AL-0001"' in alert.contents[0].text


def test_investigate_prompt_mentions_alert():
    result = run(lambda c: c.get_prompt("investigate_alert", {"alert_id": "AL-0042"}))
    assert "AL-0042" in result.messages[0].content.text
