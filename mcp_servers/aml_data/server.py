"""aml-data MCP server: read-only access to the AML transaction graph (Neo4j) and the policy documents (Qdrant).

Run over stdio (how MCP clients launch it):   python -m mcp_servers.aml_data
Try it in the MCP Inspector:                  npx @modelcontextprotocol/inspector .venv/Scripts/python -m mcp_servers.aml_data

It exposes all three MCP primitives:
- tools      - functions the model can call (search policies, look up customers, detect patterns)
- resources  - read-only context the client can attach (graph schema, an alert as JSON)
- prompts    - reusable task templates (investigate an alert end to end)
"""

import json
import logging
from collections.abc import Callable
from functools import cache
from typing import Any, Literal, Protocol

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from app.config import get_settings
from app.graph.repository import GraphRepository
from app.graph.schema import SCHEMA_DESCRIPTION

log = logging.getLogger("aml-data-mcp")

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)

INSTRUCTIONS = """Read-only access to a synthetic bank's AML data and its compliance knowledge base.
Use search_policies for rules, thresholds, typologies and internal procedures.
Use the graph tools for facts about customers, accounts, transactions and alerts.
Lookback windows count back from the dataset's as-of date, not today's date.
All data is synthetic."""


class DocSearch(Protocol):
    def search(self, query: str, limit: int) -> list: ...


def create_server(graph: Callable[[], GraphRepository], docs: Callable[[], DocSearch]) -> MCPServer:
    """Build the server. `graph` and `docs` are factories so connections open on first use, not at startup."""
    server = MCPServer("aml-data", instructions=INSTRUCTIONS, version="0.3.0")

    # ---------- knowledge base ----------

    @server.tool(annotations=READ_ONLY)
    def search_policies(query: str, top_k: int = 5) -> dict[str, Any]:
        """Search the AML/fraud knowledge base: regulations (SAR/CTR rules and deadlines), money laundering
        typologies, fraud red flags and Example Bank's internal alert-handling policy.
        Returns numbered passages with their source; cite them when answering."""
        try:
            hits = docs().search(query, _clamp(top_k, 1, 10))
        except RuntimeError as exc:
            raise ToolError(f"Knowledge base unavailable: {exc}") from exc
        return {
            "query": query,
            "passages": [
                {"number": i, "title": h.title, "section": h.section, "source": h.source, "page": h.page,
                 "score": round(h.score, 4), "text": h.text}
                for i, h in enumerate(hits, start=1)
            ],
        }

    # ---------- customers ----------

    @server.tool(annotations=READ_ONLY)
    def find_customers(name: str, limit: int = 10) -> dict[str, Any]:
        """Find customers whose name contains the given text (case-insensitive). Use this to turn a name into a
        customer_id. Customer IDs look like C01129, account IDs like A001520, alert IDs like AL-0002."""
        rows = _graph(graph).find_customers(name, _clamp(limit, 1, 50))
        return {"count": len(rows), "customers": rows}

    @server.tool(annotations=READ_ONLY)
    def get_customer_profile(customer_id: str) -> dict[str, Any]:
        """Customer KYC profile: type, occupation, risk rating, PEP status, accounts with in/out totals,
        alerts, how many other customers share their phone/device/address, and average monthly inflow
        compared with the expected volume declared at onboarding."""
        profile = _graph(graph).customer_profile(customer_id)
        if profile is None:
            raise ToolError(f"No customer with id {customer_id}")
        return profile

    @server.tool(annotations=READ_ONLY)
    def find_shared_identifiers(customer_id: str) -> dict[str, Any]:
        """Other customers who share a phone number, device or address with this customer.
        Several recently onboarded customers sharing identifiers suggests a synthetic identity ring or mule network;
        businesses sharing an address can indicate shell companies."""
        rows = _graph(graph).shared_identifiers(customer_id)
        return {"customer_id": customer_id, "count": len(rows), "linked_customers": rows}

    # ---------- transactions ----------

    @server.tool(annotations=READ_ONLY)
    def get_account_transactions(
        account_id: str,
        days: int = 90,
        direction: Literal["all", "in", "out"] = "all",
        channel: Literal["card", "ach", "p2p", "wire", "crypto", "cash_deposit", "cash_withdrawal"] | None = None,
        min_amount: float = 0,
        limit: int = 50,
    ) -> dict[str, Any]:
        """The account (type, opening date, owner's customer_id and name) and its transactions in the last
        `days` days (newest first), with totals in and out. Filter by direction, channel or minimum amount to
        focus on what matters. Use limit=1 to just find out who owns an account."""
        result = _graph(graph).account_transactions(
            account_id, _clamp(days, 1, 365), direction, channel, max(0.0, min_amount), _clamp(limit, 1, 200)
        )
        if result is None:
            raise ToolError(f"No account with id {account_id}")
        return result

    @server.tool(annotations=READ_ONLY)
    def get_counterparty_summary(account_id: str, days: int = 90) -> dict[str, Any]:
        """Who sends money to and receives money from an account: distinct sender/receiver counts, totals,
        top counterparties, and the pass-through ratio (money out / money in). Many unrelated senders plus a
        ratio near 1.0 suggests a mule or pass-through account."""
        result = _graph(graph).counterparty_summary(account_id, _clamp(days, 1, 365))
        if result is None:
            raise ToolError(f"No account with id {account_id}")
        return result

    # ---------- detection ----------

    @server.tool(annotations=READ_ONLY)
    def check_structuring(account_id: str, days: int = 60, threshold: float = 10_000, margin: float = 0.10) -> dict[str, Any]:
        """Cash deposits into one account that fall just below a reporting threshold (default: $9,000-$9,999.99
        for the $10,000 CTR threshold), with the branches used. Several such deposits across branches is the
        classic structuring pattern."""
        return _graph(graph).near_threshold_cash(account_id, _clamp(days, 1, 365), threshold, _clamp(margin, 0.01, 0.5))

    @server.tool(annotations=READ_ONLY)
    def scan_for_structuring(days: int = 60, min_deposits: int = 3, threshold: float = 10_000,
                             margin: float = 0.10, limit: int = 20) -> dict[str, Any]:
        """Bank-wide scan for accounts with at least `min_deposits` near-threshold cash deposits.
        Shows existing STRUCTURING_CASH alerts per account, so accounts with none are possible missed cases."""
        rows = _graph(graph).scan_near_threshold_cash(
            _clamp(days, 1, 365), threshold, _clamp(margin, 0.01, 0.5), _clamp(min_deposits, 2, 50), _clamp(limit, 1, 100)
        )
        return {"count": len(rows), "accounts": rows}

    @server.tool(annotations=READ_ONLY)
    def find_circular_flows(account_id: str, max_hops: int = 5, min_amount: float = 1_000, max_days: int = 30,
                            amount_tolerance: float = 0.10) -> dict[str, Any]:
        """Money that leaves an account and comes back to it through 1 to (max_hops - 1) other accounts.
        Each transfer must follow the previous one in time and move a similar amount (within
        `amount_tolerance`, default 10%), and the loop must close within `max_days`. Results are grouped by
        route, with each time the loop happened. Indicates round-tripping or layering."""
        routes = _graph(graph).circular_flows(
            account_id, _clamp(max_hops, 2, 6), max(0.0, min_amount), _clamp(max_days, 1, 180),
            _clamp(amount_tolerance, 0.0, 0.5),
        )
        return {"account_id": account_id, "routes_found": len(routes), "routes": routes}

    # ---------- alerts ----------

    @server.tool(annotations=READ_ONLY)
    def list_alerts(
        status: Literal["open", "escalated", "closed"] | None = None,
        priority: Literal["High", "Medium", "Low"] | None = None,
        rule: str | None = None,
        limit: int = 25,
    ) -> dict[str, Any]:
        """Transaction-monitoring alerts, highest priority and score first. Filter by status, priority or rule
        (e.g. STRUCTURING_CASH, FAN_IN_FAN_OUT, CIRCULAR_FLOW, MULTI_BRANCH_CASH, RAPID_MOVEMENT,
        SHARED_IDENTIFIER, PEP_LARGE_WIRE, LARGE_WIRE)."""
        rows = _graph(graph).list_alerts(status, priority, rule, _clamp(limit, 1, 100))
        return {"count": len(rows), "alerts": rows}

    @server.tool(annotations=READ_ONLY)
    def get_alert(alert_id: str) -> dict[str, Any]:
        """One alert with its account, the account owner's profile, and the transactions that triggered it."""
        result = _graph(graph).get_alert(alert_id)
        if result is None:
            raise ToolError(f"No alert with id {alert_id}")
        return result

    # ---------- resources ----------

    @server.resource("aml://graph-schema", name="graph-schema", mime_type="text/markdown",
                     description="Node labels, relationships and properties of the AML graph.")
    def graph_schema() -> str:
        return SCHEMA_DESCRIPTION

    @server.resource("aml://alerts/{alert_id}", name="alert", mime_type="application/json",
                     description="A single alert with its account, customer and flagged transactions.")
    def alert_resource(alert_id: str) -> str:
        return json.dumps(_graph(graph).get_alert(alert_id), indent=2)

    # ---------- prompts ----------

    @server.prompt(title="Investigate an alert")
    def investigate_alert(alert_id: str) -> str:
        """Step-by-step Level 1 triage of an alert, ending in a close/escalate recommendation."""
        return f"""Investigate alert {alert_id} as a Level 1 AML analyst at Example Bank.

1. Call get_alert to see the trigger, the account and the customer.
2. Call search_policies for the alert's typology and for Example Bank's triage and escalation rules.
3. Call get_customer_profile: compare activity with the expected monthly volume, note PEP status and prior alerts.
4. Pick the checks that fit the alert rule:
   - cash rules: check_structuring
   - fan-in/fan-out or rapid movement: get_counterparty_summary
   - circular flow: find_circular_flows
   - shared identifier: find_shared_identifiers
   Use get_account_transactions for detail when needed.
5. Conclude with:
   - Findings (facts with txn_ids, amounts and dates)
   - Matching typology and red flags, citing the policy passages
   - Priority under Example Bank policy
   - Recommendation: close (with a specific rationale) or escalate to Level 2
   - If escalating: a draft SAR narrative covering who, what, when, where, why and how

Base every statement on tool results. If information is missing, say what is missing."""

    return server


def _graph(factory: Callable[[], GraphRepository]) -> GraphRepository:
    try:
        return factory()
    except Exception as exc:
        raise ToolError(f"Graph database unavailable: {exc}") from exc


def _clamp(value, low, high):
    return max(low, min(high, value))


def main() -> None:
    settings = get_settings()

    @cache
    def graph() -> GraphRepository:
        from app.graph.db import connect

        return GraphRepository(connect(settings), settings.neo4j_database)

    @cache
    def docs() -> DocSearch:
        from app.rag.store import DocStore

        return DocStore(settings)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    create_server(graph, docs).run("stdio")


if __name__ == "__main__":
    main()
