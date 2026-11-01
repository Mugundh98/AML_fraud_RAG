"""Read-only queries over the AML graph. Each method backs one MCP tool."""

from datetime import date, datetime, time, timedelta
from typing import Any

from neo4j import Driver, RoutingControl

IDENTIFIER_RELS = "HAS_PHONE|USES_DEVICE|REGISTERED_AT"
# Shows who/what is on the other side of a transfer, whether it's an account or a branch.
COUNTERPARTY = """coalesce(other.account_id, other.branch_id) AS counterparty_id,
       coalesce(other.holder_name, other.city + ' branch') AS counterparty_name,
       other.country AS counterparty_country, other.tags AS counterparty_tags"""


class GraphRepository:
    def __init__(self, driver: Driver, database: str):
        self.driver = driver
        self.database = database
        self._as_of: date | None = None
        self._days: int | None = None

    # ---------- dataset ----------

    def as_of(self) -> date:
        """The data's 'today'. Synthetic data doesn't move with the clock, so lookbacks count back from here."""
        if self._as_of is None:
            rows = self._read("MATCH (d:Dataset) RETURN d.as_of AS as_of, d.days AS days")
            if not rows:
                raise RuntimeError("No dataset loaded. Run: python -m scripts.load_graph")
            self._as_of, self._days = date.fromisoformat(rows[0]["as_of"]), rows[0]["days"]
        return self._as_of

    def window_start(self, days: int) -> datetime:
        return datetime.combine(self.as_of() - timedelta(days=days - 1), time(0))

    # ---------- customers ----------

    def find_customers(self, name: str, limit: int) -> list[dict]:
        return self._read(
            """
            MATCH (c:Customer) WHERE toLower(c.name) CONTAINS toLower($name)
            OPTIONAL MATCH (c)-[:OWNS]->(a:Account)
            RETURN c.customer_id AS customer_id, c.name AS name, c.customer_type AS customer_type,
                   c.risk_rating AS risk_rating, c.is_pep AS is_pep, collect(a.account_id) AS account_ids
            ORDER BY name LIMIT $limit
            """,
            name=name, limit=limit,
        )

    def customer_profile(self, customer_id: str) -> dict | None:
        rows = self._read("MATCH (c:Customer {customer_id: $id}) RETURN c {.*} AS customer", id=customer_id)
        if not rows:
            return None
        customer = rows[0]["customer"]
        self.as_of()
        months = self._days / 30
        accounts = self._read(
            """
            MATCH (:Customer {customer_id: $id})-[:OWNS]->(a:Account)
            OPTIONAL MATCH (a)<-[tin:TRANSFER]-()
            WITH a, count(tin) AS in_count, coalesce(sum(tin.amount), 0) AS in_total
            OPTIONAL MATCH (a)-[tout:TRANSFER]->()
            RETURN a.account_id AS account_id, a.account_type AS account_type, a.opened_on AS opened_on,
                   a.home_branch AS home_branch, in_count, round(in_total, 2) AS in_total,
                   count(tout) AS out_count, round(coalesce(sum(tout.amount), 0), 2) AS out_total
            ORDER BY account_id
            """,
            id=customer_id,
        )
        alerts = self._read(
            """
            MATCH (:Customer {customer_id: $id})-[:OWNS]->(a:Account)<-[:ON_ACCOUNT]-(al:Alert)
            RETURN al.alert_id AS alert_id, al.rule AS rule, al.priority AS priority, al.status AS status,
                   al.created_at AS created_at, a.account_id AS account_id
            ORDER BY created_at DESC
            """,
            id=customer_id,
        )
        shared = self._read(
            f"""
            MATCH (c:Customer {{customer_id: $id}})-[:{IDENTIFIER_RELS}]->()<-[:{IDENTIFIER_RELS}]-(o:Customer)
            WHERE o <> c RETURN count(DISTINCT o) AS n
            """,
            id=customer_id,
        )[0]["n"]
        total_in = sum(a["in_total"] for a in accounts)
        return {
            "customer": customer,
            "accounts": accounts,
            "alerts": alerts,
            "customers_sharing_identifiers": shared,
            "activity_vs_profile": {
                "data_window_days": self._days,
                "avg_monthly_inflow": round(total_in / months, 2),
                "expected_monthly_volume": customer.get("expected_monthly_volume"),
            },
        }

    def shared_identifiers(self, customer_id: str) -> list[dict]:
        return self._read(
            f"""
            MATCH (c:Customer {{customer_id: $id}})-[rel:{IDENTIFIER_RELS}]->(i)<-[:{IDENTIFIER_RELS}]-(o:Customer)
            WHERE o <> c
            OPTIONAL MATCH (o)-[:OWNS]->(a:Account)
            RETURN o.customer_id AS customer_id, o.name AS name, o.customer_type AS customer_type,
                   o.onboarded_on AS onboarded_on, o.risk_rating AS risk_rating,
                   collect(DISTINCT {{type: type(rel), value: coalesce(i.number, i.device_id, i.text)}}) AS shared,
                   collect(DISTINCT a.account_id) AS account_ids
            ORDER BY size(shared) DESC, customer_id
            """,
            id=customer_id,
        )

    # ---------- transactions ----------

    def account_transactions(self, account_id: str, days: int, direction: str, channel: str | None,
                             min_amount: float, limit: int) -> dict | None:
        if not self._read("MATCH (a:Account {account_id: $id}) RETURN 1 AS ok", id=account_id):
            return None
        params = dict(id=account_id, start=self.window_start(days), direction=direction, channel=channel,
                      min_amount=min_amount, limit=limit)
        where = """
            WHERE t.timestamp >= $start AND t.amount >= $min_amount
              AND ($channel IS NULL OR t.channel = $channel)
              AND ($direction = 'all' OR ($direction = 'in') = (endNode(t) = a))
        """
        summary = self._read(
            f"""
            MATCH (a:Account {{account_id: $id}})-[t:TRANSFER]-(other) {where}
            WITH a, t, CASE WHEN startNode(t) = a THEN 'out' ELSE 'in' END AS dir
            RETURN count(t) AS count,
                   round(sum(CASE dir WHEN 'in' THEN t.amount ELSE 0 END), 2) AS total_in,
                   round(sum(CASE dir WHEN 'out' THEN t.amount ELSE 0 END), 2) AS total_out
            """,
            **params,
        )[0]
        transactions = self._read(
            f"""
            MATCH (a:Account {{account_id: $id}})-[t:TRANSFER]-(other) {where}
            RETURN t.txn_id AS txn_id, t.timestamp AS timestamp, t.amount AS amount, t.channel AS channel,
                   CASE WHEN startNode(t) = a THEN 'out' ELSE 'in' END AS direction,
                   t.description AS description, {COUNTERPARTY}
            ORDER BY t.timestamp DESC LIMIT $limit
            """,
            **params,
        )
        return {
            "account_id": account_id,
            "period": {"from": self.window_start(days).date().isoformat(), "to": self.as_of().isoformat()},
            "summary": summary,
            "transactions": transactions,
            "truncated": summary["count"] > len(transactions),
        }

    def counterparty_summary(self, account_id: str, days: int) -> dict | None:
        start = self.window_start(days)
        totals = self._read(
            """
            MATCH (a:Account {account_id: $id})
            OPTIONAL MATCH (s)-[tin:TRANSFER]->(a) WHERE tin.timestamp >= $start
            WITH a, count(tin) AS in_count, count(DISTINCT s) AS distinct_senders, coalesce(sum(tin.amount), 0) AS in_total
            OPTIONAL MATCH (a)-[tout:TRANSFER]->(r) WHERE tout.timestamp >= $start
            RETURN in_count, distinct_senders, round(in_total, 2) AS in_total, count(tout) AS out_count,
                   count(DISTINCT r) AS distinct_receivers, round(coalesce(sum(tout.amount), 0), 2) AS out_total
            """,
            id=account_id, start=start,
        )
        if not totals:
            return None
        top = """
            MATCH (a:Account {{account_id: $id}}){arrow}(other) WHERE t.timestamp >= $start
            RETURN {COUNTERPARTY}, count(t) AS transfers, round(sum(t.amount), 2) AS total,
                   collect(DISTINCT t.channel) AS channels
            ORDER BY total DESC LIMIT 10
        """
        summary = totals[0]
        summary["pass_through_ratio"] = (
            round(summary["out_total"] / summary["in_total"], 3) if summary["in_total"] else None
        )
        return {
            "account_id": account_id,
            "period_days": days,
            "summary": summary,
            "top_senders": self._read(top.format(arrow="<-[t:TRANSFER]-", COUNTERPARTY=COUNTERPARTY), id=account_id, start=start),
            "top_receivers": self._read(top.format(arrow="-[t:TRANSFER]->", COUNTERPARTY=COUNTERPARTY), id=account_id, start=start),
        }

    # ---------- detection ----------

    def near_threshold_cash(self, account_id: str, days: int, threshold: float, margin: float) -> dict:
        deposits = self._read(
            """
            MATCH (b:Branch)-[t:TRANSFER {channel: 'cash_deposit'}]->(:Account {account_id: $id})
            WHERE t.timestamp >= $start AND t.amount >= $low AND t.amount < $threshold
            RETURN t.txn_id AS txn_id, t.timestamp AS timestamp, t.amount AS amount,
                   b.branch_id AS branch_id, b.city + ', ' + b.state AS branch_location
            ORDER BY t.timestamp
            """,
            id=account_id, start=self.window_start(days), low=threshold * (1 - margin), threshold=threshold,
        )
        return {
            "account_id": account_id,
            "period_days": days,
            "band": {"from": threshold * (1 - margin), "below": threshold},
            "deposit_count": len(deposits),
            "distinct_branches": len({d["branch_id"] for d in deposits}),
            "total": round(sum(d["amount"] for d in deposits), 2),
            "deposits": deposits,
        }

    def scan_near_threshold_cash(self, days: int, threshold: float, margin: float, min_deposits: int, limit: int) -> list[dict]:
        return self._read(
            """
            MATCH (b:Branch)-[t:TRANSFER {channel: 'cash_deposit'}]->(a:Account)
            WHERE t.timestamp >= $start AND t.amount >= $low AND t.amount < $threshold
            WITH a, count(t) AS deposits, count(DISTINCT b) AS branches, round(sum(t.amount), 2) AS total
            WHERE deposits >= $min_deposits
            OPTIONAL MATCH (c:Customer)-[:OWNS]->(a)
            OPTIONAL MATCH (al:Alert {rule: 'STRUCTURING_CASH'})-[:ON_ACCOUNT]->(a)
            RETURN a.account_id AS account_id, c.customer_id AS customer_id, c.name AS customer_name,
                   c.customer_type AS customer_type, c.occupation AS occupation,
                   deposits, branches, total, collect(al.alert_id) AS existing_alerts
            ORDER BY deposits DESC, total DESC LIMIT $limit
            """,
            start=self.window_start(days), low=threshold * (1 - margin), threshold=threshold,
            min_deposits=min_deposits, limit=limit,
        )

    def circular_flows(self, account_id: str, max_hops: int, min_amount: float, max_days: int,
                       amount_tolerance: float) -> list[dict]:
        """Loops of transfers back to the account, grouped by route. Each hop must happen after the previous
        one and move roughly the same amount (within `amount_tolerance`), i.e. the same money going round."""
        # Quantifier bounds can't be query parameters, so max_hops is validated before formatting.
        hops = int(max_hops)
        rows = self._read(
            f"""
            MATCH p = (a:Account {{account_id: $id}})
                      ((:Account)-[r:TRANSFER WHERE r.amount >= $min_amount]->(:Account)){{2,{hops}}}
                      (a)
            WHERE NOT a IN nodes(p)[1..-1]
              AND all(i IN range(0, size(r) - 2) WHERE r[i].timestamp <= r[i + 1].timestamp
                      AND abs(r[i + 1].amount - r[i].amount) <= $tolerance * r[i].amount)
              AND r[-1].timestamp <= r[0].timestamp + duration({{days: $max_days}})
            RETURN [n IN nodes(p) | n.account_id] AS path,
                   [n IN nodes(p)[1..-1] | n.holder_name] AS intermediaries,
                   [x IN r | {{txn_id: x.txn_id, amount: x.amount, timestamp: toString(x.timestamp)}}] AS transfers,
                   round(100.0 * (r[0].amount - r[-1].amount) / r[0].amount, 2) AS value_lost_pct
            ORDER BY r[0].timestamp
            LIMIT 50
            """,
            id=account_id, min_amount=min_amount, max_days=max_days, tolerance=amount_tolerance,
        )
        routes: dict[tuple, dict] = {}
        for row in rows:
            route = routes.setdefault(tuple(row["path"]), {
                "path": row["path"], "intermediaries": row["intermediaries"], "occurrences": [],
            })
            route["occurrences"].append({"transfers": row["transfers"], "value_lost_pct": row["value_lost_pct"]})
        return list(routes.values())[:10]

    # ---------- alerts ----------

    def list_alerts(self, status: str | None, priority: str | None, rule: str | None, limit: int) -> list[dict]:
        return self._read(
            """
            MATCH (al:Alert)-[:ON_ACCOUNT]->(a:Account)
            WHERE ($status IS NULL OR al.status = $status)
              AND ($priority IS NULL OR al.priority = $priority)
              AND ($rule IS NULL OR al.rule = $rule)
            OPTIONAL MATCH (c:Customer)-[:OWNS]->(a)
            RETURN al.alert_id AS alert_id, al.rule AS rule, al.score AS score, al.priority AS priority,
                   al.status AS status, al.created_at AS created_at, al.summary AS summary,
                   a.account_id AS account_id, c.customer_id AS customer_id, c.name AS customer_name
            ORDER BY CASE al.priority WHEN 'High' THEN 0 WHEN 'Medium' THEN 1 ELSE 2 END, al.score DESC
            LIMIT $limit
            """,
            status=status, priority=priority, rule=rule, limit=limit,
        )

    def get_alert(self, alert_id: str) -> dict | None:
        rows = self._read(
            """
            MATCH (al:Alert {alert_id: $id})-[:ON_ACCOUNT]->(a:Account)
            OPTIONAL MATCH (c:Customer)-[:OWNS]->(a)
            RETURN al {.*} AS alert, a {.*} AS account,
                   c {.customer_id, .name, .customer_type, .occupation, .risk_rating, .is_pep, .onboarded_on,
                      .expected_monthly_volume} AS customer
            """,
            id=alert_id,
        )
        if not rows:
            return None
        result = rows[0]
        result["flagged_transactions"] = self._read(
            f"""
            MATCH (src)-[t:TRANSFER]->(dst) WHERE t.txn_id IN $ids
            WITH t, src, dst, CASE WHEN dst.account_id = $account THEN src ELSE dst END AS other
            RETURN t.txn_id AS txn_id, t.timestamp AS timestamp, t.amount AS amount, t.channel AS channel,
                   CASE WHEN dst.account_id = $account THEN 'in' ELSE 'out' END AS direction,
                   t.description AS description, {COUNTERPARTY}
            ORDER BY t.timestamp
            """,
            ids=result["alert"]["txn_ids"], account=result["account"]["account_id"],
        )
        return result

    # ---------- plumbing ----------

    def _read(self, query: str, **params) -> list[dict]:
        records, _, _ = self.driver.execute_query(
            query, params, database_=self.database, routing_=RoutingControl.READ
        )
        return [_plain(record.data()) for record in records]


def _plain(value: Any) -> Any:
    """Convert Neo4j temporal types to ISO strings so results are JSON-serialisable."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if hasattr(value, "iso_format"):
        return value.iso_format()
    return value
