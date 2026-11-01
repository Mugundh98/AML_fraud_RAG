"""Load the generated CSVs into Neo4j with batched UNWIND queries."""

import csv
import json
from collections.abc import Callable
from pathlib import Path

from neo4j import Driver

from app.graph.schema import CONSTRAINTS

BATCH = 2_000

CUSTOMERS = """
UNWIND $rows AS r
CREATE (c:Customer {customer_id: r.customer_id, name: r.name, customer_type: r.customer_type,
                    occupation: r.occupation, state: r.state, country: r.country, risk_rating: r.risk_rating,
                    is_pep: r.is_pep = 'True', onboarded_on: date(r.onboarded_on),
                    expected_monthly_volume: toInteger(r.expected_monthly_volume)})
MERGE (p:Phone {number: r.phone})
MERGE (d:Device {device_id: r.device_id})
MERGE (ad:Address {text: r.address})
CREATE (c)-[:HAS_PHONE]->(p), (c)-[:USES_DEVICE]->(d), (c)-[:REGISTERED_AT]->(ad)
"""

ACCOUNTS = """
UNWIND $rows AS r
CREATE (a:Account {account_id: r.account_id, account_type: r.account_type,
                   opened_on: CASE r.opened_on WHEN '' THEN null ELSE date(r.opened_on) END,
                   home_branch: r.home_branch, bank: r.bank, country: r.country,
                   is_external: r.is_external = 'True', holder_name: r.holder_name, tags: r.tags})
WITH a, r WHERE r.customer_id <> ''
MATCH (c:Customer {customer_id: r.customer_id})
CREATE (c)-[:OWNS]->(a)
"""

BRANCHES = "UNWIND $rows AS r CREATE (:Branch {branch_id: r.branch_id, city: r.city, state: r.state})"

# One query per endpoint combination keeps every MATCH on an indexed label.
TRANSFER = """
UNWIND $rows AS r
MATCH (src:{src} {{{src_key}: r.from_id}})
MATCH (dst:{dst} {{{dst_key}: r.to_id}})
CREATE (src)-[:TRANSFER {{txn_id: r.txn_id, amount: toFloat(r.amount), timestamp: localdatetime(r.timestamp),
                         channel: r.channel, description: r.description}}]->(dst)
"""
ENDPOINTS = {"account": ("Account", "account_id"), "branch": ("Branch", "branch_id")}

ALERTS = """
UNWIND $rows AS r
MATCH (a:Account {account_id: r.account_id})
CREATE (al:Alert {alert_id: r.alert_id, rule: r.rule, score: toInteger(r.score), priority: r.priority,
                  status: r.status, created_at: localdatetime(r.created_at), summary: r.summary,
                  txn_ids: split(r.txn_ids, ';')})
CREATE (al)-[:ON_ACCOUNT]->(a)
"""


def load_dataset(driver: Driver, data_dir: Path, database: str, log: Callable[[str], None] = print) -> dict:
    def run(query: str, **params):
        driver.execute_query(query, params, database_=database)

    log("Clearing existing data…")
    # Delete in chunks: one huge transaction can exceed AuraDB Free's memory limit.
    while True:
        records, _, _ = driver.execute_query(
            "MATCH (n) WITH n LIMIT 5000 DETACH DELETE n RETURN count(*) AS deleted", database_=database
        )
        if records[0]["deleted"] == 0:
            break

    log("Creating constraints and indexes…")
    for statement in CONSTRAINTS:
        run(statement)

    counts = {}
    for name, query in [("branches", BRANCHES), ("customers", CUSTOMERS), ("accounts", ACCOUNTS)]:
        rows = _read(data_dir / f"{name}.csv")
        _batched(run, query, rows)
        counts[name] = len(rows)
        log(f"Loaded {len(rows)} {name}")

    transfers = _read(data_dir / "transfers.csv")
    for from_kind, to_kind in [("account", "account"), ("branch", "account"), ("account", "branch")]:
        rows = [t for t in transfers if t["from_kind"] == from_kind and t["to_kind"] == to_kind]
        (src, src_key), (dst, dst_key) = ENDPOINTS[from_kind], ENDPOINTS[to_kind]
        _batched(run, TRANSFER.format(src=src, src_key=src_key, dst=dst, dst_key=dst_key), rows,
                 progress=lambda done, total: log(f"  transfers {from_kind}->{to_kind}: {done}/{total}"))
    counts["transfers"] = len(transfers)
    log(f"Loaded {len(transfers)} transfers")

    alerts = _read(data_dir / "alerts.csv")
    _batched(run, ALERTS, alerts)
    counts["alerts"] = len(alerts)
    log(f"Loaded {len(alerts)} alerts")

    meta = json.loads((data_dir / "dataset.json").read_text())
    run("CREATE (:Dataset {as_of: date($as_of), days: $days, seed: $seed, transactions: $transactions})",
        as_of=meta["as_of"], days=meta["days"], seed=meta["seed"], transactions=meta["transactions"])
    return counts


def _batched(run, query: str, rows: list[dict], progress=None) -> None:
    for start in range(0, len(rows), BATCH):
        run(query, rows=rows[start : start + BATCH])
        if progress:
            progress(min(start + BATCH, len(rows)), len(rows))


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))
