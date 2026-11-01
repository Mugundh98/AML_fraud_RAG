"""Check that the graph queries behind the MCP tools surface every planted scenario in scenarios.csv.

Run from the project root (after load_graph):  python -m scripts.check_detection
"""

import csv

from app.config import get_settings
from app.graph.db import connect
from app.graph.generator import BORDER_BRANCHES
from app.graph.repository import GraphRepository


def check(repo: GraphRepository, typology: str, customers: list[str], accounts: list[str]) -> tuple[bool, str]:
    account = accounts[0]
    if typology == "structuring":
        flagged = {r["account_id"]: r for r in repo.scan_near_threshold_cash(180, 10_000, 0.10, 3, 100)}
        hit = flagged.get(account)
        return bool(hit), f"{hit['deposits']} deposits at {hit['branches']} branches" if hit else "not in scan"
    if typology == "money_mule":
        s = repo.counterparty_summary(account, 90)["summary"]
        return s["distinct_senders"] >= 8 and s["pass_through_ratio"] >= 0.8, (
            f"{s['distinct_senders']} senders, pass-through {s['pass_through_ratio']}")
    if typology == "circular_flow":
        routes = repo.circular_flows(account, 6, 1_000, 30, 0.10)
        full = [r for r in routes if set(r["path"]) == set(accounts)]
        return bool(full), f"{len(routes)} route(s), {sum(len(r['occurrences']) for r in full)} full loop(s)"
    if typology == "funnel_account":
        tx = repo.account_transactions(account, 180, "all", None, 0, 200)["transactions"]
        deposit_branches = {t["counterparty_id"] for t in tx if t["channel"] == "cash_deposit"}
        border_out = [t for t in tx if t["channel"] == "cash_withdrawal" and t["counterparty_id"] in BORDER_BRANCHES]
        return len(deposit_branches) >= 5 and bool(border_out), (
            f"deposits at {len(deposit_branches)} branches, {len(border_out)} border withdrawals")
    if typology == "pass_through":
        s = repo.counterparty_summary(account, 180)["summary"]
        ratio = s["pass_through_ratio"] or 0
        return 0.9 <= ratio <= 1.0 and s["in_total"] >= 100_000, f"${s['in_total']:,.0f} in, ratio {ratio}"
    if typology == "synthetic_identity_ring":
        linked = repo.shared_identifiers(customers[0])
        return len(linked) == len(customers) - 1, f"{len(linked)} linked customers (ring of {len(customers)})"
    if typology == "pep_offshore_wires":
        p = repo.customer_profile(customers[0])
        inflow, expected = p["activity_vs_profile"]["avg_monthly_inflow"], p["customer"]["expected_monthly_volume"]
        return p["customer"]["is_pep"] and inflow > 3 * expected, f"inflow ${inflow:,.0f}/mo vs expected ${expected:,}"
    return False, "unknown typology"


def main() -> None:
    settings = get_settings()
    driver = connect(settings)
    repo = GraphRepository(driver, settings.neo4j_database)
    with (settings.synthetic_dir / "scenarios.csv").open(encoding="utf-8") as f:
        scenarios = list(csv.DictReader(f))

    found = 0
    print(f"{'id':4} {'typology':24} {'alerted':8} {'found':6} evidence")
    for s in scenarios:
        ok, evidence = check(repo, s["typology"], s["customer_ids"].split(";"), s["account_ids"].split(";"))
        found += ok
        print(f"{s['scenario_id']:4} {s['typology']:24} {s['alerted']:8} {'yes' if ok else 'NO':6} {evidence}")
    print(f"\nFound {found}/{len(scenarios)} planted scenarios "
          f"({sum(s['alerted'] == 'False' for s in scenarios)} of them have no alert).")

    extra = [r for r in repo.scan_near_threshold_cash(180, 10_000, 0.10, 3, 100)
             if r["account_id"] not in {s["account_ids"].split(";")[0] for s in scenarios if s["typology"] == "structuring"}]
    print(f"Structuring scan also flagged {len(extra)} non-planted account(s): "
          + ", ".join(f"{r['customer_name']} ({r['occupation']}, {r['deposits']} deposits)" for r in extra))
    driver.close()


if __name__ == "__main__":
    main()
