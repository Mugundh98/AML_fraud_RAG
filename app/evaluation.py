"""Evaluation dataset and scoring for the agent (Langfuse datasets and experiments).

Dataset aml-investigations-v0, 12 items built from the generator's answer key (data/synthetic/scenarios.csv),
so expected outputs are ground truth by construction, not written by an AI:

- 7 positive: one planted scenario per typology, preferring ones that have no alert
- 3 negative: 2 ordinary customers, and a cash-intensive business whose deposits look like structuring but
  match its profile (a realistic false positive)
- 2 policy questions with a known answer from the knowledge base

Scoring is plain code (no LLM judge): free, fast and reproducible.
"""

import csv
import re
from collections import Counter
from pathlib import Path
from typing import Any

from langfuse import Evaluation

DATASET_NAME = "aml-investigations-v0"

VERDICT_REQUEST = ("Finish with one line: 'Verdict: suspicious' or 'Verdict: not suspicious', "
                   "followed by the typology if any.")

# Words an answer must use to count as naming the typology.
TYPOLOGY_WORDS = {
    "structuring": ["structuring", "smurfing"],
    "money_mule": ["mule"],
    "circular_flow": ["circular", "round-trip", "round trip", "cycle"],
    "funnel_account": ["funnel"],
    "pass_through": ["pass-through", "pass through", "rapid movement"],
    "synthetic_identity_ring": ["synthetic identit", "identity ring"],
    "pep_offshore_wires": ["pep", "politically exposed"],
}

# The detection tool an analyst would expect to see used for each typology.
TYPOLOGY_TOOLS = {
    "structuring": ["check_structuring", "scan_for_structuring"],
    "money_mule": ["get_counterparty_summary"],
    "circular_flow": ["find_circular_flows"],
    "funnel_account": ["get_account_transactions", "check_structuring"],
    "pass_through": ["get_counterparty_summary"],
    "synthetic_identity_ring": ["find_shared_identifiers"],
    "pep_offshore_wires": ["get_customer_profile"],
}

# Scenario per typology: no-alert ones where they exist, since those test the agent, not the alert queue.
POSITIVE_SCENARIOS = ["S04", "S07", "S10", "S11", "S14", "S17", "S18"]

POLICY_ITEMS = [
    {"key": "policy-sar-deadline",
     "question": "How long does a bank have to file a SAR after first detecting suspicious activity?",
     "must_mention": ["30"], "note": "31 CFR 1020.320: no later than 30 calendar days after initial detection"},
    {"key": "policy-ctr-threshold",
     "question": "Above what amount of cash in one business day must a bank file a CTR?",
     "must_mention": ["10,000"], "note": "Currency in excess of $10,000 in a single business day"},
]


def build_items(synthetic_dir: Path) -> list[dict[str, Any]]:
    scenarios = {s["scenario_id"]: s for s in _read(synthetic_dir / "scenarios.csv")}
    items = []
    for scenario_id in POSITIVE_SCENARIOS:
        s = scenarios[scenario_id]
        customer, account = s["customer_ids"].split(";")[0], s["account_ids"].split(";")[0]
        items.append(_item(
            f"positive-{scenario_id}", _review_question(customer, account),
            {"verdict": "suspicious", "typology": s["typology"], "must_mention": [account],
             "expected_tools": TYPOLOGY_TOOLS[s["typology"]]},
            {"kind": "positive", "scenario_id": scenario_id, "alerted": s["alerted"] == "True",
             "description": s["description"]},
        ))

    for customer, account, why in _negatives(synthetic_dir, scenarios):
        items.append(_item(
            f"negative-{customer}", _review_question(customer, account),
            {"verdict": "not_suspicious", "typology": None, "must_mention": [account], "expected_tools": []},
            {"kind": "negative", "why": why},
        ))

    for p in POLICY_ITEMS:
        items.append(_item(p["key"], p["question"],
                           {"verdict": None, "typology": None, "must_mention": p["must_mention"], "expected_tools": []},
                           {"kind": "policy", "note": p["note"]}))
    return items


def _negatives(synthetic_dir: Path, scenarios: dict) -> list[tuple[str, str, str]]:
    """Two ordinary, active individual customers with no alert, plus the false-positive car wash."""
    planted = {c for s in scenarios.values() for c in s["customer_ids"].split(";")}
    accounts = _read(synthetic_dir / "accounts.csv")
    alerted = {a["account_id"] for a in _read(synthetic_dir / "alerts.csv")}
    activity = Counter()
    for t in _read(synthetic_dir / "transfers.csv"):
        activity[t["from_id"]] += 1
        activity[t["to_id"]] += 1
    own_accounts = {}
    for a in accounts:
        if a["is_external"] == "False" and a["account_type"] == "checking":
            own_accounts.setdefault(a["customer_id"], a["account_id"])

    ordinary = [
        (c["customer_id"], own_accounts[c["customer_id"]], "ordinary individual, no alert")
        for c in _read(synthetic_dir / "customers.csv")
        if c["customer_type"] == "individual" and c["risk_rating"] == "low" and c["is_pep"] == "False"
        and c["customer_id"] not in planted and c["customer_id"] in own_accounts
        and own_accounts[c["customer_id"]] not in alerted and activity[own_accounts[c["customer_id"]]] >= 15
    ][:2]
    # The false positive is chosen by behaviour: the unplanted account with the most cash deposits just under
    # the $10,000 CTR threshold (a cash-intensive business, whose deposits match its declared profile).
    planted_accounts = {a for s in scenarios.values() for a in s["account_ids"].split(";")}
    near_threshold = Counter(
        t["to_id"] for t in _read(synthetic_dir / "transfers.csv")
        if t["channel"] == "cash_deposit" and 9_000 <= float(t["amount"]) < 10_000 and t["to_id"] not in planted_accounts
    )
    account = sorted(near_threshold, key=lambda a: (-near_threshold[a], a))[0]
    owner = next(a["customer_id"] for a in accounts if a["account_id"] == account)
    false_positive = (owner, account, f"cash-intensive business with {near_threshold[account]} near-threshold "
                                      "cash deposits, consistent with its profile (flagged by the structuring scan)")
    return ordinary + [false_positive]


def _review_question(customer: str, account: str) -> str:
    return f"Review customer {customer} (account {account}). Is there suspicious activity? {VERDICT_REQUEST}"


def _item(key: str, question: str, expected: dict, metadata: dict) -> dict:
    return {"id": f"{DATASET_NAME}-{key}", "input": {"question": question}, "expected_output": expected,
            "metadata": {**metadata, "source": "data/synthetic/scenarios.csv (generator answer key)"}}


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------- evaluators (item level) ----------

VERDICT = re.compile(r"verdict\W{0,5}(not\s+suspicious|suspicious)", re.IGNORECASE)


def parse_verdict(answer: str) -> str | None:
    matches = VERDICT.findall(answer)
    if not matches:
        return None
    return "not_suspicious" if matches[-1].lower().startswith("not") else "suspicious"


def verdict_correct(*, output, expected_output, **kwargs) -> Evaluation | None:
    if expected_output.get("verdict") is None:
        return None
    verdict = parse_verdict(output["answer"])
    return Evaluation(name="verdict_correct", value=float(verdict == expected_output["verdict"]),
                      comment=f"expected {expected_output['verdict']}, got {verdict or 'no verdict line'}")


def typology_named(*, output, expected_output, **kwargs) -> Evaluation | None:
    typology = expected_output.get("typology")
    if not typology:
        return None
    text = output["answer"].lower()
    found = [w for w in TYPOLOGY_WORDS[typology] if w in text]
    return Evaluation(name="typology_named", value=float(bool(found)),
                      comment=f"{typology}: found {found}" if found else f"{typology}: none of {TYPOLOGY_WORDS[typology]}")


def right_tool_used(*, output, expected_output, **kwargs) -> Evaluation | None:
    expected = expected_output.get("expected_tools") or []
    if not expected:
        return None
    used = [t for t in output["tools"] if t in expected]
    return Evaluation(name="right_tool_used", value=float(bool(used)),
                      comment=f"used {output['tools']}; expected one of {expected}")


def facts_mentioned(*, output, expected_output, **kwargs) -> Evaluation | None:
    needed = expected_output.get("must_mention") or []
    if not needed:
        return None
    found = [f for f in needed if f in output["answer"]]
    return Evaluation(name="facts_mentioned", value=len(found) / len(needed), comment=f"found {found} of {needed}")


def tool_calls(*, output, **kwargs) -> Evaluation:
    return Evaluation(name="tool_calls", value=float(len(output["tools"])))


ITEM_EVALUATORS = [verdict_correct, typology_named, right_tool_used, facts_mentioned, tool_calls]


# ---------- run level ----------

def averages(*, item_results, **kwargs) -> list[Evaluation]:
    """Mean of each item-level score across the run, e.g. verdict accuracy."""
    values: dict[str, list[float]] = {}
    for result in item_results:
        for evaluation in result.evaluations:
            values.setdefault(evaluation.name, []).append(float(evaluation.value))
    return [Evaluation(name=f"avg_{name}", value=sum(v) / len(v), comment=f"{len(v)} items")
            for name, v in values.items()]
