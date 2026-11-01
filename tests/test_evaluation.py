"""The evaluation dataset builder and the code-based scorers."""

from types import SimpleNamespace

import pytest

from app.evaluation import (
    averages, build_items, facts_mentioned, parse_verdict, right_tool_used, tool_calls, typology_named,
    verdict_correct,
)
from app.graph.generator import SyntheticBank


@pytest.fixture(scope="module")
def items(tmp_path_factory):
    path = tmp_path_factory.mktemp("synthetic")
    SyntheticBank(seed=42, n_transactions=20_000).generate().write_csv(path)
    return build_items(path)


def test_dataset_has_the_approved_mix(items):
    kinds = [i["metadata"]["kind"] for i in items]
    assert (kinds.count("positive"), kinds.count("negative"), kinds.count("policy")) == (7, 3, 2)
    assert len({i["expected_output"]["typology"] for i in items if i["metadata"]["kind"] == "positive"}) == 7
    assert len({i["id"] for i in items}) == 12  # stable, unique IDs make uploads idempotent


def test_unalerted_scenarios_are_preferred(items):
    positives = [i for i in items if i["metadata"]["kind"] == "positive"]
    assert sum(not i["metadata"]["alerted"] for i in positives) == 5  # funnel and PEP scenarios are all alerted


def test_false_positive_is_the_business_the_structuring_scan_flags(items):
    false_positive = next(i for i in items if "near-threshold" in i["metadata"].get("why", ""))
    assert false_positive["expected_output"]["verdict"] == "not_suspicious"


@pytest.mark.parametrize("answer,verdict", [
    ("... lots of analysis.\nVerdict: suspicious — structuring", "suspicious"),
    ("**Verdict:** Not suspicious", "not_suspicious"),
    ("verdict - NOT SUSPICIOUS (no typology)", "not_suspicious"),
    ("Verdict: suspicious at first glance... Final verdict: not suspicious", "not_suspicious"),  # last one wins
    ("No verdict line at all", None),
])
def test_parse_verdict(answer, verdict):
    assert parse_verdict(answer) == verdict


def output(answer, tools=()):
    return {"answer": answer, "tools": list(tools)}


def test_scorers():
    expected = {"verdict": "suspicious", "typology": "money_mule", "must_mention": ["A001523"],
                "expected_tools": ["get_counterparty_summary"]}
    good = output("A001523 forwards 90% of P2P inflows: a money mule.\nVerdict: suspicious — money mule",
                  ["get_account_transactions", "get_counterparty_summary"])
    assert verdict_correct(output=good, expected_output=expected).value == 1.0
    assert typology_named(output=good, expected_output=expected).value == 1.0
    assert right_tool_used(output=good, expected_output=expected).value == 1.0
    assert facts_mentioned(output=good, expected_output=expected).value == 1.0
    assert tool_calls(output=good).value == 2.0

    bad = output("Looks fine.\nVerdict: not suspicious", ["get_customer_profile"])
    assert verdict_correct(output=bad, expected_output=expected).value == 0.0
    assert right_tool_used(output=bad, expected_output=expected).value == 0.0


def test_scorers_skip_what_does_not_apply():
    policy = {"verdict": None, "typology": None, "must_mention": ["30"], "expected_tools": []}
    answer = output("A SAR is due within 30 calendar days.")
    assert verdict_correct(output=answer, expected_output=policy) is None
    assert typology_named(output=answer, expected_output=policy) is None
    assert right_tool_used(output=answer, expected_output=policy) is None
    assert facts_mentioned(output=answer, expected_output=policy).value == 1.0


def test_run_averages():
    results = [SimpleNamespace(evaluations=[SimpleNamespace(name="verdict_correct", value=v)]) for v in (1, 1, 0, 1)]
    (avg,) = averages(item_results=results)
    assert (avg.name, avg.value) == ("avg_verdict_correct", 0.75)
