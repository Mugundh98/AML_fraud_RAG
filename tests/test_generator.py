import pytest

from app.graph.generator import CTR_THRESHOLD, SyntheticBank


@pytest.fixture(scope="module")
def bank():
    return SyntheticBank(seed=7, n_transactions=6_000).generate()


def test_same_seed_gives_same_data(bank):
    again = SyntheticBank(seed=7, n_transactions=6_000).generate()
    assert [(t.txn_id, t.amount, t.timestamp) for t in again.transfers] == [
        (t.txn_id, t.amount, t.timestamp) for t in bank.transfers
    ]


def test_transaction_budget_is_exact(bank):
    assert len(bank.transfers) == 6_000


def test_ids_unique_and_endpoints_exist(bank):
    account_ids = {a.account_id for a in bank.accounts}
    branch_ids = {t.from_id for t in bank.transfers if t.from_kind == "branch"} | {
        t.to_id for t in bank.transfers if t.to_kind == "branch"
    }
    assert len({t.txn_id for t in bank.transfers}) == len(bank.transfers)
    assert branch_ids <= {f"BR-{i:02d}" for i in range(1, 21)}
    for t in bank.transfers:
        assert t.from_id in account_ids or t.from_kind == "branch"
        assert t.to_id in account_ids or t.to_kind == "branch"


def test_every_typology_is_planted(bank):
    assert {s.typology for s in bank.scenarios} == {
        "structuring", "money_mule", "circular_flow", "funnel_account",
        "pass_through", "synthetic_identity_ring", "pep_offshore_wires",
    }
    # Some scenarios are deliberately left un-alerted so investigations have something to find.
    assert any(not s.alerted for s in bank.scenarios)


def test_structuring_deposits_sit_just_below_ctr_threshold(bank):
    for scenario in (s for s in bank.scenarios if s.typology == "structuring"):
        deposits = [t for t in bank.transfers if t.to_id == scenario.account_ids[0] and t.channel == "cash_deposit"
                    and t.amount >= 9_000]
        assert len(deposits) >= 5
        assert all(t.amount < CTR_THRESHOLD for t in deposits)
        assert len({t.from_id for t in deposits}) >= 2


def test_circular_flow_returns_to_origin(bank):
    scenario = next(s for s in bank.scenarios if s.typology == "circular_flow")
    ring = scenario.account_ids
    hops = {(t.from_id, t.to_id) for t in bank.transfers if t.from_id in ring and t.to_id in ring}
    for i, src in enumerate(ring):
        assert (src, ring[(i + 1) % len(ring)]) in hops


def test_alerts_reference_real_transfers(bank):
    txn_ids = {t.txn_id for t in bank.transfers}
    for alert in bank.alerts:
        assert alert.alert_id.startswith("AL-")
        assert alert.transfers and all(t.txn_id in txn_ids for t in alert.transfers)
        assert alert.priority in {"High", "Medium", "Low"}


def test_pep_alerts_are_high_priority(bank):
    assert {a.priority for a in bank.alerts if a.rule == "PEP_LARGE_WIRE"} == {"High"}
