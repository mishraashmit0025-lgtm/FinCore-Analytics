import datetime as dt
import json
import sqlite3

import pandas as pd
import pytest

from fincore.recon import evaluate, generate, match, pipeline, quality


@pytest.fixture(scope="module")
def case():
    return generate.generate(3000, seed=101)


@pytest.fixture(scope="module")
def result(case):
    return match.reconcile(case.ledger, case.bank)


def test_matcher_accuracy(case, result):
    s = evaluate.score(case, result)
    assert s["pair_precision"] >= 0.999
    assert s["pair_recall"] >= 0.995
    assert s["label_accuracy"] >= 0.995


def test_every_break_type_is_detected(case, result):
    by_type = evaluate.score(case, result)["by_type"]
    for t in generate.BREAK_TYPES:
        assert by_type.loc[t, "accuracy"] >= 0.95, t


def test_beats_naive_baseline(case, result):
    p0, r0 = evaluate.pair_scores(evaluate.naive_amount_date(case.ledger, case.bank), case.truth_pairs)
    p1, r1 = evaluate.pair_scores(result.pairs(), case.truth_pairs)
    assert p1 > p0 and r1 > r0 + 0.15


def test_fuzzy_names_help(case):
    exact = evaluate.score(case, match.reconcile(case.ledger, case.bank, fuzzy_names=False))
    fuzzy = evaluate.score(case, match.reconcile(case.ledger, case.bank, fuzzy_names=True))
    assert fuzzy["pair_recall"] > exact["pair_recall"]


def test_input_order_does_not_change_result(case, result):
    shuffled = match.reconcile(case.ledger.sample(frac=1, random_state=1), case.bank.sample(frac=1, random_state=2))
    assert shuffled.pairs() == result.pairs()
    assert shuffled.labels() == result.labels()


@pytest.mark.parametrize(
    ("ledger_name", "bank_name", "same"),
    [
        ("Aarav Logistics Pvt Ltd", "AARAV LOGISTICS PRIV", True),
        ("Western Ghats Coffee", "WESTERN GHATS COFFEE", True),
        ("Malabar Spices Co", "MALABAR SPICES COMPA", True),
        ("Kaveri Foods Pvt Ltd", "KAVERI FOODS PRIVATE", True),
        ("Gupta Textiles", "GUPTA TEXTILFS", True),  # one-letter typo
        ("Gupta Textiles", "GUPTA TRADERS", False),
        ("Narmada Cables Ltd", "NARMADA CEMENT LTD", False),
    ],
)
def test_counterparty_matching(ledger_name, bank_name, same):
    assert match.same_counterparty(ledger_name.upper(), bank_name) is same


def test_transposition_detector():
    assert match._is_transposition(1234500, 1324500)
    assert not match._is_transposition(1234500, 1234509)
    assert not match._is_transposition(1234500, 4231500)  # non-adjacent swap


def test_matches_are_one_to_one_except_split_and_aggregate(result):
    seen_l, seen_b = set(), set()
    for m in result.matches:
        assert not (set(m.ledger_ids) & seen_l)
        assert not (set(m.bank_ids) & seen_b)
        seen_l |= set(m.ledger_ids)
        seen_b |= set(m.bank_ids)
        if m.rule in ("exact", "no_ref"):
            assert m.ledger_amount == m.bank_amount
        if m.rule in ("split", "aggregate"):
            assert m.ledger_amount == m.bank_amount


def _write(case, tmp_path):
    lp, bp = tmp_path / "ledger.csv", tmp_path / "bank.csv"
    case.ledger.to_csv(lp, index=False)
    case.bank.to_csv(bp, index=False)
    return lp, bp


def test_pipeline_outputs_are_consistent(case, tmp_path):
    lp, bp = _write(case, tmp_path)
    out = pipeline.run(lp, bp, tmp_path / "runs", as_of=dt.date(2026, 6, 30))
    manifest = json.loads((out.run_dir / "manifest.json").read_text())
    assert manifest["inputs"]["ledger"]["rows"] == len(case.ledger)
    assert manifest["quality"]["ledger_score"] == 100.0
    with sqlite3.connect(out.run_dir / "recon.sqlite") as con:
        n_lines = con.execute("SELECT COUNT(DISTINCT record_id) FROM match_lines WHERE side='bank'").fetchone()[0]
        n_open = con.execute("SELECT COUNT(*) FROM breaks WHERE break_type='missing_ledger'").fetchone()[0]
        dup_rows = con.execute("SELECT COUNT(*) FROM breaks WHERE break_type='duplicate'").fetchone()[0]
    # Every bank line is either matched or an open item.
    assert n_lines + n_open == len(case.bank)
    # Each duplicate booking appears once in breaks, not twice.
    assert dup_rows == len(out.result.duplicates)
    assert "Match rate by rule" in (out.run_dir / "mis.md").read_text()


def test_quality_gate_blocks_duplicate_keys_and_float_amounts(case, tmp_path):
    bad = case.ledger.copy()
    bad.loc[1, "entry_id"] = bad.loc[0, "entry_id"]
    bad["amount_paise"] = bad["amount_paise"].astype(object)
    bad.loc[2, "amount_paise"] = "1250.50"
    lp, bp = tmp_path / "l.csv", tmp_path / "b.csv"
    bad.to_csv(lp, index=False)
    case.bank.to_csv(bp, index=False)
    with pytest.raises(quality.DataQualityError) as exc:
        pipeline.run(lp, bp, tmp_path / "runs", as_of=dt.date(2026, 6, 30))
    msg = str(exc.value)
    assert "unique_entry_id" in msg and "integer_paise" in msg


def test_non_blocking_checks_lower_score_only():
    df = pd.DataFrame({
        "entry_id": ["a", "b"], "book_date": ["2026-05-01", "2030-01-01"], "amount_paise": [100, 200],
        "counterparty": ["X", "Y"], "reference": ["SBINR26121123456781234", None], "currency": ["INR", "USD"],
    })
    res, score = quality.run_checks(df, quality.ledger_checks(dt.date(2026, 4, 1), dt.date(2026, 6, 30)), "entry_id")
    quality.gate(res, "ledger")  # does not raise
    failed = {r.name for r in res if r.failed}
    assert failed == {"date_in_period", "currency_inr", "reference_present"}
    assert score < 100
