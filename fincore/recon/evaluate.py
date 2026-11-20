"""Score a reconciliation against the generator's truth table."""

from __future__ import annotations

from collections import Counter

import pandas as pd

from fincore.recon.generate import ReconCase
from fincore.recon.match import ReconResult


def score(case: ReconCase, result: ReconResult) -> dict:
    pred = result.pairs()
    truth = case.truth_pairs
    tp = len(pred & truth)
    precision = tp / len(pred) if pred else 1.0
    recall = tp / len(truth) if truth else 1.0

    labels = result.labels()
    all_ids = list(case.ledger["entry_id"]) + list(case.bank["txn_id"])
    rows = []
    for rid in all_ids:
        rows.append({"id": rid, "truth": case.truth_breaks.get(rid, "clean"), "pred": labels.get(rid, "clean")})
    df = pd.DataFrame(rows)
    by_type = (
        df.assign(ok=df["truth"] == df["pred"])
        .groupby("truth")["ok"].agg(["count", "mean"])
        .rename(columns={"count": "records", "mean": "accuracy"})
    )
    confusions = Counter(
        (t, p) for t, p in zip(df["truth"], df["pred"], strict=True) if t != p
    )
    return {
        "pairs_truth": len(truth),
        "pairs_pred": len(pred),
        "pair_precision": precision,
        "pair_recall": recall,
        "label_accuracy": float((df["truth"] == df["pred"]).mean()),
        "by_type": by_type,
        "confusions": confusions.most_common(10),
    }


def naive_amount_date(ledger: pd.DataFrame, bank: pd.DataFrame, window: int = 3) -> set[tuple[str, str]]:
    """The spreadsheet approach: first open bank line with the same amount within
    ``window`` days. No references, no names. Used as a baseline only."""
    bank = bank.assign(value_date=pd.to_datetime(bank["value_date"]))
    by_amt: dict[int, list] = {}
    for row in bank.sort_values(["value_date", "txn_id"]).itertuples():
        by_amt.setdefault(row.amount_paise, []).append(row)
    used: set[str] = set()
    pairs = set()
    led = ledger.assign(book_date=pd.to_datetime(ledger["book_date"])).sort_values(["book_date", "entry_id"])
    for row in led.itertuples():
        for b in by_amt.get(row.amount_paise, []):
            if b.txn_id not in used and abs((b.value_date - row.book_date).days) <= window:
                used.add(b.txn_id)
                pairs.add((row.entry_id, b.txn_id))
                break
    return pairs


def pair_scores(pred: set, truth: set) -> tuple[float, float]:
    tp = len(pred & truth)
    return (tp / len(pred) if pred else 1.0, tp / len(truth) if truth else 1.0)
