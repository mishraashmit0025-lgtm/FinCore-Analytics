"""Input data-quality gates, run before any matching.

Each check returns the failing row ids, so a failure points at records rather
than just a percentage. Checks marked ``blocking`` stop the run: matching a
file with duplicate primary keys or float-formatted amounts produces a
reconciliation that looks fine and isn't.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Check:
    name: str
    blocking: bool
    weight: float
    fn: Callable[[pd.DataFrame], pd.Series]  # True where the row FAILS


@dataclass(frozen=True)
class CheckResult:
    name: str
    blocking: bool
    failed: int
    total: int
    sample_ids: list[str]

    @property
    def pass_rate(self) -> float:
        return 1 - self.failed / self.total if self.total else 1.0


def _required(cols):
    def fn(df):
        return df[cols].isna().any(axis=1) | (df[cols].astype(str).apply(lambda s: s.str.strip()) == "").any(axis=1)
    return fn


def _integer_amount(df):
    a = df["amount_paise"]
    return ~a.map(lambda v: isinstance(v, int) or (hasattr(v, "is_integer") and float(v).is_integer()))


def _dup_key(key):
    return lambda df: df[key].duplicated(keep=False)


def _date_in(col, start, end):
    def fn(df):
        d = pd.to_datetime(df[col], errors="coerce")
        return d.isna() | (d < pd.Timestamp(start)) | (d > pd.Timestamp(end))
    return fn


def ledger_checks(start: dt.date, end: dt.date) -> list[Check]:
    return [
        Check("required_fields", True, 3, _required(["entry_id", "book_date", "amount_paise", "counterparty"])),
        Check("unique_entry_id", True, 3, _dup_key("entry_id")),
        Check("integer_paise", True, 3, _integer_amount),
        Check("date_in_period", False, 2, _date_in("book_date", start, end)),
        Check("currency_inr", False, 1, lambda df: df["currency"] != "INR"),
        Check("positive_amount", False, 1, lambda df: pd.to_numeric(df["amount_paise"], errors="coerce") <= 0),
        Check("reference_present", False, 1,
              lambda df: df["reference"].isna() | (df["reference"].astype(str).str.len() < 6)),
    ]


def bank_checks(start: dt.date, end: dt.date) -> list[Check]:
    return [
        Check("required_fields", True, 3, _required(["txn_id", "value_date", "amount_paise", "narration"])),
        Check("unique_txn_id", True, 3, _dup_key("txn_id")),
        Check("integer_paise", True, 3, _integer_amount),
        Check("date_in_period", False, 2, _date_in("value_date", start, end)),
        Check("currency_inr", False, 1, lambda df: df["currency"] != "INR"),
    ]


def run_checks(df: pd.DataFrame, checks: list[Check], id_col: str) -> tuple[list[CheckResult], float]:
    """Return per-check results and a 0-100 weighted quality score."""
    results = []
    for c in checks:
        mask = c.fn(df).fillna(True).astype(bool)
        bad = df.loc[mask, id_col].astype(str).tolist()
        results.append(CheckResult(c.name, c.blocking, len(bad), len(df), bad[:5]))
    wsum = sum(c.weight for c in checks)
    score = 100 * sum(c.weight * r.pass_rate for c, r in zip(checks, results, strict=True)) / wsum
    return results, score


class DataQualityError(RuntimeError):
    pass


def gate(results: list[CheckResult], source: str) -> None:
    blocking = [r for r in results if r.blocking and r.failed]
    if blocking:
        detail = "; ".join(f"{r.name}: {r.failed} rows (e.g. {', '.join(r.sample_ids)})" for r in blocking)
        raise DataQualityError(f"{source} failed blocking checks: {detail}")
