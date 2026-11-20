"""Multi-pass ledger-to-bank matcher.

Passes run from most to least certain, and each record can be consumed once.
Running a loose rule before a strict one is how reconciliations end up with
confident wrong matches, so the order is part of the design:

    1  exact        same UTR, same amount, value date within the window
    2  split        one ledger entry = sum of all bank lines carrying its UTR
    3  ref_diff     same UTR, different amount: transposition or bank charges
    4  no_ref       bank line without UTR: same amount and counterparty,
                    nearest date, only if the candidate is unique
    5  aggregate    one bulk credit = exact subset sum of a counterparty's
                    open ledger entries in the window
    6  duplicate    open ledger entry identical to one already matched

Whatever is left is a break: missing_bank for ledger entries, missing_ledger
for bank lines. Every match records the rule that made it and the evidence
(amount difference, date lag, which test classified it), which is the audit
trail a reviewer signs off against.
"""

from __future__ import annotations

import difflib
import hashlib
import itertools
import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

UTR = re.compile(r"\b([A-Z]{4}R\d{13})\b")
DATE_WINDOW = 7  # calendar days
CHARGE_FEES = (1500, 2500, 5000)  # paise; bank fee schedule, before GST
GST = 0.18
MAX_SUBSET = 5
MAX_CANDIDATES = 16


@dataclass
class Match:
    match_id: str
    rule: str
    ledger_ids: list[str]
    bank_ids: list[str]
    ledger_amount: int
    bank_amount: int
    lag_days: int
    break_type: str | None
    evidence: str


@dataclass
class ReconResult:
    matches: list[Match]
    open_ledger: pd.DataFrame
    open_bank: pd.DataFrame
    duplicates: list[Match] = field(default_factory=list)

    def pairs(self) -> set[tuple[str, str]]:
        return {(lid, bid) for m in self.matches for lid in m.ledger_ids for bid in m.bank_ids}

    def labels(self) -> dict[str, str]:
        """Predicted break type per record id (records matched cleanly are absent)."""
        out: dict[str, str] = {}
        for m in self.matches + self.duplicates:
            if m.break_type:
                for x in m.ledger_ids + m.bank_ids:
                    out[x] = m.break_type
        for x in self.open_ledger["entry_id"]:
            out.setdefault(x, "missing_bank")
        for x in self.open_bank["txn_id"]:
            out.setdefault(x, "missing_ledger")
        return out

    def matches_frame(self) -> pd.DataFrame:
        rows = [m.__dict__ | {"ledger_ids": ",".join(m.ledger_ids), "bank_ids": ",".join(m.bank_ids)}
                for m in self.matches + self.duplicates]
        return pd.DataFrame(rows)


LEGAL_SUFFIXES = {"PVT", "PRIVATE", "LTD", "LIMITED", "LLP", "P", "CO", "COMPANY"}


def cp_key(name: str) -> str:
    return re.sub(r"[^A-Z0-9 ]", "", name.upper())[:20].strip()


@lru_cache(maxsize=65536)
def _canon(name: str) -> str:
    toks = re.sub(r"[^A-Z0-9 ]", " ", name.upper()).split()
    # The 20-char cut in bank narrations often lands inside the legal form
    # ("AARAV LOGISTICS PRIV"), so a trailing fragment of a suffix goes too.
    if len(toks) > 1 and any(s.startswith(toks[-1]) for s in LEGAL_SUFFIXES if len(s) > 2):
        toks = toks[:-1]
    return " ".join(t for t in toks if t not in LEGAL_SUFFIXES)


@lru_cache(maxsize=65536)
def same_counterparty(ledger_name: str, bank_name: str, threshold: float = 0.88) -> bool:
    """Bank narrations truncate at 20 chars and spell legal forms freely.

    Compare names with legal suffixes removed, cutting the ledger name to the
    length of the bank's version so truncation alone never fails a match.
    """
    a, b = _canon(ledger_name), _canon(bank_name)
    if not a or not b:
        return False
    if a == b or a.startswith(b) and len(b) >= 8:
        return True
    a = a[: max(len(b), 8)]
    return difflib.SequenceMatcher(None, a, b).ratio() >= threshold


def _prepare(ledger: pd.DataFrame, bank: pd.DataFrame):
    led = ledger.copy()
    bk = bank.copy()
    led["book_date"] = pd.to_datetime(led["book_date"])
    bk["value_date"] = pd.to_datetime(bk["value_date"])
    led["cp"] = led["counterparty"].str.upper()
    led["utr"] = led["reference"].where(led["reference"].str.match(UTR.pattern), None)
    bk["utr"] = bk["narration"].str.extract(UTR, expand=False)
    bk["cp"] = bk["narration"].str.rsplit("/", n=1).str[-1].str.strip()
    return led, bk


def _is_transposition(a: int, b: int) -> bool:
    if (a - b) % 9 or sorted(str(a)) != sorted(str(b)):
        return False
    sa, sb = str(a), str(b)
    diff = [i for i in range(len(sa)) if sa[i] != sb[i]]
    return len(diff) == 2 and diff[1] == diff[0] + 1


def _is_bank_charge(ledger_amt: int, bank_amt: int) -> int | None:
    shortfall = ledger_amt - bank_amt
    for fee in CHARGE_FEES:
        if shortfall == fee + round(fee * GST):
            return fee
    return None


def reconcile(ledger: pd.DataFrame, bank: pd.DataFrame, fuzzy_names: bool = True) -> ReconResult:
    """Match ``ledger`` to ``bank``.

    ``fuzzy_names=False`` requires the bank narration name to equal the
    truncated vendor-master name; it exists for the ablation in the README.
    """
    led, bk = _prepare(ledger, bank)
    if fuzzy_names:
        cp_ok = same_counterparty
    else:
        def cp_ok(lname: str, bname: str) -> bool:
            return cp_key(lname) == cp_key(bname)
    L = led.set_index("entry_id").to_dict("index")
    B = bk.set_index("txn_id").to_dict("index")
    open_l = set(L)
    open_b = set(B)
    matches: list[Match] = []

    def lag(lid: str, bid: str) -> int:
        return int((B[bid]["value_date"] - L[lid]["book_date"]).days)

    def bdays(lid: str, bid: str) -> int:
        return int(np.busday_count(L[lid]["book_date"].date(), B[bid]["value_date"].date()))

    def emit(rule, lids, bids, break_type, evidence):
        la = sum(L[x]["amount_paise"] for x in lids)
        ba = sum(B[x]["amount_paise"] for x in bids)
        mid = hashlib.sha1(("|".join(sorted(lids)) + "#" + "|".join(sorted(bids))).encode()).hexdigest()[:12]
        matches.append(Match(mid, rule, list(lids), list(bids), la, ba,
                             max(lag(x, y) for x in lids for y in bids), break_type, evidence))
        open_l.difference_update(lids)
        open_b.difference_update(bids)

    bank_by_utr: dict[str, list[str]] = defaultdict(list)
    for bid, row in B.items():
        if isinstance(row["utr"], str):
            bank_by_utr[row["utr"]].append(bid)

    def in_window(lid, bid):
        return -1 <= lag(lid, bid) <= DATE_WINDOW

    # Deterministic order: by book date, then id.
    order = sorted(L, key=lambda x: (L[x]["book_date"], x))

    # 1. exact
    for lid in order:
        u = L[lid]["utr"]
        if lid not in open_l or not isinstance(u, str):
            continue
        cands = [b for b in bank_by_utr.get(u, []) if b in open_b
                 and B[b]["amount_paise"] == L[lid]["amount_paise"] and in_window(lid, b)]
        if len(cands) == 1:
            b = cands[0]
            d = bdays(lid, b)
            emit("exact", [lid], [b], "timing" if d >= 2 else None, f"utr={u} lag_bd={d}")

    # 2. split: every open bank line with this UTR together equals the entry
    for lid in order:
        u = L[lid]["utr"]
        if lid not in open_l or not isinstance(u, str):
            continue
        cands = [b for b in bank_by_utr.get(u, []) if b in open_b and in_window(lid, b)]
        if len(cands) >= 2 and sum(B[b]["amount_paise"] for b in cands) == L[lid]["amount_paise"]:
            emit("split", [lid], sorted(cands), "split", f"utr={u} parts={len(cands)}")

    # 3. same UTR, amount differs
    for lid in order:
        u = L[lid]["utr"]
        if lid not in open_l or not isinstance(u, str):
            continue
        cands = [b for b in bank_by_utr.get(u, []) if b in open_b and in_window(lid, b)]
        if len(cands) != 1:
            continue
        b = cands[0]
        la, ba = L[lid]["amount_paise"], B[b]["amount_paise"]
        if _is_transposition(la, ba):
            emit("ref_diff", [lid], [b], "transposition", f"utr={u} diff={la - ba} digits swapped")
        elif (fee := _is_bank_charge(la, ba)) is not None:
            emit("ref_diff", [lid], [b], "bank_charges", f"utr={u} fee={fee}+gst")
        else:
            emit("ref_diff", [lid], [b], "amount_mismatch", f"utr={u} diff={la - ba} unexplained")

    # 4. no reference on the bank side
    led_by_amount: dict[int, list[str]] = defaultdict(list)
    for lid in order:
        if lid in open_l:
            led_by_amount[L[lid]["amount_paise"]].append(lid)
    for bid in sorted(open_b, key=lambda x: (B[x]["value_date"], x)):
        if isinstance(B[bid]["utr"], str):
            continue
        cands = [x for x in led_by_amount.get(B[bid]["amount_paise"], [])
                 if x in open_l and in_window(x, bid) and cp_ok(L[x]["cp"], B[bid]["cp"])]
        if not cands:
            continue
        # Nearest date wins; ties go to the lower id so reruns are identical.
        # When there are several candidates the extra ones are usually
        # duplicate bookings, which pass 6 picks up.
        cands.sort(key=lambda x: (abs(lag(x, bid)), x))
        emit("no_ref", [cands[0]], [bid], "no_reference", f"cp={B[bid]['cp']} candidates={len(cands)}")

    # 5. aggregate (one bank line, many ledger entries)
    for bid in sorted(open_b, key=lambda x: (B[x]["value_date"], x)):
        target = B[bid]["amount_paise"]
        pool = [x for x in order if x in open_l and 0 < L[x]["amount_paise"] < target
                and in_window(x, bid) and cp_ok(L[x]["cp"], B[bid]["cp"])]
        pool = sorted(pool, key=lambda x: abs(lag(x, bid)))[:MAX_CANDIDATES]
        found = None
        for k in range(2, min(MAX_SUBSET, len(pool)) + 1):
            for combo in itertools.combinations(pool, k):
                if sum(L[x]["amount_paise"] for x in combo) == target:
                    found = combo
                    break
            if found:
                break
        if found:
            emit("aggregate", sorted(found), [bid], "aggregate", f"cp={B[bid]['cp']} n={len(found)}")

    # 6. duplicates of entries that already matched
    matched_sig = {}
    for m in matches:
        for lid in m.ledger_ids:
            r = L[lid]
            matched_sig[(r["cp"], r["amount_paise"], r["book_date"], r["reference"])] = lid
    duplicates = []
    for lid in order:
        if lid not in open_l:
            continue
        r = L[lid]
        orig = matched_sig.get((r["cp"], r["amount_paise"], r["book_date"], r["reference"]))
        if orig:
            duplicates.append(Match(f"dup-{lid}", "duplicate", [lid], [], r["amount_paise"], 0, 0,
                                    "duplicate", f"same as {orig}"))
            open_l.discard(lid)

    # Once a duplicate is known, the original entry is part of that break too.
    for d in duplicates:
        orig = d.evidence.removeprefix("same as ")
        for m in matches:
            if orig in m.ledger_ids and m.break_type is None:
                m.break_type = "duplicate"

    return ReconResult(
        matches=matches,
        open_ledger=ledger[ledger["entry_id"].isin(open_l)].reset_index(drop=True),
        open_bank=bank[bank["txn_id"].isin(open_b)].reset_index(drop=True),
        duplicates=duplicates,
    )
