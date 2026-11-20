"""Seeded generator of a general ledger and a bank statement with known breaks.

Real ledger data can't be published, so the matcher is measured against a
synthetic pair of files where every difference is injected on purpose and
recorded in a truth table. The break mix follows what shows up in a
receivables/payables bank reconciliation:

    timing          bank value date 1-4 business days after the book date
    bank_charges    bank amount net of a flat charge plus 18% GST
    transposition   two adjacent digits swapped when the entry was keyed
    split           one invoice settled by 2-3 bank credits
    aggregate       several invoices settled by one bulk credit
    duplicate       the same receipt booked twice in the ledger
    missing_bank    booked but not yet through the bank (uncleared cheque)
    missing_ledger  on the statement, never booked (interest, sweep, charges)
    no_reference    UTR missing from the bank narration

Noise that is not a break but makes matching hard, as in real statements:
bank narrations spell counterparty names differently from the vendor master
("PVT LTD" vs "PRIVATE LIMITED", dropped suffixes, the odd typo), and some
counterparties pay the same retainer amount every few weeks, so amount plus
counterparty is not unique.

Amounts are integer paise throughout; nothing in the recon path touches float.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

COUNTERPARTIES = [
    "Aarav Logistics Pvt Ltd", "Bharat Polymers Ltd", "Chandra Agro Exports",
    "Deccan Steel Traders", "Eastern Pharma Distributors", "Gupta Textiles",
    "Himalaya Cold Chain", "Indus Auto Components", "Jaipur Ceramics LLP",
    "Kaveri Foods Pvt Ltd", "Lakshmi Engineering Works", "Malabar Spices Co",
    "Narmada Cables Ltd", "Orient Paper Mills", "Pune Precision Tools",
    "Rajput Builders", "Sahyadri Dairy", "Tirupati Packaging",
    "Udaipur Marble House", "Vardhman Chemicals", "Western Ghats Coffee",
    "Yamuna Fertilisers", "Zenith IT Services",
]
BANK_CODES = ["SBIN", "ICIC", "UTIB", "KKBK", "PUNB", "BARB", "YESB", "IDIB"]
BREAK_TYPES = [
    "timing", "bank_charges", "transposition", "split", "aggregate",
    "duplicate", "missing_bank", "missing_ledger", "no_reference",
]


@dataclass
class ReconCase:
    ledger: pd.DataFrame
    bank: pd.DataFrame
    truth_pairs: set[tuple[str, str]]  # (ledger_id, bank_id) that belong together
    truth_breaks: dict[str, str]  # record id -> injected break type


def _business_days(start: dt.date, n: int) -> list[dt.date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def _utr(rng: np.random.Generator, code: str, day: dt.date) -> str:
    return f"{code}R{day:%y%j}{rng.integers(10**7, 10**8)}"


def _bank_name(cp: str, rng: np.random.Generator) -> str:
    name = cp.upper()
    u = rng.random()
    if u < 0.12:
        name = name.replace("PVT LTD", "PRIVATE LIMITED").replace(" LTD", " LIMITED")
    elif u < 0.22:
        name = re.sub(r" CO$", " COMPANY", name.replace("PVT LTD", "P LTD").replace(" LLP", ""))
    elif u < 0.30:
        name = name.replace(" PVT LTD", "").replace(" LTD", "")
    if rng.random() < 0.04 and len(name) > 6:
        i = int(rng.integers(1, min(len(name), 18) - 1))
        if name[i].isalpha():
            name = name[:i] + chr((ord(name[i]) - 65 + 1) % 26 + 65) + name[i + 1 :]
    return name[:20]


def _transpose(amount: int, rng: np.random.Generator) -> int:
    s = list(str(amount))
    candidates = [i for i in range(len(s) - 1) if s[i] != s[i + 1] and not (i == 0 and s[1] == "0")]
    if not candidates:
        return amount + 900  # still a multiple of 9 away; rare
    i = int(rng.choice(candidates))
    s[i], s[i + 1] = s[i + 1], s[i]
    return int("".join(s))


def generate(n: int = 4000, seed: int = 7, break_rate: float = 0.25, start=dt.date(2026, 4, 1)) -> ReconCase:
    rng = np.random.default_rng(seed)
    days = _business_days(start, 70)
    ledger, bank = [], []
    pairs: set[tuple[str, str]] = set()
    breaks: dict[str, str] = {}
    lid = bid = 0

    def new_l(**kw):
        nonlocal lid
        lid += 1
        row = {"entry_id": f"GL{lid:06d}", **kw}
        ledger.append(row)
        return row["entry_id"]

    def new_b(**kw):
        nonlocal bid
        bid += 1
        row = {"txn_id": f"BK{bid:06d}", **kw}
        bank.append(row)
        return row["txn_id"]

    def narration(cp: str, ref: str | None, mode: str) -> str:
        head = f"{mode}/{ref}" if ref else f"{mode}"
        return f"{head}/{_bank_name(cp, rng)}"

    retainer = {cp: int(round(np.exp(rng.normal(11.0, 0.6)))) * 100 for cp in COUNTERPARTIES}

    weights = np.array([0.40, 0.10, 0.06, 0.08, 0.06, 0.06, 0.08, 0.08, 0.08])
    i = 0
    while i < n:
        day = days[int(rng.integers(0, 50))]
        cp = COUNTERPARTIES[int(rng.integers(len(COUNTERPARTIES)))]
        code = BANK_CODES[int(rng.integers(len(BANK_CODES)))]
        # Log-normal invoice sizes: median ~Rs 85k, long right tail.
        amt = int(round(np.exp(rng.normal(11.35, 1.1)))) * 100
        if rng.random() < 0.15:
            amt = retainer[cp]
        ref = _utr(rng, code, day)
        mode = "NEFT" if amt < 2e7 else "RTGS"
        kind = "clean" if rng.random() > break_rate else str(rng.choice(BREAK_TYPES, p=weights / weights.sum()))
        lag = int(rng.integers(0, 2))
        bday = days[days.index(day) + lag]
        i += 1

        if kind in ("clean", "timing", "bank_charges", "transposition", "no_reference", "duplicate"):
            if kind == "timing":
                bday = days[days.index(day) + int(rng.integers(2, 5))]
            l_amt = _transpose(amt, rng) if kind == "transposition" else amt
            b_amt = amt
            if kind == "bank_charges":
                fee = int(rng.choice([1500, 2500, 5000]))
                b_amt = amt - fee - round(fee * 0.18)
            le = new_l(book_date=day, amount_paise=l_amt, counterparty=cp, reference=ref, narration=f"Receipt {cp}")
            be = new_b(
                value_date=bday,
                amount_paise=b_amt,
                narration=narration(cp, None if kind == "no_reference" else ref, mode),
            )
            pairs.add((le, be))
            if kind != "clean":
                breaks[le] = kind
                breaks[be] = kind
            if kind == "duplicate":
                dup = new_l(book_date=day, amount_paise=l_amt, counterparty=cp, reference=ref,
                            narration=f"Receipt {cp}")
                breaks[dup] = "duplicate"
        elif kind == "split":
            parts = int(rng.integers(2, 4))
            cuts = np.sort(rng.choice(np.arange(1, amt // 100), size=parts - 1, replace=False)) * 100
            pieces = np.diff(np.concatenate([[0], cuts, [amt]]))
            le = new_l(book_date=day, amount_paise=amt, counterparty=cp, reference=ref, narration=f"Receipt {cp}")
            breaks[le] = "split"
            for k, p in enumerate(pieces):
                pday = days[days.index(day) + k]
                be = new_b(value_date=pday, amount_paise=int(p), narration=narration(cp, ref, mode))
                pairs.add((le, be))
                breaks[be] = "split"
        elif kind == "aggregate":
            count = int(rng.integers(2, 5))
            sizes = [int(round(np.exp(rng.normal(10.5, 0.8)))) * 100 for _ in range(count)]
            lids = [
                new_l(book_date=day, amount_paise=s, counterparty=cp,
                      reference=_utr(rng, code, day), narration=f"Invoice {cp}")
                for s in sizes
            ]
            be = new_b(value_date=bday, amount_paise=sum(sizes), narration=narration(cp, None, "BULK NEFT"))
            breaks[be] = "aggregate"
            for x in lids:
                pairs.add((x, be))
                breaks[x] = "aggregate"
        elif kind == "missing_bank":
            le = new_l(book_date=days[int(rng.integers(40, 55))], amount_paise=amt, counterparty=cp,
                       reference=f"CHQ{rng.integers(100000, 999999)}", narration=f"Cheque {cp}")
            breaks[le] = "missing_bank"
        elif kind == "missing_ledger":
            fee = int(rng.choice([11800, 29500, 5900, 250000]))
            be = new_b(value_date=day, amount_paise=-fee if rng.random() < 0.6 else fee,
                       narration=str(rng.choice(["SMS CHARGES", "INT CREDIT", "CASH MGMT FEE", "AUTO SWEEP"])))
            breaks[be] = "missing_ledger"

    led = pd.DataFrame(ledger)
    bk = pd.DataFrame(bank)
    # Shuffle so nothing downstream can rely on generation order.
    led = led.sample(frac=1, random_state=seed).reset_index(drop=True)
    bk = bk.sample(frac=1, random_state=seed + 1).reset_index(drop=True)
    led["currency"] = "INR"
    bk["currency"] = "INR"
    return ReconCase(ledger=led, bank=bk, truth_pairs=pairs, truth_breaks=breaks)
