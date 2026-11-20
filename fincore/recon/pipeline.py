"""End-to-end reconciliation run: load, gate, match, persist, report.

Each run writes to its own directory:

    manifest.json   input file hashes, row counts, quality scores, code version
    recon.sqlite    ledger, bank, matches, match_lines and breaks tables
    matches.csv     one row per match with rule and evidence
    breaks.csv      open items with age and suggested action
    mis.md          management summary built from the SQL in sql/

The manifest plus the database is enough for a reviewer to trace any number
in the MIS back to the source rows and the rule that produced it.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fincore.recon import match, quality
from fincore.report import md_table

SQL_DIR = Path(__file__).with_name("sql")

ACTIONS = {
    "missing_bank": "Chase clearing / check for stale cheque",
    "missing_ledger": "Book the bank entry (charge, interest or sweep)",
    "duplicate": "Reverse the duplicate journal",
    "transposition": "Correct the keyed amount",
    "bank_charges": "Book bank charges and GST input credit",
    "amount_mismatch": "Investigate with counterparty",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True,
            cwd=Path(__file__).parent,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_inputs(ledger_path: Path, bank_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Amounts are read as strings and converted explicitly: letting pandas
    # parse "1250.50" as float and multiplying by 100 is how paise go missing.
    led = pd.read_csv(ledger_path, dtype={"amount_paise": str, "reference": str})
    bk = pd.read_csv(bank_path, dtype={"amount_paise": str})
    for df in (led, bk):
        ok = df["amount_paise"].str.fullmatch(r"-?\d+").fillna(False)
        df["amount_paise"] = [int(v) if good else v for v, good in zip(df["amount_paise"], ok, strict=True)]
    return led, bk


@dataclass
class RunOutput:
    run_dir: Path
    result: match.ReconResult
    manifest: dict


def run(ledger_path: Path, bank_path: Path, out_dir: Path, as_of: dt.date,
        period_start: dt.date | None = None) -> RunOutput:
    ledger, bank = load_inputs(ledger_path, bank_path)
    period_start = period_start or as_of - dt.timedelta(days=120)

    l_res, l_score = quality.run_checks(ledger, quality.ledger_checks(period_start, as_of), "entry_id")
    b_res, b_score = quality.run_checks(bank, quality.bank_checks(period_start, as_of), "txn_id")
    quality.gate(l_res, "ledger")
    quality.gate(b_res, "bank")

    result = match.reconcile(ledger, bank)

    run_id = f"{as_of:%Y%m%d}-{_sha256(ledger_path)[:8]}"
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    matches = result.matches_frame()
    breaks = _breaks_frame(result, ledger, bank, as_of)
    matches.to_csv(run_dir / "matches.csv", index=False)
    breaks.to_csv(run_dir / "breaks.csv", index=False)

    db = run_dir / "recon.sqlite"
    db.unlink(missing_ok=True)
    with sqlite3.connect(db) as con:
        ledger.to_sql("ledger", con, index=False)
        bank.to_sql("bank", con, index=False)
        matches.to_sql("matches", con, index=False)
        breaks.to_sql("breaks", con, index=False)
        lines = [
            {"match_id": m.match_id, "side": side, "record_id": rid}
            for m in result.matches + result.duplicates
            for side, ids in (("ledger", m.ledger_ids), ("bank", m.bank_ids))
            for rid in ids
        ]
        pd.DataFrame(lines).to_sql("match_lines", con, index=False)
        mis = build_mis(con, as_of)
    (run_dir / "mis.md").write_text(mis, encoding="utf-8")

    manifest = {
        "run_id": run_id,
        "as_of": as_of.isoformat(),
        "code_version": _git_rev(),
        "inputs": {
            "ledger": {"path": str(ledger_path), "sha256": _sha256(ledger_path), "rows": len(ledger)},
            "bank": {"path": str(bank_path), "sha256": _sha256(bank_path), "rows": len(bank)},
        },
        "quality": {
            "ledger_score": round(l_score, 2),
            "bank_score": round(b_score, 2),
            "ledger_checks": {r.name: r.failed for r in l_res},
            "bank_checks": {r.name: r.failed for r in b_res},
        },
        "results": {
            "matches": len(result.matches),
            "by_rule": matches["rule"].value_counts().to_dict() if len(matches) else {},
            "open_breaks": len(breaks),
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return RunOutput(run_dir, result, manifest)


def _breaks_frame(result: match.ReconResult, ledger, bank, as_of: dt.date) -> pd.DataFrame:
    rows = []
    for _, r in result.open_ledger.iterrows():
        rows.append({"record_id": r["entry_id"], "side": "ledger", "date": str(r["book_date"]),
                     "amount_paise": r["amount_paise"], "counterparty": r["counterparty"],
                     "break_type": "missing_bank"})
    for _, r in result.open_bank.iterrows():
        rows.append({"record_id": r["txn_id"], "side": "bank", "date": str(r["value_date"]),
                     "amount_paise": r["amount_paise"], "counterparty": r["narration"].rsplit("/", 1)[-1],
                     "break_type": "missing_ledger"})
    # The original of a duplicated entry is a good match; only the extra
    # booking is an open item, and it lives in result.duplicates.
    flagged = [m for m in result.matches if m.break_type in ACTIONS and m.break_type != "duplicate"]
    for m in flagged + result.duplicates:
        if m.break_type in ACTIONS:
            rid = (m.ledger_ids or m.bank_ids)[0]
            src = ledger.set_index("entry_id").loc[rid] if m.ledger_ids else bank.set_index("txn_id").loc[rid]
            rows.append({"record_id": rid, "side": "ledger" if m.ledger_ids else "bank",
                         "date": str(src.get("book_date", src.get("value_date"))),
                         "amount_paise": m.ledger_amount - m.bank_amount if m.bank_ids else m.ledger_amount,
                         "counterparty": src.get("counterparty", ""), "break_type": m.break_type})
    df = pd.DataFrame(rows, columns=["record_id", "side", "date", "amount_paise", "counterparty", "break_type"])
    df["age_days"] = (pd.Timestamp(as_of) - pd.to_datetime(df["date"])).dt.days
    df["action"] = df["break_type"].map(ACTIONS)
    return df


def build_mis(con: sqlite3.Connection, as_of: dt.date) -> str:
    def q(name: str) -> pd.DataFrame:
        return pd.read_sql_query((SQL_DIR / f"{name}.sql").read_text(), con)

    def rupees(df, cols):
        df = df.copy()
        for c in cols:
            df[c] = (df[c] / 100).map(lambda v: f"{v:,.2f}")
        return df

    parts = [f"# Bank reconciliation MIS, as of {as_of:%d %b %Y}\n"]
    parts.append("## Match rate by rule\n\n" + md_table(q("match_rate"), floatfmt="{:.2f}") + "\n")
    parts.append("## Open breaks by type and age (Rs)\n\n"
                 + md_table(rupees(q("break_aging"), ["d0_7", "d8_30", "d31_90", "d90_plus", "total"])) + "\n")
    parts.append("## Counterparties with the largest unreconciled value (Rs)\n\n"
                 + md_table(rupees(q("top_counterparties"), ["open_value"])) + "\n")
    return "\n".join(parts)
