"""Download daily OHLCV for the research universe from Yahoo's chart endpoint.

Writes data/nse_prices.csv in long format (date, symbol, close, adj_close, volume).
The snapshot is committed so every number in the README can be reproduced
offline; rerun this script to refresh it.

    python scripts/fetch_nse.py --start 2015-01-01 --end 2026-10-31
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fincore.data.universe import BENCHMARK, UNIVERSE  # noqa: E402

URL = (
    "https://query2.finance.yahoo.com/v8/finance/chart/{sym}"
    "?period1={p1}&period2={p2}&interval=1d&events=div%7Csplit"
)


def fetch(symbol: str, start: dt.date, end: dt.date) -> list[dict]:
    p1 = int(dt.datetime.combine(start, dt.time()).replace(tzinfo=dt.UTC).timestamp())
    p2 = int(dt.datetime.combine(end, dt.time()).replace(tzinfo=dt.UTC).timestamp())
    req = urllib.request.Request(
        URL.format(sym=urllib.request.quote(symbol), p1=p1, p2=p2),
        headers={"User-Agent": "Mozilla/5.0 (fincore research fetcher)"},
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.load(resp)
            break
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 4:
                raise
            time.sleep(2**attempt)
    result = payload["chart"]["result"][0]
    ts = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    adj = result["indicators"].get("adjclose", [{}])[0].get("adjclose", quote["close"])
    # Yahoo stamps NSE bars at 09:15 IST; convert to the IST calendar date.
    ist = dt.timezone(dt.timedelta(hours=5, minutes=30))
    rows = []
    for t, c, a, v in zip(ts, quote["close"], adj, quote["volume"], strict=True):
        if c is None or a is None:
            continue  # holiday placeholders and halted sessions
        rows.append(
            {
                "date": dt.datetime.fromtimestamp(t, ist).date().isoformat(),
                "symbol": symbol.removesuffix(".NS"),
                "close": round(c, 4),
                "adj_close": round(a, 4),
                "volume": int(v or 0),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=dt.date.fromisoformat, default=dt.date(2015, 1, 1))
    ap.add_argument("--end", type=dt.date.fromisoformat, default=dt.date(2026, 10, 31))
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "nse_prices.csv")
    args = ap.parse_args()

    symbols = [s + ".NS" for s in UNIVERSE] + [BENCHMARK]
    rows: list[dict] = []
    for sym in symbols:
        got = fetch(sym, args.start, args.end)
        print(f"{sym:16s} {len(got):5d} bars  {got[0]['date']} .. {got[-1]['date']}")
        rows.extend(got)
        time.sleep(0.5)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["date", "symbol", "close", "adj_close", "volume"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["symbol"], r["date"])))
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
