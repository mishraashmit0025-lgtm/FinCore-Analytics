"""fincore command line.

    fincore research   [--data data/nse_prices.csv] [--out reports/]
    fincore var        [--strategy min_var_shrunk] [--method filtered_hs] [--alpha 0.99]
    fincore recon      --ledger L.csv --bank B.csv --as-of 2026-06-30 [--out runs/]
    fincore recon-demo [--seed 7] [--n 4000]
    fincore value      examples/valuation/example_industrials.toml [--out reports/]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path


def _research(args) -> None:
    from fincore.data import market
    from fincore.research import report

    md = market.load(args.data)
    report.build(md, args.out, charts=not args.no_charts)
    print(f"wrote {args.out / 'research.md'}")


def _var(args) -> None:
    from fincore.data import market
    from fincore.research import strategies, walkforward
    from fincore.risk import backtest, var

    md = market.load(args.data)
    run = walkforward.run(md, strategies.STRATEGIES[args.strategy], args.strategy)
    f = var.rolling_var(run.net, args.method, args.alpha, args.window)
    r = backtest.evaluate(f["var"], f["realised"], args.alpha)
    nxt = var.ESTIMATORS[args.method](run.net.to_numpy()[-args.window:], args.alpha)
    print(f"{args.strategy}: {args.method} {args.alpha:.0%} 1-day VaR for the next session "
          f"{nxt.var:.2%} (ES {nxt.es:.2%})")
    print(f"backtest: {r.exceptions} exceptions in {r.n} days (expected {r.expected:.1f}), "
          f"Kupiec p={r.p_pof:.3f}, Christoffersen p={r.p_ind:.3f}, zone={r.zone}")


def _recon(args) -> None:
    from fincore.recon import pipeline, quality

    try:
        out = pipeline.run(args.ledger, args.bank, args.out, args.as_of)
    except quality.DataQualityError as exc:
        print(f"blocked: {exc}", file=sys.stderr)
        sys.exit(2)
    print(json.dumps(out.manifest["results"], indent=2))
    print(f"run written to {out.run_dir}")


def _recon_demo(args) -> None:
    from fincore.recon import evaluate, generate, match

    case = generate.generate(args.n, args.seed)
    res = match.reconcile(case.ledger, case.bank)
    s = evaluate.score(case, res)
    p0, r0 = evaluate.pair_scores(evaluate.naive_amount_date(case.ledger, case.bank), case.truth_pairs)
    print(f"ledger {len(case.ledger)} rows, bank {len(case.bank)} rows, {len(case.truth_pairs)} true links")
    print(f"naive amount+date : precision {p0:.4f} recall {r0:.4f}")
    print(f"fincore matcher   : precision {s['pair_precision']:.4f} recall {s['pair_recall']:.4f} "
          f"break-label accuracy {s['label_accuracy']:.4f}")
    if s["confusions"]:
        print("misclassified (truth -> predicted):", s["confusions"])


def _value(args) -> None:
    from fincore.valuation import model, report

    company = model.load_company(args.company)
    report.build(company, args.out)
    print(f"wrote {args.out / ('valuation_' + company.ticker.lower() + '.md')}")


def main(argv: list[str] | None = None) -> None:
    from fincore.data.market import DEFAULT_PATH

    ap = argparse.ArgumentParser(prog="fincore", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("research", help="strategy backtests, VaR backtests, stress, attribution")
    p.add_argument("--data", type=Path, default=DEFAULT_PATH)
    p.add_argument("--out", type=Path, default=Path("reports"))
    p.add_argument("--no-charts", action="store_true")
    p.set_defaults(fn=_research)

    p = sub.add_parser("var", help="next-day VaR for a strategy plus its backtest")
    p.add_argument("--data", type=Path, default=DEFAULT_PATH)
    p.add_argument("--strategy", default="min_var_shrunk")
    p.add_argument("--method", default="filtered_hs",
                   choices=["historical", "parametric", "cornish_fisher", "filtered_hs"])
    p.add_argument("--alpha", type=float, default=0.99)
    p.add_argument("--window", type=int, default=500)
    p.set_defaults(fn=_var)

    p = sub.add_parser("recon", help="reconcile a ledger extract against a bank statement")
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--as-of", type=dt.date.fromisoformat, required=True)
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.set_defaults(fn=_recon)

    p = sub.add_parser("recon-demo", help="generate a seeded case and score the matcher")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n", type=int, default=4000)
    p.set_defaults(fn=_recon_demo)

    p = sub.add_parser("value", help="three-statement model and DCF memo")
    p.add_argument("company", type=Path)
    p.add_argument("--out", type=Path, default=Path("reports"))
    p.set_defaults(fn=_value)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
