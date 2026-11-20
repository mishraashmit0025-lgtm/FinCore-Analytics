"""Build the portfolio research and risk report from the price snapshot."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fincore.data.market import MarketData
from fincore.portfolio import attribution, covariance, metrics
from fincore.report import md_table, pct
from fincore.research import strategies, walkforward
from fincore.research.costs import CostModel
from fincore.risk import backtest, decompose, stress, var


def strategy_table(md: MarketData, runs: dict[str, walkforward.BacktestRun]) -> pd.DataFrame:
    rows = []
    bench = md.benchmark_returns
    for name, run in runs.items():
        p = metrics.performance(run.net, bench)
        g = metrics.performance(run.gross)
        rows.append({
            "strategy": name,
            "CAGR net": pct(p.cagr), "CAGR gross": pct(g.cagr), "vol": pct(p.vol),
            "Sharpe": round(p.sharpe, 2), "max DD": pct(p.max_drawdown), "beta": round(p.beta, 2),
            "turnover/yr": round(run.annual_turnover, 2), "cost/yr": pct(run.annual_cost),
        })
    first = next(iter(runs.values())).net.index
    b = metrics.performance(bench.reindex(first).dropna())
    rows.append({"strategy": "NIFTY 50 (price index)", "CAGR net": pct(b.cagr), "CAGR gross": pct(b.cagr),
                 "vol": pct(b.vol), "Sharpe": round(b.sharpe, 2), "max DD": pct(b.max_drawdown), "beta": 1.0,
                 "turnover/yr": 0.0, "cost/yr": "-"})
    return pd.DataFrame(rows)


def var_table(port: pd.Series, alpha: float = 0.99, window: int = 500) -> tuple[pd.DataFrame, dict]:
    rows, forecasts = [], {}
    for name in var.ESTIMATORS:
        f = var.rolling_var(port, name, alpha, window)
        forecasts[name] = f
        r = backtest.evaluate(f["var"], f["realised"], alpha)
        rows.append({
            "model": name, "days": r.n, "exceptions": r.exceptions, "expected": round(r.expected, 1),
            "Kupiec p": round(r.p_pof, 3), "Christoffersen p": round(r.p_ind, 3),
            "cond. coverage p": round(r.p_cc, 3), "zone": r.zone,
            "avg VaR": pct(f["var"].mean()), "avg ES": pct(f["es"].mean()),
        })
    return pd.DataFrame(rows), forecasts


def capacity_table(md: MarketData, names=("min_var_shrunk", "risk_parity", "momentum_12_1"),
                   aums=(5e8, 5e9, 5e10)) -> pd.DataFrame:
    rows = []
    for n in names:
        row = {"strategy": n}
        for aum in aums:
            run = walkforward.run(md, strategies.STRATEGIES[n], n, costs=CostModel(aum=aum))
            row[f"Rs {aum / 1e7:,.0f} cr"] = pct(metrics.performance(run.net).cagr)
        rows.append(row)
    return pd.DataFrame(rows)


def build(md: MarketData, out_dir: Path, charts: bool = True) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = {n: walkforward.run(md, fn, n) for n, fn in strategies.STRATEGIES.items()}
    focus = runs["min_var_shrunk"]
    start, end = focus.net.index[0].date(), focus.net.index[-1].date()

    parts = [
        "# Portfolio research and risk report\n",
        f"Universe: {md.prices.shape[1]} NIFTY 50 stocks, daily adjusted closes. "
        f"Backtest {start} to {end}, monthly rebalance, 252-day estimation window, "
        "decisions at the close and execution at the next close, Rs 500 crore AUM with "
        "statutory charges and square-root impact. Constituents are today's, so absolute "
        "returns carry survivorship bias; comparisons between strategies are fairer than "
        "comparisons with the index.\n",
        "## Strategies (net of costs)\n",
        md_table(strategy_table(md, runs), floatfmt="{:.2f}"),
        "\n## Capacity: net CAGR as AUM grows\n",
        md_table(capacity_table(md)),
    ]

    port = focus.net
    vt, forecasts = var_table(port)
    parts += ["\n## 1-day 99% VaR backtest on the min-variance book\n",
              "Rolling 500-day estimation window, forecasts made before each day's return is seen.\n",
              md_table(vt, floatfmt="{:.3f}")]

    w = focus.weights.iloc[-1]
    w = w[w > 1e-6].sort_values(ascending=False)
    win = md.returns.iloc[-252:]
    cov, delta = covariance.ledoit_wolf_constant_correlation(win.to_numpy())
    cov = pd.DataFrame(cov, index=win.columns, columns=win.columns)
    comp = decompose.component_var(w.reindex(win.columns).fillna(0), cov)
    sector = decompose.by_sector(comp, md.sectors)
    sector = sector[sector["weight"] > 0]
    split = decompose.systematic_share(w, win, md.benchmark_returns)
    parts += [
        f"\n## Current book: risk by sector (Ledoit-Wolf shrinkage intensity {delta:.2f})\n",
        f"Market beta {split['beta']:.2f}; {pct(split['systematic_share'], 0)} of variance is NIFTY, "
        f"{pct(split['idiosyncratic_share'], 0)} residual.\n",
        md_table(sector.reset_index(), floatfmt="{:.3f}"),
    ]

    st = stress.replay(md, w).reset_index()
    for c in ("portfolio", "nifty", "worst_point"):
        st[c] = st[c].map(pct)
    parts += ["\n## Historical stress replay of the current book\n", md_table(st),
              f"\nBeta-mapped NIFTY -10% shock: {pct(stress.beta_shock(md, w, -0.10))}.\n"]

    bw = pd.DataFrame(1 / md.prices.shape[1], index=focus.weights.index, columns=focus.weights.columns)
    eff, summ = attribution.from_backtest(focus.weights, md.prices, bw, md.sectors)
    eff = eff.assign(total=eff.sum(axis=1)).sort_values("total")
    parts += [
        "\n## Brinson-Fachler attribution vs equal weight (Carino-linked, gross)\n",
        f"Min-variance {pct(summ['portfolio_return'], 1)} vs equal weight {pct(summ['benchmark_return'], 1)}, "
        f"excess {pct(summ['excess_return'], 1)}; linked effects sum to {pct(summ['explained'], 1)}.\n",
        md_table(eff.reset_index(), floatfmt="{:.4f}"),
    ]

    report = "\n".join(parts) + "\n"
    (out_dir / "research.md").write_text(report, encoding="utf-8")
    if charts:
        _charts(md, runs, forecasts, out_dir)
    return report


def _charts(md, runs, forecasts, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name in ("equal_weight", "min_var_shrunk", "risk_parity", "momentum_12_1"):
        ax.plot((1 + runs[name].net).cumprod(), label=name, linewidth=1.2)
    idx = runs["equal_weight"].net.index
    b = (1 + md.benchmark_returns.reindex(idx).fillna(0)).cumprod()
    ax.plot(b, label="NIFTY 50", color="black", linewidth=1.4, linestyle="--")
    ax.set_yscale("log")
    ax.set_title("Growth of Rs 1, net of costs (log scale)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "equity_curves.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    f_hs, f_p, f_f = forecasts["historical"], forecasts["parametric"], forecasts["filtered_hs"]
    ax.plot(f_hs.index, f_hs["realised"], color="0.6", linewidth=0.5, label="daily return")
    ax.plot(f_p.index, -f_p["var"], linewidth=1, label="Gaussian VaR")
    ax.plot(f_f.index, -f_f["var"], linewidth=1, label="Filtered HS VaR")
    hits = f_p["realised"] < -f_p["var"]
    ax.scatter(f_p.index[hits], f_p["realised"][hits], s=10, color="red", zorder=3, label="Gaussian exceptions")
    ax.set_title("1-day 99% VaR, min-variance book")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "var_backtest.png", dpi=130)
    plt.close(fig)
