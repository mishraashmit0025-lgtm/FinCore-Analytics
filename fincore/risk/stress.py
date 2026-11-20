"""Stress testing: replay of Indian market episodes and hypothetical shocks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fincore.data.market import MarketData


@dataclass(frozen=True)
class Scenario:
    name: str
    start: str  # last close before the shock
    end: str


# Windows run from the close before the event to the trough (or to the end of
# the episode where there was no single trough).
HISTORICAL = [
    Scenario("Demonetisation", "2016-11-08", "2016-12-26"),
    Scenario("IL&FS default / NBFC liquidity", "2018-08-28", "2018-10-26"),
    Scenario("COVID-19 crash", "2020-02-19", "2020-03-23"),
    Scenario("2022 rate-hike drawdown", "2022-01-14", "2022-06-17"),
    Scenario("Hindenburg-Adani report", "2023-01-24", "2023-02-28"),
    Scenario("2024 election result day", "2024-06-03", "2024-06-04"),
]


def replay(md: MarketData, weights: pd.Series, scenarios=HISTORICAL) -> pd.DataFrame:
    """Buy-and-hold P&L of ``weights`` over each window, vs NIFTY 50."""
    rows = []
    for sc in scenarios:
        p = md.prices.loc[sc.start : sc.end]
        b = md.benchmark.loc[sc.start : sc.end]
        if len(p) < 2:
            continue
        asset_ret = p.iloc[-1] / p.iloc[0] - 1
        w = weights.reindex(asset_ret.index).fillna(0)
        port = float((w * asset_ret).sum())
        path = (p / p.iloc[0] * w).sum(axis=1)
        rows.append(
            {
                "scenario": sc.name,
                "start": p.index[0].date(),
                "end": p.index[-1].date(),
                "days": len(p) - 1,
                "portfolio": port,
                "nifty": float(b.iloc[-1] / b.iloc[0] - 1),
                "worst_point": float(path.min() / path.iloc[0] - 1),
                "worst_name": asset_ret.loc[w[w > 0].index].idxmin(),
            }
        )
    return pd.DataFrame(rows).set_index("scenario")


def market_betas(md: MarketData, lookback: int = 504) -> pd.Series:
    r = md.returns.iloc[-lookback:]
    b = md.benchmark_returns.reindex(r.index)
    bc = b - b.mean()
    return (r.sub(r.mean()).mul(bc, axis=0).sum() / (bc**2).sum()).rename("beta")


def beta_shock(md: MarketData, weights: pd.Series, market_move: float, lookback: int = 504) -> float:
    """First-order P&L if NIFTY moves by ``market_move``, via each stock's beta.

    This ignores idiosyncratic moves and the tendency of correlations (and so
    betas) to rise in sell-offs; the historical replay is the check on that.
    """
    betas = market_betas(md, lookback)
    w = weights.reindex(betas.index).fillna(0)
    return float(np.sum(w * betas) * market_move)
