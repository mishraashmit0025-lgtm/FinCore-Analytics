"""Performance statistics on daily simple returns."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class Performance:
    cagr: float
    vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    beta: float | None
    tracking_error: float | None
    information_ratio: float | None

    def as_dict(self) -> dict:
        return asdict(self)


def drawdown(returns: pd.Series) -> pd.Series:
    wealth = (1 + returns).cumprod()
    return wealth / wealth.cummax() - 1


def performance(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    rf: float = 0.065,
) -> Performance:
    """``rf`` is an annual risk-free rate; 6.5% approximates the 91-day T-bill average."""
    r = returns.dropna()
    years = len(r) / TRADING_DAYS
    growth = float((1 + r).prod())
    cagr = growth ** (1 / years) - 1
    vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    rf_d = (1 + rf) ** (1 / TRADING_DAYS) - 1
    ex = r - rf_d
    sharpe = ex.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    downside = np.sqrt(np.mean(np.minimum(ex, 0) ** 2)) * np.sqrt(TRADING_DAYS)
    sortino = ex.mean() * TRADING_DAYS / downside
    mdd = float(drawdown(r).min())
    beta = te = ir = None
    if benchmark is not None:
        b = benchmark.reindex(r.index).dropna()
        rr = r.loc[b.index]
        beta = float(np.cov(rr, b)[0, 1] / np.var(b, ddof=1))
        active = rr - b
        te = float(active.std(ddof=1) * np.sqrt(TRADING_DAYS))
        ir = float(active.mean() * TRADING_DAYS / te)
    return Performance(
        cagr=float(cagr),
        vol=float(vol),
        sharpe=float(sharpe),
        sortino=float(sortino),
        max_drawdown=mdd,
        calmar=float(cagr / abs(mdd)) if mdd < 0 else float("inf"),
        beta=beta,
        tracking_error=te,
        information_ratio=ir,
    )
