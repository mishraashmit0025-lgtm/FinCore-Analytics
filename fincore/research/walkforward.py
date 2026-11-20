"""Walk-forward backtest engine.

Timeline for a rebalance decided on day d:

    close of d      strategy sees returns up to and including d
    close of d+1    trades execute at the closing price (costs charged here)
    d+2 onwards     the new weights earn returns

The one-day lag is deliberate. Computing weights from the close of d and
trading at that same close assumes you can observe the closing auction and
trade in it simultaneously, which inflates short-horizon strategies.

Between rebalances weights drift with prices; turnover is measured against
the drifted weights, not the previous targets, which is what a real
rebalance would trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fincore.data.market import MarketData
from fincore.research.costs import CostModel
from fincore.research.strategies import Strategy


@dataclass
class BacktestRun:
    name: str
    gross: pd.Series
    net: pd.Series
    weights: pd.DataFrame  # target weights by execution date
    turnover: pd.Series  # one-way turnover by execution date
    explicit_cost: pd.Series
    impact_cost: pd.Series
    meta: dict = field(default_factory=dict)

    @property
    def annual_turnover(self) -> float:
        years = len(self.net) / 252
        return float(self.turnover.sum() / years)

    @property
    def annual_cost(self) -> float:
        years = len(self.net) / 252
        return float((self.explicit_cost.sum() + self.impact_cost.sum()) / years)


def rebalance_dates(index: pd.DatetimeIndex, freq: str = "ME") -> pd.DatetimeIndex:
    """Last trading day of each period present in ``index``."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(s.index.to_period(freq.rstrip("E"))).max().to_numpy())


def run(
    md: MarketData,
    strategy: Strategy,
    name: str = "strategy",
    lookback: int = 252,
    freq: str = "ME",
    start: str | None = None,
    costs: CostModel | None = None,
) -> BacktestRun:
    costs = costs or CostModel()
    rets = md.returns
    adv = md.adv_value.reindex(rets.index)
    vol = rets.rolling(63, min_periods=20).std().shift(1)  # known before trading
    dates = rets.index
    decide = set(rebalance_dates(dates, freq))
    first = dates[lookback] if start is None else max(dates[lookback], pd.Timestamp(start))

    n = rets.shape[1]
    w = np.zeros(n)  # current (drifted) weights; start in cash
    pending: np.ndarray | None = None
    gross, net = [], []
    w_hist, to_hist, ex_hist, im_hist, exec_dates = [], [], [], [], []
    out_index = []

    for i, day in enumerate(dates):
        if day < first:
            continue
        r = rets.iloc[i].to_numpy()

        # 1. Earn today's return on the weights held coming into today.
        day_ret = float(w @ r)
        cost_today = 0.0
        if w.sum() > 0:
            w = w * (1 + r) / (1 + day_ret)

        # 2. Execute yesterday's decision at today's close.
        if pending is not None:
            trades = pending - w
            explicit, impact = costs.trade_cost(
                trades, vol.iloc[i].fillna(0).to_numpy(), adv.iloc[i].fillna(0).to_numpy()
            )
            cost_today = explicit + impact
            w = pending.copy()
            w_hist.append(pending)
            to_hist.append(0.5 * np.abs(trades).sum())
            ex_hist.append(explicit)
            im_hist.append(impact)
            exec_dates.append(day)
            pending = None

        gross.append(day_ret)
        net.append((1 + day_ret) * (1 - cost_today) - 1)
        out_index.append(day)

        # 3. Decide at today's close using data through today.
        if day in decide:
            window = rets.iloc[max(0, i + 1 - lookback) : i + 1]
            window = window.dropna(axis=1, how="any")
            target = np.zeros(n)
            cols = rets.columns.get_indexer(window.columns)
            target[cols] = strategy(window)
            pending = target

    idx = pd.DatetimeIndex(out_index, name="date")
    ex_idx = pd.DatetimeIndex(exec_dates, name="date")
    return BacktestRun(
        name=name,
        gross=pd.Series(gross, index=idx, name=name),
        net=pd.Series(net, index=idx, name=name),
        weights=pd.DataFrame(w_hist, index=ex_idx, columns=rets.columns),
        turnover=pd.Series(to_hist, index=ex_idx),
        explicit_cost=pd.Series(ex_hist, index=ex_idx),
        impact_cost=pd.Series(im_hist, index=ex_idx),
        meta={"lookback": lookback, "freq": freq, "aum": costs.aum},
    )
