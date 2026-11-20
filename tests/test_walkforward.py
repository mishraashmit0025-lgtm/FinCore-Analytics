import numpy as np
import pandas as pd
import pytest

from fincore.data.market import MarketData
from fincore.research import costs, walkforward


def _market(days=400, n=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=days)
    rets = rng.normal(0.0004, 0.012, size=(days, n))
    prices = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=[f"S{i}" for i in range(n)])
    volume = pd.DataFrame(1e6, index=idx, columns=prices.columns)
    bench = prices.mean(axis=1)
    return MarketData(prices, volume, bench, {c: "X" for c in prices.columns})


class FreeTrading(costs.CostModel):
    def trade_cost(self, trades, daily_vol, adv_value):
        return 0.0, 0.0


def last_column(window):
    w = np.zeros(window.shape[1])
    w[-1] = 1.0
    return w


def test_no_lookahead():
    md = _market()
    seen = []

    def spy(window):
        seen.append(window.index[-1])
        return np.full(window.shape[1], 1 / window.shape[1])

    walkforward.run(md, spy, lookback=60)
    decision = seen[3]
    # Corrupt everything after a decision date; the weights decided on it must not change.
    md2 = MarketData(md.prices.copy(), md.volume, md.benchmark, md.sectors)
    md2.prices.loc[md2.prices.index > decision] *= 3.0

    captured = {}

    def recorder(window):
        captured[window.index[-1]] = window.copy()
        return np.full(window.shape[1], 1 / window.shape[1])

    walkforward.run(md, recorder, lookback=60)
    first = dict(captured)
    captured.clear()
    walkforward.run(md2, recorder, lookback=60)
    pd.testing.assert_frame_equal(first[decision], captured[decision])


def test_one_day_execution_lag():
    md = _market()
    run = walkforward.run(md, last_column, lookback=60, costs=FreeTrading())
    first_decision = walkforward.rebalance_dates(md.returns.index)
    first_decision = first_decision[first_decision >= md.returns.index[60]][0]
    exec_day = run.weights.index[0]
    rets_idx = md.returns.index
    assert rets_idx.get_loc(exec_day) == rets_idx.get_loc(first_decision) + 1
    # Return on the execution day is still earned by the old (cash) book.
    assert run.gross.loc[exec_day] == 0.0
    nxt = rets_idx[rets_idx.get_loc(exec_day) + 1]
    assert run.gross.loc[nxt] == pytest.approx(md.returns.loc[nxt, "S4"])


def test_buy_and_hold_drift_matches_prices():
    md = _market()
    run = walkforward.run(md, lambda w: np.full(w.shape[1], 1 / w.shape[1]), lookback=60, freq="YE",
                          costs=FreeTrading())
    a = run.weights.index[0]
    b = run.weights.index[1] if len(run.weights) > 1 else run.gross.index[-1]
    path = run.gross.loc[a:b].iloc[1:]
    held = (md.prices.loc[b] / md.prices.loc[a]).mean() - 1
    assert (1 + path).prod() - 1 == pytest.approx(held, rel=1e-10)


def test_costs_reduce_returns_and_turnover_measured_on_drifted_weights():
    md = _market()
    ew = walkforward.run(md, lambda w: np.full(w.shape[1], 1 / w.shape[1]), lookback=60)
    assert (ew.net <= ew.gross + 1e-15).all()
    # First trade is from cash: one-way turnover 0.5. Later monthly trades only undo drift.
    assert ew.turnover.iloc[0] == pytest.approx(0.5)
    assert ew.turnover.iloc[1:].max() < 0.05


def test_statutory_charges():
    cm = costs.CostModel(brokerage=0.0, include_impact=False)
    buy = np.array([0.1, 0.0])
    explicit, impact = cm.trade_cost(buy, np.zeros(2), np.ones(2))
    fees = costs.EXCHANGE + costs.SEBI
    assert explicit == pytest.approx(0.1 * (costs.STT + fees * (1 + costs.GST) + costs.STAMP_BUY))
    assert impact == 0.0
    sell_explicit, _ = cm.trade_cost(-buy, np.zeros(2), np.ones(2))
    assert sell_explicit < explicit  # no stamp duty on sells


def test_impact_follows_square_root_law():
    cm = costs.CostModel(aum=1e9, brokerage=0.0, impact_k=1.0)
    t = np.array([0.01])
    _, small = cm.trade_cost(t, np.array([0.02]), np.array([1e9]))
    _, big = cm.trade_cost(4 * t, np.array([0.02]), np.array([1e9]))
    # 4x the size -> 4x the notional at 2x the per-unit impact.
    assert big == pytest.approx(8 * small)
