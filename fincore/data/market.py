"""Load the price snapshot into wide panels and derive returns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fincore.data.universe import BENCHMARK, UNIVERSE

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "nse_prices.csv"
TRADING_DAYS = 252


@dataclass(frozen=True)
class MarketData:
    prices: pd.DataFrame  # adjusted closes, dates x symbols
    volume: pd.DataFrame  # shares traded
    benchmark: pd.Series  # NIFTY 50 index level
    sectors: dict[str, str]

    @property
    def returns(self) -> pd.DataFrame:
        return simple_returns(self.prices)

    @property
    def benchmark_returns(self) -> pd.Series:
        return self.benchmark.pct_change(fill_method=None).dropna()

    @property
    def adv_value(self) -> pd.DataFrame:
        """20-day average daily traded value in rupees, lagged one day.

        Lagged so a trade sized on day t uses only volume known at t-1.
        """
        traded = self.volume * self.prices
        return traded.rolling(20, min_periods=10).mean().shift(1)

    def window(self, start: str | None = None, end: str | None = None) -> MarketData:
        sl = slice(start, end)
        return MarketData(
            prices=self.prices.loc[sl],
            volume=self.volume.loc[sl],
            benchmark=self.benchmark.loc[sl],
            sectors=self.sectors,
        )


def load(path: Path | str = DEFAULT_PATH) -> MarketData:
    raw = pd.read_csv(path, parse_dates=["date"])
    bench_sym = BENCHMARK.removesuffix(".NS")
    is_bench = raw["symbol"] == bench_sym
    stocks = raw[~is_bench]
    prices = stocks.pivot(index="date", columns="symbol", values="adj_close").sort_index()
    volume = stocks.pivot(index="date", columns="symbol", values="volume").sort_index()
    bench = raw[is_bench].set_index("date")["adj_close"].sort_index()

    # Keep sessions where the index printed. The few extra stock-only bars are
    # special sessions (e.g. Muhurat trading) that would otherwise show up as
    # one-day returns the benchmark has no counterpart for.
    common = prices.index.intersection(bench.index)
    prices = prices.loc[common]
    volume = volume.loc[common]
    bench = bench.loc[common]

    cols = [s for s in UNIVERSE if s in prices.columns]
    return MarketData(
        prices=prices[cols],
        volume=volume[cols],
        benchmark=bench.rename("NIFTY50"),
        sectors={s: UNIVERSE[s] for s in cols},
    )


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change(fill_method=None).iloc[1:]


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices).diff().iloc[1:]
