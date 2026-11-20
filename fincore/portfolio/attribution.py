"""Brinson-Fachler sector attribution with Carino linking.

Per period t and sector j (portfolio weight wp, benchmark weight wb, sector
returns rp and rb, total benchmark return Rb):

    allocation   (wp - wb) * (rb - Rb)
    selection    wb * (rp - rb)
    interaction  (wp - wb) * (rp - rb)

These add up to Rp - Rb within a period, but single-period effects do not add
up across periods because returns compound. Carino (1999) scales each period's
effects by k_t / k so that the linked effects sum exactly to the multi-period
arithmetic excess return. Without linking the effects drift away from the
excess return over a few years, which is the usual sign of a naive
implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sector_returns(weights: pd.Series, asset_ret: pd.Series, sectors: dict[str, str]):
    df = pd.DataFrame({"w": weights, "r": asset_ret})
    df["sector"] = df.index.map(sectors)
    wsec = df.groupby("sector")["w"].sum()
    contrib = (df["w"] * df["r"]).groupby(df["sector"]).sum()
    # Sectors with zero weight get the simple-average return so allocation
    # effects of not holding a sector are still priced.
    simple = df.groupby("sector")["r"].mean()
    rsec = (contrib / wsec.where(wsec > 0)).fillna(simple)
    return wsec, rsec


def single_period(
    wp: pd.Series, wb: pd.Series, asset_ret: pd.Series, sectors: dict[str, str]
) -> pd.DataFrame:
    wps, rps = _sector_returns(wp, asset_ret, sectors)
    wbs, rbs = _sector_returns(wb, asset_ret, sectors)
    rb_total = float((wbs * rbs).sum())
    out = pd.DataFrame(
        {
            "allocation": (wps - wbs) * (rbs - rb_total),
            "selection": wbs * (rps - rbs),
            "interaction": (wps - wbs) * (rps - rbs),
        }
    )
    return out


def _carino_k(rp: float, rb: float) -> float:
    if abs(rp - rb) < 1e-14:
        return 1.0 / (1.0 + rp)
    return (np.log1p(rp) - np.log1p(rb)) / (rp - rb)


def linked(
    periods: list[tuple[pd.Series, pd.Series, pd.Series]], sectors: dict[str, str]
) -> tuple[pd.DataFrame, dict]:
    """Link a list of (portfolio_w, benchmark_w, asset_returns) periods."""
    effects, rp_list, rb_list = [], [], []
    for wp, wb, ar in periods:
        e = single_period(wp, wb, ar, sectors)
        effects.append(e)
        rp_list.append(float((wp * ar).sum()))
        rb_list.append(float((wb * ar).sum()))
    rp_total = float(np.prod(1 + np.array(rp_list)) - 1)
    rb_total = float(np.prod(1 + np.array(rb_list)) - 1)
    k = _carino_k(rp_total, rb_total)
    total = None
    for e, rp, rb in zip(effects, rp_list, rb_list, strict=True):
        scaled = e * (_carino_k(rp, rb) / k)
        total = scaled if total is None else total.add(scaled, fill_value=0)
    summary = {
        "portfolio_return": rp_total,
        "benchmark_return": rb_total,
        "excess_return": rp_total - rb_total,
        "explained": float(total.to_numpy().sum()),
    }
    return total.sort_values("selection"), summary


def from_backtest(weights: pd.DataFrame, prices: pd.DataFrame, benchmark_weights: pd.DataFrame, sectors):
    """Build holding periods between consecutive execution dates."""
    dates = list(weights.index) + [prices.index[-1]]
    periods = []
    for a, b in zip(dates[:-1], dates[1:], strict=True):
        if b <= a:
            continue
        ar = prices.loc[b] / prices.loc[a] - 1
        periods.append((weights.loc[a], benchmark_weights.loc[a], ar))
    return linked(periods, sectors)
