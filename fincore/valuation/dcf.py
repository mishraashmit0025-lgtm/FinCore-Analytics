"""Discounted cash flow valuation on top of the three-statement projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize

from fincore.valuation.model import Company, project


@dataclass(frozen=True)
class WACC:
    risk_free: float
    beta: float
    equity_risk_premium: float
    pre_tax_cost_of_debt: float
    tax_rate: float
    target_debt_weight: float

    @property
    def cost_of_equity(self) -> float:
        return self.risk_free + self.beta * self.equity_risk_premium

    @property
    def value(self) -> float:
        wd = self.target_debt_weight
        return (1 - wd) * self.cost_of_equity + wd * self.pre_tax_cost_of_debt * (1 - self.tax_rate)


@dataclass(frozen=True)
class DCFResult:
    fcff: pd.Series
    discount_factors: pd.Series
    pv_explicit: float
    terminal_value: float
    pv_terminal: float
    enterprise_value: float
    net_debt: float
    equity_value: float
    per_share: float
    wacc: float
    terminal_growth: float
    implied_exit_multiple: float

    @property
    def terminal_share(self) -> float:
        return self.pv_terminal / self.enterprise_value


def fcff(proj: pd.DataFrame, tax_rate: float) -> pd.Series:
    nopat = proj["ebit"] * (1 - tax_rate)
    return (nopat + proj["da"] - proj["capex"] - proj["d_nwc"]).rename("fcff")


def value(
    company: Company,
    wacc: float,
    terminal_growth: float,
    scenario: str = "base",
    mid_year: bool = True,
    ronic: float | None = None,
) -> DCFResult:
    """Value the firm.

    With ``ronic`` set, the terminal value uses the value-driver formula
    (Koller et al., *Valuation*): TV = NOPAT_{N+1} (1 - g / RONIC) / (WACC - g).
    That ties terminal reinvestment to growth; plain Gordon on the last FCFF
    silently assumes the final forecast year's capex intensity forever, which
    is too much reinvestment if capex is still elevated in year N and too
    little if it isn't.
    """
    if terminal_growth >= wacc:
        raise ValueError("terminal growth must be below WACC")
    a = company.assumptions(scenario)
    proj = project(company, scenario)
    cf = fcff(proj, a.tax_rate)
    t = np.arange(1, len(cf) + 1, dtype=float)
    if mid_year:
        t -= 0.5
    df = pd.Series(1 / (1 + wacc) ** t, index=cf.index, name="discount_factor")
    pv_explicit = float((cf * df).sum())

    # Gordon growth on the final year's cash flow grown one more year,
    # discounted from the end of the final year (plus the mid-year shift
    # because the perpetuity's flows also arrive through each year).
    if ronic is None:
        tv = float(cf.iloc[-1] * (1 + terminal_growth) / (wacc - terminal_growth))
    else:
        nopat_next = float(proj["ebit"].iloc[-1] * (1 - a.tax_rate) * (1 + terminal_growth))
        tv = nopat_next * (1 - terminal_growth / ronic) / (wacc - terminal_growth)
    n = len(cf)
    tv_discount = 1 / (1 + wacc) ** (n - 0.5 if mid_year else n)
    pv_tv = tv * tv_discount

    ev = pv_explicit + pv_tv
    o = company.opening
    net_debt = o.term_debt + o.revolver - o.cash
    equity = ev - net_debt
    return DCFResult(
        fcff=cf,
        discount_factors=df,
        pv_explicit=pv_explicit,
        terminal_value=tv,
        pv_terminal=pv_tv,
        enterprise_value=ev,
        net_debt=net_debt,
        equity_value=equity,
        per_share=equity / company.shares_outstanding,
        wacc=wacc,
        terminal_growth=terminal_growth,
        implied_exit_multiple=tv / float(proj["ebitda"].iloc[-1]),
    )


def implied_terminal_growth(
    company: Company,
    wacc: float,
    target_per_share: float,
    scenario: str = "base",
    ronic: float | None = None,
) -> float:
    """Reverse DCF: the perpetual growth rate that makes the DCF equal a price.

    Useful when DCF and multiples disagree: it states the disagreement as a
    growth assumption that can be argued about.
    """
    def gap(g):
        return value(company, wacc, g, scenario, ronic=ronic).per_share - target_per_share

    lo, hi = -0.05, wacc - 1e-4
    if gap(lo) > 0 or gap(hi) < 0:
        raise ValueError("target not reachable for growth below WACC")
    return float(optimize.brentq(gap, lo, hi, xtol=1e-10))


def sensitivity(
    company: Company,
    waccs: list[float],
    growths: list[float],
    scenario: str = "base",
    ronic: float | None = None,
) -> pd.DataFrame:
    """Per-share value over a WACC x terminal-growth grid (NaN where g >= WACC)."""
    grid = pd.DataFrame(index=pd.Index(waccs, name="wacc"), columns=pd.Index(growths, name="g"), dtype=float)
    for w in waccs:
        for g in growths:
            grid.loc[w, g] = value(company, w, g, scenario, ronic=ronic).per_share if g < w else np.nan
    return grid


def comps(company: Company, scenario: str = "base") -> pd.DataFrame:
    """Implied per-share value from peer median EV/EBITDA and P/E on year-1 numbers."""
    if company.peers is None:
        raise ValueError("no peers in company file")
    proj = project(company, scenario)
    y1 = proj.iloc[0]
    o = company.opening
    net_debt = o.term_debt + o.revolver - o.cash
    rows = []
    for col, metric in (("ev_ebitda", y1["ebitda"]), ("pe", y1["net_income"])):
        q = company.peers[col].quantile([0.25, 0.5, 0.75])
        for label, mult in zip(("p25", "median", "p75"), q, strict=True):
            if col == "ev_ebitda":
                per_share = (mult * metric - net_debt) / company.shares_outstanding
            else:
                per_share = mult * metric / company.shares_outstanding
            rows.append({"method": col, "stat": label, "multiple": mult, "per_share": per_share})
    return pd.DataFrame(rows)
