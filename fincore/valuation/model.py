"""Integrated three-statement projection.

The three statements are linked the way an analyst's spreadsheet links them,
including the circular reference spreadsheets resolve with iterative
calculation: interest depends on average debt and cash, which depend on the
cash flow, which depends on interest. Here the circularity is solved
explicitly by fixed-point iteration per year, and the model refuses to return
a projection whose balance sheet doesn't balance.

Units are whatever the input file uses (Rs crore in the example).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Assumptions:
    revenue_growth: list[float]
    gross_margin: list[float]
    sga_pct: list[float]
    capex_pct: list[float]
    da_pct_of_opening_ppe: float
    dso: float
    dio: float
    dpo: float
    tax_rate: float
    payout_ratio: float
    term_debt_repayment: float
    rate_term_debt: float
    rate_revolver: float
    rate_cash: float
    min_cash: float
    revolver_limit: float

    @property
    def years(self) -> int:
        return len(self.revenue_growth)


@dataclass(frozen=True)
class Opening:
    """Last reported balance sheet plus last-year revenue."""

    revenue: float
    cash: float
    receivables: float
    inventory: float
    net_ppe: float
    payables: float
    term_debt: float
    revolver: float
    equity: float


@dataclass
class Company:
    name: str
    ticker: str
    units: str
    fiscal_year_end: str
    shares_outstanding: float  # crore shares
    first_forecast_year: int
    history: pd.DataFrame  # reported years, for ratio analysis
    opening: Opening
    base: Assumptions
    scenarios: dict[str, dict] = field(default_factory=dict)
    valuation: dict = field(default_factory=dict)
    peers: pd.DataFrame | None = None

    def assumptions(self, scenario: str = "base") -> Assumptions:
        if scenario == "base":
            return self.base
        return replace(self.base, **self.scenarios[scenario])


def load_company(path: Path | str) -> Company:
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    hist = pd.DataFrame(raw["history"]).set_index("year")
    peers = pd.DataFrame(raw["peers"]).set_index("name") if "peers" in raw else None
    return Company(
        name=raw["company"]["name"],
        ticker=raw["company"]["ticker"],
        units=raw["company"]["units"],
        fiscal_year_end=raw["company"]["fiscal_year_end"],
        shares_outstanding=raw["company"]["shares_outstanding"],
        first_forecast_year=raw["company"]["first_forecast_year"],
        history=hist,
        opening=Opening(**raw["opening"]),
        base=Assumptions(**raw["assumptions"]),
        scenarios=raw.get("scenarios", {}),
        valuation=raw.get("valuation", {}),
        peers=peers,
    )


class BalanceSheetError(AssertionError):
    pass


class FundingGapError(RuntimeError):
    """The revolver is fully drawn and cash would still go negative.

    A spreadsheet would show negative cash and carry on; a projection that
    needs financing it doesn't have isn't a valid base for a valuation.
    """


def project(company: Company, scenario: str = "base", tol: float = 1e-9) -> pd.DataFrame:
    a = company.assumptions(scenario)
    o = company.opening
    prev = {
        "revenue": o.revenue, "cash": o.cash, "receivables": o.receivables,
        "inventory": o.inventory, "net_ppe": o.net_ppe, "payables": o.payables,
        "term_debt": o.term_debt, "revolver": o.revolver, "equity": o.equity,
    }
    rows = []
    for t in range(a.years):
        y = company.first_forecast_year + t
        rev = prev["revenue"] * (1 + a.revenue_growth[t])
        cogs = rev * (1 - a.gross_margin[t])
        sga = rev * a.sga_pct[t]
        ebitda = rev - cogs - sga
        da = prev["net_ppe"] * a.da_pct_of_opening_ppe
        ebit = ebitda - da
        capex = rev * a.capex_pct[t]

        rec = rev * a.dso / 365
        inv = cogs * a.dio / 365
        pay = cogs * a.dpo / 365
        nwc = rec + inv - pay
        d_nwc = nwc - (prev["receivables"] + prev["inventory"] - prev["payables"])
        term = max(prev["term_debt"] - a.term_debt_repayment, 0.0)
        repay = prev["term_debt"] - term

        # Fixed point on (interest, revolver, cash).
        revolver, cash = prev["revolver"], prev["cash"]
        for it in range(200):  # noqa: B007 - `it` is reported as the iteration count
            int_exp = a.rate_term_debt * (prev["term_debt"] + term) / 2 + a.rate_revolver * (
                prev["revolver"] + revolver) / 2
            int_inc = a.rate_cash * (prev["cash"] + cash) / 2
            ebt = ebit - int_exp + int_inc
            tax = max(ebt, 0.0) * a.tax_rate
            ni = ebt - tax
            div = max(ni, 0.0) * a.payout_ratio
            cfo = ni + da - d_nwc
            pre_revolver_cash = prev["cash"] + cfo - capex - repay - div - prev["revolver"]
            # Revolver sweeps: draw to keep min cash, repay with any excess.
            new_rev = min(max(a.min_cash - pre_revolver_cash, 0.0), a.revolver_limit)
            new_cash = pre_revolver_cash + new_rev
            if abs(new_rev - revolver) < tol and abs(new_cash - cash) < tol:
                revolver, cash = new_rev, new_cash
                break
            revolver, cash = new_rev, new_cash
        else:
            raise RuntimeError(f"interest circularity did not converge in {y}")
        if cash < 0:
            raise FundingGapError(
                f"FY{y}: revolver fully drawn ({revolver:,.1f}) and cash still {cash:,.1f}; "
                "add equity/term debt or cut capex"
            )

        ppe = prev["net_ppe"] + capex - da
        equity = prev["equity"] + ni - div
        assets = cash + rec + inv + ppe
        liab_eq = pay + term + revolver + equity
        if abs(assets - liab_eq) > 1e-6 * max(1.0, assets):
            raise BalanceSheetError(f"{y}: assets {assets:.6f} != liabilities+equity {liab_eq:.6f}")

        row = {
            "year": y, "revenue": rev, "cogs": cogs, "gross_profit": rev - cogs, "sga": sga,
            "ebitda": ebitda, "da": da, "ebit": ebit, "interest_expense": int_exp,
            "interest_income": int_inc, "ebt": ebt, "tax": tax, "net_income": ni,
            "dividends": div, "cfo": cfo, "capex": capex, "d_nwc": d_nwc,
            "debt_repayment": repay, "revolver_change": revolver - prev["revolver"],
            "cash": cash, "receivables": rec, "inventory": inv, "net_ppe": ppe,
            "total_assets": assets, "payables": pay, "term_debt": term, "revolver": revolver,
            "equity": equity, "iterations": it + 1,
            "below_min_cash": cash < a.min_cash - 1e-9,
        }
        rows.append(row)
        prev = {k: row[k] for k in prev} | {"revenue": rev}
    return pd.DataFrame(rows).set_index("year")


def ratios(history: pd.DataFrame) -> pd.DataFrame:
    h = history
    out = pd.DataFrame(index=h.index)
    out["revenue_growth"] = h["revenue"].pct_change()
    out["gross_margin"] = 1 - h["cogs"] / h["revenue"]
    out["ebitda_margin"] = h["ebitda"] / h["revenue"]
    out["dso"] = h["receivables"] / h["revenue"] * 365
    out["dio"] = h["inventory"] / h["cogs"] * 365
    out["dpo"] = h["payables"] / h["cogs"] * 365
    out["capex_pct"] = h["capex"] / h["revenue"]
    out["roce"] = h["ebit"] / (h["total_assets"] - h["payables"])
    out["net_debt_to_ebitda"] = (h["debt"] - h["cash"]) / h["ebitda"]
    return out
