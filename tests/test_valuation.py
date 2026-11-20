from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from fincore.valuation import dcf, model

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "valuation" / "example_industrials.toml"


@pytest.fixture
def company():
    return model.load_company(EXAMPLE)


@pytest.mark.parametrize("scenario", ["base", "bear", "bull"])
def test_balance_sheet_balances(company, scenario):
    p = model.project(company, scenario)
    lhs = p["cash"] + p["receivables"] + p["inventory"] + p["net_ppe"]
    rhs = p["payables"] + p["term_debt"] + p["revolver"] + p["equity"]
    np.testing.assert_allclose(lhs, rhs, rtol=1e-9)
    assert (p["cash"] >= company.base.min_cash - 1e-9).all()


def test_cash_flow_ties_to_balance_sheet(company):
    p = model.project(company)
    prev_cash = np.concatenate([[company.opening.cash], p["cash"].to_numpy()[:-1]])
    flow = p["cfo"] - p["capex"] - p["debt_repayment"] - p["dividends"] + p["revolver_change"]
    np.testing.assert_allclose(p["cash"] - prev_cash, flow, atol=1e-9)


def test_interest_uses_average_balances(company):
    p = model.project(company)
    a = company.base
    prev_debt = np.concatenate([[company.opening.term_debt], p["term_debt"].to_numpy()[:-1]])
    prev_rev = np.concatenate([[company.opening.revolver], p["revolver"].to_numpy()[:-1]])
    expected = a.rate_term_debt * (prev_debt + p["term_debt"]) / 2 + a.rate_revolver * (prev_rev + p["revolver"]) / 2
    np.testing.assert_allclose(p["interest_expense"], expected, atol=1e-8)


def test_revolver_draws_when_cash_runs_short(company):
    stressed = replace(company, base=replace(company.base, capex_pct=[0.09] * 5))
    p = model.project(stressed)
    assert p["revolver"].max() > 0
    assert not p["below_min_cash"].any()


def test_revolver_limit_binds_before_cash_goes_negative(company):
    stressed = replace(company, base=replace(company.base, capex_pct=[0.10] * 5))
    p = model.project(stressed)
    assert p["revolver"].max() == pytest.approx(company.base.revolver_limit)
    assert p["below_min_cash"].iloc[-1] and p["cash"].min() >= 0


def test_funding_gap_is_an_error_not_negative_cash(company):
    stressed = replace(company, base=replace(company.base, capex_pct=[0.20] * 5))
    with pytest.raises(model.FundingGapError):
        model.project(stressed)


def test_unbalanced_opening_is_rejected(company):
    broken = replace(company, opening=replace(company.opening, cash=company.opening.cash + 10))
    with pytest.raises(model.BalanceSheetError):
        model.project(broken)


def test_dcf_equals_closed_form_for_constant_growth(company, monkeypatch):
    """If FCFF grows at g forever from year 1, EV must equal the growing perpetuity."""
    g, w, f1 = 0.05, 0.11, 100.0
    years = 5
    fake = model.project(company).iloc[:years].copy()

    def fake_fcff(proj, tax_rate):
        import pandas as pd
        return pd.Series([f1 * (1 + g) ** t for t in range(years)], index=proj.index)

    monkeypatch.setattr(dcf, "fcff", fake_fcff)
    monkeypatch.setattr(dcf, "project", lambda c, s="base": fake)
    end_year = dcf.value(company, w, g, mid_year=False)
    assert end_year.enterprise_value == pytest.approx(f1 / (w - g))
    mid = dcf.value(company, w, g, mid_year=True)
    assert mid.enterprise_value == pytest.approx(f1 / (w - g) * (1 + w) ** 0.5)


def test_value_moves_the_right_way(company):
    base = dcf.value(company, 0.12, 0.05).per_share
    assert dcf.value(company, 0.13, 0.05).per_share < base
    assert dcf.value(company, 0.12, 0.06).per_share > base
    assert dcf.value(company, 0.12, 0.05, "bear").per_share < base < dcf.value(company, 0.12, 0.05, "bull").per_share


def test_reverse_dcf_inverts(company):
    target = dcf.value(company, 0.12, 0.057, ronic=0.18).per_share
    g = dcf.implied_terminal_growth(company, 0.12, target, ronic=0.18)
    assert g == pytest.approx(0.057, abs=1e-8)


def test_growth_above_wacc_rejected(company):
    with pytest.raises(ValueError):
        dcf.value(company, 0.08, 0.09)


def test_value_driver_growth_adds_nothing_when_ronic_equals_wacc(company):
    # TV = NOPAT_{N+1} / WACC when RONIC = WACC: growth beyond year N+1 is worth
    # nothing, so TV / (1 + g) must not depend on g.
    a = dcf.value(company, 0.12, 0.03, ronic=0.12)
    b = dcf.value(company, 0.12, 0.07, ronic=0.12)
    assert a.terminal_value / 1.03 == pytest.approx(b.terminal_value / 1.07, rel=1e-12)
