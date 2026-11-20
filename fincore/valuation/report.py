"""Valuation memo: projection, DCF, sensitivity, comps, scenarios, reverse DCF."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fincore.report import md_table, pct
from fincore.valuation import dcf
from fincore.valuation.model import Company, project, ratios


def _fy(cols) -> list[str]:
    return [f"FY{str(c)[2:]}" for c in cols]


def build(company: Company, out_dir: Path) -> str:
    v = company.valuation
    a = company.base
    w = dcf.WACC(v["risk_free"], v["beta"], v["equity_risk_premium"], v["pre_tax_cost_of_debt"],
                 a.tax_rate, v["target_debt_weight"])
    g, ronic = v["terminal_growth"], v.get("ronic")
    base = dcf.value(company, w.value, g, ronic=ronic)
    proj = project(company)

    lines = proj[["revenue", "ebitda", "ebit", "net_income", "cfo", "capex", "cash",
                  "term_debt", "revolver", "equity"]].T
    lines.columns = _fy(lines.columns)
    flows = pd.DataFrame({"FCFF": base.fcff, "discount factor": base.discount_factors,
                          "PV": base.fcff * base.discount_factors}).T
    flows.columns = lines.columns

    scen = []
    for s in ["bear", "base", "bull"]:
        if s != "base" and s not in company.scenarios:
            continue
        r = dcf.value(company, w.value, g, s, ronic=ronic)
        scen.append({"scenario": s, "EV": round(r.enterprise_value, 1), "equity": round(r.equity_value, 1),
                     "per share": round(r.per_share, 1), "TV share of EV": pct(r.terminal_share, 0)})

    waccs = [w.value + d for d in (-0.02, -0.01, 0.0, 0.01, 0.02)]
    grid = dcf.sensitivity(company, waccs, [g - 0.01, g, g + 0.01], ronic=ronic)
    grid.index = [pct(x) for x in grid.index]
    grid.columns = [f"g={pct(c, 1)}" for c in grid.columns]

    comps = dcf.comps(company)
    med = float(comps.loc[comps["stat"] == "median", "per_share"].mean())
    try:
        implied = pct(dcf.implied_terminal_growth(company, w.value, med, ronic=ronic))
    except ValueError:
        implied = "not reachable below WACC"

    hist = ratios(company.history).T
    hist.columns = _fy(hist.columns)
    tv_method = f"value-driver, RONIC {pct(ronic, 0)}" if ronic else "Gordon growth"

    parts = [
        f"# {company.name} ({company.ticker}): valuation memo\n",
        f"All figures in {company.units} unless stated. Fiscal year ends {company.fiscal_year_end}.\n",
        "## Historical ratios\n",
        md_table(hist.reset_index(names="metric"), floatfmt="{:.3f}"),
        "\n## Projection (base case)\n",
        md_table(lines.reset_index(names="line"), floatfmt="{:,.1f}"),
        "\nThe balance sheet balances in every year (checked in code); interest on average "
        "debt and cash is solved iteratively and converged in "
        f"{int(proj['iterations'].max())} iterations or fewer.\n",
        "## Cost of capital\n",
        f"Cost of equity {pct(w.cost_of_equity)} = {pct(v['risk_free'])} + {v['beta']} x "
        f"{pct(v['equity_risk_premium'])}; after-tax cost of debt "
        f"{pct(v['pre_tax_cost_of_debt'] * (1 - a.tax_rate))}; target debt weight "
        f"{pct(v['target_debt_weight'], 0)}; WACC {pct(w.value)}.\n",
        "## DCF (mid-year convention)\n",
        md_table(flows.reset_index(names="line"), floatfmt="{:,.2f}"),
        f"\nTerminal value {base.terminal_value:,.1f} ({tv_method}, g = {pct(g, 1)}), implying "
        f"{base.implied_exit_multiple:.1f}x terminal EBITDA. Enterprise value {base.enterprise_value:,.1f}, "
        f"less net debt {base.net_debt:,.1f}, equity {base.equity_value:,.1f}, "
        f"**{base.per_share:,.1f} per share**. Terminal value is {pct(base.terminal_share, 0)} of EV.\n",
        "## Scenarios\n",
        md_table(pd.DataFrame(scen), floatfmt="{:,.1f}"),
        "\n## Sensitivity: value per share, WACC x terminal growth\n",
        md_table(grid.reset_index(names="WACC"), floatfmt="{:,.1f}"),
        "\n## Trading comparables (year-1 metrics)\n",
        md_table(comps, floatfmt="{:,.1f}"),
        f"\nPeers at the median imply about {med:,.1f} per share. Reverse DCF: matching that needs "
        f"perpetual growth of {implied} at the base WACC. That is the growth the peer multiples embed; "
        "whether it is credible for this business, given its return on new capital, is the judgement "
        "the memo has to make, and the scenario table is where to argue it.\n",
    ]
    report = "\n".join(parts)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"valuation_{company.ticker.lower()}.md").write_text(report, encoding="utf-8")
    return report
