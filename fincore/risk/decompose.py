"""Where the risk comes from: Euler (component) VaR and a market/sector split.

Component VaR under the Gaussian model is
    CVaR_i = w_i * (Σw)_i / σ_p * z_α
and the components add up exactly to portfolio VaR, so they can be rolled up
by sector. The Gaussian number understates the level of tail risk (see the VaR
backtest) but the *allocation* across holdings is what this view is for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def component_var(weights: pd.Series, cov: pd.DataFrame, alpha: float = 0.99) -> pd.DataFrame:
    w = weights.reindex(cov.index).fillna(0).to_numpy()
    c = cov.to_numpy()
    sigma = float(np.sqrt(w @ c @ w))
    z = stats.norm.ppf(alpha)
    marginal = (c @ w) / sigma * z
    comp = w * marginal
    out = pd.DataFrame(
        {"weight": w, "marginal_var": marginal, "component_var": comp},
        index=cov.index,
    )
    out["pct_of_var"] = out["component_var"] / comp.sum()
    return out


def by_sector(component: pd.DataFrame, sectors: dict[str, str]) -> pd.DataFrame:
    g = component.assign(sector=component.index.map(sectors)).groupby("sector")
    out = g[["weight", "component_var", "pct_of_var"]].sum()
    out["risk_to_weight"] = out["pct_of_var"] / out["weight"].where(out["weight"] > 0)
    return out.sort_values("pct_of_var", ascending=False)


def systematic_share(weights: pd.Series, returns: pd.DataFrame, market: pd.Series) -> dict:
    """Split portfolio variance into market (beta) and residual parts by OLS."""
    port = returns.mul(weights.reindex(returns.columns).fillna(0), axis=1).sum(axis=1)
    mkt = market.reindex(port.index)
    beta = float(np.cov(port, mkt)[0, 1] / np.var(mkt, ddof=1))
    resid = port - beta * mkt
    total = float(np.var(port, ddof=1))
    sys_var = beta**2 * float(np.var(mkt, ddof=1))
    return {
        "beta": beta,
        "systematic_share": sys_var / total,
        "idiosyncratic_share": float(np.var(resid, ddof=1)) / total,
    }
