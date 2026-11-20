"""Value at Risk and Expected Shortfall.

Conventions used throughout:

* Inputs are simple returns of a portfolio (fractions, not percent).
* VaR and ES are reported as positive loss fractions. A 1-day 99% VaR of
  0.031 means "on 1 day in 100 we expect to lose at least 3.1%".
* ``alpha`` is the confidence level (0.99), not the tail probability.

Four estimators are provided because they fail in different ways and a risk
report should show the disagreement rather than pick one silently:

historical      Empirical quantile of the window. No distributional
                assumption, but slow to react and blind to anything worse
                than the worst day in the window.
parametric      Gaussian. Fast and smooth, badly understates fat tails.
cornish_fisher  Gaussian quantile adjusted for sample skew and excess
                kurtosis. Better tails, but the expansion is only valid for
                moderate skew/kurtosis and can turn non-monotone.
filtered_hs     Filtered historical simulation (Barone-Adesi et al., 1999):
                standardise returns by an EWMA volatility, take the
                empirical quantile of the standardised residuals and rescale
                by today's volatility. Reacts to volatility regimes while
                keeping the empirical tail shape.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

Estimator = Callable[[np.ndarray, float], "VaRResult"]


@dataclass(frozen=True)
class VaRResult:
    var: float
    es: float


def _as_array(returns) -> np.ndarray:
    x = np.asarray(returns, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < 30:
        raise ValueError(f"need at least 30 observations, got {x.size}")
    return x


def _empirical_tail(x: np.ndarray, alpha: float) -> VaRResult:
    # Lower-tail quantile with the "lower" rule: VaR is an observed loss, so
    # the backtest exception count is not distorted by interpolation.
    q = np.quantile(x, 1 - alpha, method="lower")
    tail = x[x <= q]
    return VaRResult(var=-q, es=-tail.mean())


def historical(returns, alpha: float = 0.99) -> VaRResult:
    return _empirical_tail(_as_array(returns), alpha)


def parametric(returns, alpha: float = 0.99) -> VaRResult:
    x = _as_array(returns)
    mu, sigma = x.mean(), x.std(ddof=1)
    z = stats.norm.ppf(1 - alpha)
    var = -(mu + z * sigma)
    es = -(mu - sigma * stats.norm.pdf(z) / (1 - alpha))
    return VaRResult(var=var, es=es)


def cornish_fisher(returns, alpha: float = 0.99) -> VaRResult:
    x = _as_array(returns)
    mu, sigma = x.mean(), x.std(ddof=1)
    s = stats.skew(x, bias=False)
    k = stats.kurtosis(x, fisher=True, bias=False)
    z = stats.norm.ppf(1 - alpha)
    z_cf = (
        z
        + (z**2 - 1) * s / 6
        + (z**3 - 3 * z) * k / 24
        - (2 * z**3 - 5 * z) * s**2 / 36
    )
    var = -(mu + z_cf * sigma)
    # ES under CF has no closed form worth trusting; integrate the adjusted
    # quantile function numerically over the tail.
    u = np.linspace(1e-6, 1 - alpha, 400)
    zu = stats.norm.ppf(u)
    zu_cf = (
        zu
        + (zu**2 - 1) * s / 6
        + (zu**3 - 3 * zu) * k / 24
        - (2 * zu**3 - 5 * zu) * s**2 / 36
    )
    es = -(mu + sigma * np.trapezoid(zu_cf, u) / (1 - alpha))
    return VaRResult(var=var, es=max(es, var))


def ewma_volatility(returns, lam: float = 0.94) -> np.ndarray:
    """RiskMetrics EWMA volatility forecasts.

    Returns n+1 values for n returns: element t is the forecast for day t made
    with returns before t, and the last element is the forecast for the day
    after the sample. The recursion is seeded with the variance of the first
    30 observations.
    """
    x = np.asarray(returns, dtype=float)
    f = np.empty(x.size + 1)
    f[0] = np.var(x[: min(x.size, 30)])
    for t in range(x.size):
        f[t + 1] = lam * f[t] + (1 - lam) * x[t] ** 2
    return np.sqrt(f)


def filtered_hs(returns, alpha: float = 0.99, lam: float = 0.94) -> VaRResult:
    x = _as_array(returns)
    sig = ewma_volatility(x, lam)
    z = x / sig[:-1]  # each return over the volatility forecast it was made against
    tail = _empirical_tail(z, alpha)
    return VaRResult(var=tail.var * sig[-1], es=tail.es * sig[-1])


ESTIMATORS: dict[str, Estimator] = {
    "historical": historical,
    "parametric": parametric,
    "cornish_fisher": cornish_fisher,
    "filtered_hs": filtered_hs,
}


def rolling_var(
    returns: pd.Series,
    method: str = "historical",
    alpha: float = 0.99,
    window: int = 500,
) -> pd.DataFrame:
    """One-day-ahead VaR forecasts aligned to the day they are tested against.

    Row t holds the forecast built from returns[t-window : t] (exclusive of t)
    and the realised return on t, ready for ``backtest.kupiec``.
    """
    fn = ESTIMATORS[method]
    r = returns.dropna()
    vals = r.to_numpy()
    out_var, out_es = [], []
    for t in range(window, len(vals)):
        res = fn(vals[t - window : t], alpha)
        out_var.append(res.var)
        out_es.append(res.es)
    idx = r.index[window:]
    return pd.DataFrame({"var": out_var, "es": out_es, "realised": vals[window:]}, index=idx)


def scale_horizon(one_day: float, days: int) -> float:
    """Square-root-of-time scaling. Only valid for iid returns; documented as such."""
    return one_day * np.sqrt(days)
