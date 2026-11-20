"""VaR model backtests.

kupiec          Proportion-of-failures likelihood ratio (Kupiec, 1995).
                Tests whether the exception rate equals 1 - alpha.
christoffersen  Independence LR (Christoffersen, 1998). Tests whether an
                exception today makes one tomorrow more likely. A model can
                pass Kupiec with the right count but fail here when its
                exceptions cluster in crises, which is the failure that
                actually hurts.
traffic_light   Basel zone for 250 days at 99% (green 0-4, yellow 5-9,
                red 10+), scaled to other sample sizes by the binomial CDF
                thresholds the Basel Committee used to set those bands.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class BacktestResult:
    n: int
    exceptions: int
    expected: float
    lr_pof: float
    p_pof: float
    lr_ind: float
    p_ind: float
    lr_cc: float
    p_cc: float
    zone: str

    @property
    def rate(self) -> float:
        return self.exceptions / self.n


def _xlogy(x: float, y: float) -> float:
    return 0.0 if x == 0 else x * np.log(y)


def kupiec(exceptions: np.ndarray, alpha: float) -> tuple[float, float]:
    hits = np.asarray(exceptions, dtype=bool)
    n, x = hits.size, int(hits.sum())
    p = 1 - alpha
    phat = x / n
    ll_null = _xlogy(n - x, 1 - p) + _xlogy(x, p)
    ll_alt = _xlogy(n - x, 1 - phat) + _xlogy(x, phat)
    lr = -2 * (ll_null - ll_alt)
    return lr, float(stats.chi2.sf(lr, 1))


def christoffersen(exceptions: np.ndarray) -> tuple[float, float]:
    h = np.asarray(exceptions, dtype=int)
    prev, cur = h[:-1], h[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))
    if n01 + n11 == 0:
        return 0.0, 1.0  # no exceptions after day 0: nothing to cluster
    pi0 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi1 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll_null = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    ll_alt = (
        _xlogy(n00, 1 - pi0) + _xlogy(n01, pi0) + _xlogy(n10, 1 - pi1) + _xlogy(n11, pi1)
    )
    lr = -2 * (ll_null - ll_alt)
    return lr, float(stats.chi2.sf(lr, 1))


def traffic_light(exceptions: int, n: int, alpha: float = 0.99) -> str:
    # Basel: green while cumulative prob < 95%, red once >= 99.99%.
    cdf = stats.binom.cdf(exceptions, n, 1 - alpha)
    if cdf < 0.95:
        return "green"
    if cdf < 0.9999:
        return "yellow"
    return "red"


def evaluate(var_forecast: np.ndarray, realised: np.ndarray, alpha: float) -> BacktestResult:
    var_forecast = np.asarray(var_forecast, dtype=float)
    realised = np.asarray(realised, dtype=float)
    hits = realised < -var_forecast
    lr_pof, p_pof = kupiec(hits, alpha)
    lr_ind, p_ind = christoffersen(hits)
    lr_cc = lr_pof + lr_ind
    return BacktestResult(
        n=hits.size,
        exceptions=int(hits.sum()),
        expected=hits.size * (1 - alpha),
        lr_pof=lr_pof,
        p_pof=p_pof,
        lr_ind=lr_ind,
        p_ind=p_ind,
        lr_cc=lr_cc,
        p_cc=float(stats.chi2.sf(lr_cc, 2)),
        zone=traffic_light(int(hits.sum()), hits.size, alpha),
    )
