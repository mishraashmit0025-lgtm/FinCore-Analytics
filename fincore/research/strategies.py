"""Portfolio construction rules used in the walk-forward study.

A strategy sees only the trailing return window handed to it by the engine
and returns long-only weights over that window's columns.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from fincore.portfolio import covariance, optimize

Strategy = Callable[[pd.DataFrame], np.ndarray]


def equal_weight(window: pd.DataFrame) -> np.ndarray:
    return optimize.equal_weight(window.shape[1])


def inverse_vol(window: pd.DataFrame) -> np.ndarray:
    return optimize.inverse_volatility(covariance.sample(window.to_numpy()))


def min_variance_sample(window: pd.DataFrame, cap: float = 0.10) -> np.ndarray:
    return optimize.min_variance(covariance.sample(window.to_numpy()), cap=cap)


def min_variance_shrunk(window: pd.DataFrame, cap: float = 0.10) -> np.ndarray:
    cov, _ = covariance.ledoit_wolf_constant_correlation(window.to_numpy())
    return optimize.min_variance(cov, cap=cap)


def risk_parity_shrunk(window: pd.DataFrame) -> np.ndarray:
    cov, _ = covariance.ledoit_wolf_constant_correlation(window.to_numpy())
    return optimize.risk_parity(cov)


def momentum_12_1(window: pd.DataFrame, top: int = 8) -> np.ndarray:
    """Equal-weight the ``top`` names by 12-month return skipping the last month.

    Skipping the most recent ~21 days avoids the short-term reversal effect
    (Jegadeesh 1990) contaminating the momentum signal.
    """
    if len(window) < 252:
        return equal_weight(window)
    growth = (1 + window.iloc[-252:-21]).prod() - 1
    chosen = growth.rank(ascending=False, method="first") <= top
    w = chosen.to_numpy(dtype=float)
    return w / w.sum()


STRATEGIES: dict[str, Strategy] = {
    "equal_weight": equal_weight,
    "inverse_vol": inverse_vol,
    "min_var_sample": min_variance_sample,
    "min_var_shrunk": min_variance_shrunk,
    "risk_parity": risk_parity_shrunk,
    "momentum_12_1": momentum_12_1,
}
