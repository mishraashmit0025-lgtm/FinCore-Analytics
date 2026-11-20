"""Long-only portfolio construction.

All optimisers take a covariance matrix and return weights that are >= 0 and
sum to 1. Position caps are enforced because an uncapped min-variance book on
Indian large caps ends up 40%+ in two FMCG names, which no investment
committee would sign off.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize


def equal_weight(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def inverse_volatility(cov: np.ndarray) -> np.ndarray:
    w = 1 / np.sqrt(np.diag(cov))
    return w / w.sum()


def min_variance(cov: np.ndarray, cap: float = 0.10) -> np.ndarray:
    n = cov.shape[0]
    if cap * n < 1:
        raise ValueError(f"cap {cap} infeasible for {n} assets")
    res = optimize.minimize(
        lambda w: w @ cov @ w,
        x0=equal_weight(n),
        jac=lambda w: 2 * cov @ w,
        bounds=[(0.0, cap)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(n)}],
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not res.success:
        raise RuntimeError(f"min_variance did not converge: {res.message}")
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def risk_parity(cov: np.ndarray, budget: np.ndarray | None = None, tol: float = 1e-10) -> np.ndarray:
    """Equal (or budgeted) risk contribution weights.

    Solves the convex problem of Spinu (2013),
        min_y 0.5 y'Σy - Σ b_i log y_i,   y > 0,
    whose solution normalised to sum to one has risk contributions w_i(Σw)_i
    proportional to b_i. Newton's method on this objective converges in a
    handful of steps from the inverse-vol start and needs no constraints.
    """
    n = cov.shape[0]
    b = equal_weight(n) if budget is None else np.asarray(budget, float) / np.sum(budget)
    y = inverse_volatility(cov) / np.sqrt(inverse_volatility(cov) @ cov @ inverse_volatility(cov))
    for _ in range(100):
        grad = cov @ y - b / y
        hess = cov + np.diag(b / y**2)
        step = np.linalg.solve(hess, grad)
        # Backtrack to stay in the positive orthant.
        a = 1.0
        while np.any(y - a * step <= 0):
            a /= 2
        y = y - a * step
        if np.max(np.abs(grad)) < tol:
            break
    return y / y.sum()


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Euler decomposition of portfolio volatility; sums to sigma_p."""
    sigma = np.sqrt(w @ cov @ w)
    return w * (cov @ w) / sigma
