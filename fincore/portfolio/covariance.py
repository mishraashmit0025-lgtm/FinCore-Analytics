"""Covariance estimators.

With 25 stocks and a one-year window (~250 days) the sample covariance has
325 free parameters estimated from 250 observations per series, and the
minimum-variance optimiser will happily exploit its noise. Shrinkage pulls the
estimate towards a structured target and is the standard fix.
"""

from __future__ import annotations

import numpy as np


def sample(returns: np.ndarray) -> np.ndarray:
    x = np.asarray(returns, dtype=float)
    return np.cov(x, rowvar=False, ddof=1)


def ewma(returns: np.ndarray, lam: float = 0.97) -> np.ndarray:
    x = np.asarray(returns, dtype=float)
    x = x - x.mean(axis=0)
    w = lam ** np.arange(x.shape[0] - 1, -1, -1)
    w /= w.sum()
    return (x * w[:, None]).T @ x


def ledoit_wolf_constant_correlation(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit & Wolf (2004), "Honey, I shrunk the sample covariance matrix".

    Target F keeps each asset's sample variance and replaces every pairwise
    correlation by the average correlation. The optimal intensity
    delta = max(0, min(1, kappa / T)) with kappa = (pi - rho) / gamma, using
    the paper's consistent estimators. Returns (shrunk matrix, delta).
    """
    x = np.asarray(returns, dtype=float)
    t, n = x.shape
    x = x - x.mean(axis=0)
    s = x.T @ x / t  # the paper uses the 1/T estimator
    var = np.diag(s)
    std = np.sqrt(var)
    corr = s / np.outer(std, std)
    rbar = (corr.sum() - n) / (n * (n - 1))
    f = rbar * np.outer(std, std)
    np.fill_diagonal(f, var)

    y = x**2
    pi_mat = y.T @ y / t - s**2
    pi_hat = pi_mat.sum()

    # rho: asymptotic covariance between target and sample entries.
    term1 = (x**3).T @ x / t  # E[x_i^3 x_j]
    theta_ii = term1 - var[:, None] * s  # theta_{ii,ij}
    theta_jj = theta_ii.T  # theta_{jj,ij}
    ratio = np.outer(1 / std, std)  # sqrt(s_jj / s_ii)
    off = rbar / 2 * (ratio * theta_ii + ratio.T * theta_jj)
    np.fill_diagonal(off, 0.0)
    rho_hat = np.trace(pi_mat) + off.sum()

    gamma_hat = np.linalg.norm(s - f, "fro") ** 2
    kappa = (pi_hat - rho_hat) / gamma_hat
    delta = float(np.clip(kappa / t, 0.0, 1.0))
    shrunk = delta * f + (1 - delta) * s
    return shrunk * t / (t - 1), delta


def nearest_psd(cov: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Clip negative eigenvalues; used before Cholesky in simulations."""
    vals, vecs = np.linalg.eigh((cov + cov.T) / 2)
    return (vecs * np.maximum(vals, eps)) @ vecs.T
