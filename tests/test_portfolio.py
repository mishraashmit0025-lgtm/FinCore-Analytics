import numpy as np
import pandas as pd
import pytest

from fincore.portfolio import attribution, covariance, metrics, optimize


def _cov_cor_reference(x):
    """Loop transcription of Ledoit & Wolf's covCor.m, kept deliberately naive
    so it shares no vectorised code with the implementation under test."""
    t, n = x.shape
    x = x - x.mean(axis=0)
    s = x.T @ x / t
    var = np.diag(s).copy()
    sd = np.sqrt(var)
    rbar = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                rbar += s[i, j] / (sd[i] * sd[j])
    rbar /= n * (n - 1)
    f = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            f[i, j] = var[i] if i == j else rbar * sd[i] * sd[j]
    pi = 0.0
    pi_mat = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            pi_mat[i, j] = np.mean((x[:, i] * x[:, j] - s[i, j]) ** 2)
            pi += pi_mat[i, j]
    rho = sum(pi_mat[i, i] for i in range(n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            th_ii = np.mean((x[:, i] ** 2 - var[i]) * (x[:, i] * x[:, j] - s[i, j]))
            th_jj = np.mean((x[:, j] ** 2 - var[j]) * (x[:, i] * x[:, j] - s[i, j]))
            rho += rbar / 2 * (sd[j] / sd[i] * th_ii + sd[i] / sd[j] * th_jj)
    gamma = np.sum((f - s) ** 2)
    delta = max(0.0, min(1.0, (pi - rho) / gamma / t))
    return (delta * f + (1 - delta) * s) * t / (t - 1), delta


@pytest.fixture
def returns():
    rng = np.random.default_rng(4)
    n, t = 8, 120
    a = rng.normal(size=(n, n))
    cov = a @ a.T / n * 1e-4 + np.eye(n) * 5e-5
    return rng.multivariate_normal(np.zeros(n), cov, size=t)


def test_ledoit_wolf_matches_reference(returns):
    got, d = covariance.ledoit_wolf_constant_correlation(returns)
    ref, d_ref = _cov_cor_reference(returns)
    assert d == pytest.approx(d_ref, rel=1e-10)
    np.testing.assert_allclose(got, ref, rtol=1e-10)
    assert 0 < d < 1
    assert np.all(np.linalg.eigvalsh(got) > 0)


def test_shrinkage_falls_as_sample_grows():
    rng = np.random.default_rng(5)
    # Correlations differ by pair, so the constant-correlation target is wrong
    # and the optimal intensity should go to zero as T grows.
    a = rng.normal(size=(6, 6))
    cov = a @ a.T + np.eye(6)
    short = rng.multivariate_normal(np.zeros(6), cov, size=60)
    long = rng.multivariate_normal(np.zeros(6), cov, size=6000)
    assert covariance.ledoit_wolf_constant_correlation(long)[1] < covariance.ledoit_wolf_constant_correlation(short)[1]


def test_risk_parity_equalises_contributions(returns):
    cov = covariance.sample(returns)
    w = optimize.risk_parity(cov)
    rc = optimize.risk_contributions(w, cov)
    assert w.sum() == pytest.approx(1)
    assert np.all(w > 0)
    np.testing.assert_allclose(rc / rc.sum(), 1 / len(w), atol=1e-8)
    assert rc.sum() == pytest.approx(np.sqrt(w @ cov @ w))


def test_risk_parity_with_budget(returns):
    cov = covariance.sample(returns)
    b = np.arange(1, cov.shape[0] + 1, dtype=float)
    w = optimize.risk_parity(cov, budget=b)
    rc = optimize.risk_contributions(w, cov)
    np.testing.assert_allclose(rc / rc.sum(), b / b.sum(), atol=1e-8)


def test_min_variance_respects_cap_and_beats_alternatives(returns):
    cov = covariance.sample(returns)
    w = optimize.min_variance(cov, cap=0.25)
    assert w.sum() == pytest.approx(1)
    assert w.max() <= 0.25 + 1e-9 and w.min() >= -1e-12
    v = w @ cov @ w
    for other in (optimize.equal_weight(len(w)), optimize.inverse_volatility(cov), optimize.risk_parity(cov)):
        assert v <= other @ cov @ other + 1e-15


def test_min_variance_infeasible_cap():
    with pytest.raises(ValueError):
        optimize.min_variance(np.eye(5), cap=0.1)


def test_performance_known_series():
    idx = pd.bdate_range("2020-01-01", periods=504)
    r = pd.Series(0.0004, index=idx)
    r.iloc[100] = -0.2
    p = metrics.performance(r, rf=0.0)
    assert p.max_drawdown == pytest.approx(-0.2, abs=1e-12)
    growth = (1.0004**503) * 0.8
    assert p.cagr == pytest.approx(growth ** (1 / 2) - 1)


def test_brinson_single_period_adds_up():
    sectors = {"A": "X", "B": "X", "C": "Y", "D": "Z"}
    ar = pd.Series({"A": 0.05, "B": -0.02, "C": 0.10, "D": 0.01})
    wp = pd.Series({"A": 0.5, "B": 0.0, "C": 0.3, "D": 0.2})
    wb = pd.Series(0.25, index=ar.index)
    e = attribution.single_period(wp, wb, ar, sectors)
    assert e.to_numpy().sum() == pytest.approx((wp * ar).sum() - (wb * ar).sum())


def test_carino_linking_sums_to_compound_excess():
    rng = np.random.default_rng(6)
    names = list("ABCDEFGH")
    sectors = {n: "S" + str(i % 3) for i, n in enumerate(names)}
    periods = []
    for _ in range(36):
        ar = pd.Series(rng.normal(0.01, 0.06, len(names)), index=names)
        wp = pd.Series(rng.dirichlet(np.ones(len(names))), index=names)
        wb = pd.Series(1 / len(names), index=names)
        periods.append((wp, wb, ar))
    eff, summary = attribution.linked(periods, sectors)
    assert summary["explained"] == pytest.approx(summary["excess_return"], abs=1e-12)
    # Without linking, plain summation drifts away from the compound excess.
    naive = sum(attribution.single_period(*p, sectors).to_numpy().sum() for p in periods)
    assert abs(naive - summary["excess_return"]) > 1e-3
