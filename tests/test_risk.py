import numpy as np
import pytest
from scipy import stats

from fincore.risk import backtest, var


@pytest.fixture
def normal_returns():
    rng = np.random.default_rng(0)
    return rng.normal(0.0005, 0.01, 200_000)


def test_parametric_matches_closed_form(normal_returns):
    r = var.parametric(normal_returns, 0.99)
    mu, sd = normal_returns.mean(), normal_returns.std(ddof=1)
    assert r.var == pytest.approx(-(mu + stats.norm.ppf(0.01) * sd))
    # Gaussian ES = -mu + sd * phi(z) / (1 - alpha)
    assert r.es == pytest.approx(-mu + sd * stats.norm.pdf(stats.norm.ppf(0.01)) / 0.01)


def test_all_estimators_agree_on_gaussian_data(normal_returns):
    ref = var.parametric(normal_returns, 0.99)
    for name in ("historical", "cornish_fisher"):
        r = var.ESTIMATORS[name](normal_returns, 0.99)
        assert r.var == pytest.approx(ref.var, rel=0.02), name
        assert r.es == pytest.approx(ref.es, rel=0.03), name


def test_historical_uses_an_observed_loss():
    x = np.arange(-50, 50) / 1000.0  # -0.050 .. 0.049
    r = var.historical(x, 0.95)
    assert r.var == pytest.approx(0.046)  # 5th lowest of 100
    assert r.es == pytest.approx(np.mean([0.050, 0.049, 0.048, 0.047, 0.046]))


def test_cornish_fisher_widens_tail_for_fat_tails():
    rng = np.random.default_rng(1)
    t = stats.t.rvs(df=4, size=50_000, random_state=rng) * 0.01
    assert var.cornish_fisher(t, 0.99).var > var.parametric(t, 0.99).var


def test_filtered_hs_scales_with_volatility():
    rng = np.random.default_rng(2)
    x = rng.standard_t(5, 1500) * 0.01
    base = var.filtered_hs(x, 0.99)
    doubled = var.filtered_hs(2 * x, 0.99)
    assert doubled.var == pytest.approx(2 * base.var)
    assert doubled.es == pytest.approx(2 * base.es)


def test_ewma_recursion_and_alignment():
    x = np.array([0.01, -0.02, 0.03] + [0.0] * 40)
    sig = var.ewma_volatility(x, lam=0.9)
    assert sig.shape == (x.size + 1,)
    f0 = np.var(x[:30])
    assert sig[0] ** 2 == pytest.approx(f0)
    assert sig[1] ** 2 == pytest.approx(0.9 * f0 + 0.1 * 0.01**2)
    assert sig[2] ** 2 == pytest.approx(0.9 * sig[1] ** 2 + 0.1 * 0.02**2)


def test_rolling_var_never_sees_the_day_it_forecasts():
    import pandas as pd

    rng = np.random.default_rng(3)
    s = pd.Series(rng.normal(0, 0.01, 300), index=pd.bdate_range("2020-01-01", periods=300))
    f = var.rolling_var(s, "historical", 0.99, window=100)
    s2 = s.copy()
    s2.iloc[150] = -0.5  # a crash on day 150
    f2 = var.rolling_var(s2, "historical", 0.99, window=100)
    day = s.index[150]
    assert f.loc[day, "var"] == f2.loc[day, "var"]
    assert f2.loc[s.index[151], "var"] > f.loc[s.index[151], "var"]


def test_too_few_observations():
    with pytest.raises(ValueError):
        var.historical(np.zeros(10))


def test_kupiec_against_hand_calculation():
    hits = np.zeros(250, dtype=bool)
    hits[:8] = True
    lr, p = backtest.kupiec(hits, 0.99)
    p0, ph = 0.01, 8 / 250
    expected = -2 * ((242 * np.log(1 - p0) + 8 * np.log(p0)) - (242 * np.log(1 - ph) + 8 * np.log(ph)))
    assert lr == pytest.approx(expected)
    assert p == pytest.approx(stats.chi2.sf(expected, 1))


def test_kupiec_zero_exceptions_is_finite():
    lr, p = backtest.kupiec(np.zeros(250, dtype=bool), 0.99)
    assert np.isfinite(lr) and 0 < p < 1


def test_christoffersen_flags_clustering():
    n = 1000
    spread = np.zeros(n, dtype=bool)
    spread[::100] = True  # 10 isolated hits
    clustered = np.zeros(n, dtype=bool)
    clustered[500:510] = True  # same count, back to back
    _, p_spread = backtest.christoffersen(spread)
    _, p_clust = backtest.christoffersen(clustered)
    assert p_spread > 0.5
    assert p_clust < 1e-6
    # Kupiec cannot tell them apart.
    assert backtest.kupiec(spread, 0.99) == backtest.kupiec(clustered, 0.99)


@pytest.mark.parametrize(("exceptions", "zone"), [(4, "green"), (5, "yellow"), (9, "yellow"), (10, "red")])
def test_basel_traffic_light_250_days(exceptions, zone):
    assert backtest.traffic_light(exceptions, 250, 0.99) == zone
