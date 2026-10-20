import numpy as np
import pandas as pd
import pytest
from scipy import stats

import multiarch as ma
from multiarch import DCC, GOGARCH, simulate_dcc

QBAR = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])
MARGINS = dict(mu=0.05, omega=[0.02, 0.03, 0.01], alpha=[0.08, 0.1, 0.06],
               beta=[0.9, 0.87, 0.92])


@pytest.fixture(scope="module")
def sim():
    return simulate_dcc(4000, Qbar=QBAR, a=0.04, b=0.93, seed=11, **MARGINS)[0]


@pytest.fixture(scope="module")
def fit(sim):
    return DCC(sim).fit(se=None)


# ------------------------------------------------------------ scale, fix, apply


def test_scale_invariance(sim):
    a = DCC(sim).fit()
    b = DCC(sim / 100).fit()
    np.testing.assert_allclose(a.params["corr"], b.params["corr"], atol=1e-6)
    np.testing.assert_allclose(a.std_err["corr"], b.std_err["corr"], rtol=1e-3)
    om = a.params.xs("omega", level=1).to_numpy()
    np.testing.assert_allclose(b.params.xs("omega", level=1).to_numpy() * 1e4, om, rtol=1e-4)
    assert abs(a.loglik - (b.loglik - sim.size * np.log(100))) < 1e-4


def test_fix_and_apply_roundtrip(sim, fit):
    assert abs(DCC(sim).fix(fit.params.to_numpy()).loglik - fit.loglik) < 1e-8
    ap = fit.apply(sim)
    assert abs(ap.loglik - fit.loglik) < 1e-8
    # filtering a prefix gives the same path as the full filter on that prefix
    pre = fit.apply(sim[:500], sigma2_0=fit.sigma2[0])
    np.testing.assert_allclose(pre.conditional_covariance, fit.conditional_covariance[:500])


# ------------------------------------------------------------ copulas & distributions


def test_gaussian_copula_with_normal_margins_equals_dcc(sim):
    """A Gaussian copula with Gaussian margins is the Gaussian DCC model."""
    m1 = DCC(sim)
    m2 = DCC(sim, copula="normal")
    p = m1.fit(se=None).params.to_numpy()
    assert abs(m1.fix(p).loglik - m2.fix(p).loglik) < 1e-6


def test_t_copula_density_matches_scipy():
    rng = np.random.default_rng(2)
    y = rng.standard_normal((300, 2))
    m = DCC(y, model="ccc", dist="normal", copula="t")
    p = np.array([0.0, 0.1, 0.1, 0.8] * 2 + [6.0])
    r = m.fix(p)
    nu = 6.0
    u = stats.norm.cdf(r.std_resid_arr)
    x = stats.t.ppf(u, nu)
    R = np.corrcoef(x, rowvar=False) * 0 + r.Rbar  # same R as the model
    cop = stats.multivariate_t(shape=R, df=nu).logpdf(x) - stats.t.logpdf(x, nu).sum(1)
    margins = (stats.norm.logpdf(r.std_resid_arr) - 0.5 * np.log(r.sigma2)).sum(1)
    np.testing.assert_allclose(r.loglik_obs, cop + margins, rtol=1e-9)


def test_copula_fit_and_simulation(sim):
    r = DCC(sim, dist="t", copula="t").fit(se="two-step")
    assert r.nu > 10  # data are Gaussian
    assert np.all(np.isfinite(r.std_err))
    out = r.simulate(50, n_paths=100, start="end", seed=0)
    assert np.all(np.isfinite(out["returns"]))


def test_corr_dist_t_with_normal_margins():
    y = simulate_dcc(5000, Qbar=QBAR, a=0.04, b=0.93, nu=6.0, seed=3, **MARGINS)[0]
    r = DCC(y, corr_dist="t").fit(se=None)
    assert "nu" not in r.params.drop("corr", level=0).index.get_level_values(1)
    assert abs(r.nu - 6.0) < 1.5


@pytest.mark.parametrize("convention", ["arch", "rmgarch", "tsmarch"])
@pytest.mark.parametrize("model", ["dcc", "adcc", "ccc"])
def test_conventions_run(sim, convention, model):
    r = DCC(sim, model=model, convention=convention).fit(se="naive")
    assert r.converged and np.isfinite(r.loglik)


def test_composite_with_copula(sim):
    r = DCC(sim, estimator="composite", copula="t", dist="t").fit(se=None)
    assert np.isfinite(r.loglik)


# ------------------------------------------------------------ risk


@pytest.mark.parametrize("kw", [dict(), dict(corr_dist="t")])
def test_var_analytic_matches_simulation(sim, kw):
    r = DCC(sim, **kw).fit(se=None)
    w = np.array([0.5, 0.3, 0.2])
    a = r.value_at_risk(w, 0.01)
    s = r.value_at_risk(w, 0.01, method="simulation", n_sims=400_000, seed=1)
    assert abs(a["VaR"] / s["VaR"] - 1) < 0.02
    assert abs(a["ES"] / s["ES"] - 1) < 0.03


def test_var_path_and_backtest(sim, fit):
    w = np.array([1 / 3] * 3)
    v = fit.value_at_risk_path(w, 0.05)
    bt = ma.var_backtest(sim @ w, v["VaR"], 0.05)
    assert 0.7 < bt["violation_ratio"] < 1.3  # correctly specified model
    assert bt["kupiec_pvalue"] > 0.01


def test_var_backtest_statistics():
    r = np.array([-2.0] * 10 + [0.0] * 90)
    out = ma.var_backtest(r, np.ones(100), level=0.05)
    assert out["violations"] == 10
    p, x, T = 0.05, 10, 100
    lr = -2 * ((T - x) * np.log(1 - p) + x * np.log(p)
               - (T - x) * np.log(1 - x / T) - x * np.log(x / T))
    assert abs(out["kupiec_stat"] - lr) < 1e-10
    assert out["independence_pvalue"] < 1e-6  # all hits clustered


def test_rolling_forecast_has_no_lookahead(sim):
    df = pd.DataFrame(sim[:1200])
    rf = ma.rolling_forecast(df, window=1000, refit_every=50)
    changed = df.copy()
    changed.iloc[1100:] *= 5  # alter the future
    rf2 = ma.rolling_forecast(changed, window=1000, refit_every=50)
    np.testing.assert_allclose(rf.covariance[:101], rf2.covariance[:101])
    assert rf.covariance.shape == (200, 3, 3)
    assert rf.value_at_risk([1, 0, 0]).shape == (200, 2)


# ------------------------------------------------------------ diagnostics


def test_constant_correlation_test_size_and_power():
    ccc = simulate_dcc(3000, Qbar=QBAR, a=0.0, b=0.0, seed=5, **MARGINS)[0]
    dcc = simulate_dcc(3000, Qbar=QBAR, a=0.08, b=0.9, seed=5, **MARGINS)[0]
    assert ma.constant_correlation_test(ccc)["pvalue"] > 0.01
    assert ma.constant_correlation_test(dcc)["pvalue"] < 0.001


def test_hosking():
    x = np.random.default_rng(0).standard_normal((2000, 3))
    assert ma.hosking_test(x, lags=10)["pvalue"] > 0.01
    ar = np.zeros_like(x)
    for t in range(1, len(x)):
        ar[t] = 0.3 * ar[t - 1] + x[t]
    assert ma.hosking_test(ar, lags=10)["pvalue"] < 1e-6


# ------------------------------------------------------------ GO-GARCH


def test_gogarch_recovers_structure():
    rng = np.random.default_rng(7)
    T, n = 6000, 3
    params = [(0.05, 0.10, 0.85), (0.02, 0.05, 0.93), (0.1, 0.15, 0.75)]
    f = np.empty((T, n))
    for i, (om, al, be) in enumerate(params):
        s2 = om / (1 - al - be)
        for t in range(T):
            e = np.sqrt(s2) * rng.standard_normal()
            f[t, i] = e
            s2 = om + al * e * e + be * s2
    f /= f.std(0)
    A = np.array([[1.0, 0.5, 0.2], [0.3, 1.0, 0.4], [0.1, 0.2, 1.0]])
    y = f @ A.T
    g = GOGARCH(y).fit()
    M = np.linalg.inv(g.A) @ A  # should be a signed permutation
    assert np.allclose(np.sort(np.abs(M).max(axis=1)), 1, atol=0.05)
    fc = g.forecast(3)
    assert fc["covariance"].shape == (3, 3, 3)
    v = g.value_at_risk([1 / 3] * 3)
    s = g.value_at_risk([1 / 3] * 3, method="simulation", n_sims=200_000, seed=0)
    assert abs(v["VaR"] / s["VaR"] - 1) < 0.02


def test_gogarch_unmixing_fixed():
    y = simulate_dcc(1500, Qbar=QBAR, a=0.03, b=0.95, seed=8, **MARGINS)[0]
    g = GOGARCH(y).fit()
    g2 = GOGARCH(y).fit(unmixing=np.linalg.inv(g.A))
    assert abs(g.loglik - g2.loglik) < 1e-6
