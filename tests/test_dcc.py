import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from multiarch import DCC, UnivariateSpec, ewma_covariance, fit_univariate, simulate_dcc
from multiarch import _kernels as K

QBAR = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])
MARGINS = dict(mu=0.05, omega=[0.02, 0.03, 0.01], alpha=[0.08, 0.1, 0.06],
               beta=[0.9, 0.87, 0.92])


@pytest.fixture(scope="module")
def sim():
    y, _, _ = simulate_dcc(8000, Qbar=QBAR, a=0.04, b=0.93, seed=1, **MARGINS)
    return y


@pytest.fixture(scope="module")
def sp500():
    from arch.data import sp500
    return 100 * sp500.load()["Adj Close"].pct_change().dropna().to_numpy()


# ------------------------------------------------------------ univariate


@pytest.mark.parametrize("vol,o", [("garch", 0), ("gjr", 1)])
@pytest.mark.parametrize("dist", ["normal", "t"])
def test_univariate_matches_arch(sp500, vol, o, dist):
    from arch import arch_model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = arch_model(sp500, p=1, o=o, q=1, dist=dist).fit(disp="off")
    m = fit_univariate(sp500, UnivariateSpec(vol=vol, dist=dist))
    np.testing.assert_allclose(m.params, a.params.to_numpy(), atol=2e-3, rtol=1e-3)
    assert abs(m.loglik - a.loglikelihood) < 0.05


# ------------------------------------------------------------ kernels


def _reference_dcc_ll(z, a, b, nu=None):
    T, n = z.shape
    Qbar = z.T @ z / T
    Q = Qbar.copy()
    out = []
    for t in range(T):
        if t:
            Q = (1 - a - b) * Qbar + a * np.outer(z[t - 1], z[t - 1]) + b * Q
        d = 1 / np.sqrt(np.diag(Q))
        R = Q * np.outer(d, d)
        if nu is None:
            out.append(stats.multivariate_normal(cov=R).logpdf(z[t]))
        else:
            out.append(stats.multivariate_t(shape=R * (nu - 2) / nu, df=nu).logpdf(z[t]))
    return np.array(out)


@pytest.mark.parametrize("nu", [None, 7.0])
def test_dcc_kernel_matches_scipy(nu):
    rng = np.random.default_rng(0)
    z = rng.standard_normal((300, 4))
    Qbar = z.T @ z / len(z)
    got = K.dcc_loglik_obs(z, Qbar, np.zeros_like(Qbar), 0.05, 0.9, 0.0, nu or 0.0, Qbar)
    np.testing.assert_allclose(got, _reference_dcc_ll(z, 0.05, 0.9, nu), rtol=1e-10)


def test_composite_equals_full_for_two_assets():
    rng = np.random.default_rng(3)
    z = rng.standard_normal((500, 2))
    Qbar = z.T @ z / len(z)
    Nbar = np.minimum(z, 0).T @ np.minimum(z, 0) / len(z)
    pairs = np.array([[0, 1]])
    for nu in (0.0, 6.0):
        full = K.dcc_loglik_obs(z, Qbar, Nbar, 0.03, 0.9, 0.02, nu, Qbar).sum()
        comp = K.dcc_composite_loglik(z, Qbar, Nbar, pairs, 0.03, 0.9, 0.02, nu, Qbar)
        assert abs(full - comp) < 1e-8


# ------------------------------------------------------------ estimation


def test_dcc_recovers_parameters(sim):
    r = DCC(sim).fit(se="two-step")
    a, b = r.params["corr"]
    assert abs(a - 0.04) < 0.01 and abs(b - 0.93) < 0.02
    np.testing.assert_allclose(r.params.xs("alpha", level=1), MARGINS["alpha"], atol=0.02)
    assert r.converged
    assert np.all(r.std_err > 0)


def test_composite_close_to_full(sim):
    full = DCC(sim).fit(se=None).params["corr"]
    comp = DCC(sim, estimator="composite").fit(se=None).params["corr"]
    np.testing.assert_allclose(comp, full, atol=0.01)


def test_adcc_recovers_asymmetry():
    rng_q = QBAR
    z0 = np.random.default_rng(9).standard_normal((100000, 3)) @ np.linalg.cholesky(rng_q).T
    n0 = np.minimum(z0, 0)
    Nbar = n0.T @ n0 / len(n0)
    y, _, _ = simulate_dcc(8000, Qbar=QBAR, Nbar=Nbar, a=0.02, b=0.93, g=0.04,
                           seed=4, **MARGINS)
    r = DCC(y, model="adcc").fit(se=None)
    a, b, g = r.params["corr"]
    assert abs(g - 0.04) < 0.02 and abs(a - 0.02) < 0.015 and abs(b - 0.93) < 0.03


def test_t_recovers_nu():
    y, _, _ = simulate_dcc(8000, Qbar=QBAR, a=0.04, b=0.93, nu=7.0, seed=5, **MARGINS)
    r = DCC(y, dist="t").fit(se=None)
    assert abs(r.params[("corr", "nu")] - 7.0) < 1.5


def test_ccc_is_sample_correlation(sim):
    r = DCC(sim, model="ccc").fit(se="naive")
    z = r.std_resid.to_numpy()
    np.testing.assert_allclose(r.conditional_correlation[10], np.corrcoef(z.T), atol=1e-2)
    assert "corr" not in r.params.index.get_level_values(0)


def test_dcc_beats_ccc_in_likelihood(sim):
    assert DCC(sim).fit(se=None).loglik > DCC(sim, model="ccc").fit(se=None).loglik


# ------------------------------------------------------------ outputs


def test_shapes_and_dataframe_io(sim):
    df = pd.DataFrame(sim, columns=["A", "B", "C"],
                      index=pd.date_range("2000-01-01", periods=len(sim)))
    r = DCC(df).fit(se=None)
    T = len(df)
    assert r.conditional_covariance.shape == (T, 3, 3)
    assert list(r.conditional_volatility.columns) == ["A", "B", "C"]
    assert r.correlation("A", "C").index.equals(df.index)
    H = r.conditional_covariance
    assert np.all(np.linalg.eigvalsh(H) > 0)
    np.testing.assert_allclose(np.diagonal(H, axis1=1, axis2=2), r.sigma2)


def test_forecast(sim):
    r = DCC(sim).fit(se=None)
    f = r.forecast(500)
    np.testing.assert_allclose(f["correlation"][0], r._R[-1])
    d = 1 / np.sqrt(np.diag(r.Qbar))
    np.testing.assert_allclose(f["correlation"][-1], r.Qbar * np.outer(d, d), atol=1e-6)
    ups = np.array(r.uparams)
    uncond = ups[:, 1] / (1 - ups[:, 2] - ups[:, 3])
    np.testing.assert_allclose(f["variance"][-1], uncond, rtol=1e-3)
    assert f["covariance"].shape == (500, 3, 3)


def test_simulate_from_result(sim):
    r = DCC(sim).fit(se=None)
    out = r.simulate(1000, seed=0)
    assert out["returns"].shape == (1, 1000, 3) and np.all(out["sigma2"] > 0)
    end = r.simulate(5, n_paths=200, start="end", seed=0)
    # first simulated variance is the one-step forecast
    np.testing.assert_allclose(end["sigma2"][:, 0], np.tile(r.forecast(1)["variance"][0], (200, 1)))
    np.testing.assert_allclose(end["correlation"][0, 0], r.forecast(1)["correlation"][0])


def test_ewma():
    x = np.random.default_rng(1).standard_normal((50, 2))
    S = ewma_covariance(x, lam=0.9)
    manual = 0.9 * S[-2] + 0.1 * np.outer(x[-1], x[-1])
    np.testing.assert_allclose(S[-1], manual)
