"""Two-step CCC / DCC / aDCC estimation (Engle 2002; Cappiello, Engle &
Sheppard 2006), optionally as a Gaussian or Student-t copula (Patton 2006),
with composite likelihood for large cross-sections (Engle, Shephard &
Sheppard 2008) and two-step-corrected standard errors (Engle & Sheppard
2001)."""

import copy
import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from . import _kernels as K
from ._numdiff import gradient, hessian, jacobian
from .univariate import (
    UnivariateSpec,
    filter_series,
    fit_univariate,
    loglik_obs,
    rescale,
    robust_cov,
)

_PENALTY = 1e10
_EPS_U = 1e-12


def _targets(x):
    T = x.shape[0]
    Qbar = x.T @ x / T
    n = np.minimum(x, 0.0)
    Nbar = n.T @ n / T
    return Qbar, Nbar


def _normalise(Q):
    d = 1 / np.sqrt(np.diag(Q))
    return Q * np.outer(d, d)


def _asym_bound(Qbar, Nbar):
    """delta = max eigenvalue of Qbar^{-1/2} Nbar Qbar^{-1/2}; stationarity
    of aDCC needs a + b + delta * g < 1."""
    Li = np.linalg.inv(np.linalg.cholesky(Qbar))
    return float(np.linalg.eigvalsh(Li @ Nbar @ Li.T).max())


def _const_corr_ll_obs(x, R, nu):
    L = np.linalg.cholesky(R)
    y = np.linalg.solve(L, x.T)
    quad = np.sum(y * y, axis=0)
    logdet = 2 * np.log(np.diag(L)).sum()
    n = x.shape[1]
    if nu <= 0:
        return -0.5 * (n * np.log(2 * np.pi) + logdet + quad)
    c = math.lgamma(0.5 * (nu + n)) - math.lgamma(0.5 * nu) - 0.5 * n * math.log(math.pi * (nu - 2))
    return c - 0.5 * logdet - 0.5 * (nu + n) * np.log1p(quad / (nu - 2))


def _std_t_logpdf(x, nu):
    """Unit-variance Student-t log density, elementwise."""
    c = math.lgamma(0.5 * (nu + 1)) - math.lgamma(0.5 * nu) - 0.5 * math.log(math.pi * (nu - 2))
    return c - 0.5 * (nu + 1) * np.log1p(x * x / (nu - 2))


def _std_t_ppf(u, nu):
    return stats.t.ppf(u, nu) * math.sqrt((nu - 2) / nu)


def _std_t_cdf(x, nu):
    return stats.t.cdf(x * math.sqrt(nu / (nu - 2)), nu)


class DCC:
    """Two-step multivariate GARCH with dynamic conditional correlation.

    Parameters
    ----------
    returns : (T, N) array or DataFrame, in any units (percent or
        fractions; estimation is scale-free).
    model : "dcc", "adcc" or "ccc".
    vol : univariate variance model, "garch" or "gjr" (both (1,1)).
    mean : "constant" or "zero".
    dist : margin distribution, "normal" or "t" (unit-variance Student-t).
    corr_dist : distribution of the correlation stage without a copula,
        "normal" or "t" (multivariate t with its own shape `nu`). Defaults
        to `dist`; e.g. dist="normal", corr_dist="t" is tsmarch's and
        rmgarch's "mvt" with Gaussian QML margins.
    copula : None (standardised residuals enter the correlation stage
        directly), "normal" or "t". With a copula the margins are mapped
        to uniforms through their fitted CDFs and the correlation dynamics
        drive a Gaussian or Student-t copula, as in rmgarch's cgarch.
    estimator : "full" (exact likelihood) or "composite" (sum of pairwise
        bivariate likelihoods; use for large N).
    pairs : for the composite estimator, "all" or "adjacent" (N-1 pairs).
    convention : start-up conventions. "arch" (default): exponential
        backcast for the variances (as the `arch` package), Qbar = Z'Z/T and
        Q_1 = Qbar. "rmgarch": reproduce rmgarch's likelihood exactly
        (variances start at the sample mean of squared residuals,
        Qbar = cov(Z), rmgarch's pre-sample Q step and copula clipping).
        "tsmarch": match the tsmarch/tsgarch packages (sigma2_1 = omega +
        persistence * mean squared residual, Qbar = cov(Z), Q_1 = Qbar).
    """

    def __init__(self, returns, model="dcc", vol="garch", mean="constant",
                 dist="normal", corr_dist=None, copula=None, estimator="full",
                 pairs="all", convention="arch"):
        if model not in ("dcc", "adcc", "ccc"):
            raise ValueError("model must be 'dcc', 'adcc' or 'ccc'")
        if estimator not in ("full", "composite"):
            raise ValueError("estimator must be 'full' or 'composite'")
        if copula not in (None, "normal", "t"):
            raise ValueError("copula must be None, 'normal' or 't'")
        if isinstance(returns, pd.DataFrame):
            self.index = returns.index
            self.columns = [str(c) for c in returns.columns]
            y = returns.to_numpy(dtype=float)
        else:
            y = np.asarray(returns, dtype=float)
            self.index = pd.RangeIndex(y.shape[0])
            self.columns = [f"x{i}" for i in range(y.shape[1])]
        if y.ndim != 2 or y.shape[1] < 2:
            raise ValueError("returns must be (T, N) with N >= 2")
        if not np.isfinite(y).all():
            raise ValueError("returns contain NaN or inf")
        self.y = y
        self.model = model
        self.dist = dist
        self.copula = copula
        self.estimator = estimator
        if convention not in ("arch", "rmgarch", "tsmarch"):
            raise ValueError("convention must be 'arch', 'rmgarch' or 'tsmarch'")
        self.convention = convention
        self.uspec = UnivariateSpec(mean=mean, vol=vol, dist=dist,
                                    init={"arch": "arch", "rmgarch": "rugarch",
                                          "tsmarch": "tsgarch"}[convention])
        if corr_dist not in (None, "normal", "t"):
            raise ValueError("corr_dist must be None, 'normal' or 't'")
        if corr_dist is not None and copula is not None:
            raise ValueError("corr_dist is set by `copula` when a copula is used")
        self.corr_dist = corr_dist or dist
        self._mvt = copula == "t" or (copula is None and self.corr_dist == "t")
        n = y.shape[1]
        if pairs == "all":
            self.pairs = np.array(list(itertools.combinations(range(n), 2)))
        elif pairs == "adjacent":
            self.pairs = np.array([(i, i + 1) for i in range(n - 1)])
        else:
            raise ValueError("pairs must be 'all' or 'adjacent'")
        self._pairs_arg = pairs

    def _clone(self, y, index=None):
        m = copy.copy(self)
        m.y = np.asarray(y, float)
        m.index = pd.RangeIndex(m.y.shape[0]) if index is None else index
        return m

    def _targets(self, x):
        if self.convention in ("rmgarch", "tsmarch"):
            n = np.minimum(x, 0.0)
            return np.cov(x, rowvar=False), np.cov(n, rowvar=False)
        return _targets(x)

    def _q0(self, Qbar, Nbar, a, b, g):
        """Q for the first observation. rmgarch's C code feeds a pre-sample
        shock of ones into DCC and of zeros into copula-DCC."""
        if self.convention == "rmgarch":
            shock = 0.0 if self.copula else 1.0
            return (1 - a - b) * Qbar - g * Nbar + (a + g) * shock * np.ones_like(Qbar) + b * Qbar
        return Qbar

    # ------------------------------------------------------------ names

    @property
    def corr_names(self):
        names = {"dcc": ["a", "b"], "adcc": ["a", "b", "g"], "ccc": []}[self.model]
        return names + (["nu"] if self._mvt else [])

    def _unpack_corr(self, phi):
        a = b = g = nu = 0.0
        k = 0
        if self.model in ("dcc", "adcc"):
            a, b = phi[0], phi[1]
            k = 2
        if self.model == "adcc":
            g = phi[2]
            k = 3
        if self._mvt:
            nu = phi[k]
        return a, b, g, nu

    # ------------------------------------------------------------ stage-2 data

    def _std_resid(self, uparams, sigma2_0=None):
        cols = []
        for i, p in enumerate(uparams):
            s0 = None if sigma2_0 is None else sigma2_0[i]
            cols.append(filter_series(self.uspec, p, self.y[:, i], s0)[2])
        return np.column_stack(cols)

    def _stage2_input(self, uparams, z=None, sigma2_0=None):
        """Standardised residuals, or their PIT uniforms under a copula."""
        if z is None:
            z = self._std_resid(uparams, sigma2_0)
        if self.copula is None:
            return z
        u = np.empty_like(z)
        for i, p in enumerate(uparams):
            nu_i = self.uspec.unpack(p)[5]
            u[:, i] = _std_t_cdf(z[:, i], nu_i) if nu_i > 0 else stats.norm.cdf(z[:, i])
        if self.convention == "rmgarch":
            return np.clip(u, 2.220446049250313e-16, 0.99999)
        return np.clip(u, _EPS_U, 1 - _EPS_U)

    def _transform(self, e, nu):
        """-> (x, adj): variables driving the correlation recursion and the
        per-observation log-density correction (minus the copula's
        marginal densities; zero without a copula)."""
        if self.copula is None:
            return e, None
        if self.copula == "normal":
            x = stats.norm.ppf(e)
            return x, -stats.norm.logpdf(x)
        x = _std_t_ppf(e, nu)
        return x, -_std_t_logpdf(x, nu)

    def _ll_obs(self, phi, e, targets=None):
        """Per-observation stage-2 log density (full likelihood).
        `targets` = (Qbar, Nbar) to hold the intercept fixed (filtering new
        data); by default they are re-estimated from the data."""
        a, b, g, nu = self._unpack_corr(phi)
        x, adj = self._transform(e, nu)
        Qbar, Nbar = self._targets(x) if targets is None else targets
        if self.model == "ccc":
            ll = _const_corr_ll_obs(x, _normalise(Qbar), nu)
        else:
            ll = K.dcc_loglik_obs(x, Qbar, Nbar, a, b, g, nu, self._q0(Qbar, Nbar, a, b, g))
        return ll if adj is None else ll + adj.sum(axis=1)

    def _objective(self, phi, e):
        try:
            if self.estimator == "composite" and self.model != "ccc":
                a, b, g, nu = self._unpack_corr(phi)
                x, adj = self._transform(e, nu)
                Qbar, Nbar = self._targets(x)
                ll = K.dcc_composite_loglik(x, Qbar, Nbar, self.pairs, a, b, g, nu,
                                            self._q0(Qbar, Nbar, a, b, g))
                if adj is not None:
                    counts = np.bincount(self.pairs.ravel(), minlength=x.shape[1])
                    ll += float(adj.sum(axis=0) @ counts)
            else:
                ll = self._ll_obs(phi, e).sum()
        except np.linalg.LinAlgError:
            return _PENALTY
        return -ll if np.isfinite(ll) else _PENALTY

    def _fit_corr(self, e):
        names = self.corr_names
        if not names:
            return np.array([]), True
        if self.model == "adcc":
            x, _ = self._transform(e, 8.0)
            delta = _asym_bound(*self._targets(x))
        else:
            delta = 0.0
        bounds = {"a": (0.0, 0.5), "b": (0.0, 0.9999), "g": (0.0, 0.5), "nu": (2.05, 200.0)}
        bnds = [bounds[n] for n in names]
        cons = []
        if self.model != "ccc":
            def gap(phi):
                a, b, g, _ = self._unpack_corr(phi)
                return 0.9999 - a - b - delta * g
            cons.append({"type": "ineq", "fun": gap})

        obj = lambda phi: self._objective(phi, e)
        if self.model == "ccc":
            starts = [[nu0] for nu0 in (5.0, 8.0, 15.0)]
        else:
            starts = []
            for a, b in itertools.product((0.01, 0.03, 0.06), (0.85, 0.93, 0.97)):
                if a + b >= 0.999:
                    continue
                s = [a, b] + ([0.01] if self.model == "adcc" else [])
                starts.append(s + ([8.0] if self._mvt else []))
        x0 = min(starts, key=lambda s: obj(np.array(s)))
        res = minimize(obj, np.array(x0, float), method="SLSQP", bounds=bnds,
                       constraints=cons, options={"maxiter": 1000, "ftol": 1e-11})
        return res.x, bool(res.success)

    # ------------------------------------------------------------ fit

    def fit(self, se="two-step"):
        """Estimate the model.

        se : "two-step" (Engle-Sheppard covariance that accounts for the
            first-stage estimation error; full estimator only), "naive"
            (stage-wise robust covariances, ignoring the first stage),
            or None to skip standard errors (fastest).
        """
        if se not in ("two-step", "naive", None):
            raise ValueError("se must be 'two-step', 'naive' or None")
        if se == "two-step" and self.estimator == "composite":
            se = "naive"  # the corrected form is for the full likelihood
        # estimate on unit-variance data for a well-conditioned problem,
        # then map the margin parameters (and their covariance) back
        scale = self.y.std(axis=0)
        inner = self._clone(self.y / scale, self.index)
        uparams, phi, cov, ok = inner._estimate(se)
        Js = []
        for i, p in enumerate(uparams):
            uparams[i], J = rescale(self.uspec, p, scale[i])
            Js.append(J)
        J = np.concatenate(Js + [np.ones(phi.size)])
        return DCCResult(self, uparams, phi, cov * np.outer(J, J), converged=ok)

    def _estimate(self, se):
        ufits = [
            fit_univariate(self.y[:, i], self.uspec, compute_cov=se is not None)
            for i in range(self.y.shape[1])
        ]
        uparams = [f.params for f in ufits]
        e = self._stage2_input(uparams)
        phi, ok = self._fit_corr(e)

        k1 = len(self.uspec.names)
        n = self.y.shape[1]
        npar = k1 * n + phi.size
        cov = np.full((npar, npar), np.nan)
        if se == "naive":
            for i, f in enumerate(ufits):
                cov[i * k1:(i + 1) * k1, i * k1:(i + 1) * k1] = f.cov
            if phi.size:
                cov[k1 * n:, k1 * n:] = self._naive_corr_cov(phi, e)
        elif se == "two-step":
            cov = self._two_step_cov(uparams, phi)
        return uparams, phi, cov, ok and all(f.converged for f in ufits)

    def fix(self, params):
        """Build a result from given parameters without estimation.
        `params` is a vector in the order of `DCCResult.params`."""
        params = np.asarray(params, float)
        k1 = len(self.uspec.names)
        n = self.y.shape[1]
        uparams = [params[i * k1:(i + 1) * k1] for i in range(n)]
        phi = params[k1 * n:]
        return DCCResult(self, uparams, phi, np.full((params.size,) * 2, np.nan))

    # ------------------------------------------------------------ inference

    def _naive_corr_cov(self, phi, e):
        if self.estimator == "composite":
            # Hessian-only for the composite likelihood (no per-obs scores)
            H = hessian(lambda q: -self._objective(q, e), phi)
            return np.linalg.inv(-H)
        return robust_cov(lambda q: self._ll_obs(q, e), phi)

    def _two_step_cov(self, uparams, phi):
        """Engle & Sheppard (2001) / Newey-McFadden two-step GMM covariance.

        Stacked moments g_t = [s_1t(theta_1); ...; s_Nt(theta_N); s_t(phi | theta)].
        A is block lower-triangular: univariate Hessians on the diagonal,
        the correlation Hessian bottom-right and the cross derivative
        d s(phi) / d theta in the last block row. Cov = A^{-1} B A^{-T} / T.
        The correlation targets are re-estimated inside every evaluation,
        so their dependence on theta is part of the cross derivative.
        """
        n = self.y.shape[1]
        k1 = len(self.uspec.names)
        T = self.y.shape[0]
        theta = np.concatenate(uparams)
        kc = phi.size
        p = theta.size + kc

        def corr_ll_obs(theta_vec, phi_vec):
            return self._ll_obs(phi_vec, self._stage2_input(theta_vec.reshape(n, k1)))

        scores = np.empty((T, p))
        A = np.zeros((p, p))
        for i in range(n):
            sl = slice(i * k1, (i + 1) * k1)
            f = lambda q, i=i: loglik_obs(self.uspec, q, self.y[:, i])
            scores[:, sl] = jacobian(f, uparams[i])
            A[sl, sl] = hessian(lambda q: f(q).sum(), uparams[i]) / T
        if kc:
            scores[:, theta.size:] = jacobian(lambda q: corr_ll_obs(theta, q), phi)
            A[theta.size:, theta.size:] = hessian(
                lambda q: corr_ll_obs(theta, q).sum(), phi) / T
            g2 = lambda t: gradient(lambda q: corr_ll_obs(t, q).sum(), phi)
            A[theta.size:, :theta.size] = jacobian(g2, theta) / T
        B = scores.T @ scores / T
        Ainv = np.linalg.inv(A)
        return Ainv @ B @ Ainv.T / T


@dataclass
class DCCResult:
    """Fitted (or fixed-parameter) DCC model. Series arrays are aligned with
    the input: row t of `conditional_covariance` is the forecast of the
    covariance of returns at t made with information up to t-1."""

    model: DCC
    uparams: list
    corr_params: np.ndarray
    cov: np.ndarray = field(repr=False)
    converged: bool = True
    targets: tuple = field(default=None, repr=False)
    sigma2_0: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        m = self.model
        self.uparams = [np.asarray(p, float) for p in self.uparams]
        self.corr_params = np.asarray(self.corr_params, float)
        a, b, g, nu = m._unpack_corr(self.corr_params)
        fs = []
        for i, p in enumerate(self.uparams):
            s0 = None if self.sigma2_0 is None else self.sigma2_0[i]
            fs.append(filter_series(m.uspec, p, m.y[:, i], s0))
        self.resid = np.column_stack([f[0] for f in fs])
        self.sigma2 = np.column_stack([f[1] for f in fs])
        self.std_resid_arr = np.column_stack([f[2] for f in fs])
        self._e = m._stage2_input(self.uparams, self.std_resid_arr)
        self._x, self._adj = m._transform(self._e, nu)
        if self.targets is None:
            self.targets = m._targets(self._x)
        self.Qbar, self.Nbar = self.targets
        self.Rbar = _normalise(self.Qbar)
        if m.model == "ccc":
            T = m.y.shape[0]
            self._R = np.broadcast_to(self.Rbar, (T + 1,) + self.Rbar.shape)
            self._Q = self._R
        else:
            self._Q, self._R = K.dcc_filter(self._x, self.Qbar, self.Nbar, a, b, g,
                                            m._q0(self.Qbar, self.Nbar, a, b, g))

    # ------------------------------------------------------------ tables

    @property
    def params(self):
        m = self.model
        idx, vals = [], []
        for c, p in zip(m.columns, self.uparams):
            idx += [(c, nm) for nm in m.uspec.names]
            vals += list(p)
        idx += [("corr", nm) for nm in m.corr_names]
        vals += list(self.corr_params)
        return pd.Series(vals, index=pd.MultiIndex.from_tuples(idx), name="params")

    @property
    def std_err(self):
        return pd.Series(np.sqrt(np.diag(self.cov)), index=self.params.index, name="std_err")

    @property
    def pvalues(self):
        t = self.params / self.std_err
        return pd.Series(2 * stats.norm.sf(np.abs(t)), index=t.index, name="pvalue")

    @property
    def nu(self):
        """Shape of the multivariate t / t copula (None if Gaussian)."""
        return self.model._unpack_corr(self.corr_params)[3] or None

    @property
    def loglik_obs(self):
        """Per-observation joint log-likelihood of the returns."""
        m = self.model
        a, b, g, nu = m._unpack_corr(self.corr_params)
        if m.model == "ccc":
            ll = _const_corr_ll_obs(self._x, self.Rbar, nu)
        else:
            ll = K.dcc_loglik_obs(self._x, self.Qbar, self.Nbar, a, b, g, nu,
                                  m._q0(self.Qbar, self.Nbar, a, b, g))
        if self._adj is not None:
            # copula: add back the fitted margins' own log densities
            return ll + self._adj.sum(axis=1) + self._margin_ll().sum(axis=1)
        return ll - 0.5 * np.log(self.sigma2).sum(axis=1)

    def _margin_ll(self):
        m = self.model
        out = []
        for i, p in enumerate(self.uparams):
            nu_i = m.uspec.unpack(p)[5]
            z, s2 = self.std_resid_arr[:, i], self.sigma2[:, i]
            if nu_i > 0:
                out.append(_std_t_logpdf(z, nu_i) - 0.5 * np.log(s2))
            else:
                out.append(stats.norm.logpdf(z) - 0.5 * np.log(s2))
        return np.column_stack(out)

    @property
    def loglik(self):
        return float(self.loglik_obs.sum())

    @property
    def nobs(self):
        return self.model.y.shape[0]

    @property
    def aic(self):
        return -2 * self.loglik + 2 * self.params.size

    @property
    def bic(self):
        return -2 * self.loglik + np.log(self.nobs) * self.params.size

    def summary(self):
        p, se = self.params, self.std_err
        df = pd.DataFrame({"coef": p, "std_err": se, "t": p / se, "p": self.pvalues})
        m = self.model
        cop = f"  copula={m.copula}" if m.copula else ""
        head = (f"{m.model.upper()}-{m.uspec.vol.upper()}  dist={m.dist}{cop}  "
                f"estimator={m.estimator}  T={self.nobs}  N={len(m.columns)}\n"
                f"loglik={self.loglik:.3f}  AIC={self.aic:.3f}  BIC={self.bic:.3f}  "
                f"converged={self.converged}\n")
        return head + df.to_string(float_format=lambda x: f"{x: .5g}")

    def __repr__(self):
        return self.summary()

    # ------------------------------------------------------------ series

    @property
    def std_resid(self):
        return pd.DataFrame(self.std_resid_arr, index=self.model.index, columns=self.model.columns)

    @property
    def conditional_volatility(self):
        return pd.DataFrame(np.sqrt(self.sigma2), index=self.model.index,
                            columns=self.model.columns)

    @property
    def conditional_correlation(self):
        """(T, N, N) array of R_t (the copula correlation under a copula)."""
        return np.asarray(self._R[:-1])

    @property
    def conditional_covariance(self):
        """(T, N, N) array of H_t = D_t R_t D_t."""
        s = np.sqrt(self.sigma2)
        return self.conditional_correlation * s[:, :, None] * s[:, None, :]

    @property
    def conditional_mean(self):
        mu = np.array([self.model.uspec.unpack(p)[0] for p in self.uparams])
        return np.broadcast_to(mu, self.sigma2.shape)

    def correlation(self, i, j):
        """Conditional correlation path between two assets as a Series."""
        m = self.model
        i = m.columns.index(i) if isinstance(i, str) else i
        j = m.columns.index(j) if isinstance(j, str) else j
        return pd.Series(self.conditional_correlation[:, i, j], index=m.index,
                         name=f"rho({m.columns[i]},{m.columns[j]})")

    def portfolio_variance(self, weights):
        """In-sample one-step-ahead portfolio variance w' H_t w."""
        w = np.asarray(weights, float)
        v = np.einsum("i,tij,j->t", w, self.conditional_covariance, w)
        return pd.Series(v, index=self.model.index, name="portfolio_variance")

    def value_at_risk_path(self, weights, level=0.01, value=1.0):
        """In-sample one-step-ahead VaR and ES of a portfolio (positive
        numbers are losses), for backtesting. Analytic, so only for models
        without a copula. Returns a DataFrame with columns VaR, ES."""
        if self.model.copula is not None:
            raise ValueError("analytic VaR is not available under a copula")
        w = np.asarray(weights, float)
        mean = self.conditional_mean @ w
        sd = np.sqrt(self.portfolio_variance(w).to_numpy())
        var, es = _analytic_var_es(mean, sd, level, self.nu)
        return pd.DataFrame({"VaR": value * var, "ES": value * es}, index=self.model.index)

    # ------------------------------------------------------------ forecast

    def _state_forecast_variance(self, horizon):
        m = self.model
        n = len(m.columns)
        var = np.empty((horizon, n))
        for i, p in enumerate(self.uparams):
            mu, omega, alpha, gamma, beta, _ = m.uspec.unpack(p)
            e = self.resid[-1, i]
            s1 = omega + alpha * e * e + beta * self.sigma2[-1, i] + (gamma * e * e if e < 0 else 0.0)
            pers = alpha + 0.5 * gamma + beta
            var[0, i] = s1
            for h in range(1, horizon):
                var[h, i] = omega + pers * var[h - 1, i]
        return var

    def forecast(self, horizon=1):
        """Multi-step forecasts from the end of the sample.

        Variances use the closed-form GJR-GARCH recursion. Correlations use
        the Engle & Sheppard (2001) approximation
        R_{T+h} = (1 - (a+b)^{h-1}) Rbar + (a+b)^{h-1} R_{T+1}, as in rmgarch.
        Returns dict with "mean" (h, N), "variance" (h, N),
        "correlation" (h, N, N) and "covariance" (h, N, N).
        """
        var = self._state_forecast_variance(horizon)
        mean = np.tile(self.conditional_mean[0], (horizon, 1))
        a, b, _, _ = self.model._unpack_corr(self.corr_params)
        R1 = np.asarray(self._R[-1])
        w = (a + b) ** np.arange(horizon)
        corr = (1 - w)[:, None, None] * self.Rbar + w[:, None, None] * R1
        s = np.sqrt(var)
        cov = corr * s[:, :, None] * s[:, None, :]
        return {"mean": mean, "variance": var, "correlation": corr, "covariance": cov}

    def simulate(self, nobs, n_paths=1, start="unconditional", burn=500, seed=None):
        """Simulate return paths from the fitted model.

        start : "unconditional" (stationary start with `burn` discarded
            draws) or "end" (continue from the last observation, so the
            first simulated period is T+1).
        Returns dict with "returns" and "sigma2", each (n_paths, nobs, N),
        and "correlation" (n_paths, nobs, N, N).
        """
        rng = np.random.default_rng(seed)
        m = self.model
        n = len(m.columns)
        ups = np.array([m.uspec.unpack(p) for p in self.uparams])
        mu, omega, alpha, gamma, beta, nus = ups.T
        a, b, g, nu = m._unpack_corr(self.corr_params)
        if start == "end":
            Q = np.broadcast_to(np.asarray(self._Q[-1]), (n_paths, n, n)).copy()
            s2 = np.tile(self._state_forecast_variance(1)[0], (n_paths, 1))
            burn = 0
        elif start == "unconditional":
            Q = np.broadcast_to(self.Qbar, (n_paths, n, n)).copy()
            s2 = np.tile(omega / (1 - alpha - 0.5 * gamma - beta), (n_paths, 1))
        else:
            raise ValueError("start must be 'unconditional' or 'end'")
        C = (1 - a - b) * self.Qbar - g * self.Nbar
        out_r = np.empty((n_paths, nobs, n))
        out_s = np.empty((n_paths, nobs, n))
        out_R = np.empty((n_paths, nobs, n, n))
        for t in range(burn + nobs):
            d = 1 / np.sqrt(np.diagonal(Q, axis1=1, axis2=2))
            R = Q * d[:, :, None] * d[:, None, :]
            L = np.linalg.cholesky(R)
            x = np.einsum("pij,pj->pi", L, _draw(rng, n_paths, n, nu))
            z = m._copula_to_margins(x, nu, nus) if m.copula else x
            eps = np.sqrt(s2) * z
            if t >= burn:
                k = t - burn
                out_r[:, k] = mu + eps
                out_s[:, k] = s2
                out_R[:, k] = R
            nx = np.minimum(x, 0.0)
            Q = (C + a * x[:, :, None] * x[:, None, :] + g * nx[:, :, None] * nx[:, None, :]
                 + b * Q)
            s2 = omega + (alpha + gamma * (eps < 0)) * eps ** 2 + beta * s2
        return {"returns": out_r, "sigma2": out_s, "correlation": out_R}

    def value_at_risk(self, weights, level=0.01, horizon=1, method=None,
                      n_sims=100_000, value=1.0, seed=None):
        """VaR and ES of a portfolio over the next `horizon` periods
        (cumulative return), as positive losses scaled by `value`.

        method : "analytic" (one period, no copula), "simulation", or None
            to choose analytic when possible.
        """
        w = np.asarray(weights, float)
        if method is None:
            method = "analytic" if horizon == 1 and self.model.copula is None else "simulation"
        if method == "analytic":
            if horizon != 1 or self.model.copula is not None:
                raise ValueError("analytic VaR needs horizon=1 and no copula")
            f = self.forecast(1)
            mean = f["mean"][0] @ w
            sd = math.sqrt(w @ f["covariance"][0] @ w)
            var, es = _analytic_var_es(mean, sd, level, self.nu)
        else:
            sims = self.simulate(horizon, n_paths=n_sims, start="end", seed=seed)
            pr = (sims["returns"] @ w).sum(axis=1)
            q = np.quantile(pr, level)
            var, es = -q, -pr[pr <= q].mean()
        return {"VaR": float(value * var), "ES": float(value * es)}

    # ------------------------------------------------------------ new data

    def apply(self, returns, keep_targets=True, sigma2_0=None):
        """Filter new (or extended) data with the fitted parameters.

        keep_targets : keep the estimated Qbar/Nbar (default) instead of
            re-computing them from the new data.
        sigma2_0 : optional first-period variances; default re-initialises
            from the new data with the model's `init` rule.
        """
        if isinstance(returns, pd.DataFrame):
            m = self.model._clone(returns.to_numpy(float), returns.index)
        else:
            m = self.model._clone(returns)
        return DCCResult(m, self.uparams, self.corr_params, self.cov,
                         self.converged, self.targets if keep_targets else None,
                         sigma2_0)


def _copula_to_margins(self, x, nu, nus):
    """Map copula draws x to standardised margin shocks z (per column)."""
    u = _std_t_cdf(x, nu) if self.copula == "t" else stats.norm.cdf(x)
    u = np.clip(u, _EPS_U, 1 - _EPS_U)
    z = np.empty_like(x)
    for i, nu_i in enumerate(nus):
        z[:, i] = _std_t_ppf(u[:, i], nu_i) if nu_i > 0 else stats.norm.ppf(u[:, i])
    return z


DCC._copula_to_margins = _copula_to_margins


def _draw(rng, n_paths, n, nu):
    """Uncorrelated unit-variance shocks; multivariate t if nu > 0."""
    x = rng.standard_normal((n_paths, n))
    if nu and nu > 0:
        w = rng.chisquare(nu, size=n_paths) / nu
        x = x / np.sqrt(w)[:, None] * math.sqrt((nu - 2) / nu)
    return x


def _analytic_var_es(mean, sd, level, nu=None):
    """VaR/ES (positive losses) of mean + sd * Z, Z normal or unit-variance t."""
    if not nu:
        q = stats.norm.ppf(level)
        return -(mean + sd * q), -(mean - sd * stats.norm.pdf(q) / level)
    k = math.sqrt((nu - 2) / nu)
    q = stats.t.ppf(level, nu)
    es_std = (nu + q * q) / (nu - 1) * stats.t.pdf(q, nu) / level
    return -(mean + sd * k * q), -(mean - sd * k * es_std)
