"""GO-GARCH (van der Weide 2002) with the ICA rotation of Broda & Paolella
(2009) / rmgarch: returns are linear combinations of independent factors,
each following its own univariate GARCH."""

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .dcc import _analytic_var_es, _normalise, _std_t_ppf
from .univariate import UnivariateSpec, filter_series, fit_univariate


def _sym_decorrelate(W):
    s, u = np.linalg.eigh(W @ W.T)
    return u @ np.diag(1 / np.sqrt(s)) @ u.T @ W


def fastica(X, seed=0, max_iter=2000, tol=1e-10):
    """Symmetric FastICA with the log-cosh contrast on whitened data X
    (N, T). Returns the orthogonal unmixing matrix W (factors = W @ X)."""
    n, T = X.shape
    rng = np.random.default_rng(seed)
    W = _sym_decorrelate(rng.standard_normal((n, n)))
    for _ in range(max_iter):
        G = np.tanh(W @ X)
        W1 = _sym_decorrelate(G @ X.T / T - np.diag((1 - G * G).mean(axis=1)) @ W)
        lim = np.max(np.abs(np.abs(np.einsum("ij,ij->i", W1, W)) - 1))
        W = W1
        if lim < tol:
            break
    return W


class GOGARCH:
    """Generalized orthogonal GARCH.

    eps_t = A f_t, with f_it independent GARCH processes of unit
    unconditional variance and A = P Lambda^{1/2} U' from the PCA of the
    sample covariance and an ICA rotation U.

    Parameters
    ----------
    returns : (T, N) array or DataFrame.
    vol : factor variance model, "garch" or "gjr".
    dist : factor distribution, "normal" or "t".
    mean : "constant" (sample mean, as rmgarch) or "zero".
    seed : seed for the ICA start (ICA is identified only up to factor
        order and sign).
    convention : factor variance start-up, "arch" or "rmgarch" (see DCC).
    """

    def __init__(self, returns, vol="garch", dist="normal", mean="constant", seed=0,
                 convention="arch"):
        if isinstance(returns, pd.DataFrame):
            self.index = returns.index
            self.columns = [str(c) for c in returns.columns]
            y = returns.to_numpy(float)
        else:
            y = np.asarray(returns, float)
            self.index = pd.RangeIndex(y.shape[0])
            self.columns = [f"x{i}" for i in range(y.shape[1])]
        if y.ndim != 2 or y.shape[1] < 2:
            raise ValueError("returns must be (T, N) with N >= 2")
        self.y = y
        self.mean = mean
        self.seed = seed
        init = {"arch": "arch", "rmgarch": "rugarch", "tsmarch": "tsgarch"}[convention]
        self.fspec = UnivariateSpec(mean="zero", vol=vol, dist=dist, init=init)

    def fit(self, compute_cov=True, unmixing=None):
        """Estimate the rotation by ICA and the factor GARCH models.
        `unmixing` (N, N) fixes the map from demeaned returns to factors
        (f_t = unmixing @ eps_t) instead of estimating it."""
        y = self.y
        mu = y.mean(axis=0) if self.mean == "constant" else np.zeros(y.shape[1])
        eps = y - mu
        if unmixing is None:
            Sigma = eps.T @ eps / len(eps)
            lam, P = np.linalg.eigh(Sigma)
            K = np.diag(lam ** -0.5) @ P.T          # whitening
            W = fastica(K @ eps.T, seed=self.seed)
            A = P @ np.diag(lam ** 0.5) @ W.T       # mixing: eps = A f
        else:
            A = np.linalg.inv(np.asarray(unmixing, float))
        f = eps @ np.linalg.inv(A).T
        fits = [fit_univariate(f[:, i], self.fspec, compute_cov=compute_cov)
                for i in range(f.shape[1])]
        return GOGARCHResult(self, mu, A, f, fits)


@dataclass
class GOGARCHResult:
    model: GOGARCH
    mu: np.ndarray
    A: np.ndarray
    factors: np.ndarray = field(repr=False)
    fits: list = field(repr=False)

    def __post_init__(self):
        fs = [filter_series(self.model.fspec, ft.params, self.factors[:, i])
              for i, ft in enumerate(self.fits)]
        self.factor_sigma2 = np.column_stack([s[1] for s in fs])

    @property
    def params(self):
        idx, vals = [], []
        for i, ft in enumerate(self.fits):
            idx += [(f"f{i}", nm) for nm in self.model.fspec.names]
            vals += list(ft.params)
        return pd.Series(vals, index=pd.MultiIndex.from_tuples(idx), name="params")

    @property
    def std_err(self):
        return pd.Series(np.concatenate([ft.std_err for ft in self.fits]),
                         index=self.params.index, name="std_err")

    @property
    def loglik(self):
        """sum of factor log-likelihoods minus T log|det A|."""
        T = self.factors.shape[0]
        return float(sum(ft.loglik for ft in self.fits)
                     - T * np.log(abs(np.linalg.det(self.A))))

    @property
    def nobs(self):
        return self.factors.shape[0]

    @property
    def aic(self):
        k = self.params.size + self.A.size
        return -2 * self.loglik + 2 * k

    def summary(self):
        p, se = self.params, self.std_err
        df = pd.DataFrame({"coef": p, "std_err": se, "t": p / se})
        head = (f"GO-GARCH ({self.model.fspec.vol}, {self.model.fspec.dist})  "
                f"T={self.nobs}  N={len(self.model.columns)}  loglik={self.loglik:.3f}\n")
        return head + df.to_string(float_format=lambda x: f"{x: .5g}")

    def __repr__(self):
        return self.summary()

    @property
    def conditional_covariance(self):
        """(T, N, N) array of H_t = A diag(h_t) A'."""
        return np.einsum("ik,tk,jk->tij", self.A, self.factor_sigma2, self.A)

    @property
    def conditional_correlation(self):
        H = self.conditional_covariance
        d = 1 / np.sqrt(np.diagonal(H, axis1=1, axis2=2))
        return H * d[:, :, None] * d[:, None, :]

    @property
    def conditional_volatility(self):
        H = self.conditional_covariance
        return pd.DataFrame(np.sqrt(np.diagonal(H, axis1=1, axis2=2)),
                            index=self.model.index, columns=self.model.columns)

    def _factor_forecast(self, horizon):
        out = np.empty((horizon, len(self.fits)))
        for i, ft in enumerate(self.fits):
            _, omega, alpha, gamma, beta, _ = self.model.fspec.unpack(ft.params)
            e = self.factors[-1, i]
            s = omega + alpha * e * e + beta * self.factor_sigma2[-1, i] + (gamma * e * e if e < 0 else 0)
            pers = alpha + 0.5 * gamma + beta
            for h in range(horizon):
                out[h, i] = s
                s = omega + pers * s
        return out

    def forecast(self, horizon=1):
        h = self._factor_forecast(horizon)
        cov = np.einsum("ik,tk,jk->tij", self.A, h, self.A)
        return {"mean": np.tile(self.mu, (horizon, 1)), "covariance": cov,
                "correlation": np.array([_normalise(c) for c in cov]),
                "variance": np.diagonal(cov, axis1=1, axis2=2).copy()}

    def simulate(self, nobs, n_paths=1, start="end", burn=500, seed=None):
        """Simulate return paths; start "end" continues from the sample."""
        rng = np.random.default_rng(seed)
        ups = np.array([self.model.fspec.unpack(ft.params) for ft in self.fits])
        _, omega, alpha, gamma, beta, nus = ups.T
        n = len(self.fits)
        if start == "end":
            s2 = np.tile(self._factor_forecast(1)[0], (n_paths, 1))
            burn = 0
        else:
            s2 = np.tile(omega / (1 - alpha - 0.5 * gamma - beta), (n_paths, 1))
        out = np.empty((n_paths, nobs, len(self.mu)))
        for t in range(burn + nobs):
            z = rng.standard_normal((n_paths, n))
            for i, nu in enumerate(nus):
                if nu > 0:
                    z[:, i] = _std_t_ppf(rng.uniform(size=n_paths), nu)
            fct = np.sqrt(s2) * z
            if t >= burn:
                out[:, t - burn] = self.mu + fct @ self.A.T
            s2 = omega + (alpha + gamma * (fct < 0)) * fct ** 2 + beta * s2
        return {"returns": out}

    def value_at_risk(self, weights, level=0.01, horizon=1, method=None,
                      n_sims=100_000, value=1.0, seed=None):
        """Portfolio VaR/ES (positive losses). Analytic for one period with
        normal factors, otherwise by simulation."""
        w = np.asarray(weights, float)
        if method is None:
            method = ("analytic" if horizon == 1 and self.model.fspec.dist == "normal"
                      else "simulation")
        if method == "analytic":
            f = self.forecast(1)
            var, es = _analytic_var_es(self.mu @ w, math.sqrt(w @ f["covariance"][0] @ w), level)
        else:
            pr = (self.simulate(horizon, n_paths=n_sims, seed=seed)["returns"] @ w).sum(axis=1)
            q = np.quantile(pr, level)
            var, es = -q, -pr[pr <= q].mean()
        return {"VaR": float(value * var), "ES": float(value * es)}
