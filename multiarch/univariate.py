"""Univariate GARCH(1,1) / GJR-GARCH(1,1) margins, the first stage of DCC.

Conventions (start-up backcast, unit-variance Student-t, parameter order)
follow the `arch` package so results are directly comparable.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from . import _kernels as K
from ._numdiff import hessian, jacobian


def backcast(resid):
    """arch's exponential backcast for the pre-sample variance."""
    tau = min(75, resid.shape[0])
    w = 0.94 ** np.arange(tau)
    w /= w.sum()
    return float(np.sum(resid[:tau] ** 2 * w))


@dataclass
class UnivariateSpec:
    mean: str = "constant"  # "constant" or "zero"
    vol: str = "garch"  # "garch" or "gjr"
    dist: str = "normal"  # "normal" or "t"
    # start-up of the variance recursion: "arch" (exponential backcast,
    # sigma2_0 = omega + persistence * backcast), "rugarch"
    # (sigma2_0 = full-sample mean of squared residuals) or "tsgarch"
    # (sigma2_0 = omega + persistence * mean of squared residuals)
    init: str = "arch"

    def __post_init__(self):
        if self.mean not in ("constant", "zero"):
            raise ValueError("mean must be 'constant' or 'zero'")
        if self.vol not in ("garch", "gjr"):
            raise ValueError("vol must be 'garch' or 'gjr'")
        if self.dist not in ("normal", "t"):
            raise ValueError("dist must be 'normal' or 't'")
        if self.init not in ("arch", "rugarch", "tsgarch"):
            raise ValueError("init must be 'arch', 'rugarch' or 'tsgarch'")

    @property
    def names(self):
        n = ["mu"] if self.mean == "constant" else []
        n += ["omega", "alpha"]
        if self.vol == "gjr":
            n.append("gamma")
        n.append("beta")
        if self.dist == "t":
            n.append("nu")
        return n

    def unpack(self, p):
        """-> mu, omega, alpha, gamma, beta, nu  (nu=0 means normal)."""
        k = 0
        mu = 0.0
        if self.mean == "constant":
            mu = p[0]
            k = 1
        omega, alpha = p[k], p[k + 1]
        k += 2
        gamma = 0.0
        if self.vol == "gjr":
            gamma = p[k]
            k += 1
        beta = p[k]
        nu = p[k + 1] if self.dist == "t" else 0.0
        return mu, omega, alpha, gamma, beta, nu


def _sigma2_0(spec, omega, alpha, gamma, beta, resid):
    if spec.init == "rugarch":
        return float(np.mean(resid * resid))
    if spec.init == "tsgarch":
        return omega + (alpha + 0.5 * gamma + beta) * float(np.mean(resid * resid))
    return omega + (alpha + 0.5 * gamma + beta) * backcast(resid)


def loglik_obs(spec, p, y):
    mu, omega, alpha, gamma, beta, nu = spec.unpack(p)
    resid = y - mu
    s0 = _sigma2_0(spec, omega, alpha, gamma, beta, resid)
    sigma2 = K.garch_filter(omega, alpha, gamma, beta, resid, s0)
    if nu > 0:
        return K.std_t_loglik_obs(resid, sigma2, nu)
    return K.normal_loglik_obs(resid, sigma2)


def filter_series(spec, p, y, sigma2_0=None):
    """-> (residuals, conditional variance, standardised residuals).
    `sigma2_0` overrides the start-up variance."""
    mu, omega, alpha, gamma, beta, _ = spec.unpack(p)
    resid = y - mu
    s0 = _sigma2_0(spec, omega, alpha, gamma, beta, resid) if sigma2_0 is None else sigma2_0
    sigma2 = K.garch_filter(omega, alpha, gamma, beta, resid, s0)
    return resid, sigma2, resid / np.sqrt(sigma2)


@dataclass
class UnivariateFit:
    spec: UnivariateSpec
    params: np.ndarray
    loglik: float
    cov: np.ndarray = field(repr=False)
    converged: bool = True

    @property
    def std_err(self):
        return np.sqrt(np.diag(self.cov))


def _start_values(spec, y):
    var = float(np.var(y))
    best, best_ll = None, -np.inf
    for alpha in (0.03, 0.08, 0.15):
        for beta in (0.75, 0.85, 0.92):
            gamma = 0.05 if spec.vol == "gjr" else 0.0
            if alpha + beta + gamma / 2 >= 0.995:
                continue
            p = []
            if spec.mean == "constant":
                p.append(float(np.mean(y)))
            p += [var * (1 - alpha - beta - gamma / 2), alpha]
            if spec.vol == "gjr":
                p.append(gamma)
            p.append(beta)
            if spec.dist == "t":
                p.append(8.0)
            p = np.array(p)
            ll = loglik_obs(spec, p, y).sum()
            if ll > best_ll:
                best, best_ll = p, ll
    return best


def rescale(spec, p, s):
    """Map parameters estimated on y/s back to the scale of y.
    Returns (params, jacobian diagonal)."""
    J = np.array([{"mu": s, "omega": s * s}.get(nm, 1.0) for nm in spec.names])
    return p * J, J


def fit_univariate(y, spec=None, compute_cov=True):
    """QML / ML fit of a single GARCH margin. Covariance is the robust
    (Bollerslev-Wooldridge) sandwich, matching arch's default.

    The fit runs on y / sd(y) and is mapped back, so results do not depend
    on whether returns are in percent or fractions."""
    spec = spec or UnivariateSpec()
    y = np.asarray(y, dtype=float)
    s = float(np.std(y))
    f = _fit_unit(y / s, spec, compute_cov)
    p, J = rescale(spec, f.params, s)
    cov = f.cov * np.outer(J, J)
    return UnivariateFit(spec, p, f.loglik - y.shape[0] * np.log(s), cov, f.converged)


def _fit_unit(y, spec, compute_cov):
    names = spec.names
    var = float(np.var(y))
    scale = max(abs(float(np.mean(y))), 1e-8)

    bounds = []
    for nm in names:
        bounds.append(
            {
                "mu": (-10 * scale - 10 * np.sqrt(var), 10 * scale + 10 * np.sqrt(var)),
                "omega": (1e-8 * var, 2.0 * var),
                "alpha": (0.0, 1.0),
                "gamma": (-1.0, 2.0),
                "beta": (0.0, 1.0),
                "nu": (2.05, 500.0),
            }[nm]
        )
    ia, ib = names.index("alpha"), names.index("beta")
    ig = names.index("gamma") if "gamma" in names else None

    def persistence_gap(p):
        g = p[ig] / 2 if ig is not None else 0.0
        return 1.0 - 1e-6 - p[ia] - p[ib] - g

    cons = [{"type": "ineq", "fun": persistence_gap}]
    if ig is not None:  # alpha + gamma >= 0 keeps the variance positive
        cons.append({"type": "ineq", "fun": lambda p: p[ia] + p[ig]})

    def obj(p):
        ll = loglik_obs(spec, p, y).sum()
        return -ll if np.isfinite(ll) else 1e10

    x0 = _start_values(spec, y)
    res = minimize(
        obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    p = res.x
    ll = -res.fun
    cov = np.full((len(p), len(p)), np.nan)
    if compute_cov:
        cov = robust_cov(lambda q: loglik_obs(spec, q, y), p)
    return UnivariateFit(spec, p, ll, cov, bool(res.success))


def robust_cov(ll_obs_fn, p):
    """Sandwich covariance A^{-1} B A^{-1} / T from per-observation log-likelihoods."""
    scores = jacobian(ll_obs_fn, p)  # T x k
    T = scores.shape[0]
    H = hessian(lambda q: ll_obs_fn(q).sum(), p) / T
    B = scores.T @ scores / T
    Ainv = np.linalg.inv(H)
    return Ainv @ B @ Ainv / T
