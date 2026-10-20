import numpy as np

from . import _kernels as K


def simulate_dcc(nobs, mu, omega, alpha, beta, Qbar, a, b, gamma=None, g=0.0,
                 Nbar=None, nu=None, burn=500, seed=None, Q0=None, sigma2_0=None,
                 shocks=None):
    """Simulate returns from GJR-GARCH(1,1) margins + (A)DCC correlation.

    nu : degrees of freedom for unit-variance multivariate-t shocks
         (None for Gaussian).
    Q0, sigma2_0 : starting state (default: unconditional, Qbar and
         omega / (1 - persistence)).
    shocks : optional (nobs + burn, N) iid unit-variance, uncorrelated
         innovations, e.g. copula draws; overrides `nu`.
    Returns (returns, sigma2, std_resid), each (nobs, N).
    """
    rng = np.random.default_rng(seed)
    omega = np.asarray(omega, float)
    n = omega.size
    mu = np.broadcast_to(np.asarray(mu, float), (n,)).copy()
    alpha = np.asarray(alpha, float)
    beta = np.asarray(beta, float)
    gamma = np.zeros(n) if gamma is None else np.asarray(gamma, float)
    Qbar = np.asarray(Qbar, float)
    Nbar = np.zeros_like(Qbar) if Nbar is None else np.asarray(Nbar, float)
    T = nobs + burn
    Q0 = Qbar if Q0 is None else np.asarray(Q0, float)
    if sigma2_0 is None:
        sigma2_0 = omega / (1 - alpha - 0.5 * gamma - beta)
    sigma2_0 = np.asarray(sigma2_0, float)
    if shocks is not None:
        return K.simulate_path(mu, omega, alpha, gamma, beta, Qbar, Nbar,
                               float(a), float(b), float(g), np.asarray(shocks, float),
                               burn, Q0, sigma2_0)
    x = rng.standard_normal((T, n))
    if nu is not None:
        # multivariate t: common chi-square mixing, scaled to unit variance
        w = rng.chisquare(nu, size=T) / nu
        x = x / np.sqrt(w)[:, None] * np.sqrt((nu - 2) / nu)
    return K.simulate_path(mu, omega, alpha, gamma, beta, Qbar, Nbar,
                           float(a), float(b), float(g), x, burn, Q0, sigma2_0)
