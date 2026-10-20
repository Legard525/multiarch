"""Numba kernels for the inner recursions. Everything here is O(T) loops that
would be slow in pure Python; the public API lives in the other modules."""

import math

import numpy as np
from numba import njit

LOG_2PI = math.log(2.0 * math.pi)


# ---------------------------------------------------------------- univariate


@njit(cache=True)
def garch_filter(omega, alpha, gamma, beta, resid, sigma2_0):
    """GJR-GARCH(1,1) variance recursion (gamma=0 gives GARCH), given the
    first conditional variance sigma2_0."""
    n = resid.shape[0]
    sigma2 = np.empty(n)
    sigma2[0] = sigma2_0
    for t in range(1, n):
        e = resid[t - 1]
        e2 = e * e
        s = omega + alpha * e2 + beta * sigma2[t - 1]
        if e < 0.0:
            s += gamma * e2
        sigma2[t] = s
    return sigma2


@njit(cache=True)
def normal_loglik_obs(resid, sigma2):
    return -0.5 * (LOG_2PI + np.log(sigma2) + resid * resid / sigma2)


@njit(cache=True)
def std_t_loglik_obs(resid, sigma2, nu):
    """Unit-variance Student-t, as in arch's StudentsT."""
    c = (
        math.lgamma(0.5 * (nu + 1.0))
        - math.lgamma(0.5 * nu)
        - 0.5 * math.log(math.pi * (nu - 2.0))
    )
    return c - 0.5 * np.log(sigma2) - 0.5 * (nu + 1.0) * np.log(
        1.0 + resid * resid / (sigma2 * (nu - 2.0))
    )


# ---------------------------------------------------------------- correlation


@njit(cache=True)
def _mv_logpdf(z_t, R, nu):
    """Log density of z_t ~ N(0, R) (nu <= 0) or unit-variance MV-t(nu)."""
    n = z_t.shape[0]
    L = np.linalg.cholesky(R)
    logdet = 0.0
    for i in range(n):
        logdet += 2.0 * math.log(L[i, i])
    # forward substitution: y = L^{-1} z, quad = y'y
    y = np.empty(n)
    quad = 0.0
    for i in range(n):
        s = z_t[i]
        for k in range(i):
            s -= L[i, k] * y[k]
        y[i] = s / L[i, i]
        quad += y[i] * y[i]
    if nu <= 0.0:
        return -0.5 * (n * LOG_2PI + logdet + quad)
    return (
        math.lgamma(0.5 * (nu + n))
        - math.lgamma(0.5 * nu)
        - 0.5 * n * math.log(math.pi * (nu - 2.0))
        - 0.5 * logdet
        - 0.5 * (nu + n) * math.log(1.0 + quad / (nu - 2.0))
    )


@njit(cache=True)
def _q_to_r(Q, R):
    n = Q.shape[0]
    for i in range(n):
        for j in range(n):
            R[i, j] = Q[i, j] / math.sqrt(Q[i, i] * Q[j, j])


@njit(cache=True)
def dcc_loglik_obs(z, Qbar, Nbar, a, b, g, nu, Q0):
    """Per-observation log density of standardised residuals under (A)DCC.

    Q_t = (1-a-b) Qbar - g Nbar + a z z' + g n n' + b Q_{t-1},  Q_0 = Qbar,
    with n = z * 1{z<0}. g = 0 gives plain DCC; Q0 is the Q used for the
    first observation (normally Qbar).
    """
    T, n = z.shape
    out = np.empty(T)
    C = (1.0 - a - b) * Qbar - g * Nbar
    Q = Q0.copy()
    R = np.empty((n, n))
    for t in range(T):
        if t > 0:
            zp = z[t - 1]
            for i in range(n):
                ni = zp[i] if zp[i] < 0.0 else 0.0
                for j in range(i, n):
                    nj = zp[j] if zp[j] < 0.0 else 0.0
                    v = C[i, j] + a * zp[i] * zp[j] + g * ni * nj + b * Q[i, j]
                    Q[i, j] = v
                    Q[j, i] = v
        _q_to_r(Q, R)
        out[t] = _mv_logpdf(z[t], R, nu)
    return out


@njit(cache=True)
def dcc_filter(z, Qbar, Nbar, a, b, g, Q0):
    """Q_t and R_t for t = 0..T (index T is the one-step-ahead forecast)."""
    T, n = z.shape
    Qs = np.empty((T + 1, n, n))
    Rs = np.empty((T + 1, n, n))
    C = (1.0 - a - b) * Qbar - g * Nbar
    Q = Q0.copy()
    for t in range(T + 1):
        if t > 0:
            zp = z[t - 1]
            for i in range(n):
                ni = zp[i] if zp[i] < 0.0 else 0.0
                for j in range(i, n):
                    nj = zp[j] if zp[j] < 0.0 else 0.0
                    v = C[i, j] + a * zp[i] * zp[j] + g * ni * nj + b * Q[i, j]
                    Q[i, j] = v
                    Q[j, i] = v
        Qs[t] = Q
        _q_to_r(Q, Rs[t])
    return Qs, Rs


@njit(cache=True)
def dcc_composite_loglik(z, Qbar, Nbar, pairs, a, b, g, nu, Q0):
    """Composite likelihood (Engle, Shephard & Sheppard 2008): the sum of
    bivariate DCC log-likelihoods over the given pairs. O(T * #pairs) with
    scalar 2x2 algebra, so it scales to hundreds of assets."""
    T = z.shape[0]
    total = 0.0
    if nu > 0.0:
        c = (
            math.lgamma(0.5 * (nu + 2.0))
            - math.lgamma(0.5 * nu)
            - math.log(math.pi * (nu - 2.0))
        )
    else:
        c = -LOG_2PI
    for p in range(pairs.shape[0]):
        i = pairs[p, 0]
        j = pairs[p, 1]
        cii = (1.0 - a - b) * Qbar[i, i] - g * Nbar[i, i]
        cjj = (1.0 - a - b) * Qbar[j, j] - g * Nbar[j, j]
        cij = (1.0 - a - b) * Qbar[i, j] - g * Nbar[i, j]
        qii = Q0[i, i]
        qjj = Q0[j, j]
        qij = Q0[i, j]
        for t in range(T):
            if t > 0:
                zi = z[t - 1, i]
                zj = z[t - 1, j]
                ni = zi if zi < 0.0 else 0.0
                nj = zj if zj < 0.0 else 0.0
                qii = cii + a * zi * zi + g * ni * ni + b * qii
                qjj = cjj + a * zj * zj + g * nj * nj + b * qjj
                qij = cij + a * zi * zj + g * ni * nj + b * qij
            rho = qij / math.sqrt(qii * qjj)
            det = 1.0 - rho * rho
            if det <= 0.0:
                return -np.inf
            x = z[t, i]
            y = z[t, j]
            quad = (x * x - 2.0 * rho * x * y + y * y) / det
            if nu > 0.0:
                total += c - 0.5 * math.log(det) - 0.5 * (nu + 2.0) * math.log(
                    1.0 + quad / (nu - 2.0)
                )
            else:
                total += c - 0.5 * math.log(det) - 0.5 * quad
    return total


# ---------------------------------------------------------------- simulation


@njit(cache=True)
def simulate_path(mu, omega, alpha, gamma, beta, Qbar, Nbar, a, b, g, shocks, burn,
                  Q0, s2_0):
    """Simulate returns from GJR-GARCH margins + (A)DCC correlation.

    `shocks` are iid unit-variance draws (T+burn, n) with identity
    correlation; they are rotated by chol(R_t) each step. Q0 and s2_0 are
    the state for the first simulated period (Qbar and the unconditional
    variances for an unconditional start, Q_{T+1} and sigma2_{T+1} to
    simulate forward from the end of a sample).
    """
    Ttot, n = shocks.shape
    eps = np.empty((Ttot, n))
    sig2 = np.empty((Ttot, n))
    zs = np.empty((Ttot, n))
    C = (1.0 - a - b) * Qbar - g * Nbar
    Q = Q0.copy()
    R = np.empty((n, n))
    for t in range(Ttot):
        if t > 0:
            zp = zs[t - 1]
            for i in range(n):
                ni = zp[i] if zp[i] < 0.0 else 0.0
                for j in range(i, n):
                    nj = zp[j] if zp[j] < 0.0 else 0.0
                    v = C[i, j] + a * zp[i] * zp[j] + g * ni * nj + b * Q[i, j]
                    Q[i, j] = v
                    Q[j, i] = v
            ep = eps[t - 1]
            for i in range(n):
                e2 = ep[i] * ep[i]
                s = omega[i] + alpha[i] * e2 + beta[i] * sig2[t - 1, i]
                if ep[i] < 0.0:
                    s += gamma[i] * e2
                sig2[t, i] = s
        else:
            sig2[0] = s2_0
        _q_to_r(Q, R)
        L = np.linalg.cholesky(R)
        zt = L @ shocks[t]
        zs[t] = zt
        for i in range(n):
            eps[t, i] = math.sqrt(sig2[t, i]) * zt[i]
    return eps[burn:] + mu, sig2[burn:], zs[burn:]
