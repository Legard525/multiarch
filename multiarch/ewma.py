import numpy as np
from numba import njit


@njit(cache=True)
def _ewma(x, lam, S0):
    T, n = x.shape
    out = np.empty((T + 1, n, n))
    out[0] = S0
    for t in range(T):
        out[t + 1] = lam * out[t] + (1 - lam) * np.outer(x[t], x[t])
    return out


def ewma_covariance(returns, lam=0.94, demean=False, init=None):
    """RiskMetrics EWMA covariance: S_{t+1} = lam S_t + (1-lam) x_t x_t'.

    Returns a (T+1, N, N) array; element t is the forecast for period t
    made with data up to t-1, so the last element is the one-step-ahead
    forecast. `init` defaults to the sample covariance.
    """
    x = np.asarray(returns, float)
    if demean:
        x = x - x.mean(0)
    S0 = np.cov(x, rowvar=False, bias=True) if init is None else np.asarray(init, float)
    return _ewma(x, float(lam), S0)
