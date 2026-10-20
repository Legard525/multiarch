"""Specification tests: constant correlation (Engle & Sheppard 2001),
multivariate portmanteau (Hosking 1980) and VaR backtests (Kupiec 1995,
Christoffersen 1998)."""

import numpy as np
import pandas as pd
from scipy import stats

from .univariate import UnivariateSpec, filter_series, fit_univariate


def _sqrtm_inv(R):
    w, V = np.linalg.eigh(R)
    return V @ np.diag(w ** -0.5) @ V.T


def constant_correlation_test(returns=None, lags=5, vol="garch", mean="constant",
                              std_resid=None, whiten=True):
    """Engle & Sheppard (2001) test of H0: constant conditional correlation.

    Fits univariate GARCH margins, whitens the standardised residuals with
    the CCC correlation matrix, and regresses the stacked off-diagonal
    outer products on a constant and `lags` own lags. Under H0 all
    coefficients are zero; the statistic is chi2(lags + 1).
    Pass `std_resid` to skip the GARCH step (and `whiten=False` if they
    are already whitened). Returns dict(statistic, pvalue, df).
    """
    if std_resid is None:
        y = returns.to_numpy(float) if isinstance(returns, pd.DataFrame) else np.asarray(returns, float)
        spec = UnivariateSpec(mean=mean, vol=vol)
        z = np.column_stack([
            filter_series(spec, fit_univariate(y[:, i], spec, compute_cov=False).params, y[:, i])[2]
            for i in range(y.shape[1])
        ])
    else:
        z = np.asarray(std_resid, float)
    w = z @ _sqrtm_inv(np.corrcoef(z, rowvar=False)) if whiten else z
    n = w.shape[1]
    Y, X = [], []
    for i in range(n - 1):
        for j in range(i + 1, n):
            p = w[:, i] * w[:, j]
            Y.append(p[lags:])
            X.append(np.column_stack([np.ones(len(p) - lags)]
                                     + [p[lags - k:len(p) - k] for k in range(1, lags + 1)]))
    Y = np.concatenate(Y)
    X = np.vstack(X)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    s2 = resid @ resid / len(Y)
    stat = float(beta @ X.T @ X @ beta / s2)
    df = lags + 1
    return {"statistic": stat, "pvalue": float(stats.chi2.sf(stat, df)), "df": df}


def hosking_test(x, lags=10, fitted_params=0):
    """Multivariate portmanteau test (Hosking 1980) for serial correlation
    in a (T, N) series, e.g. standardised residuals or their squares.
    Returns dict(statistic, pvalue, df)."""
    x = np.asarray(x, float)
    x = x - x.mean(0)
    T, n = x.shape
    C0 = x.T @ x / T
    C0i = np.linalg.inv(C0)
    stat = 0.0
    for k in range(1, lags + 1):
        Ck = x[k:].T @ x[:-k] / T
        stat += np.trace(Ck.T @ C0i @ Ck @ C0i) / (T - k)
    stat *= T * T
    df = n * n * lags - fitted_params
    return {"statistic": float(stat), "pvalue": float(stats.chi2.sf(stat, df)), "df": df}


def var_backtest(returns, var, level=0.01):
    """Backtest a VaR forecast path.

    returns : realised portfolio returns; var : VaR as positive losses,
    aligned with `returns`. Reports the violation ratio (observed / expected
    violations), Kupiec's unconditional coverage test, Christoffersen's
    independence test and their sum (conditional coverage).
    """
    r = np.asarray(returns, float)
    v = np.asarray(var, float)
    hit = (r < -v).astype(int)
    T = hit.size
    x = int(hit.sum())
    p = level
    pi = x / T

    def _xlogy(a, b):
        return a * np.log(b) if a > 0 else 0.0

    lr_uc = -2 * (_xlogy(T - x, 1 - p) + _xlogy(x, p) - _xlogy(T - x, 1 - pi) - _xlogy(x, pi))
    h0, h1 = hit[:-1], hit[1:]
    n00 = int(((h0 == 0) & (h1 == 0)).sum())
    n01 = int(((h0 == 0) & (h1 == 1)).sum())
    n10 = int(((h0 == 1) & (h1 == 0)).sum())
    n11 = int(((h0 == 1) & (h1 == 1)).sum())
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    p1 = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    lr_ind = -2 * (_xlogy(n00 + n10, 1 - p1) + _xlogy(n01 + n11, p1)
                   - _xlogy(n00, 1 - p01) - _xlogy(n01, p01)
                   - _xlogy(n10, 1 - p11) - _xlogy(n11, p11))
    return {
        "violations": x,
        "expected": p * T,
        "violation_ratio": x / (p * T),
        "kupiec_stat": float(lr_uc),
        "kupiec_pvalue": float(stats.chi2.sf(lr_uc, 1)),
        "independence_stat": float(lr_ind),
        "independence_pvalue": float(stats.chi2.sf(lr_ind, 1)),
        "cc_stat": float(lr_uc + lr_ind),
        "cc_pvalue": float(stats.chi2.sf(lr_uc + lr_ind, 2)),
    }
