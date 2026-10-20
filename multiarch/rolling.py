"""Rolling-window estimation and one-step-ahead forecasts, for backtesting."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dcc import DCC, _analytic_var_es


@dataclass
class RollingForecast:
    """One-step-ahead forecasts for periods window..T-1.

    mean (n, N), covariance (n, N, N), and nu (n,) for multivariate-t models
    (NaN otherwise). Row k is the forecast for `index[k]`, made with data up
    to the previous period only.
    """

    index: pd.Index
    columns: list
    mean: np.ndarray
    covariance: np.ndarray
    nu: np.ndarray

    def portfolio_variance(self, weights):
        w = np.asarray(weights, float)
        return pd.Series(np.einsum("i,tij,j->t", w, self.covariance, w), index=self.index)

    def value_at_risk(self, weights, level=0.01, value=1.0):
        """Analytic one-step VaR/ES path (positive numbers are losses)."""
        w = np.asarray(weights, float)
        mean = self.mean @ w
        sd = np.sqrt(self.portfolio_variance(w).to_numpy())
        var = np.empty(len(mean))
        es = np.empty(len(mean))
        for k in range(len(mean)):
            nu = None if np.isnan(self.nu[k]) else self.nu[k]
            var[k], es[k] = _analytic_var_es(mean[k], sd[k], level, nu)
        return pd.DataFrame({"VaR": value * var, "ES": value * es}, index=self.index)


def rolling_forecast(returns, window, refit_every=20, expanding=False, **model_kwargs):
    """Re-estimate a DCC model every `refit_every` periods on a rolling (or
    expanding) window and filter forward between refits with the parameters
    held fixed. Uses no information from the forecast period.

    `model_kwargs` are passed to `DCC` (model, vol, dist, ...). Copula models
    are supported for the covariance path, but their VaR needs simulation.
    """
    df = returns if isinstance(returns, pd.DataFrame) else pd.DataFrame(np.asarray(returns, float))
    y = df.to_numpy(float)
    T, n = y.shape
    if window >= T:
        raise ValueError("window must be shorter than the sample")
    means, covs, nus = [], [], []
    for k in range(window, T, refit_every):
        lo = 0 if expanding else k - window
        res = DCC(y[lo:k], **model_kwargs).fit(se=None)
        hi = min(k + refit_every, T)
        # filter the estimation window plus the next block; row t of the
        # filtered covariance only uses data up to t-1
        ap = res.apply(y[lo:hi], sigma2_0=res.sigma2[0])
        sl = slice(k - lo, hi - lo)
        covs.append(ap.conditional_covariance[sl])
        means.append(ap.conditional_mean[sl])
        nus.append(np.full(hi - k, res.nu if res.nu else np.nan))
    return RollingForecast(df.index[window:], [str(c) for c in df.columns],
                           np.concatenate(means), np.concatenate(covs), np.concatenate(nus))
