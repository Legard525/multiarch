"""The multivariate-volatility workflow from a typical risk course in Python:
EWMA, CCC and DCC correlations, a test for constant correlation, portfolio
VaR, and a rolling VaR backtest.

Runs on simulated data; to use real returns, replace `y` with a DataFrame
of returns (any units, e.g. `prices.pct_change().dropna()`).
"""
import numpy as np
import pandas as pd

import multiarch as ma

# two assets with time-varying correlation (fractional daily returns)
y, _, _ = ma.simulate_dcc(
    3000, mu=0.0003, omega=[2e-6, 3e-6], alpha=[0.08, 0.10], beta=[0.90, 0.88],
    Qbar=np.array([[1.0, 0.4], [0.4, 1.0]]), a=0.04, b=0.94, seed=1,
)
y = pd.DataFrame(y, columns=["JPM", "INTC"],
                 index=pd.bdate_range("2014-01-01", periods=len(y)))

# 1. EWMA (RiskMetrics, lambda = 0.94); element t+1 is the forecast for t+1
S = ma.ewma_covariance(y, lam=0.94)
rho_ewma = S[:-1, 0, 1] / np.sqrt(S[:-1, 0, 0] * S[:-1, 1, 1])

# 2. CCC and DCC with GARCH(1,1) margins
ccc = ma.DCC(y, model="ccc").fit()
dcc = ma.DCC(y, model="dcc").fit()
print(dcc.summary())
print(f"\nCCC correlation {ccc.Rbar[0, 1]:.3f}; DCC correlation range "
      f"{dcc.correlation('JPM', 'INTC').min():.3f} to {dcc.correlation('JPM', 'INTC').max():.3f}; "
      f"EWMA range {rho_ewma.min():.3f} to {rho_ewma.max():.3f}")
print(f"log-likelihood CCC {ccc.loglik:.1f}, DCC {dcc.loglik:.1f}")

# 3. Is correlation constant? (Engle & Sheppard 2001)
print("constant-correlation test:", ma.constant_correlation_test(y, lags=5))

# 4. Next-day 1% VaR and ES of a 50/50 portfolio worth 1,000,000
w = [0.5, 0.5]
print("DCC VaR:", dcc.value_at_risk(w, level=0.01, value=1e6))
print("DCC 10-day VaR (simulated):", dcc.value_at_risk(w, level=0.01, horizon=10, value=1e6, seed=0))
ewma_sd = np.sqrt(np.array(w) @ S[-1] @ np.array(w))
print(f"EWMA VaR (normal): {1e6 * 2.326 * ewma_sd:,.0f}")

# 5. Rolling backtest: re-estimate every 50 days on a 1000-day window
rf = ma.rolling_forecast(y, window=1000, refit_every=50)
var_path = rf.value_at_risk(w, level=0.01)
realised = y.iloc[1000:] @ np.array(w)
print("backtest:", ma.var_backtest(realised, var_path["VaR"], level=0.01))
