"""Compare multiarch with rmgarch (run `Rscript rmgarch_fit.R .` first).

For every case it reports
  ll(rm)      log-likelihood reported by rmgarch
  ll(ours@rm) multiarch's log-likelihood evaluated at rmgarch's estimates
              (equal to ll(rm) if both implement the same model)
  ll(ours)    multiarch's own maximum
plus the largest coefficient and conditional-correlation differences and
the fit times.
"""
import time

import numpy as np
import pandas as pd

import multiarch as ma

X10 = pd.read_csv("dji10.csv", index_col=0)
X5 = X10.iloc[:, :5]
RAW2 = pd.read_csv("raw2.csv")

CASES = {
    "dcc_norm": (X5, dict()),
    "adcc_norm": (X5, dict(model="adcc")),
    "dcc_t": (X5, dict(dist="t")),
    "dcc_gjr": (X5, dict(vol="gjr")),
    "dcc_zero": (X5, dict(mean="zero")),
    "dcc_n10": (X10, dict()),
    "dcc_raw2": (RAW2, dict()),
    "cop_t": (X5, dict(dist="t", copula="t")),
    "cop_norm": (X5, dict(dist="t", copula="normal")),
}
RENAME = {"alpha1": "alpha", "beta1": "beta", "gamma1": "gamma", "shape": "nu",
          "dcca1": "a", "dccb1": "b", "dccg1": "g", "mshape": "nu"}


def rm_params(coef, res):
    """rmgarch estimates as a Series indexed like res.params."""
    out = {}
    for name, c in zip(coef.name, coef.coef):
        if name.startswith("[Joint]"):
            out[("corr", RENAME[name[7:]])] = c
        else:
            tick, par = name[1:].split("].")
            out[(tick, RENAME.get(par, par))] = c
    return pd.Series(out).reindex(res.params.index)


rows = []
for case, (X, kw) in CASES.items():
    coef = pd.read_csv(f"coef_{case}.csv")
    model = ma.DCC(X, convention="rmgarch", **kw)
    model.fit(se=None)  # warm-up for numba
    t0 = time.time()
    res = model.fit(se="two-step")
    secs = time.time() - t0
    p_rm = rm_params(coef, res)
    at_rm = model.fix(p_rm.to_numpy())
    rc = pd.read_csv(f"rcor_{case}.csv").to_numpy()
    n = X.shape[1]
    iu = np.triu_indices(n, 1)
    # rmgarch's upper.tri is column-major; sort pairs to match
    order = np.lexsort((iu[0], iu[1]))
    ours = res.conditional_correlation[:, iu[0][order], iu[1][order]]
    ours_at = at_rm.conditional_correlation[:, iu[0][order], iu[1][order]]
    se_ratio = (res.std_err / pd.Series(coef.se.to_numpy(), index=p_rm.index)).to_numpy()
    rows.append({
        "case": case, "ll(rm)": coef.loglik[0], "ll(ours@rm)": at_rm.loglik,
        "ll(ours)": res.loglik,
        "max|coef diff|": np.abs(p_rm - res.params).max(),
        "max|rho diff|": np.abs(rc - ours).max(),
        "max|rho diff| @rm": np.abs(rc - ours_at).max(),
        "SE ratio median": np.median(se_ratio),
        "rm secs": coef.secs[0], "ours secs": secs,
    })
    if case == "dcc_norm":
        fc_rm = pd.read_csv("forecast_dcc_norm.csv").to_numpy()
        iu2 = np.triu_indices(5)
        o2 = np.lexsort((iu2[0], iu2[1]))
        fc = at_rm.forecast(10)["covariance"][:, iu2[0][o2], iu2[1][o2]]
        fc_rel = np.abs(fc / fc_rm - 1).max()

pd.set_option("display.width", 200)
print(pd.DataFrame(rows).set_index("case").round(5).to_string())
print(f"\n10-step DCC covariance forecast at rmgarch params: max relative diff {fc_rel:.2e}")

from multiarch.univariate import loglik_obs

g_rm = pd.read_csv("gogarch.csv")
W = pd.read_csv("gogarch_W.csv").to_numpy()
gc = pd.read_csv("gogarch_coef.csv", index_col=0)
t0 = time.time()
g = ma.GOGARCH(X5, convention="rmgarch").fit()
gsecs = time.time() - t0
g_w = ma.GOGARCH(X5, convention="rmgarch").fit(unmixing=W)
ll_at = (sum(loglik_obs(g_w.model.fspec, gc.iloc[:, i].to_numpy(), g_w.factors[:, i]).sum()
             for i in range(W.shape[0]))
         + len(X5) * np.log(abs(np.linalg.det(W))))
print(f"\nGO-GARCH: ll(rm) {g_rm.loglik[0]:.3f} | ours at rm's rotation and params {ll_at:.3f} | "
      f"ours with rm's rotation {g_w.loglik:.3f} | ours (own ICA) {g.loglik:.3f}  "
      f"time rm {g_rm.secs[0]:.2f}s ours {gsecs:.2f}s")

# rmgarch's DCCtest regresses the regressors on the regressand (qr.solve
# argument order) and divides by sqrt(sigma^2); replicate it to show the
# inputs agree, then report the textbook Engle-Sheppard statistic.
dt = pd.read_csv("dcctest.csv")
Z = pd.read_csv("dcctest_Z.csv").to_numpy()
L = 5
Y, Xr = [], []
for i in range(Z.shape[1] - 1):
    for j in range(i + 1, Z.shape[1]):
        p = Z[:, i] * Z[:, j]
        Y.append(p[L:])
        Xr.append(np.column_stack([np.ones(len(p) - L)] + [p[L - k:len(p) - k] for k in range(1, L + 1)]))
Y, Xr = np.concatenate(Y), np.vstack(Xr)
brev = Xr.T @ Y / (Y @ Y)
e = Y - Xr @ brev
replica = brev @ Xr.T @ Xr @ brev / np.sqrt(e @ e / len(Y))
ours = ma.constant_correlation_test(std_resid=Z, lags=L, whiten=False)
print(f"\nEngle-Sheppard test on rmgarch's residuals: rmgarch {dt.stat[0]:.4f}, "
      f"replica of rmgarch's formula {replica:.4f}, textbook statistic {ours['statistic']:.4f} "
      f"(p = {ours['pvalue']:.2e})")
