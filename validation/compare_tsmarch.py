"""Compare multiarch with tsmarch, the R package used in the Financial Risk
Forecasting notes (run `Rscript tsmarch_fit.R .` first). Raw returns."""
import time

import numpy as np
import pandas as pd

import multiarch as ma

X = pd.read_csv("dji5.csv", index_col=0) / 100
CASES = {"ts_dcc2": (2, dict()), "ts_dcc5": (5, dict()),
         "ts_adcc5": (5, dict(model="adcc")), "ts_dcct5": (5, dict(corr_dist="t"))}
NAMES = {"alpha1": "alpha", "beta1": "beta", "alpha_1": "a", "beta_1": "b",
         "gamma_1": "g", "shape": "nu"}

rows = []
for case, (n, kw) in CASES.items():
    coef = pd.read_csv(f"ts_coef_{case}.csv")
    ref = pd.read_csv(f"ts_{case}.csv")
    model = ma.DCC(X.iloc[:, :n], convention="tsmarch", **kw)
    model.fit(se=None)
    t0 = time.time()
    res = model.fit(se=None)
    secs = time.time() - t0
    p = {}
    for name, c in zip(coef.name, coef.coef):
        if name.startswith("[Joint]"):
            p[("corr", NAMES[name[7:]])] = c
        else:
            tick, par = name[1:].split("].")
            p[(tick, NAMES.get(par, par))] = c
    p = pd.Series(p).reindex(res.params.index)
    at = model.fix(p.to_numpy())
    rows.append({
        "case": case, "ll(ts)": ref.loglik[0], "ll(ours@ts)": at.loglik, "ll(ours)": res.loglik,
        "max|coef diff| (corr)": np.abs(p["corr"] - res.params["corr"]).max(),
        "max|rho12 diff| @ts": np.abs(ref.rho12 - at.conditional_correlation[:, 0, 1]).max(),
        "max|rho12 diff|": np.abs(ref.rho12 - res.conditional_correlation[:, 0, 1]).max(),
        "ts secs": ref.secs[0], "ours secs": secs,
    })
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).set_index("case").to_string(float_format=lambda x: f"{x:.5f}"))
