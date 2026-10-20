"""Monte Carlo check: do two-step standard errors match the sampling SD?"""
import numpy as np
from multiarch import DCC, simulate_dcc

Qbar = np.array([[1, .5, .3], [.5, 1, .4], [.3, .4, 1.]])
est, se = [], []
for s in range(300):
    y, _, _ = simulate_dcc(2000, mu=0.05, omega=[.02, .03, .01], alpha=[.08, .1, .06],
                           beta=[.9, .87, .92], Qbar=Qbar, a=.04, b=.93, seed=s)
    r = DCC(y).fit(se="two-step")
    est.append(r.params.to_numpy()); se.append(r.std_err.to_numpy())
est, se = np.array(est), np.array(se)
names = DCC(y).fit(se=None).params.index
for k in [1, 2, 3, 12, 13]:
    print(f"{str(names[k]):22s} MC sd {est[:, k].std():.4f}   median SE {np.nanmedian(se[:, k]):.4f}")
