"""Fit time vs number of assets (T=2500, no standard errors)."""
import time
import numpy as np
from multiarch import DCC, simulate_dcc

for n in (5, 10, 25, 50, 100):
    rng = np.random.default_rng(n)
    F = rng.standard_normal((n, 2))
    C = F @ F.T + np.eye(n)
    d = 1 / np.sqrt(np.diag(C))
    Qbar = C * np.outer(d, d)
    y, _, _ = simulate_dcc(2500, mu=0.0, omega=np.full(n, .02), alpha=np.full(n, .08),
                           beta=np.full(n, .9), Qbar=Qbar, a=.03, b=.95, seed=n)
    row = [f"N={n:4d}"]
    for est, pairs in (("full", "all"), ("composite", "adjacent")):
        if est == "full" and n > 50:
            row.append("full: skipped"); continue
        m = DCC(y, estimator=est, pairs=pairs)
        t0 = time.time(); r = m.fit(se=None); dt = time.time() - t0
        a, b = r.params["corr"]
        row.append(f"{est}: {dt:6.2f}s (a={a:.3f}, b={b:.3f})")
    print("   ".join(row))
