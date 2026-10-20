"""Central finite differences with arch-style relative step sizes."""

import numpy as np


def _steps(p):
    return np.maximum(np.abs(p), 1e-4) * 1e-5


def jacobian(f, p):
    """d f(p) / d p for vector-valued f -> (len(f), len(p))."""
    p = np.asarray(p, dtype=float)
    h = _steps(p)
    cols = []
    for k in range(p.size):
        e = np.zeros_like(p)
        e[k] = h[k]
        cols.append((f(p + e) - f(p - e)) / (2 * h[k]))
    return np.column_stack(cols)


def gradient(f, p):
    return jacobian(lambda q: np.atleast_1d(f(q)), p)[0]


def hessian(f, p):
    """Hessian of scalar f by second-order central differences."""
    p = np.asarray(p, dtype=float)
    k = p.size
    h = np.maximum(np.abs(p), 1e-4) * 1e-4
    H = np.empty((k, k))
    f0 = f(p)
    for i in range(k):
        ei = np.zeros(k)
        ei[i] = h[i]
        H[i, i] = (f(p + ei) - 2 * f0 + f(p - ei)) / h[i] ** 2
        for j in range(i):
            ej = np.zeros(k)
            ej[j] = h[j]
            H[i, j] = H[j, i] = (
                f(p + ei + ej) - f(p + ei - ej) - f(p - ei + ej) + f(p - ei - ej)
            ) / (4 * h[i] * h[j])
    return H
