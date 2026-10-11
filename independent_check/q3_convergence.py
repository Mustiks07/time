"""Q3 check: compare optimiser with the same-discretisation value of the analytic bang-bang profile,
and show convergence of the analytic-profile quadrature with N (trapezoid bias ~ h^2)."""
import numpy as np
from q3_twin import optimise, ratio

def discrete_tent(x, N):
    a = x; h = 1.0 / N
    u = np.concatenate([np.ones(N//4), -np.ones(N//2), np.ones(N//4)])
    e = np.concatenate([[0.0], np.cumsum(u) * a * h])
    w = np.full(N + 1, h); w[0] = w[-1] = h/2
    return np.dot(w, np.cosh(e)), np.dot(w, np.sinh(e))

for x in [8.0, 18.0]:
    for N in [80, 160]:
        num, u = optimise(x, N=N, nstart=6)
        dt, ds = discrete_tent(x, N)
        print(f"x={x:5.1f} N={N:4d}  optimiser={num:.10f}  discrete-tent={dt:.10f}  diff={num-dt:.2e}  exact={ratio(x):.10f}")
for N in [160, 640, 2560]:
    dt, _ = discrete_tent(18.0, N)
    print(f"tent quadrature N={N}: {dt:.10f}  (exact {ratio(18.0):.10f})")
