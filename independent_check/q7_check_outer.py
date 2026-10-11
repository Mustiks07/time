"""Check outer-radius ball maxima by also sampling along every BH direction (+ local refinement)."""
import numpy as np
from scipy.optimize import minimize
from q7_mp import fibonacci, icosahedron, accel
for name, P in [('ico12', icosahedron()), ('fib100', fibonacci(100)), ('fib1000', fibonacci(1000))]:
    m = 9.0 / len(P)
    for s in [0.7, 0.8, 0.9]:
        vals = accel(s * P, P, m)
        k = np.argmax(vals)
        best = vals.max()
        for kk in np.argsort(vals)[-5:]:
            n = P[kk]; th0 = np.arccos(n[2]); ph0 = np.arctan2(n[1], n[0])
            f = lambda q: -accel(s*np.array([np.sin(q[0])*np.cos(q[1]), np.sin(q[0])*np.sin(q[1]), np.cos(q[0])]), P, m)[0]
            best = max(best, -minimize(f, [th0, ph0], method='Nelder-Mead', options={'xatol':1e-11,'fatol':1e-16}).fun)
        print(f"{name} s={s}: max along BH directions = {vals.max():.6e}, refined = {best:.6e} c^2/rho0")
