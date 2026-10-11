"""Exploratory (round 1, experiment E19).  Heuristic NEC-only family: (1) thin layer raising p_r with rho=0 (p_t>0),
(2) massless region with p_t=0 where g decays as g' = -g^2/2 (NEC_t saturated),
(3) thin end layer with p_t=-rho (NEC_t saturated) releasing g to 0 with mass.
State (y,g), s = ln r.  y_s = rho~ - y ; g_s = 2 pt~ - (rho~+g) P ; P=(y+g)/(2(1-y)); lnZ_s = P."""
import math

import numpy as np
from scipy.integrate import solve_ivp

def run(g0, s_mid, rho_end=4000.0):
    # region 2 starts with g=g0 (layer 1 idealised as a jump in g with rho=0, which costs no mass)
    def f2(s, Y):
        y, g, L = Y
        P = (y + g) / (2 * (1 - y))
        return [-y, -(g) * P, P]          # rho=0, pt=0
    sol = solve_ivp(f2, (0, s_mid), [0.0, g0, 0.0], rtol=1e-10, atol=1e-12)
    y, g, L = sol.y[:, -1]
    # region 3: rho large, pt = -rho (NEC_t saturated), integrate until g = 0
    # NEC_t: rho + pt >= 0 -> pt >= -rho ; NEC_r: rho + p_r = rho+g >=0 fine
    def f3b(s, Y):
        y, g, L = Y
        P = (y + g) / (2 * (1 - y))
        pt = -rho_end
        return [rho_end - y, 2 * pt - (rho_end + g) * P, P]
    ev = lambda s, Y: Y[1]; ev.terminal = True; ev.direction = -1
    hz = lambda s, Y: 0.999 - Y[0]; hz.terminal = True
    sol3 = solve_ivp(f3b, (s_mid, s_mid + 1.0), [y, g, L], events=[ev, hz], rtol=1e-10, atol=1e-12, max_step=1e-5)
    y, g, L = sol3.y[:, -1]
    s_end = sol3.t[-1]
    if y >= 0.999 or g > 1e-6:
        return None
    lnZ = L - 0.5 * math.log(1 - y)
    return dict(Z=math.exp(lnZ), M=0.5 * y * math.exp(s_end), y_end=y, s_end=s_end)


if __name__ == '__main__':
    # Compare with the Theorem-2 bound 0.5 (1 - 1/Z) (which assumes p_r <= rho, violated here).
    for g0 in [2, 5, 10, 20, 50]:
        best = None
        for s_mid in np.geomspace(1e-4, 1.0, 60):
            r = run(g0, s_mid, rho_end=4e4)
            if r and (best is None or r['M'] < best['M']):
                best = dict(r, s_mid=s_mid)
        if best:
            b = 0.5 * (1 - 1 / best['Z'])
            print(f"g0={g0:5}  Z={best['Z']:.3f}  M/r_in={best['M']:.4f}  thm2_bound={b:.4f}  "
                  f"ratio={best['M'] / b:.3f}  s_mid={best['s_mid']:.2e}")
