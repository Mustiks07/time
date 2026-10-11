"""Q7 sensitivity: for Fibonacci point sets the origin is not a critical point of U (residual dipole).
Find a genuine critical point (grad U = 0) inside the shell by multistart minimisation of |grad U|^2
with an analytic Hessian-based Newton polish, then redo the minimum-mass calculation with the
habitat centred there (coordinate radius 1 m / U(x*))."""
import numpy as np
from scipy.optimize import minimize, brentq
from q7_mp import fibonacci, U_grad, ball_max, accel
from consts import *

def hess(x, P, m):
    d = x[None, :] - P                       # x - x_i
    r = np.linalg.norm(d, axis=1)
    # d^2/dx_a dx_b (1/r) = (3 d_a d_b - r^2 delta_ab)/r^5
    H = np.einsum('i,ia,ib->ab', m * 3 / r**5, d, d) - np.eye(3) * np.sum(m / r**3)
    return H

def crit_points(P, m, nstart=400, rmax=0.8, seed=0):
    rng = np.random.default_rng(seed)
    found = []
    for k in range(nstart):
        x0 = rng.normal(size=3); x0 *= rmax * rng.uniform() ** (1 / 3) / np.linalg.norm(x0)
        f = lambda x: np.sum(U_grad(np.atleast_2d(x), P, m)[1][0] ** 2)
        res = minimize(f, x0, method='BFGS', options={'gtol': 1e-14})
        x = res.x
        for _ in range(30):                  # Newton polish on grad U = 0
            g = U_grad(np.atleast_2d(x), P, m)[1][0]
            try:
                x = x - np.linalg.solve(hess(x, P, m), g)
            except np.linalg.LinAlgError:
                break
        g = U_grad(np.atleast_2d(x), P, m)[1][0]
        if np.linalg.norm(g) < 1e-12 and np.linalg.norm(x) < 0.9:
            if not any(np.linalg.norm(x - y) < 1e-6 for y in found):
                found.append(x)
    return found

g_geom = g0 / c**2
for N in [100, 1000]:
    P = fibonacci(N); m = 9.0 / N
    cps = crit_points(P, m)
    cps.sort(key=np.linalg.norm)
    print(f"N={N}: {len(cps)} critical points of U with |x|<0.9 rho0; nearest ones:")
    for x in cps[:5]:
        U = U_grad(np.atleast_2d(x), P, m)[0][0]
        ev = np.linalg.eigvalsh(hess(x, P, m))
        print(f"   |x*|={np.linalg.norm(x):.4f} rho0, U={U:.8f}, Hessian eigs={ev}")
    if not cps:
        continue
    xc = cps[0]
    Uc = U_grad(np.atleast_2d(xc), P, m)[0][0]
    rc = 1.0 / Uc   # coordinate radius (m) for 1 m proper radius
    def g(lr):
        rho0 = np.exp(lr)
        amax, _ = ball_max(rc / rho0, P, m, centre=xc, J=4, K=2000, refine=3)
        return np.log(amax / rho0 / g_geom)
    lr = brentq(g, np.log(1.0), np.log(1e20), xtol=1e-10)
    rho0 = np.exp(lr); Mt = 9 * rho0 * KG_PER_M
    # small-r law at the critical point: a ~ ||H|| r / U^2 (linear)
    Hn = np.max(np.abs(np.linalg.eigvalsh(hess(xc, P, m))))
    print(f"   habitat centred at nearest critical point: rho0_min = {rho0:.5e} m, M_total = {Mt:.5e} kg = {Mt/M_sun:.5e} Msun")
    print(f"   (linear law check: rho0 ~ sqrt(|H|max r c^2/(U^2 g0)) = {np.sqrt(Hn*rc/Uc**2/g_geom):.5e} m)")
