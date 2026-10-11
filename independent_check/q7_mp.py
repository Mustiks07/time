"""Q7: Majumdar-Papapetrou static observers' acceleration inside a shell of extremal BHs.

ds^2 = -U^{-2} dt^2 + U^2 dx.dx,  U = 1 + sum_i m_i/|x - x_i|.
Static observer 4-acceleration: a_mu = d_mu ln sqrt(-g_tt) = -d_mu U / U;
|a| = sqrt(g^{ij} a_i a_j) = (1/U) |grad U| / U = |grad U| / U^2  (flat gradient), units 1/length
-> multiply by c^2 for SI.
N equal masses at radius rho0, U(0) = 1 + N m/rho0 = 10  => m = 9 rho0/N. Total (ADM) mass = N m = 9 rho0.
Icosahedron = spherical 5-design: multipoles l=1..5 vanish, so near the centre
   U = 10 + (m/rho0) x^6 S6(n) + ...,  S6(n) = sum_i P6(n.n_i), x=r/rho0
   |grad U| = (m/rho0^2) x^5 sqrt(36 S6^2 + |grad_Omega S6|^2) + ...
   => a ~ C x^5 / rho0 (C computed below), so the leading behaviour is a quintic.
Fibonacci-sphere points are not exactly balanced: a residual dipole sum_i n_i != 0 gives a(0) != 0.
Habitat: proper radius 1 m ~ U(0) r => coordinate radius r = 0.1 m. Condition: max_{|x|<=r} a <= g0/c^2.
"""
import numpy as np
import mpmath as mp
from scipy.optimize import minimize, brentq
from consts import *

# ------------------------------------------------------------- point sets (rho0 = 1)
def icosahedron():
    phi = (1 + 5**0.5) / 2
    v = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            v += [(0, s1, s2 * phi), (s1, s2 * phi, 0), (s2 * phi, 0, s1)]
    v = np.array(v, float)
    return v / np.linalg.norm(v, axis=1)[:, None]

def fibonacci(N, convention='half'):
    i = np.arange(N)
    if convention == 'half':        # z_i = 1 - (2i+1)/N  (most common)
        z = 1 - (2 * i + 1) / N
    else:                           # z_i = 1 - 2i/(N-1)  (includes the poles)
        z = 1 - 2 * i / (N - 1)
    ga = np.pi * (3 - np.sqrt(5))
    th = ga * i
    rr = np.sqrt(np.clip(1 - z * z, 0, None))
    return np.stack([rr * np.cos(th), rr * np.sin(th), z], axis=1)

# ------------------------------------------------------------- field
def U_grad(x, P, m):
    d = P[None, :, :] - x[:, None, :]           # x_i - x
    dist = np.linalg.norm(d, axis=2)
    U = 1 + m * np.sum(1 / dist, axis=1)
    gU = m * np.sum(d / dist[..., None]**3, axis=1)   # grad of 1/|x-x_i| = (x_i - x)/|x-x_i|^3
    return U, gU

def accel(x, P, m):
    U, gU = U_grad(np.atleast_2d(x), P, m)
    return np.linalg.norm(gU, axis=1) / U**2

def sphere_max(s, P, m, centre, K=6000, refine=16, seed=1):
    if s == 0:
        return accel(centre, P, m)[0]
    rng = np.random.default_rng(seed)
    n = rng.normal(size=(K, 3)); n /= np.linalg.norm(n, axis=1)[:, None]
    n = np.vstack([n, P / np.linalg.norm(P, axis=1)[:, None]])   # also seed along every BH direction
    vals = accel(centre + s * n, P, m)
    best = vals.max()
    idx = np.argsort(vals)[-refine:]
    for k in idx:
        th0 = np.arccos(np.clip(n[k, 2], -1, 1)); ph0 = np.arctan2(n[k, 1], n[k, 0])
        f = lambda q: -accel(centre + s * np.array([np.sin(q[0]) * np.cos(q[1]), np.sin(q[0]) * np.sin(q[1]), np.cos(q[0])]), P, m)[0]
        res = minimize(f, [th0, ph0], method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-16, 'maxiter': 4000})
        best = max(best, -res.fun)
    return best

def ball_max(r, P, m, centre=np.zeros(3), J=12, **kw):
    """max over |x - centre| <= r = max over spheres s in (0, r] (radial grid) and centre."""
    vals = [sphere_max(0.0, P, m, centre)] + [sphere_max(r * j / J, P, m, centre, **kw) for j in range(1, J + 1)]
    return max(vals), int(np.argmax(vals))

# ------------------------------------------------------------- high-precision (for the icosahedron)
mp.mp.dps = 40
def accel_mp(x, P, m):
    x = [mp.mpf(float(v)) for v in x]
    U = mp.mpf(1); g = [mp.mpf(0)] * 3
    for p in P:
        d = [mp.mpf(repr(p[k])) - x[k] for k in range(3)]
        dist = mp.sqrt(d[0]**2 + d[1]**2 + d[2]**2)
        U += m / dist
        for k in range(3):
            g[k] += m * d[k] / dist**3
    return mp.sqrt(g[0]**2 + g[1]**2 + g[2]**2) / U**2

def ico_points_mp():
    phi = (1 + mp.sqrt(5)) / 2
    v = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            v += [(0, s1, s2 * phi), (s1, s2 * phi, 0), (s2 * phi, 0, s1)]
    out = []
    for p in v:
        nrm = mp.sqrt(sum(mp.mpf(c_)**2 for c_ in p))
        out.append([mp.mpf(c_) / nrm for c_ in p])
    return out

def accel_mp_exact(xdir, s, Pm, m):
    x = [s * mp.mpf(float(v)) for v in xdir]
    U = mp.mpf(1); g = [mp.mpf(0)] * 3
    for p in Pm:
        d = [p[k] - x[k] for k in range(3)]
        dist = mp.sqrt(d[0]**2 + d[1]**2 + d[2]**2)
        U += m / dist
        for k in range(3):
            g[k] += m * d[k] / dist**3
    return mp.sqrt(g[0]**2 + g[1]**2 + g[2]**2) / U**2

def ico_C_multipole():
    """Leading coefficient C in a_max ~ C x^5 (units 1/rho0) from the l=6 multipole."""
    P = icosahedron(); m = 9 / 12
    from scipy.special import eval_legendre
    def S6_and_grad(nv):
        nv = nv / np.linalg.norm(nv)
        c_ = P @ nv
        S = np.sum(eval_legendre(6, c_))
        # dP6/dc
        P6p = np.polynomial.legendre.Legendre.basis(6).deriv()
        dP = np.sum(P6p(c_)[:, None] * (P - np.outer(c_, nv)), axis=0)   # tangential gradient of S6
        return S, dP
    rng = np.random.default_rng(3)
    best = 0; bestn = None
    n = rng.normal(size=(20000, 3)); n /= np.linalg.norm(n, axis=1)[:, None]
    for nv in n[:4000]:
        S, dS = S6_and_grad(nv)
        val = np.sqrt(36 * S * S + dS @ dS)
        if val > best:
            best, bestn = val, nv
    f = lambda q: -np.sqrt(36 * S6_and_grad(np.array([np.sin(q[0]) * np.cos(q[1]), np.sin(q[0]) * np.sin(q[1]), np.cos(q[0])]))[0]**2
                           + np.sum(S6_and_grad(np.array([np.sin(q[0]) * np.cos(q[1]), np.sin(q[0]) * np.sin(q[1]), np.cos(q[0])]))[1]**2))
    th0 = np.arccos(bestn[2]); ph0 = np.arctan2(bestn[1], bestn[0])
    res = minimize(f, [th0, ph0], method='Nelder-Mead', options={'xatol': 1e-12, 'fatol': 1e-14})
    q = res.x
    nbest = np.array([np.sin(q[0]) * np.cos(q[1]), np.sin(q[0]) * np.sin(q[1]), np.cos(q[0])])
    return m * (-res.fun) / 100.0, nbest   # U^2 = 100 at leading order

if __name__ == "__main__":
    g_geom = g0 / c**2
    r_coord = 1.0 / 10.0          # proper 1 m ~ U(0) r
    print(f"g0/c^2 = {g_geom:.6e} 1/m ; habitat coordinate radius r = {r_coord} m")
    xs = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    configs = {'ico12': icosahedron(), 'fib100': fibonacci(100), 'fib1000': fibonacci(1000)}
    summary = {}
    for name, P in configs.items():
        N = len(P); m = 9.0 / N
        U0 = 1 + m * np.sum(1 / np.linalg.norm(P, axis=1))
        dip = np.linalg.norm(P.sum(axis=0))
        # minimum pairwise separation (coordinate) vs m
        D = np.linalg.norm(P[:, None] - P[None], axis=2); D[D == 0] = 9
        print(f"\n=== {name}: N={N}, m={m:.6g} rho0, U(0)={U0:.12f}, |sum n_i|={dip:.3e}, min sep={D.min():.4f} rho0")
        a0 = accel(np.zeros(3), P, m)[0]
        print(f"   a(0) = {a0:.6e} c^2/rho0   (expected 0.09*|sum n|/N = {0.09*dip/N:.6e})")
        tab = []
        for x in xs:
            amax, jarg = ball_max(x, P, m)
            tab.append((x, amax))
            print(f"   r/rho0={x:5.2f}: a_max over ball = {amax:.6e} c^2/rho0  (argmax shell {jarg}/12)")
        summary[name] = dict(N=N, m=m, dip=dip, a0=a0, tab=tab)

    # ---- icosahedron small-x law
    C, nb = ico_C_multipole()
    print(f"\nicosahedron l=6 multipole: a_max ~ C x^5 / rho0, C = {C:.8f}  (direction {nb})")
    Pm = ico_points_mp(); mm = mp.mpf(9) / 12
    for x in [1e-4, 1e-3, 1e-2]:
        v = accel_mp_exact(nb, mp.mpf(x), Pm, mm)
        print(f"   mpmath direct at x={x:g} along argmax direction: a/x^5 = {mp.nstr(v / mp.mpf(x)**5, 10)}")
    # double-precision ball max at moderate x, divided by x^5
    for x in [0.02, 0.05, 0.1]:
        print(f"   double-precision ball max at x={x}: a/x^5 = {summary['ico12']['tab'][xs.index(x)][1] / x**5:.6f}")

    # ---- minimum total mass, icosahedron: C (r/rho0)^5 / rho0 = g/c^2 -> rho0 = (C r^5 c^2/g0)^(1/6)
    rho0_ico = (C * r_coord**5 / g_geom)**(1 / 6)
    # refine with mpmath exact field (maximum over the ball is at the boundary in this regime)
    def f_ico(lr):
        rho0 = mp.e**lr
        x = mp.mpf(r_coord) / rho0
        return mp.log(accel_mp_exact(nb, x, Pm, mm) / rho0 / mp.mpf(g_geom))
    lr = mp.findroot(f_ico, mp.log(rho0_ico))
    rho0_ico_exact = float(mp.e**lr)
    Mtot = 9 * rho0_ico_exact * KG_PER_M
    print(f"\nICO: rho0_min = {rho0_ico:.6f} m (leading law), {rho0_ico_exact:.6f} m (exact mp field);"
          f" r/rho0 = {r_coord/rho0_ico_exact:.3e}; each m = {0.75*rho0_ico_exact:.4f} m")
    print(f"     M_total = 9 rho0 c^2/G = {Mtot:.5e} kg = {Mtot/M_sun:.5f} Msun = {Mtot/M_J:.2f} M_J")
    # verify the boundary maximum is the ball maximum at this scale via double precision is not possible
    # (cancellation); check with mpmath on a few interior radii along the best direction & random directions
    rng = np.random.default_rng(7)
    xb = mp.mpf(r_coord) / rho0_ico_exact
    worst = max(accel_mp_exact(n_ / np.linalg.norm(n_), xb, Pm, mm) for n_ in rng.normal(size=(200, 3)))
    print(f"     check: random-direction max on boundary {mp.nstr(worst/mp.mpf(rho0_ico_exact)/mp.mpf(g_geom), 6)} g0 (<=1 expected),"
          f" along argmax {mp.nstr(accel_mp_exact(nb, xb, Pm, mm)/mp.mpf(rho0_ico_exact)/mp.mpf(g_geom), 8)} g0")
    summary['ico12']['rho0'] = rho0_ico_exact; summary['ico12']['Mtot'] = Mtot; summary['ico12']['C'] = C

    # ---- Fibonacci: solve a_max(r/rho0)/rho0 = g/c^2 (centred at the coordinate origin)
    for name in ['fib100', 'fib1000']:
        P = configs[name]; N = len(P); m = 9.0 / N
        def g(lr):
            rho0 = np.exp(lr)
            amax, _ = ball_max(r_coord / rho0, P, m, J=4, K=2000, refine=3)
            return np.log(amax / rho0 / g_geom)
        lr = brentq(g, np.log(1.0), np.log(1e20), xtol=1e-10)
        rho0 = np.exp(lr); Mtot = 9 * rho0 * KG_PER_M
        print(f"\n{name} (origin-centred): rho0_min = {rho0:.6e} m, r/rho0 = {r_coord/rho0:.2e};"
              f" M_total = {Mtot:.5e} kg = {Mtot/M_sun:.5e} Msun")
        print(f"     (dominated by residual dipole: a(0)*c^2 at this rho0 = {summary[name]['a0']/rho0/g_geom:.6f} g0)")
        summary[name]['rho0'] = rho0; summary[name]['Mtot'] = Mtot
        # (critical-point-centred variant: see q7_crit.py)
        # convention sensitivity
        P2 = fibonacci(N, 'poles')
        print(f"     (alt. Fibonacci convention z=1-2i/(N-1): |sum n|={np.linalg.norm(P2.sum(axis=0)):.3e})")
    np.save('/home/user/time/independent_check/q7_summary.npy', summary, allow_pickle=True)
