"""Q4: Kerr prograde circular equatorial geodesics: u^t, ISCO, ZAMO speed, tidal eigenvalues.

Units G=c=M=1 unless stated. Boyer-Lindquist, equatorial plane:
  g_tt = -(1-2/r), g_tphi = -2a/r, g_phiphi = r^2 + a^2 + 2a^2/r.
Keplerian prograde: Omega = 1/(r^{3/2} + a).
Direct: u^t = 1/sqrt(-(g_tt + 2 g_tphi Omega + g_phiphi Omega^2)).
BPT (1972): u^t = (r^{3/2} + a) / ( r^{3/4} sqrt(r^{3/2} - 3 r^{1/2} + 2a) ).
ISCO (BPT): Z1 = 1 + (1-a^2)^{1/3}[(1+a)^{1/3}+(1-a)^{1/3}], Z2 = sqrt(3a^2+Z1^2),
            r_isco = 3 + Z2 - sqrt((3-Z1)(3+Z1+2Z2)).
ZAMO-measured speed: v = (Omega-omega) sqrt(g_phiphi)/alpha_lapse, omega=-g_tphi/g_phiphi,
            alpha_lapse^2 = -g_tt + g_tphi^2/g_phiphi.  Then u^t = gamma/alpha_lapse.
BPT closed form: v = (r^2 - 2a sqrt(r) + a^2) / ( sqrt(Delta) (r^{3/2} + a) ).
Tidal tensor: E^a_c = R^a_{b c d} u^b u^d (geodesic deviation: D^2 xi/dtau^2 = -E xi).
  Its eigenvalues on the 3-space orthogonal to u are the tidal eigenvalues (u itself has eigenvalue 0).
  The Riemann tensor is computed symbolically (sympy) from the full Kerr metric and evaluated
  with mpmath at high precision.
"""
import sympy as sp
import mpmath as mp
import pickle, os
from consts import *

mp.mp.dps = 50

# ------------------------------------------------------------------ circular orbits
def metric_eq(r, a):
    return -(1 - 2 / r), -2 * a / r, r**2 + a**2 + 2 * a**2 / r

def Omega(r, a):
    return 1 / (r**mp.mpf(1.5) + a)

def ut_direct(r, a):
    gtt, gtp, gpp = metric_eq(r, a)
    W = Omega(r, a)
    return 1 / mp.sqrt(-(gtt + 2 * gtp * W + gpp * W**2))

def ut_bpt(r, a):
    return (r**mp.mpf(1.5) + a) / (r**mp.mpf(0.75) * mp.sqrt(r**mp.mpf(1.5) - 3 * mp.sqrt(r) + 2 * a))

def r_isco(a):
    a = mp.mpf(a)
    Z1 = 1 + mp.cbrt(1 - a**2) * (mp.cbrt(1 + a) + mp.cbrt(1 - a))
    Z2 = mp.sqrt(3 * a**2 + Z1**2)
    return 3 + Z2 - mp.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2))

def zamo(r, a):
    gtt, gtp, gpp = metric_eq(r, a)
    om = -gtp / gpp
    lapse = mp.sqrt(-gtt + gtp**2 / gpp)
    v = (Omega(r, a) - om) * mp.sqrt(gpp) / lapse
    Delta = r**2 - 2 * r + a**2
    v_bpt = (r**2 - 2 * a * mp.sqrt(r) + a**2) / (mp.sqrt(Delta) * (r**mp.mpf(1.5) + a))
    return v, v_bpt, lapse, 1 / mp.sqrt(1 - v**2)

def E_L(r, a):
    den = r**mp.mpf(0.75) * mp.sqrt(r**mp.mpf(1.5) - 3 * mp.sqrt(r) + 2 * a)
    E = (r**mp.mpf(1.5) - 2 * mp.sqrt(r) + a) / den
    L = (r**2 - 2 * a * mp.sqrt(r) + a**2) / den
    return E, L

# ------------------------------------------------------------------ Riemann (symbolic)
t, r, th, ph = sp.symbols('t r theta phi')
A = sp.symbols('a')
X = [t, r, th, ph]
CACHE = '/home/user/time/independent_check/kerr_riemann.pkl'

def build_riemann():
    if os.path.exists(CACHE):
        with open(CACHE, 'rb') as f:
            return pickle.load(f)
    Sig = r**2 + A**2 * sp.cos(th)**2
    Del = r**2 - 2 * r + A**2
    g = sp.zeros(4)
    g[0, 0] = -(1 - 2 * r / Sig)
    g[0, 3] = g[3, 0] = -2 * A * r * sp.sin(th)**2 / Sig
    g[1, 1] = Sig / Del
    g[2, 2] = Sig
    g[3, 3] = (r**2 + A**2 + 2 * A**2 * r * sp.sin(th)**2 / Sig) * sp.sin(th)**2
    ginv = sp.simplify(g.inv())
    dg = [[[sp.diff(g[i, j], X[k]) for k in range(4)] for j in range(4)] for i in range(4)]
    Gam = [[[sp.simplify(sum(ginv[i, l] * (dg[l][j][k] + dg[l][k][j] - dg[j][k][l]) for l in range(4)) / 2)
             for k in range(4)] for j in range(4)] for i in range(4)]
    # R^a_{bcd} = d_c Gam^a_{db} - d_d Gam^a_{cb} + Gam^a_{ce} Gam^e_{db} - Gam^a_{de} Gam^e_{cb}
    Riem = {}
    for a_ in range(4):
        for b in range(4):
            for c_ in range(4):
                for d in range(c_ + 1, 4):
                    expr = (sp.diff(Gam[a_][d][b], X[c_]) - sp.diff(Gam[a_][c_][b], X[d])
                            + sum(Gam[a_][c_][e] * Gam[e][d][b] - Gam[a_][d][e] * Gam[e][c_][b] for e in range(4)))
                    e0 = sp.simplify(expr.subs(th, sp.pi / 2))
                    if e0.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
                        # 0*inf artefact from 1/cos(theta) factors: take the limit theta -> pi/2
                        e0 = sp.simplify(sp.limit(sp.simplify(expr), th, sp.pi / 2))
                        assert not e0.has(sp.nan, sp.zoo, sp.oo, -sp.oo)
                    expr = e0
                    if expr != 0:
                        Riem[(a_, b, c_, d)] = expr
    with open(CACHE, 'wb') as f:
        pickle.dump(Riem, f)
    return Riem

RIEM = build_riemann()
RIEM_F = {k: sp.lambdify((r, A), v, 'mpmath') for k, v in RIEM.items()}

def riemann_num(rv, av):
    R = {}
    for (a_, b, c_, d), f in RIEM_F.items():
        val = f(rv, av)
        R[(a_, b, c_, d)] = val
        R[(a_, b, d, c_)] = -val
    return R

def tidal_eigs(rv, av, return_matrix=False):
    """E_ij = R_{abcd} e_i^a u^b e_j^c u^d in an orthonormal tetrad comoving with the orbiter;
    tidal acceleration = -E_ij xi^j. Returns eigenvalues of the symmetric 3x3 matrix."""
    rv, av = mp.mpf(rv), mp.mpf(av)
    gtt, gtp, gpp = metric_eq(rv, av)
    grr = rv**2 / (rv**2 - 2 * rv + av**2)
    gthth = rv**2
    g = mp.matrix(4, 4)
    g[0, 0] = gtt; g[0, 3] = g[3, 0] = gtp; g[3, 3] = gpp; g[1, 1] = grr; g[2, 2] = gthth
    W = Omega(rv, av)
    ut = ut_direct(rv, av)
    u = [ut, 0, 0, ut * W]
    # spatial tetrad
    e_r = [0, 1 / mp.sqrt(grr), 0, 0]
    e_th = [0, 0, 1 / mp.sqrt(gthth), 0]
    k = -(gtp + W * gpp) / (gtt + W * gtp)       # w = d_phi + k d_t is orthogonal to u
    w = [k, 0, 0, 1]
    nw = mp.sqrt(sum(g[i, j] * w[i] * w[j] for i in range(4) for j in range(4)))
    e_ph = [x / nw for x in w]
    tet = [e_r, e_th, e_ph]
    # orthonormality check
    for i, ei in enumerate(tet + [u]):
        for j, ej in enumerate(tet + [u]):
            val = sum(g[p, q] * ei[p] * ej[q] for p in range(4) for q in range(4))
            target = (1 if i == j else 0) if i < 3 else (-1 if j == 3 else 0)
            if i == 3 and j < 3: target = 0
            assert abs(val - target) < mp.mpf(10)**(-30), (i, j, val)
    Rup = riemann_num(rv, av)
    # lower first index: R_{abcd} = g_{ae} R^e_{bcd}
    Rdn = {}
    for (e, b, c_, d), val in Rup.items():
        for a_ in range(4):
            if g[a_, e] != 0:
                Rdn[(a_, b, c_, d)] = Rdn.get((a_, b, c_, d), 0) + g[a_, e] * val
    E = mp.matrix(3, 3)
    for i in range(3):
        for j in range(3):
            s = 0
            for (a_, b, c_, d), val in Rdn.items():
                s += val * tet[i][a_] * u[b] * tet[j][c_] * u[d]
            E[i, j] = s
    asym = max(abs(E[i, j] - E[j, i]) for i in range(3) for j in range(3))
    assert asym < mp.mpf(10)**(-30), asym
    ev = mp.eigsy(E, eigvals_only=True)
    ev = [ev[i] for i in range(3)]
    if return_matrix:
        return ev, E
    return ev

def marck(rv, av):
    """Closed form (Marck 1983) for equatorial circular geodesics, K=(L - aE)^2:
       {1, 1+3K/r^2, -(2+3K/r^2)} * M/r^3   (phi-like, vertical, radial)."""
    E, L = E_L(rv, av)
    K = (L - av * E)**2
    return [1 / rv**3, (1 + 3 * K / rv**2) / rv**3, -(2 + 3 * K / rv**2) / rv**3]

def M_min_tidal(lmax_geom, L=2.0):
    """lambda = lmax_geom / M^2 (geom). Tidal accel over length L: lambda c^2 L <= g0
       => M_geom >= sqrt(lmax_geom c^2 L / g0)."""
    Mg = mp.sqrt(lmax_geom * c**2 * L / g0)
    return Mg * KG_PER_M / M_sun

if __name__ == "__main__":
    print("Kerr Riemann components (equatorial) computed:", len(RIEM))
    # --- Schwarzschild check of tidal eigenvalues
    print("\nSchwarzschild check (a=0): numeric Riemann vs M/r^3 {1, r/(r-3), -(2r-3)/(r-3)}")
    for rv in [6, 10, 4]:
        num = tidal_eigs(rv, 0)
        exp = [mp.mpf(1) / rv**3, mp.mpf(rv) / (rv - 3) / rv**3, -mp.mpf(2 * rv - 3) / (rv - 3) / rv**3]
        print(f" r={rv}: numeric {[mp.nstr(x, 15) for x in sorted(num)]}  expected {[mp.nstr(x, 15) for x in sorted(exp)]}")
    print("\nKerr check: numeric Riemann vs Marck closed form at a=0.7, r=5:")
    print("  ", [mp.nstr(x, 15) for x in sorted(tidal_eigs(5, 0.7))], [mp.nstr(x, 15) for x in sorted(marck(mp.mpf(5), mp.mpf(0.7)))])

    # --- u^t direct vs BPT
    print("\nu^t direct vs BPT checks:")
    for av, rv in [(0, 6), (0.5, 4.3), (0.9, 2.5), (0.99, 1.6)]:
        print(f"  a={av} r={rv}: direct={mp.nstr(ut_direct(mp.mpf(rv), mp.mpf(av)), 18)} BPT={mp.nstr(ut_bpt(mp.mpf(rv), mp.mpf(av)), 18)}")

    def report(av):
        av = mp.mpf(av)
        ri = r_isco(av)
        ud, ub = ut_direct(ri, av), ut_bpt(ri, av)
        v, vb, lapse, gam = zamo(ri, av)
        ev = sorted(tidal_eigs(ri, av))
        evm = sorted(marck(ri, av))
        lmax = max(abs(x) for x in ev)
        return ri, ud, ub, v, vb, lapse, gam, ev, evm, lmax

    # --- (a) solve u^t(ISCO)=10, parametrise by delta = 1-a (log scale)
    f = lambda ld: ut_bpt(r_isco(1 - mp.e**ld), 1 - mp.e**ld) - 10
    ld = mp.findroot(f, mp.log(1e-5))
    a10 = 1 - mp.e**ld
    print(f"\n(a) chi with u^t(ISCO)=10: chi = {mp.nstr(a10, 15)}  (1-chi = {mp.nstr(1-a10, 10)})")
    results = {}
    for name, av in [('chi_ut10', a10), ('0', 0), ('0.9', 0.9), ('0.99', 0.99), ('0.998', 0.998)]:
        ri, ud, ub, v, vb, lapse, gam, ev, evm, lmax = report(av)
        Mmin = M_min_tidal(lmax)
        results[name] = dict(chi=av, r=ri, ut=ud, ut_bpt=ub, v=v, v_bpt=vb, lapse=lapse, gam=gam,
                             eig=ev, eig_marck=evm, lmax=lmax, Mmin=Mmin, Mmin_1m=M_min_tidal(lmax, 1.0))
        print(f"\nchi={mp.nstr(av, 12)}: r_isco={mp.nstr(ri, 12)} M ; u^t direct={mp.nstr(ud, 12)} BPT={mp.nstr(ub, 12)}")
        print(f"   ZAMO speed v={mp.nstr(v, 10)} c (BPT formula {mp.nstr(vb, 10)}), ZAMO gamma={mp.nstr(gam, 10)}, lapse={mp.nstr(lapse, 10)}  [u^t = gamma/lapse = {mp.nstr(gam/lapse, 10)}]")
        print(f"   tidal eigenvalues (units 1/M^2): {[mp.nstr(x, 10) for x in ev]}")
        print(f"   Marck closed form             : {[mp.nstr(x, 10) for x in evm]}")
        print(f"   |lambda|max = {mp.nstr(lmax, 10)} /M^2 -> M_min (2 m, 1 g0) = {mp.nstr(Mmin, 8)} Msun ; (1 m) = {mp.nstr(M_min_tidal(lmax, 1.0), 8)} Msun")
    with open('/home/user/time/independent_check/q4_results.pkl', 'wb') as fh:
        pickle.dump({k: {kk: (str(vv) if not isinstance(vv, list) else [str(x) for x in vv]) for kk, vv in d.items()} for k, d in results.items()}, fh)

    # sensitivity: how does u^t(ISCO) scale near extremality
    print("\nnear-extremal scaling u^t(ISCO) vs 1-chi:")
    for dlt in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8]:
        av = 1 - mp.mpf(dlt)
        ri = r_isco(av)
        print(f"   1-chi={dlt:.0e}: r_isco-1={mp.nstr(ri-1, 6)}, u^t={mp.nstr(ut_bpt(ri, av), 8)}")
