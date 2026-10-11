"""Static spherically symmetric spacetimes: inverse problem, energy conditions,
a cavity-redshift mass bound, and optimisation of the mass needed for a given
interior time-dilation factor.

Metric:  ds² = -e^{2Φ(r)} dt² + dr²/(1 - 2m(r)/r) + r² dΩ²,   G = c = 1.
Einstein equations (verified symbolically in einstein_tensor_check()):
    ρ   = m' / (4π r²)
    p_r = [Φ' r (r - 2m) - m] / (4π r³)
    p_t = p_r + (r/2) [p_r' + (ρ + p_r) Φ']        (∇_μ T^{μr} = 0)

A *cavity* is a vacuum region r < r_in, which is flat (m = 0) with constant lapse
α = e^{Φ(r_in)}; static clocks there run at rate α relative to clocks at infinity.
Z ≡ 1/α is the time-dilation factor requested by the programme (Z = 10).
"""
from __future__ import annotations

import math

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

PI = math.pi

# ---------------------------------------------------------------------------
# 1. Symbolic check of the field equations
# ---------------------------------------------------------------------------

def einstein_tensor_check():
    """Return the mixed Einstein tensor components for the general static metric and the
    residuals against the formulas in the module docstring (all should simplify to 0)."""
    t, r, th, ph = sp.symbols('t r theta phi', positive=True)
    Phi = sp.Function('Phi')(r)
    m = sp.Function('m')(r)
    X = [t, r, th, ph]
    g = sp.diag(-sp.exp(2 * Phi), 1 / (1 - 2 * m / r), r**2, r**2 * sp.sin(th) ** 2)
    ginv = g.inv()
    n = 4
    Gam = [[[sum(ginv[i, l] * (sp.diff(g[l, j], X[k]) + sp.diff(g[l, k], X[j]) - sp.diff(g[j, k], X[l]))
                 for l in range(n)) / 2 for k in range(n)] for j in range(n)] for i in range(n)]
    Ric = sp.zeros(n, n)
    for j in range(n):
        for l in range(n):
            Ric[j, l] = sp.simplify(sum(sp.diff(Gam[i][j][l], X[i]) - sp.diff(Gam[i][j][i], X[l])
                                        + sum(Gam[i][i][k] * Gam[k][j][l] - Gam[i][l][k] * Gam[k][j][i]
                                              for k in range(n)) for i in range(n)))
    Rs = sp.simplify(sum(ginv[i, j] * Ric[i, j] for i in range(n) for j in range(n)))
    Gmix = sp.simplify(ginv * Ric - sp.eye(n) * Rs / 2)
    rho = sp.diff(m, r) / (4 * sp.pi * r**2)
    pr = (sp.diff(Phi, r) * r * (r - 2 * m) - m) / (4 * sp.pi * r**3)
    pt = pr + r / 2 * (sp.diff(pr, r) + (rho + pr) * sp.diff(Phi, r))
    res = [sp.simplify(Gmix[0, 0] / (8 * sp.pi) + rho),
           sp.simplify(Gmix[1, 1] / (8 * sp.pi) - pr),
           sp.simplify(Gmix[2, 2] / (8 * sp.pi) - pt)]
    return res


# ---------------------------------------------------------------------------
# 2. Numerical inverse problem
# ---------------------------------------------------------------------------

def stress_energy_from_profiles(r, m, Phi):
    """Given samples of m(r) and Φ(r), return ρ, p_r, p_t via finite differences."""
    dm = np.gradient(m, r, edge_order=2)
    dPhi = np.gradient(Phi, r, edge_order=2)
    rho = dm / (4 * PI * r**2)
    pr = (dPhi * r * (r - 2 * m) - m) / (4 * PI * r**3)
    dpr = np.gradient(pr, r, edge_order=2)
    pt = pr + 0.5 * r * (dpr + (rho + pr) * dPhi)
    return rho, pr, pt


def energy_conditions(rho, pr, pt, tol=0.0):
    return dict(
        NEC=bool(np.all(rho + pr >= -tol) and np.all(rho + pt >= -tol)),
        WEC=bool(np.all(rho >= -tol) and np.all(rho + pr >= -tol) and np.all(rho + pt >= -tol)),
        SEC=bool(np.all(rho + pr >= -tol) and np.all(rho + pt >= -tol) and np.all(rho + pr + 2 * pt >= -tol)),
        DEC=bool(np.all(rho >= np.abs(pr) - tol) and np.all(rho >= np.abs(pt) - tol)),
    )


# ---------------------------------------------------------------------------
# 3. Cavity-redshift bound:  α_cavity ≥ 1 - 2M/r_in   (if ρ ≥ 0 and p_r ≤ ρ)
# ---------------------------------------------------------------------------

def alpha_lower_bound(M, r_in):
    """Lower bound on the cavity lapse, valid for ρ ≥ 0, p_r ≤ ρ, no horizons.

    Proof sketch (full proof in reports/REPORT_05):  with y = 2m/r,
        Φ' = (m + 4πr³p_r)/(r(r-2m)) ≤ y/(r(1-y)) + y'/(2(1-y)),
    the second term integrates to zero between the cavity (y=0) and infinity (y=0), and
    y ≤ 2M/r (m ≤ M because ρ ≥ 0), so  ln(1/α) ≤ ∫_{r_in}^∞ (2M/r²)/(1-2M/r) dr = -ln(1-2M/r_in).
    """
    return 1.0 - 2.0 * M / r_in


def min_mass_bound(Z, r_in=1.0):
    """M ≥ (1 - 1/Z) r_in / 2."""
    return 0.5 * (1.0 - 1.0 / Z) * r_in


def integrate_profile(r_grid, rho, pr, r_in=None):
    """Integrate m and Φ outward for *given* ρ(r), p_r(r) (p_t is then implied by conservation).
    Returns ln α at the cavity, M, and max 2m/r.  Outer boundary: Schwarzschild match."""
    r = np.asarray(r_grid)
    m = np.concatenate(([0.0], np.cumsum(0.5 * (4 * PI * r[1:]**2 * rho[1:] + 4 * PI * r[:-1]**2 * rho[:-1]) * np.diff(r))))
    y = 2 * m / r
    if np.any(y >= 1):
        return None
    dPhi = (m + 4 * PI * r**3 * pr) / (r * (r - 2 * m))
    integral = np.sum(0.5 * (dPhi[1:] + dPhi[:-1]) * np.diff(r))
    M = m[-1]
    lnalpha = 0.5 * math.log(1 - 2 * M / r[-1]) - integral
    return dict(lnalpha=lnalpha, M=M, ymax=float(y.max()), m=m, dPhi=dPhi)


def random_profile_test(n_trials=3000, seed=1):
    """Monte-Carlo test of the bound with random ρ ≥ 0, p_r ≤ ρ (p_t unconstrained).

    Returns (max tightness, worst case) where tightness = ln(1/α) / (-ln(1 - 2M/r_in)).
    The theorem states tightness ≤ 1."""
    rng = np.random.default_rng(seed)
    worst = -math.inf
    worst_case = None
    for _ in range(n_trials):
        r_out = 1 + rng.uniform(0.01, 20)
        r = np.linspace(1.0, r_out, 800)
        nb = rng.integers(1, 6)
        rho = np.zeros_like(r)
        for _k in range(nb):
            c0 = rng.uniform(1, r_out)
            w = rng.uniform(0.01, 3)
            rho += rng.uniform(0, 1) * np.exp(-0.5 * ((r - c0) / w) ** 2)
        m_tot = np.trapezoid(4 * PI * r**2 * rho, r)
        rho *= rng.uniform(0.05, 1.0) * 0.49 * r_out / max(m_tot, 1e-300)
        frac = rng.choice([1.0, rng.uniform(-1, 1)])     # p_r = ρ is the extreme admissible case
        pr = frac * rho
        res = integrate_profile(r, rho, pr)
        if res is None or 2 * res['M'] >= 1.0:
            continue
        tight = -res['lnalpha'] / (-math.log(1 - 2 * res['M']))
        if tight > worst:
            worst, worst_case = tight, dict(alpha=math.exp(res['lnalpha']), M=res['M'], ymax=res['ymax'])
    return worst, worst_case


def adversarial_bound_search(n_bumps=4, n_starts=20, seed=3):
    """Try hard (Nelder–Mead over Gaussian-bump profiles, p_r = ρ) to *maximise* the tightness
    ratio; a value > 1 would falsify the bound."""
    from scipy.optimize import minimize as _min
    rng = np.random.default_rng(seed)
    r = np.linspace(1.0, 30.0, 3000)

    def tight(x):
        x = np.clip(x, -30, 30)
        rho = np.zeros_like(r)
        for k in range(n_bumps):
            amp, c0, w = math.exp(x[3 * k]), 1 + 29 / (1 + math.exp(-x[3 * k + 1])), math.exp(x[3 * k + 2])
            rho += amp * np.exp(-0.5 * ((r - c0) / w) ** 2)
        frac = 1 / (1 + math.exp(-x[-1]))                 # total mass fraction of the r_in/2 limit
        m_tot = np.trapezoid(4 * PI * r**2 * rho, r)
        rho *= frac * 0.4999 / max(m_tot, 1e-300)
        if not np.all(np.isfinite(rho)):
            return 0.0
        res = integrate_profile(r, rho, rho)
        if res is None or 2 * res['M'] >= 0.999 or res['M'] < 1e-9:
            return 0.0
        return -res['lnalpha'] / (-math.log(1 - 2 * res['M']))

    best = 0.0
    for _ in range(n_starts):
        x0 = rng.normal(0, 1, 3 * n_bumps + 1)
        res = _min(lambda x: -tight(x), x0, method='Nelder-Mead', options=dict(maxiter=4000, xatol=1e-6, fatol=1e-9))
        best = max(best, -res.fun)
    return best


# ---------------------------------------------------------------------------
# 4. Nested thin shells (p_r = 0 between shells): mass minimisation under DEC
# ---------------------------------------------------------------------------

def _shell_sp(a, b, R):
    fa = math.sqrt(1 - 2 * a / R)
    fb = math.sqrt(1 - 2 * b / R)
    sigma = (fa - fb) / (4 * PI * R)
    p = ((1 - b / R) / fb - (1 - a / R) / fa) / (8 * PI * R)
    return sigma, p


def nested_shells(q, d):
    """q_i = 2 m_i / R_i (outer side of shell i), R_1 = 1, R_{i+1} = R_i e^{d_i}.
    Returns radii, cumulative masses, per-shell (σ, p), ln α at the centre."""
    K_ = len(q)
    R = np.concatenate(([1.0], np.exp(np.cumsum(d))))
    m = 0.5 * np.asarray(q) * R
    sp_list = []
    prev = 0.0
    for i in range(K_):
        sp_list.append(_shell_sp(prev, m[i], R[i]))
        prev = m[i]
    lnalpha = 0.5 * math.log(1 - 2 * m[-1] / R[-1])
    for i in range(K_ - 1):
        lnalpha += 0.5 * (math.log(1 - 2 * m[i] / R[i]) - math.log(1 - 2 * m[i] / R[i + 1]))
    return R, m, sp_list, lnalpha


def optimise_nested_shells(Z, K_shells, w_max=1.0, n_starts=30, seed=0, beta2_max=None):
    """Minimise M/R_1 for K thin shells, each obeying |p| ≤ w_max σ (DEC: w_max = 1;
    causal kinetic 2-D gas: w_max = 1/2) and σ ≥ 0, with central α ≤ 1/Z.
    If beta2_max is given, every shell must also be radially stable for some barotropic
    EOS with dp/dσ ≤ beta2_max (shells decouple at linear order, see
    thin_shell.stability_beta2_min_general)."""
    from .thin_shell import stability_beta2_min_general
    rng = np.random.default_rng(seed)
    target = math.log(Z)
    nv = 2 * K_shells - 1

    def split(x):
        return x[:K_shells], x[K_shells:]

    def obj(x):
        q, d = split(x)
        R = np.concatenate(([1.0], np.exp(np.cumsum(d))))
        return 0.5 * q[-1] * R[-1]

    def cons(x):
        q, d = split(x)
        try:
            R, m, spl, lna = nested_shells(q, d)
        except ValueError:
            return -np.ones(3 * K_shells + 1 + (K_shells if beta2_max is not None else 0))
        c = [-lna - target]
        for (s, p) in spl:
            c += [w_max * s - p, w_max * s + p, s]
        if beta2_max is not None:
            prev = 0.0
            for i in range(K_shells):
                try:
                    b2, above = stability_beta2_min_general(prev, m[i], R[i])
                    c.append(beta2_max - b2 if above else -1.0)
                except (ValueError, ZeroDivisionError):
                    c.append(-1.0)
                prev = m[i]
        return np.array(c)

    best = None
    for _ in range(n_starts):
        q0 = np.sort(rng.uniform(0.2, 0.9, K_shells))
        d0 = rng.uniform(0.05, 1.0, K_shells - 1)
        x0 = np.concatenate((q0, d0))
        bounds = [(1e-6, 1 - 1e-6)] * K_shells + [(1e-5, 6.0)] * (K_shells - 1)
        try:
            res = minimize(obj, x0, method='SLSQP', bounds=bounds,
                           constraints=[{'type': 'ineq', 'fun': cons}],
                           options=dict(maxiter=3000, ftol=1e-12))
        except Exception:
            continue
        if res.success and np.all(cons(res.x) > -1e-7):
            if best is None or res.fun < best.fun:
                best = res
    if best is None:
        return None
    q, d = split(best.x)
    R, m, spl, lna = nested_shells(q, d)
    return dict(M_over_rin=best.fun, R=R, m=m, shells=spl, Z=math.exp(-lna))


# ---------------------------------------------------------------------------
# 5. Continuum anisotropic shells: direct inverse-problem optimisation
# ---------------------------------------------------------------------------
#
# Work with s = ln(r/r_in) and the dimensionless functions
#     y(s) = 2m/r,   P(s) = r Φ'(r) = dΦ/ds.
# Then (all multiplied by 8π r²):
#     ρ̃   = y + y_s
#     p̃_r = 2P(1 - y) - y
#     p̃_t = ½ [ d p̃_r/ds + (ρ̃ + p̃_r) P ]
# and  ln Z = ∫_0^{s_out} P ds - ½ ln(1 - y_out),   M / r_in = ½ y_out e^{s_out}.
# Boundary conditions: y(0) = 0 and p̃_r(0) = 0 (⇒ P(0) = 0) at the cavity wall;
# p̃_r(s_out) = 0 (⇒ P = y/(2(1-y))) at the outer edge, matching Schwarzschild.
# y and P are piecewise linear on N uniform nodes in s, so every quantity is analytic.

def continuum_fields(y_nodes, P_nodes, s_out, n_sub=8):
    N = len(y_nodes) - 1
    s_nodes = np.linspace(0.0, s_out, N + 1)
    h = s_out / N
    out = []
    for k in range(N):
        ys = (y_nodes[k + 1] - y_nodes[k]) / h
        Ps = (P_nodes[k + 1] - P_nodes[k]) / h
        for j in range(n_sub + 1):
            t = j / n_sub
            y = y_nodes[k] + t * (y_nodes[k + 1] - y_nodes[k])
            P = P_nodes[k] + t * (P_nodes[k + 1] - P_nodes[k])
            rho = y + ys
            pr = 2 * P * (1 - y) - y
            dpr = 2 * Ps * (1 - y) - 2 * P * ys - ys
            pt = 0.5 * (dpr + (rho + pr) * P)
            out.append((s_nodes[k] + t * h, y, P, rho, pr, pt))
    return np.array(out)


def continuum_lnZ(y_nodes, P_nodes, s_out):
    N = len(y_nodes) - 1
    h = s_out / N
    integral = h * (0.5 * P_nodes[0] + np.sum(P_nodes[1:-1]) + 0.5 * P_nodes[-1])   # exact for linear P
    return integral - 0.5 * math.log(1 - y_nodes[-1])


def optimise_continuum(Z, N=24, mode='DEC', n_starts=12, seed=0, maxiter=600, w_rad=1.0):
    """Minimise M/r_in subject to ln Z ≥ target and pointwise conditions:
    mode 'DEC'      : ρ ≥ |p_r|, ρ ≥ |p_t|
    mode 'DEC_pr0'  : DEC with p_r ≡ 0 imposed (Einstein-cluster-like, tangential stress only)
    mode 'RAD'      : ρ ≥ 0 and p_r ≤ w_rad ρ only (the hypotheses of the cavity bound)
    mode 'NEC'      : ρ + p_r ≥ 0, ρ + p_t ≥ 0 only (ρ may be negative; objective = ADM mass)
    """
    rng = np.random.default_rng(seed)
    target = math.log(Z)

    def unpack(x):
        s_out = x[0]
        y_int = x[1:N + 1]                  # y_1..y_N (y_0 = 0)
        P_int = x[N + 1:2 * N]              # P_1..P_{N-1} (P_0 = 0, P_N from boundary)
        y_nodes = np.concatenate(([0.0], y_int))
        yN = y_nodes[-1]
        P_nodes = np.concatenate(([0.0], P_int, [yN / (2 * (1 - yN))]))
        if mode == 'DEC_pr0':
            P_nodes = y_nodes / (2 * (1 - y_nodes))
        return s_out, y_nodes, P_nodes

    def obj(x):
        s_out, y_nodes, _ = unpack(x)
        return 0.5 * y_nodes[-1] * math.exp(s_out)

    def ineq(x):
        s_out, y_nodes, P_nodes = unpack(x)
        F = continuum_fields(y_nodes, P_nodes, s_out)
        _, y, P, rho, pr, pt = F.T
        c = [continuum_lnZ(y_nodes, P_nodes, s_out) - target]
        if mode in ('DEC', 'DEC_pr0'):
            c = np.concatenate((c, rho - pr, rho + pr, rho - pt, rho + pt))
        elif mode == 'RAD':
            c = np.concatenate((c, rho, w_rad * rho - pr))
        elif mode == 'NEC':
            c = np.concatenate((c, rho + pr, rho + pt))
        return np.concatenate((c, 0.995 - y))

    best = None
    for _ in range(n_starts):
        s0 = rng.uniform(0.2, 1.5)
        yk = np.clip(np.linspace(0, 1, N + 1)[1:] ** rng.uniform(0.2, 1.0) * rng.uniform(0.3, 0.9), 1e-3, 0.95)
        Pk = yk[:-1] / (2 * (1 - yk[:-1])) * rng.uniform(0.5, 1.5)
        x0 = np.concatenate(([s0], yk, Pk))
        lo_y = -2.0 if mode == 'NEC' else 0.0
        bounds = [(1e-3, 6.0)] + [(lo_y, 0.995)] * N + [(-50.0, 50.0)] * (N - 1)
        try:
            res = minimize(obj, x0, method='SLSQP', bounds=bounds,
                           constraints=[{'type': 'ineq', 'fun': ineq}],
                           options=dict(maxiter=maxiter, ftol=1e-12))
        except Exception:
            continue
        if np.all(ineq(res.x) > -1e-7) and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        return None
    s_out, y_nodes, P_nodes = unpack(best.x)
    F = continuum_fields(y_nodes, P_nodes, s_out)
    return dict(M_over_rin=best.fun, s_out=s_out, r_out=math.exp(s_out), y=y_nodes, P=P_nodes,
                fields=F, Z=math.exp(continuum_lnZ(y_nodes, P_nodes, s_out)),
                m_max_over_rin=float(np.max(0.5 * F[:, 1] * np.exp(F[:, 0]))))


# ---------------------------------------------------------------------------
# 6. Second continuum parametrisation: nodes for y and p̃_r, free node spacing
# ---------------------------------------------------------------------------

def _fields_yg(s_nodes, y_nodes, g_nodes, n_sub=6):
    """Piecewise-linear y(s) and g(s) = p̃_r(s).  P = (y + g)/(2(1-y)) is then implied.
    Vectorised: returns array (K*(n_sub+1), 6) with columns s, y, P, ρ̃, p̃_r, p̃_t."""
    h = np.diff(s_nodes)
    ys = np.diff(y_nodes) / h
    gs = np.diff(g_nodes) / h
    t = np.linspace(0.0, 1.0, n_sub + 1)[None, :]
    s = s_nodes[:-1, None] + t * h[:, None]
    y = y_nodes[:-1, None] + t * np.diff(y_nodes)[:, None]
    g = g_nodes[:-1, None] + t * np.diff(g_nodes)[:, None]
    P = (y + g) / (2 * (1 - y))
    rho = y + ys[:, None]
    pt = 0.5 * (gs[:, None] + (rho + g) * P)
    return np.stack([a.ravel() for a in (s, y, P, rho, g, pt)], axis=1)


def _lnZ_yg(F, n_sub=6):
    k = n_sub + 1
    s = F[:, 0].reshape(-1, k)
    P = F[:, 2].reshape(-1, k)
    return float(np.sum(np.trapezoid(P, s, axis=1))) - 0.5 * math.log(1 - F[-1, 1])


def optimise_continuum_yg(Z, N=30, n_starts=10, seed=0, maxiter=1500, warm=None, mode='DEC'):
    """Minimise M/r_in under pointwise DEC (or 'RAD' / 'NEC') with free node spacing.

    Variables: log node spacings (N), y_1..y_N, g_1..g_{N-1}; y_0 = g_0 = g_N = 0."""
    rng = np.random.default_rng(seed)
    target = math.log(Z)
    n_sub = 6

    def unpack(x):
        h = np.exp(x[:N])
        s_nodes = np.concatenate(([0.0], np.cumsum(h)))
        y_nodes = np.concatenate(([0.0], x[N:2 * N]))
        g_nodes = np.concatenate(([0.0], x[2 * N:3 * N - 1], [0.0]))
        return s_nodes, y_nodes, g_nodes

    def obj(x):
        s_nodes, y_nodes, _ = unpack(x)
        return 0.5 * y_nodes[-1] * math.exp(s_nodes[-1])

    def ineq(x):
        s_nodes, y_nodes, g_nodes = unpack(x)
        F = _fields_yg(s_nodes, y_nodes, g_nodes, n_sub)
        _, y, P, rho, pr, pt = F.T
        c = [_lnZ_yg(F, n_sub) - target]
        if mode == 'DEC':
            c = np.concatenate((c, rho - pr, rho + pr, rho - pt, rho + pt))
        elif mode == 'RAD':
            c = np.concatenate((c, rho, rho - pr))
        elif mode == 'NEC':
            c = np.concatenate((c, rho + pr, rho + pt))
        return np.concatenate((c, 0.995 - y))

    starts = []
    if warm is not None:
        starts.append(np.asarray(warm, dtype=float))
    for _ in range(n_starts):
        s_out = rng.uniform(0.4, 1.2)
        frac = rng.dirichlet(np.ones(N) * 0.7)
        h = np.maximum(frac * s_out, 1e-4)
        y = np.empty(N)
        y[0] = rng.uniform(0.5, 0.96)
        y[1:] = rng.uniform(0.75, 0.85, N - 1)
        g = rng.uniform(-0.2, 0.2, N - 1)
        starts.append(np.concatenate((np.log(h), y, g)))
    lo_y = -2.0 if mode == 'NEC' else 0.0
    bounds = [(-12.0, 1.0)] * N + [(lo_y, 0.995)] * N + [(-20.0, 20.0)] * (N - 1)
    best = None
    for x0 in starts:
        try:
            res = minimize(obj, x0, method='SLSQP', bounds=bounds,
                           constraints=[{'type': 'ineq', 'fun': ineq}],
                           options=dict(maxiter=maxiter, ftol=1e-12))
        except Exception:
            continue
        if np.all(ineq(res.x) > -1e-7) and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        return None
    s_nodes, y_nodes, g_nodes = unpack(best.x)
    F = _fields_yg(s_nodes, y_nodes, g_nodes, n_sub)
    return dict(M_over_rin=best.fun, r_out=math.exp(s_nodes[-1]), s=s_nodes, y=y_nodes, g=g_nodes,
                fields=F, Z=math.exp(_lnZ_yg(F, n_sub)), x=best.x,
                m_max_over_rin=float(np.max(0.5 * F[:, 1] * np.exp(F[:, 0]))))


def thin_layer_dec_profile(n=2000):
    """Analytic thin-layer limit of a pointwise-DEC wall around a flat cavity, saturating
    p_t = ρ:  g' + (y + g)/(2(1 - y)) = 2,  g(0) = 0.  Returns (y, g) and the y where g
    returns to zero (= 24/25 analytically, reproducing the thin-shell DEC bound)."""
    y = np.linspace(0, 0.999, n)
    u = np.sqrt(1 - y)
    g = u * (6 - 5 * u - 1 / u)          # closed-form solution  g(1-y)^{-1/2} = 6 - 5√(1-y) - 1/√(1-y)
    return y, g


def warm_start_shell_plus_cluster(Z, N=20, layer=0.01, n_layer=5, n_gap=6):
    """Warm start: analytic DEC thin layer (0 → 24/25), vacuum gap (y ∝ e^{-s}) down to 0.8,
    then an Einstein-cluster plateau y = 0.8 (p_r = 0, p_t = ρ) long enough to reach ln Z."""
    y_top, y_pl = 0.96, 0.8
    gap = math.log(y_top / y_pl)
    lnZ_fixed = 0.5 * math.log((1 - y_pl) / (1 - y_top)) + (-0.5 * math.log(1 - y_pl))
    L = max((math.log(Z) - lnZ_fixed) / 2.0, 0.05)
    n_pl = N - n_layer - n_gap
    s_nodes = np.concatenate((np.linspace(0, layer, n_layer + 1),
                              layer + np.linspace(0, gap, n_gap + 1)[1:],
                              layer + gap + np.linspace(0, L, n_pl + 1)[1:]))
    yl = np.linspace(0, y_top, n_layer + 1)
    ul = np.sqrt(1 - yl)
    gl = ul * (6 - 5 * ul - 1 / ul)
    gl[-1] = 0.0
    yg = y_top * np.exp(-(s_nodes[n_layer + 1:n_layer + n_gap + 1] - layer))
    y_nodes = np.concatenate((yl, yg, np.full(n_pl, y_pl)))
    g_nodes = np.concatenate((gl, np.zeros(n_gap + n_pl)))
    h = np.diff(s_nodes)
    x = np.concatenate((np.log(h), y_nodes[1:], g_nodes[1:-1]))
    return x


# ---------------------------------------------------------------------------
# 7. Optimal-control formulation (controls bounded by DEC by construction)
# ---------------------------------------------------------------------------
#   state (y, g = p̃_r),  independent variable s = ln(r/r_in)
#   y_s = ρ̃ - y,     g_s = 2 p̃_t - (ρ̃ + g) P,     P = (y + g)/(2(1 - y)),   (ln Z)_s = P
#   controls per segment: ρ̃ ≥ 0, p̃_t = λ ρ̃ with |λ| ≤ 1.   Path constraint |g| ≤ ρ̃.

def simulate_controls(h, rho, lam, n_rk=8):
    """RK4 integration over segments; returns arrays of (s, y, g, P, ρ̃) at sub-steps and lnZ, M."""
    y = g = 0.0
    s = 0.0
    lnZ_int = 0.0
    rec = []
    for hk, rk, lk in zip(h, rho, lam):
        dt = hk / n_rk
        ptk = lk * rk

        def f(Y):
            yy, gg = Y
            P = (yy + gg) / (2 * (1 - yy))
            return (rk - yy, 2 * ptk - (rk + gg) * P, P)

        for _ in range(n_rk):
            k1 = f((y, g))
            k2 = f((y + 0.5 * dt * k1[0], g + 0.5 * dt * k1[1]))
            k3 = f((y + 0.5 * dt * k2[0], g + 0.5 * dt * k2[1]))
            k4 = f((y + dt * k3[0], g + dt * k3[1]))
            y += dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
            g += dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
            lnZ_int += dt * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6
            s += dt
            if not (y < 0.999):
                return None
            rec.append((s, y, g, rk))
    lnZ = lnZ_int - 0.5 * math.log(1 - y)
    return dict(rec=np.array(rec), lnZ=lnZ, M=0.5 * y * math.exp(s), y_end=y, g_end=g, s_end=s)


def optimise_controls_dec(Z, x0, K_seg, maxiter=400):
    """x = [log h_k, log ρ̃_k (or -30 for vacuum), atanh-ish λ_k]."""
    target = math.log(Z)

    def unpack(x):
        h = np.exp(x[:K_seg])
        rho = np.exp(x[K_seg:2 * K_seg])
        lam = np.tanh(x[2 * K_seg:3 * K_seg])
        return h, rho, lam

    def run(x):
        return simulate_controls(*unpack(x))

    cache = {}

    def cached(x):
        key = x.tobytes()
        if key not in cache:
            cache.clear()
            cache[key] = run(x)
        return cache[key]

    def obj(x):
        r = cached(x)
        return 5.0 if r is None else r['M']

    def ineq(x):
        r = cached(x)
        if r is None:
            return -np.ones(1 + 8 * K_seg)
        rec = r['rec']
        return np.concatenate(([r['lnZ'] - target], rec[:, 3] - np.abs(rec[:, 2]) + 1e-12))

    def eq(x):
        r = cached(x)
        return np.array([1.0]) if r is None else np.array([r['g_end']])

    bounds = [(-12, 1)] * K_seg + [(-30, 8)] * K_seg + [(-6, 6)] * K_seg
    res = minimize(obj, x0, method='SLSQP', bounds=bounds,
                   constraints=[{'type': 'ineq', 'fun': ineq}, {'type': 'eq', 'fun': eq}],
                   options=dict(maxiter=maxiter, ftol=1e-12))
    r = run(res.x)
    feas = r is not None and np.all(ineq(res.x) > -1e-7) and abs(r['g_end']) < 1e-7
    return dict(M_over_rin=r['M'] if r else None, Z=math.exp(r['lnZ']) if r else None, feasible=feas,
                x=res.x, sim=r, h=unpack(res.x)[0], rho=unpack(res.x)[1], lam=unpack(res.x)[2])


def warm_controls_shell_cluster(Z, n_layer=8, n_gap=4, n_pl=18, layer=0.01):
    y_top, y_pl = 0.96, 0.8
    gap = math.log(y_top / y_pl)
    lnZ_fixed = 0.5 * math.log((1 - y_pl) / (1 - y_top)) - 0.5 * math.log(1 - y_pl)
    L = max((math.log(Z) - lnZ_fixed) / 2.0, 0.05)
    h = np.concatenate((np.full(n_layer, layer / n_layer), np.full(n_gap, gap / n_gap), np.full(n_pl, L / n_pl)))
    rho = np.concatenate((np.full(n_layer, y_top / layer), np.full(n_gap, math.exp(-30)), np.full(n_pl, y_pl)))
    lam = np.concatenate((np.full(n_layer, 0.999), np.zeros(n_gap), np.full(n_pl, 0.999)))
    return np.concatenate((np.log(h), np.log(rho), np.arctanh(lam))), n_layer + n_gap + n_pl


# ---------------------------------------------------------------------------
# 8. Causally stable nested shells: analytic family (round-1 result E20)
# ---------------------------------------------------------------------------

def test_shell_critical_compactness(beta2_max):
    """Compactness y_c = 2m/R above which a light (test) thin shell around mass m cannot be
    radially stable with dp/dσ ≤ beta2_max.  y_c = 2/3 (photon sphere) for beta2_max = 1."""
    from scipy.optimize import brentq
    from .thin_shell import stability_beta2_min_general
    return brentq(lambda y: stability_beta2_min_general(y / 2, y / 2 * (1 + 1e-7))[0] - beta2_max,
                  0.02, 0.95, xtol=1e-12)


def single_shell_critical_Z(beta2_max):
    """Largest Z of a stable single thin shell with flat interior: β²_min(1/Z) = beta2_max."""
    from scipy.optimize import brentq
    from .thin_shell import stability_beta2_min
    return brentq(lambda Z: stability_beta2_min(1 / Z)[0] - beta2_max, 1.0001, 1e3, xtol=1e-12)


def stable_family_mass(Z, beta2_max=1.0):
    """Mass (units r_in c²/G) of the family: marginally stable inner shell (Z_1), vacuum gap
    down to the test-shell critical compactness y_c, then a 'plateau' of marginally stable
    light shells at y = y_c (α ∝ r^n, n = y_c/(2(1-y_c))):
        M/r_in = (1 - Z_1^{-2})/2 · (Z/Z_1)^{1/n}       (Z ≥ Z_1)."""
    Z1 = single_shell_critical_Z(beta2_max)
    if Z <= Z1:
        return 0.5 * (1 - 1 / Z**2), dict(Z1=Z1, y_c=None, n=None)
    yc = test_shell_critical_compactness(beta2_max)
    n = yc / (2 * (1 - yc))
    return 0.5 * (1 - 1 / Z1**2) * (Z / Z1) ** (1 / n), dict(Z1=Z1, y_c=yc, n=n)
