"""Thin spherical shells in GR (Israel–Lanczos junction formalism).

Geometry: interior metric  -f_in dt^2 + dr^2/f_in + r^2 dΩ^2,
          exterior metric  -f_out dt^2 + dr^2/f_out + r^2 dΩ^2,
glued along a (possibly moving) timelike hypersurface r = R(τ).

The junction quantities are *derived* symbolically (extrinsic curvature computed
from Christoffel symbols), not typed in, so that the closed forms used elsewhere
are checked against first principles (see tests/test_thin_shell.py).

Units: G = c = 1, lengths in metres.  Surface energy density σ and surface
pressure p have dimension 1/length; multiply by c^4/G to get N/m.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import sympy as sp

from . import constants as K

# ---------------------------------------------------------------------------
# 1. Symbolic derivation of extrinsic curvature and Lanczos equations
# ---------------------------------------------------------------------------

def _christoffel(g, coords):
    ginv = g.inv()
    n = len(coords)
    Gam = [[[0] * n for _ in range(n)] for _ in range(n)]
    for a in range(n):
        for b in range(n):
            for c_ in range(n):
                Gam[a][b][c_] = sp.simplify(
                    sum(ginv[a, d] * (sp.diff(g[d, b], coords[c_]) + sp.diff(g[d, c_], coords[b])
                                      - sp.diff(g[b, c_], coords[d])) for d in range(n)) / 2)
    return Gam


def extrinsic_curvature_symbolic(f, r):
    """Return (K^τ_τ, K^θ_θ) for the hypersurface r=R(τ) in metric -f dt^2 + dr^2/f + r^2 dΩ^2.

    f is a sympy expression in the coordinate symbol r.  Results use R, Rdot, Rddot.
    Sign convention: K_ab = -n_μ (∂_a∂_b x^μ + Γ^μ_{αβ} ∂_a x^α ∂_b x^β), n outward (towards
    increasing r), so a round sphere in flat space has K^θ_θ = +1/R.
    """
    t, th, ph = sp.symbols('t theta phi')
    tau = sp.symbols('tau')
    Rf = sp.Function('R')(tau)
    coords = [t, r, th, ph]
    g = sp.diag(-f, 1 / f, r**2, r**2 * sp.sin(th)**2)
    Gam = _christoffel(g, coords)

    fR = f.subs(r, Rf)
    Rd = sp.diff(Rf, tau)
    Td = sp.sqrt(fR + Rd**2) / fR             # from normalisation u·u = -1
    # embedding x^μ(τ, θ, φ)
    u = [Td, Rd, 0, 0]
    # derivative of u along τ
    du = [sp.diff(Td, tau), sp.diff(Rd, tau), 0, 0]
    # outward unit normal (covector): n_μ ∝ (-Rdot, Tdot, 0, 0), normalised
    n_cov = [-Rd, Td, 0, 0]
    sub = {t: sp.Symbol('T'), r: Rf, th: sp.pi / 2, ph: 0}
    # check normalisation: g^{μν} n_μ n_ν = +1
    ginv = g.inv().subs(sub)
    norm = sp.simplify(sum(ginv[i, i] * n_cov[i]**2 for i in range(2)))
    assert sp.simplify(norm - 1) == 0, norm

    # K_ττ
    acc = [du[m] + sum(Gam[m][a][b].subs(sub) * u[a] * u[b] for a in range(4) for b in range(4))
           for m in range(4)]
    K_tautau = -sum(n_cov[m] * acc[m] for m in range(4))
    # K_θθ : x^μ_{,θθ} = 0 for (t,r); only Christoffel term Γ^r_θθ contributes
    K_thth = -sum(n_cov[m] * Gam[m][2][2].subs(sub) for m in range(4))
    # mixed components: h^ττ = -1, h^θθ = 1/R^2
    Rs, Rds, Rdds = sp.symbols('R Rdot Rddot', positive=True)
    repl = {sp.diff(Rf, tau, 2): Rdds, sp.diff(Rf, tau): Rds}
    Ktt = sp.simplify((-K_tautau).subs(repl).subs(Rf, Rs))
    Kthth = sp.simplify((K_thth / Rf**2).subs(repl).subs(Rf, Rs))
    return Ktt, Kthth, (Rs, Rds, Rdds)


def lanczos_symbolic(f_in, f_out, r):
    """Surface energy density σ and pressure p from [K] (Lanczos equations).

    S^a_b = -(1/8π)([K^a_b] - δ^a_b [K]);  σ = -S^τ_τ ,  p = S^θ_θ .
    """
    Kt_i, Kth_i, syms = extrinsic_curvature_symbolic(f_in, r)
    Kt_o, Kth_o, _ = extrinsic_curvature_symbolic(f_out, r)
    dKt, dKth = Kt_o - Kt_i, Kth_o - Kth_i
    dK = dKt + 2 * dKth
    S_tt = -(dKt - dK) / (8 * sp.pi)
    S_thth = -(dKth - dK) / (8 * sp.pi)
    sigma = sp.simplify(-S_tt)
    p = sp.simplify(S_thth)
    return sigma, p, syms


# ---------------------------------------------------------------------------
# 2. Closed forms (verified against the symbolic derivation in the tests)
# ---------------------------------------------------------------------------

def static_shell_flat_interior(M, R):
    """Static thin shell, Minkowski interior, Schwarzschild(M) exterior.  G=c=1.

    Returns dict with interior lapse α (= dτ_inside/dt_infinity), σ, p.
    """
    u = math.sqrt(1.0 - 2.0 * M / R)
    sigma = (1.0 - u) / (4.0 * math.pi * R)
    p = (1.0 - u) ** 2 / (16.0 * math.pi * R * u)
    return dict(alpha=u, sigma=sigma, p=p, p_over_sigma=(1.0 - u) / (4.0 * u))


def shell_mass_for_alpha(alpha, R):
    """M such that a thin shell of areal radius R gives interior lapse alpha: M = R(1-α²)/2."""
    return 0.5 * R * (1.0 - alpha**2)


def p_over_sigma(alpha):
    return (1.0 - alpha) / (4.0 * alpha)


def max_Z_for_p_over_sigma(w):
    """Largest Z = 1/α for which a static thin shell needs p/σ ≤ w:  α ≥ 1/(1+4w)."""
    return 1.0 + 4.0 * w


def great_circle_force(alpha):
    """Total tangential force transmitted across an equatorial cut, F = 2πR p, in units of c^4/G."""
    return (1.0 - alpha) ** 2 / (8.0 * alpha)


def energy_conditions_shell(sigma, p):
    return dict(
        WEC=(sigma >= 0) and (sigma + p >= 0),
        NEC=(sigma + p >= 0),
        SEC=(sigma + p >= 0) and (sigma + 2 * p >= 0),
        DEC=(sigma >= abs(p)),
    )


# ---------------------------------------------------------------------------
# 3. Radial stability (Brady–Louko–Poisson style potential analysis)
# ---------------------------------------------------------------------------

def stability_beta2_min(alpha):
    """Minimal β² = dp/dσ for which the static shell (flat interior, Schwarzschild exterior)
    is linearly stable under radial perturbations.

    Equation of motion (flat interior):  Ṙ² + V(R) = 0,   V = 1 - (M/m + m/(2R))²,
    with m(R) = 4πR²σ(R),  m' = -8πRp,  m'' = -8πp + 16π β² (σ+p).
    V'' is linear in β², so the threshold is solved for exactly.
    """
    R = 1.0
    M = shell_mass_for_alpha(alpha, R)
    s = static_shell_flat_interior(M, R)
    sig, p = s['sigma'], s['p']
    m = 4 * math.pi * R**2 * sig
    m1 = -8 * math.pi * R * p

    def Vpp(beta2):
        m2 = -8 * math.pi * p + 16 * math.pi * beta2 * (sig + p)
        A = M / m + m / (2 * R)
        A1 = -M * m1 / m**2 + m1 / (2 * R) - m / (2 * R**2)
        A2 = (2 * M * m1**2 / m**3 - M * m2 / m**2 + m2 / (2 * R) - m1 / R**2 + m / R**3)
        return -2 * (A1**2 + A * A2)

    v0, v1 = Vpp(0.0), Vpp(1.0)
    slope = v1 - v0
    return -v0 / slope, slope > 0     # threshold, and whether stability is for β² ABOVE it


def equilibrium_check(alpha):
    """V(R0) and V'(R0) must vanish for the static solution (consistency test)."""
    R = 1.0
    M = shell_mass_for_alpha(alpha, R)
    s = static_shell_flat_interior(M, R)
    m = 4 * math.pi * R**2 * s['sigma']
    m1 = -8 * math.pi * R * s['p']
    A = M / m + m / (2 * R)
    A1 = -M * m1 / m**2 + m1 / (2 * R) - m / (2 * R**2)
    return 1 - A**2, -2 * A * A1


# ---------------------------------------------------------------------------
# 4. Einstein-cluster (counter-rotating particle) shell
# ---------------------------------------------------------------------------

def einstein_shell_speed(alpha):
    """Particle speed (relative to static shell frame) for a 2-D gas: p/σ = v²/2."""
    v2 = 2.0 * p_over_sigma(alpha)
    return math.sqrt(v2) if v2 < 1 else float('nan'), v2


# ---------------------------------------------------------------------------
# 5. SI tables
# ---------------------------------------------------------------------------

@dataclass
class ShellRow:
    Z: float
    R_m: float
    M_kg: float
    M_jup: float
    M_over_R_geo: float
    sigma_kg_m2: float
    tension_N_m: float
    stress_1pct_Pa: float
    p_over_sigma: float
    F_over_c4G: float
    DEC: bool


def shell_table(Zs=(2, 10, 100, 1000), Rs=(1, 10, 100, 1000)):
    rows = []
    for Z in Zs:
        a = 1.0 / Z
        for Rm in Rs:
            M = shell_mass_for_alpha(a, Rm)
            s = static_shell_flat_interior(M, Rm)
            sigma_SI = s['sigma'] * K.KG_PER_M          # kg/m^2   (σ[1/m] * c^2/G)
            tension = s['p'] * K.F_PLANCK_GR             # N/m      (p[1/m] * c^4/G)
            rows.append(ShellRow(Z, Rm, K.mass_geo_to_kg(M), K.mass_geo_to_kg(M) / K.M_jup,
                                 M / Rm, sigma_SI, tension, tension / (0.01 * Rm),
                                 s['p_over_sigma'], great_circle_force(a),
                                 energy_conditions_shell(s['sigma'], s['p'])['DEC']))
    return rows
