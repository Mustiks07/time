"""Kerr black hole: prograde circular equatorial geodesics, time dilation and tides.

Units: G = c = M = 1 unless stated.  Boyer–Lindquist coordinates.
Time-dilation factor of a circular orbiter relative to static observers at infinity
is Z = u^t = dt/dτ (exact, since the orbit is stationary and the far observer is static).
"""
from __future__ import annotations

import math

import numpy as np
import sympy as sp
from scipy.optimize import brentq

from . import constants as K


def r_isco(chi, prograde=True):
    Z1 = 1 + (1 - chi**2) ** (1 / 3) * ((1 + chi) ** (1 / 3) + (1 - chi) ** (1 / 3))
    Z2 = math.sqrt(3 * chi**2 + Z1**2)
    s = -1 if prograde else 1
    return 3 + Z2 + s * math.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2))


def metric_eq(r, chi):
    """Equatorial (θ=π/2) Kerr metric components g_tt, g_tφ, g_φφ, g_rr (M=1)."""
    a = chi
    Delta = r * r - 2 * r + a * a
    gtt = -(1 - 2 / r)
    gtp = -2 * a / r
    gpp = r * r + a * a + 2 * a * a / r
    grr = r * r / Delta
    return gtt, gtp, gpp, grr


def omega_kepler(r, chi):
    return 1.0 / (r**1.5 + chi)


def ut_from_metric(r, chi):
    gtt, gtp, gpp, _ = metric_eq(r, chi)
    W = omega_kepler(r, chi)
    return 1.0 / math.sqrt(-(gtt + 2 * gtp * W + gpp * W * W))


def ut_bpt(r, chi):
    """Bardeen–Press–Teukolsky (1972) closed form for prograde orbits."""
    return (r**1.5 + chi) / (r**0.75 * math.sqrt(r**1.5 - 3 * r**0.5 + 2 * chi))


def energy_L(r, chi):
    den = r**0.75 * math.sqrt(r**1.5 - 3 * r**0.5 + 2 * chi)
    E = (r**1.5 - 2 * r**0.5 + chi) / den
    L = (r**2 - 2 * chi * r**0.5 + chi**2) / den
    return E, L


def zamo_quantities(r, chi):
    gtt, gtp, gpp, _ = metric_eq(r, chi)
    omega = -gtp / gpp                       # frame-dragging angular velocity
    varpi = math.sqrt(gpp)
    lapse = math.sqrt(gtp**2 / gpp - gtt)
    return lapse, omega, varpi


def zamo_speed(r, chi):
    lapse, omega, varpi = zamo_quantities(r, chi)
    W = omega_kepler(r, chi)
    return varpi * (W - omega) / lapse


def chi_for_isco_dilation(Z_target):
    return brentq(lambda x: ut_bpt(r_isco(x), x) - Z_target, 0.0, 1 - 1e-12, xtol=1e-14)


def r_for_dilation(Z_target, chi):
    """Outermost prograde circular orbit radius (≥ ISCO) with u^t = Z_target, or None."""
    ri = r_isco(chi)
    if ut_bpt(ri, chi) < Z_target:
        return None
    return brentq(lambda r: ut_bpt(r, chi) - Z_target, ri, 1e3)


def tidal_eigenvalues_marck(r, chi):
    """Tidal eigenvalues (units M/r^3 * M^-2 → 1/M^2) for circular equatorial geodesics.

    Known closed form (Marck 1983) with K = (L - aE)^2:
        λ_vertical = (1/r^3)(1 + 3K/r^2),  λ_radial = -(1/r^3)(2 + 3K/r^2),  λ_along = 1/r^3.
    These are *checked against a direct Riemann-tensor computation* in kerr_tidal_numeric().
    Sign: E_ij = R_{i0j0}; tidal acceleration a_i = -E_ij ξ^j, so negative eigenvalue = stretching.
    """
    E, L = energy_L(r, chi)
    Kc = (L - chi * E) ** 2
    return np.array([1 + 3 * Kc / r**2, -(2 + 3 * Kc / r**2), 1.0]) / r**3


# --------------------------------------------------------------------------
# Direct symbolic Riemann computation (independent check of the Marck formula)
# --------------------------------------------------------------------------
_RIEMANN_CACHE = {}


def _kerr_riemann_func():
    if 'f' in _RIEMANN_CACHE:
        return _RIEMANN_CACHE['f']
    t, r, th, ph, a = sp.symbols('t r theta phi a', real=True)
    Sigma = r**2 + a**2 * sp.cos(th) ** 2
    Delta = r**2 - 2 * r + a**2
    g = sp.zeros(4, 4)
    g[0, 0] = -(1 - 2 * r / Sigma)
    g[0, 3] = g[3, 0] = -2 * a * r * sp.sin(th) ** 2 / Sigma
    g[1, 1] = Sigma / Delta
    g[2, 2] = Sigma
    g[3, 3] = (r**2 + a**2 + 2 * a**2 * r * sp.sin(th) ** 2 / Sigma) * sp.sin(th) ** 2
    X = [t, r, th, ph]
    ginv = g.inv()
    ginv = ginv.applyfunc(sp.simplify)
    Gam = [[[sp.simplify(sum(ginv[i, l] * (sp.diff(g[l, j], X[k]) + sp.diff(g[l, k], X[j]) - sp.diff(g[j, k], X[l]))
                             for l in range(4)) / 2) for k in range(4)] for j in range(4)] for i in range(4)]
    # R^i_{jkl} = ∂_k Γ^i_{jl} - ∂_l Γ^i_{jk} + Γ^i_{km}Γ^m_{jl} - Γ^i_{lm}Γ^m_{jk}
    Rup = {}
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(k + 1, 4):
                    expr = (sp.diff(Gam[i][j][l], X[k]) - sp.diff(Gam[i][j][k], X[l])
                            + sum(Gam[i][k][m] * Gam[m][j][l] - Gam[i][l][m] * Gam[m][j][k] for m in range(4)))
                    Rup[(i, j, k, l)] = expr
    # Evaluate slightly off the equator on both sides and average: avoids 0*∞ artefacts of a
    # literal θ=π/2 substitution; error O(δ²) by reflection symmetry about the equator.
    keys = list(Rup.keys())
    fR_th = sp.lambdify((r, a, th), [Rup[key] for key in keys], 'math')
    fg_th = sp.lambdify((r, a, th), g.tolist(), 'math')
    d = 1e-6

    def fR(rv, av):
        v1 = fR_th(rv, av, math.pi / 2 - d)
        v2 = fR_th(rv, av, math.pi / 2 + d)
        return [0.5 * (x + y) for x, y in zip(v1, v2)]

    def fg(rv, av):
        return fg_th(rv, av, math.pi / 2)
    _RIEMANN_CACHE['f'] = (fR, fg, keys)
    return _RIEMANN_CACHE['f']


def kerr_tidal_numeric(r, chi):
    """Eigenvalues of E_ij = R_{μανβ} e_i^μ u^α e_j^ν u^β for the prograde circular orbit (M=1)."""
    fR, fg, keys = _kerr_riemann_func()
    vals = fR(r, chi)
    Rup = np.zeros((4, 4, 4, 4))
    for key, v in zip(keys, vals):
        i, j, k, l = key
        Rup[i, j, k, l] = v
        Rup[i, j, l, k] = -v
    g = np.array(fg(r, chi), dtype=float)
    Rdown = np.einsum('im,mjkl->ijkl', g, Rup)
    ut = ut_bpt(r, chi)
    W = omega_kepler(r, chi)
    u = np.array([ut, 0.0, 0.0, ut * W])
    # orthonormal spatial triad orthogonal to u (Gram–Schmidt with metric g)
    basis = [np.array([0, 1.0, 0, 0]), np.array([0, 0, 1.0, 0]), np.array([1.0, 0, 0, 0])]
    triad = []
    for v in basis:
        w = v + np.dot(u, g @ v) * u                 # project orthogonal to u (u·u = -1)
        for e in triad:
            w = w - np.dot(e, g @ w) * e
        w = w / math.sqrt(np.dot(w, g @ w))
        triad.append(w)
    E = np.zeros((3, 3))
    for i, ei in enumerate(triad):
        for j, ej in enumerate(triad):
            E[i, j] = np.einsum('abcd,a,b,c,d->', Rdown, ei, u, ej, u)
    return np.sort(np.linalg.eigvalsh(0.5 * (E + E.T)))


def min_mass_for_tides(lam_max_geo, body_length=2.0, a_limit=K.g0):
    """Tidal acceleration ≈ |λ| c^2 L / M_geo^2 (λ in units 1/M^2).  Solve for M (kg)."""
    M_geo = math.sqrt(abs(lam_max_geo) * K.c**2 * body_length / a_limit)
    return K.mass_geo_to_kg(M_geo)


def escape_boost(r, chi):
    """Minimal tangential (prograde) local boost from the circular orbit to an unbound (E=1)
    trajectory, measured in the ZAMO frame.  Returns (Δ rapidity, photon-rocket mass ratio)."""
    lapse, omega, varpi = zamo_quantities(r, chi)
    vc = zamo_speed(r, chi)

    def E_of_v(v):
        gam = 1 / math.sqrt(1 - v * v)
        return gam * (lapse + omega * varpi * v)

    v1 = brentq(lambda v: E_of_v(v) - 1.0, vc, 1 - 1e-15)
    dphi = math.atanh(v1) - math.atanh(vc)
    return dphi, math.exp(dphi), vc, v1
