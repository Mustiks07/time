"""TOV stars as the deepest *natural static* potential wells (reference for REPORT 01/02).

Causal "maximally stiff" EOS  p = ρ - ρ_s  (sound speed = c), the standard extreme case
used for maximum-mass / maximum-compactness bounds.  G = c = 1, lengths in km internally.
Returns the central time-dilation factor Z_c = 1/α(0) and the surface factor.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp

from . import constants as K

KM = 1e3
# geometric density unit: ρ_geo [1/m²] = G ρ_SI / c²
RHO_S_SI = 2.7e17                         # kg/m³, density at zero pressure (≈ nuclear)


def tov(rho_c_over_rhos, rho_s_si=RHO_S_SI):
    rho_s = K.G * rho_s_si / K.c**2          # 1/m²
    rho_c = rho_c_over_rhos * rho_s

    def eos_rho(p):
        return p + rho_s

    def rhs(r, y):
        m, p, Phi = y
        if p <= 0:
            return [0, 0, 0]
        rho = eos_rho(p)
        dPhi = (m + 4 * math.pi * r**3 * p) / (r * (r - 2 * m))
        return [4 * math.pi * r**2 * rho, -(rho + p) * dPhi, dPhi]

    def surface(r, y):
        return y[1]
    surface.terminal = True
    surface.direction = -1
    r0 = 1.0
    p_c = rho_c - rho_s
    y0 = [4 / 3 * math.pi * r0**3 * rho_c, p_c, 0.0]
    sol = solve_ivp(rhs, (r0, 1e6), y0, events=surface, rtol=1e-10, atol=1e-14, max_step=50)
    R = sol.t_events[0][0]
    M, _, Phi_R = sol.y_events[0][0]
    # normalise Φ so that e^{Φ(R)} = sqrt(1 - 2M/R)
    Phi_c = 0.5 * math.log(1 - 2 * M / R) - Phi_R
    return dict(R_km=R / KM, M_sun=K.mass_geo_to_kg(M) / K.M_sun, compactness=2 * M / R,
                Z_surface=1 / math.sqrt(1 - 2 * M / R), Z_centre=math.exp(-Phi_c),
                p_c_over_rho_c=p_c / rho_c)


def sequence(n=60):
    out = []
    for x in np.geomspace(1.05, 60, n):
        try:
            out.append(dict(rho_c_over_rhos=x, **tov(x)))
        except Exception:
            pass
    return out


def max_mass_config():
    """Locate the maximum-mass (onset of radial instability) model precisely."""
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(lambda x: -tov(x)['M_sun'], bounds=(2.0, 4.5), method='bounded', options=dict(xatol=1e-6))
    return dict(rho_c_over_rhos=float(r.x), **tov(r.x))
