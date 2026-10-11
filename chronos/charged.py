"""Charged (Reissner–Nordström) thin shells and the extremal "BPS" time-dilation chamber.

Thin shell: flat interior, RN exterior f = 1 - 2M/r + Q²/r² (G = c = 4πε0 = 1).
Junction (static):  σ = (1 - √f)/(4πR),  p = [(1 - M/R)/√f - 1]/(8πR).
p = 0  ⇔  Q = ±M  (extremal dust shell).  Then √f = 1 - M/R and the interior lapse is
α = 1 - M/R  for ANY R > M: the shell is in neutral equilibrium (no-force condition).
The shell's rest mass 4πR²σ = R(1 - √f) = M is independent of R: moving it quasi-
statically costs no energy, so the interior clock rate can be dialled from 1 to ~0.
"""
from __future__ import annotations

import math

from . import constants as K


def rn_static_shell(M, Q, R):
    f = 1 - 2 * M / R + Q * Q / (R * R)
    if f <= 0:
        raise ValueError('shell inside horizon')
    u = math.sqrt(f)
    sigma = (1 - u) / (4 * math.pi * R)
    p = ((1 - M / R) / u - 1) / (8 * math.pi * R)
    return dict(alpha=u, sigma=sigma, p=p, rest_mass=4 * math.pi * R * R * sigma)


def extremal_shell(Z, R_m, magnetic=False):
    """Extremal dust shell giving interior dilation Z at areal radius R_m (metres)."""
    alpha = 1.0 / Z
    M_geo = R_m * (1 - alpha)
    field_geo = M_geo / R_m**2                    # |Q|/R² just outside the shell
    out = dict(Z=Z, R_m=R_m, M_kg=K.mass_geo_to_kg(M_geo), M_jup=K.mass_geo_to_kg(M_geo) / K.M_jup,
               M_over_R=1 - alpha)
    if magnetic:
        out['B_T'] = field_geo * K.B_FIELD_PER_INV_M
        out['B_over_B_EW'] = out['B_T'] / K.B_ELECTROWEAK
        out['q_m_Am'] = K.QM_EXTREMAL_PER_KG * out['M_kg']
    else:
        out['E_V_m'] = field_geo * K.E_FIELD_PER_INV_M
        out['E_over_Schwinger'] = out['E_V_m'] / K.E_SCHWINGER
        out['Q_C'] = K.Q_EXTREMAL_PER_KG * out['M_kg']
    # mean density if the shell has thickness R/10 (to show the density problem)
    out['rho_if_thickness_R_over_10'] = out['M_kg'] / (4 * math.pi * R_m**2 * 0.1 * R_m)
    return out


def min_radius_below_schwinger(Z):
    """Smallest R for which the exterior field of an *electric* extremal shell is below E_S."""
    alpha = 1.0 / Z
    # E = (1-α)/R * E_unit ≤ E_S
    return (1 - alpha) * K.E_FIELD_PER_INV_M / K.E_SCHWINGER


def dust_shell_potential(M, Q, mu, R):
    """Dust shell (rest mass mu) with flat interior and RN exterior: Ṙ² + V(R) = 0,
    V = 1 - A², A = [(2M - Q²/R)/μ + μ/R]/2.  Returns V(R)."""
    A = ((2 * M - Q * Q / R) / mu + mu / R) / 2
    return 1 - A * A


def mistuning_acceleration(Z, R_geo, delta):
    """Shell released from rest at R with Q = M(1+δ) and rest mass μ chosen so that V(R)=0.
    Returns the proper radial acceleration R̈ = -V'(R)/2 (geometric units, 1/length)."""
    alpha = 1.0 / Z
    M = R_geo * (1 - alpha)
    Q = M * (1 + delta)
    # choose μ with A(R) = 1 (momentarily static): μ²/R - 2μ + (2M - Q²/R) = 0
    a_, b_, c_ = 1 / R_geo, -2.0, 2 * M - Q * Q / R_geo
    mu = (-b_ - math.sqrt(b_ * b_ - 4 * a_ * c_)) / (2 * a_)
    h = 1e-7 * R_geo
    dV = (dust_shell_potential(M, Q, mu, R_geo + h) - dust_shell_potential(M, Q, mu, R_geo - h)) / (2 * h)
    return -0.5 * dV, mu
