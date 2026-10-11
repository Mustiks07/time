"""Special-relativistic (kinematic) time dilation with bounded proper acceleration.

Problem: a traveller leaves an inertial base at rest, returns, and stops (closed comparison
with a common start/end event pair, so the ratio of proper times is invariant).  The
traveller's proper acceleration is bounded, |a| ≤ a_max, and the traveller's total proper
time is T.  Maximise the base time t_base.

With rapidity φ(τ):  dt/dτ = cosh φ,  dx/dτ = c sinh φ,  |dφ/dτ| ≤ a_max/c,
φ(0) = φ(T) = 0, ∫ sinh φ dτ = 0.

Claim (verified numerically below): the optimum is the 4-phase constant-|a| profile
(accelerate, decelerate, accelerate back, decelerate), giving
        t_base / T = sinh(x)/x ,   x = a_max T / (4c).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq, minimize

from . import constants as K


def ratio_four_phase(x):
    return math.sinh(x) / x if x != 0 else 1.0


def x_for_ratio(Rt):
    """Solve sinh(x)/x = Rt."""
    return brentq(lambda x: math.sinh(x) / x - Rt, 1e-9, 50.0)


def mission_for_ratio(Rt, a_max=K.g0):
    x = x_for_ratio(Rt)
    T = 4 * x * K.c / a_max
    t = Rt * T
    d_turn = 2 * (K.c**2 / a_max) * (math.cosh(x) - 1)          # turnaround distance
    gamma_peak = math.cosh(x)
    photon_mass_ratio = math.exp(4 * x)                            # ideal photon rocket, total Δφ = 4x
    return dict(x=x, T_years=T / K.year, t_years=t / K.year, d_turn_ly=d_turn / K.ly,
                gamma_peak=gamma_peak, v_peak_c=math.tanh(x),
                photon_rocket_mass_ratio=photon_mass_ratio,
                kinetic_energy_per_kg_J=(gamma_peak - 1) * K.c**2)


def a_required(Rt, T_seconds):
    x = x_for_ratio(Rt)
    return 4 * x * K.c / T_seconds


def circular_track_radius(gamma, a=K.g0):
    """Uniform circular motion: proper acceleration a = γ² v²/r  ->  r = γ² v² / a."""
    v = K.c * math.sqrt(1 - 1 / gamma**2)
    return gamma**2 * v**2 / a


def optimal_control_check(x_total=4.0, N=200, seed=0):
    """Direct numerical optimisation over discretised rapidity profiles.

    Units: c = 1, a_max = 1, so T = 4 x_total.  Returns (numerical optimum, analytic sinh(x)/x).
    """
    T = 4.0 * x_total
    dt = T / N
    rng = np.random.default_rng(seed)

    def unpack(z):
        return np.concatenate(([0.0], z, [0.0]))

    def objective(z):
        phi = unpack(z)
        mid = 0.5 * (phi[1:] + phi[:-1])
        return -np.sum(np.cosh(mid)) * dt / T

    def ret(z):
        phi = unpack(z)
        mid = 0.5 * (phi[1:] + phi[:-1])
        return np.sum(np.sinh(mid)) * dt

    def slope(z):
        phi = unpack(z)
        d = np.diff(phi)
        return np.concatenate((dt - d, dt + d))      # |Δφ| ≤ a dt

    best = None
    for k in range(4):
        z0 = rng.normal(0, 0.3, N - 1)
        res = minimize(objective, z0, method='SLSQP',
                       constraints=[{'type': 'eq', 'fun': ret}, {'type': 'ineq', 'fun': slope}],
                       options=dict(maxiter=2000, ftol=1e-12))
        if best is None or res.fun < best.fun:
            best = res
    return -best.fun, ratio_four_phase(x_total), unpack(best.x)
