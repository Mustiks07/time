"""Dynamic ("breathing") thin shells of counter-rotating particles (Einstein shells).

Shell of total rest mass m0 made of particles with specific angular momentum L (random
orientations), flat interior, Schwarzschild exterior of ADM mass M (G = c = 1).
Proper mass of the shell:   μ(R) = m0 sqrt(1 + L²/R²)       (p/σ = v²/2, γv = L/R)
Equation of motion:         Ṙ² + V(R) = 0,   V = 1 - A²,   A = M/μ + μ/(2R)
Clock rates:   interior Minkowski time  dT/dτ = sqrt(1 + Ṙ²)
               exterior Schwarzschild   dt/dτ = sqrt(f + Ṙ²)/f,   f = 1 - 2M/R
Period-averaged dilation  <Z> = ∮dt / ∮dT.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


def _A(R, M, m0, L):
    mu = m0 * math.sqrt(1 + L * L / (R * R))
    return M / mu + mu / (2 * R)


def V(R, M, m0, L):
    return 1 - _A(R, M, m0, L) ** 2


def physical_branch(R, M, m0, L):
    """B = sqrt(f + Ṙ²) must be positive (shell on our side of the Einstein–Rosen bridge):
    μ² < 2MR."""
    mu = m0 * math.sqrt(1 + L * L / (R * R))
    return mu * mu < 2 * M * R


def turning_points(M, m0, L, R_hi=1e4):
    """Find a bound pocket R_min < R_max (V < 0 inside, V = 0 at both ends, R_min > 2M)."""
    Rs = np.geomspace(2 * M * (1 + 1e-6), R_hi, 6000)
    Vs = np.array([V(r, M, m0, L) for r in Rs])
    neg = Vs < 0
    i = 0
    while i < len(Rs):
        if neg[i]:
            j = i
            while j + 1 < len(Rs) and neg[j + 1]:
                j += 1
            if i > 0 and j < len(Rs) - 1:
                Rmin = brentq(V, Rs[i - 1], Rs[i], args=(M, m0, L))
                Rmax = brentq(V, Rs[j], Rs[j + 1], args=(M, m0, L))
                if physical_branch(Rmin, M, m0, L) and physical_branch(Rmax, M, m0, L):
                    return Rmin, Rmax
            i = j + 1
        else:
            i += 1
    return None


def period_average(M, m0, L):
    tp = turning_points(M, m0, L)
    if tp is None:
        return None
    Rmin, Rmax = tp
    # substitution R = Rc - Ra cos θ removes the 1/sqrt singularities at turning points
    Rc, Ra = 0.5 * (Rmax + Rmin), 0.5 * (Rmax - Rmin)

    def integrand(theta, which):
        R = Rc - Ra * math.cos(theta)
        dRdth = Ra * math.sin(theta)
        Rd2 = max(-V(R, M, m0, L), 0.0)
        Rd = math.sqrt(Rd2)
        if Rd < 1e-300:
            return 0.0
        dtau = dRdth / Rd
        f = 1 - 2 * M / R
        if which == 'T':
            return math.sqrt(1 + Rd2) * dtau
        return math.sqrt(f + Rd2) / f * dtau

    T, _ = quad(integrand, 1e-9, math.pi - 1e-9, args=('T',), limit=400)
    t, _ = quad(integrand, 1e-9, math.pi - 1e-9, args=('t',), limit=400)
    Z_static_min = 1 / math.sqrt(1 - 2 * M / Rmin)
    return dict(Rmin=Rmin, Rmax=Rmax, Z_avg=t / T, Z_at_Rmin=Z_static_min,
                v_at_Rmin=L / math.sqrt(Rmin**2 + L**2))


def scan(n=40, seed=0):
    """Random scan of oscillating shells; returns the largest period-averaged Z found."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n):
        M = 1.0
        m0 = rng.uniform(1.0, 3.0)
        L = rng.uniform(0.3, 10.0)
        try:
            res = period_average(M, m0, L)
        except Exception:
            res = None
        if res and (best is None or res['Z_avg'] > best['Z_avg']):
            best = dict(res, m0=m0, L=L)
    return best


def W(R, m0, L):
    """Effective potential: allowed region is W(R) ≤ M, with W = μ - μ²/(2R)."""
    mu = m0 * math.sqrt(1 + L * L / (R * R))
    return mu - mu * mu / (2 * R)


def scan_pockets(n=300, seed=0):
    """Choose (m0, L), locate the local minimum of W and the barrier, and pick M inside the
    pocket.  Units: m0 = 1.  Returns the list of results (each with <Z>, Z at R_min, v(R_min))."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        L = rng.uniform(0.6, 20.0)
        Rs = np.geomspace(0.05, 2000, 20000)
        Ws = np.array([W(r, 1.0, L) for r in Rs])
        # local minima
        idx = np.where((Ws[1:-1] < Ws[:-2]) & (Ws[1:-1] < Ws[2:]))[0] + 1
        if len(idx) == 0:
            continue
        k = idx[-1]
        Wmin = Ws[k]
        # barrier: max of W at smaller R than the minimum
        Wbar = np.max(Ws[:k])
        Mtop = min(Wbar, 1.0)                     # also bound at infinity: W(∞) = m0 = 1
        if Mtop <= Wmin:
            continue
        M = Wmin + rng.uniform(0.05, 0.999) * (Mtop - Wmin)
        try:
            res = period_average(M, 1.0, L)
        except Exception:
            res = None
        if res:
            out.append(dict(res, M=M, L=L))
    return out
