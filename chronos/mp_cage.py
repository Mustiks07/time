"""Majumdar–Papapetrou (MP) "cage" of extremal black holes as a time-dilation chamber.

MP electrovac solution (G = c = 4πε0 = 1):
    ds² = -U⁻² dt² + U² (dx² + dy² + dz²),    ∇²_flat U = 0,
    U = 1 + Σ_i m_i / |x - x_i|,   A_t = 1/U  (or the dual magnetic field).
Each point x_i is the horizon of an extremal black hole of mass m_i (= |charge|).
Static clocks at x tick at rate 1/U(x) relative to infinity, so the centre of a cage
with Σ m_i/|x_i| = Z - 1 has time-dilation factor Z.

For a static observer the proper acceleration and the electric (or magnetic) field
measured in its orthonormal frame are BOTH equal to |∇U|/U² in geometric units
(this is the no-force property).  Conversions: a_SI = c² a_geo,  B_SI = B_unit a_geo
(B_unit = c²/√(4πε0 G)/c ≈ 3.48e18 T·m).  Hence B[T] ≈ 38.7 × a[m/s²] in the habitat.
"""
from __future__ import annotations

import math

import numpy as np

from . import constants as K

TESLA_PER_MS2 = K.B_FIELD_PER_INV_M / K.c**2


def icosahedron():
    p = (1 + math.sqrt(5)) / 2
    v = []
    for a in (-1, 1):
        for b in (-p, p):
            v += [(0, a, b), (a, b, 0), (b, 0, a)]
    v = np.array(v, dtype=float)
    return v / np.linalg.norm(v, axis=1)[:, None]


def geodesic_sphere(freq):
    """Vertices of a frequency-`freq` geodesic subdivision of the icosahedron (10 f² + 2 points)."""
    V = icosahedron()
    # faces of icosahedron: triangles of mutually nearest vertices
    faces = []
    n = len(V)
    d = np.linalg.norm(V[:, None] - V[None], axis=2)
    edge = np.min(d[d > 1e-9])
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if abs(d[i, j] - edge) < 1e-6 and abs(d[j, k] - edge) < 1e-6 and abs(d[i, k] - edge) < 1e-6:
                    faces.append((i, j, k))
    pts = []
    for (i, j, k) in faces:
        A, B, C = V[i], V[j], V[k]
        for a in range(freq + 1):
            for b in range(freq + 1 - a):
                c_ = freq - a - b
                P = (a * A + b * B + c_ * C) / freq
                pts.append(P / np.linalg.norm(P))
    pts = np.array(pts)
    # deduplicate
    uniq = []
    for P in pts:
        if not any(np.linalg.norm(P - Q) < 1e-9 for Q in uniq):
            uniq.append(P)
    return np.array(uniq)


def fibonacci_sphere(N):
    i = np.arange(N) + 0.5
    phi = np.arccos(1 - 2 * i / N)
    theta = math.pi * (1 + 5**0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def U_and_grad(points, masses, X):
    """U and flat gradient at positions X (n,3)."""
    diff = X[:, None, :] - points[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    U = 1 + np.sum(masses[None, :] / dist, axis=1)
    grad = -np.sum((masses[None, :] / dist**3)[:, :, None] * diff, axis=1)
    return U, grad


def cage(points_unit, Z, rho0=1.0):
    """Place equal masses on a sphere of coordinate radius rho0 so that U(0) = Z."""
    N = len(points_unit)
    m = (Z - 1) * rho0 / N
    return points_unit * rho0, np.full(N, m)


def max_acceleration_on_sphere(points, masses, r, n_dirs=4000):
    dirs = np.concatenate((fibonacci_sphere(n_dirs), points / np.linalg.norm(points, axis=1)[:, None]))
    X = dirs * r
    U, g = U_and_grad(points, masses, X)
    a = np.linalg.norm(g, axis=1) / U**2
    return float(a.max()), float(U.min()), float(U.max())


def multipole_floor(points_unit, Z, x_values):
    """a_max(x) in units of 1/ρ0 for x = r/ρ0 values (ρ0 = 1)."""
    P, m = cage(points_unit, Z)
    return np.array([max_acceleration_on_sphere(P, m, x)[0] for x in x_values])


def min_mass_for_habitat(points_unit, Z, proper_radius_m=1.0, a_limit=K.g0):
    """Find the smallest cage (coordinate radius ρ0 in metres) such that static observers within the
    habitat (proper radius ≈ Z × coordinate radius) feel ≤ a_limit.  Returns dict with mass in kg."""
    r_coord = proper_radius_m / Z
    a_geo_limit = a_limit / K.c**2

    def a_of_rho0(rho0):
        P, m = cage(points_unit, Z, rho0)
        return max_acceleration_on_sphere(P, m, r_coord, n_dirs=1500)[0]

    lo, hi = r_coord * 1.05, r_coord * 1e6
    if a_of_rho0(hi) > a_geo_limit:
        return None
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        if a_of_rho0(mid) > a_geo_limit:
            lo = mid
        else:
            hi = mid
    rho0 = hi
    M_geo = (Z - 1) * rho0
    N = len(points_unit)
    m_bh_geo = M_geo / N
    return dict(N=N, rho0_m=rho0, M_kg=K.mass_geo_to_kg(M_geo), M_jup=K.mass_geo_to_kg(M_geo) / K.M_jup,
                M_sun=K.mass_geo_to_kg(M_geo) / K.M_sun, m_bh_kg=K.mass_geo_to_kg(m_bh_geo),
                B_horizon_T=K.B_FIELD_PER_INV_M / m_bh_geo,
                B_horizon_over_EW=(K.B_FIELD_PER_INV_M / m_bh_geo) / K.B_ELECTROWEAK,
                a_limit=a_limit, B_habitat_T=a_limit * TESLA_PER_MS2)


# ---------------------------------------------------------------------------
# Multipole-cancelling cages: combine icosahedral-group orbits with weights chosen so that
# the lowest invariant harmonics (ℓ = 6, 10, ...) of U cancel at the centre.
# ---------------------------------------------------------------------------

def dodecahedron():
    V = icosahedron()
    # dodecahedron vertices = normalised centroids of icosahedron faces
    d = np.linalg.norm(V[:, None] - V[None], axis=2)
    edge = np.min(d[d > 1e-9])
    pts = []
    n = len(V)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if abs(d[i, j] - edge) < 1e-6 and abs(d[j, k] - edge) < 1e-6 and abs(d[i, k] - edge) < 1e-6:
                    P = V[i] + V[j] + V[k]
                    pts.append(P / np.linalg.norm(P))
    return np.array(pts)


def icosidodecahedron():
    V = icosahedron()
    d = np.linalg.norm(V[:, None] - V[None], axis=2)
    edge = np.min(d[d > 1e-9])
    pts = []
    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            if abs(d[i, j] - edge) < 1e-6:
                P = V[i] + V[j]
                pts.append(P / np.linalg.norm(P))
    return np.array(pts)


def legendre(l, x):
    from numpy.polynomial import legendre as L
    c = np.zeros(l + 1)
    c[l] = 1
    return L.legval(x, c)


def invariant_coefficient(points, l, probe=np.array([0.3, 0.5, 0.81])):
    n = probe / np.linalg.norm(probe)
    return float(np.sum(legendre(l, points @ n)) / len(points))


def cancelling_weights(orbits, ls):
    """Per-point mass fractions w_k (for orbit k) such that Σ_k w_k N_k = 1 and the ℓ in `ls`
    invariant harmonics cancel.  Requires len(orbits) == len(ls) + 1."""
    A = [[len(o) for o in orbits]]
    b = [1.0]
    for l in ls:
        A.append([invariant_coefficient(o, l) * len(o) for o in orbits])
        b.append(0.0)
    w = np.linalg.solve(np.array(A), np.array(b))
    return w


def weighted_cage(orbits, weights, Z, rho0=1.0):
    pts = np.concatenate(orbits) * rho0
    masses = np.concatenate([np.full(len(o), w) for o, w in zip(orbits, weights)]) * (Z - 1) * rho0
    return pts, masses


def min_mass_for_weighted(orbits, weights, Z, proper_radius_m=1.0, a_limit=K.g0):
    r_coord = proper_radius_m / Z
    a_geo_limit = a_limit / K.c**2

    def a_of_rho0(rho0):
        P, m = weighted_cage(orbits, weights, Z, rho0)
        return max_acceleration_on_sphere(P, m, r_coord, n_dirs=1500)[0]

    lo, hi = r_coord * 1.05, r_coord * 1e6
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        if a_of_rho0(mid) > a_geo_limit:
            lo = mid
        else:
            hi = mid
    M_geo = (Z - 1) * hi
    P, m = weighted_cage(orbits, weights, Z, hi)
    return dict(N=len(P), rho0_m=hi, M_kg=K.mass_geo_to_kg(M_geo), M_jup=K.mass_geo_to_kg(M_geo) / K.M_jup,
                M_sun=K.mass_geo_to_kg(M_geo) / K.M_sun, m_bh_min_kg=K.mass_geo_to_kg(float(m.min())),
                B_horizon_max_T=K.B_FIELD_PER_INV_M / float(m.min()), a_limit=a_limit)


def accel_mp(points, masses, X, dps=50):
    """High-precision |∇U|/U² at a single point X (mpmath), to escape float64 cancellation."""
    import mpmath as mpm
    mpm.mp.dps = dps
    Xm = [mpm.mpf(float(v)) for v in X]
    U = mpm.mpf(1)
    g = [mpm.mpf(0)] * 3
    for P, m in zip(points, masses):
        d = [Xm[i] - mpm.mpf(float(P[i])) for i in range(3)]
        r = mpm.sqrt(d[0]**2 + d[1]**2 + d[2]**2)
        mm = mpm.mpf(float(m))
        U += mm / r
        for i in range(3):
            g[i] -= mm * d[i] / r**3
    return float(mpm.sqrt(g[0]**2 + g[1]**2 + g[2]**2) / U**2)


def fit_multipole_law(orbits, weights, Z, x_fit=(0.15, 0.2, 0.25, 0.3, 0.4)):
    """Fit a_max(x) = C x^n (units 1/ρ0) where float64 is reliable; return (C, n)."""
    P, m = weighted_cage(orbits, weights, Z, 1.0)
    a = np.array([max_acceleration_on_sphere(P, m, x)[0] for x in x_fit])
    n, logC = np.polyfit(np.log(x_fit), np.log(a), 1)
    return math.exp(logC), n


def min_mass_from_law(C, n, Z, proper_radius_m=1.0, a_limit=K.g0):
    """Solve c² C (r/ρ0)^n / ρ0 = a_limit with r = proper_radius/Z."""
    r = proper_radius_m / Z
    rho0 = (K.c**2 * C * r**n / a_limit) ** (1.0 / (n + 1))
    M_geo = (Z - 1) * rho0
    return dict(rho0_m=rho0, M_kg=K.mass_geo_to_kg(M_geo), M_jup=K.mass_geo_to_kg(M_geo) / K.M_jup,
                M_sun=K.mass_geo_to_kg(M_geo) / K.M_sun)
