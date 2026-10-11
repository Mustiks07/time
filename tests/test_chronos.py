"""Regression and consistency tests for the CHRONOS toolkit.  Run:  python -m pytest -q"""
import math

import numpy as np
import pytest
import sympy as sp

from chronos import (charged, constants as K, dynamic_shell, kerr, kinematics, mp_cage,
                     neutron_star, static_spherical as ss, thin_shell as ts)


# --------------------------------------------------------------------------- constants
def test_unit_scales():
    assert K.KG_PER_M == pytest.approx(1.3466e27, rel=1e-3)
    assert K.F_PLANCK_GR == pytest.approx(1.2103e44, rel=1e-3)
    assert K.E_SCHWINGER == pytest.approx(1.323e18, rel=1e-3)
    assert K.B_SCHWINGER == pytest.approx(4.414e9, rel=1e-3)
    # Schwarzschild radius of the Sun ≈ 2.953 km
    assert 2 * K.mass_kg_to_geo(K.M_sun) == pytest.approx(2953.25, rel=1e-4)


# --------------------------------------------------------------------------- thin shells
def test_symbolic_junction_matches_closed_form():
    r, M = sp.symbols('r M', positive=True)
    sig, p, (R, Rd, Rdd) = ts.lanczos_symbolic(sp.Integer(1) + 0 * r, 1 - 2 * M / r, r)
    st = {Rd: 0, Rdd: 0}
    for Mv, Rv in [(0.2, 1.0), (0.45, 1.0), (0.3, 2.0)]:
        sub = {M: Mv, R: Rv}
        num = ts.static_shell_flat_interior(Mv, Rv)
        assert float(sig.subs(st).subs(sub)) == pytest.approx(num['sigma'], rel=1e-12)
        assert float(p.subs(st).subs(sub)) == pytest.approx(num['p'], rel=1e-12)


def test_dec_limits_thin_shell_to_factor_five():
    assert ts.max_Z_for_p_over_sigma(1.0) == pytest.approx(5.0)
    assert ts.p_over_sigma(0.2) == pytest.approx(1.0)
    M = ts.shell_mass_for_alpha(0.2, 1.0)
    assert 2 * M == pytest.approx(24 / 25)
    assert ts.p_over_sigma(0.1) == pytest.approx(2.25)


def test_stability_threshold_closed_form():
    for Z in [1.5, 2, 3, 5, 10]:
        a = 1 / Z
        closed = (1 - a) * (3 * a * a + 2 * a + 1) / (4 * a * a * (3 * a + 1))
        num, above = ts.stability_beta2_min(a)
        assert above and num == pytest.approx(closed, rel=1e-9)
        V0, V1 = ts.equilibrium_check(a)
        assert abs(V0) < 1e-12 and abs(V1) < 1e-12


def test_einstein_shell_causality_limit():
    _, v2 = ts.einstein_shell_speed(1 / 3)
    assert v2 == pytest.approx(1.0)


# --------------------------------------------------------------------------- kinematics
def test_kinematic_bound_optimal_control():
    num, ana, _ = kinematics.optimal_control_check(2.0, N=120)
    assert num <= ana * (1 + 1e-6)
    assert num == pytest.approx(ana, rel=2e-3)


def test_kinematic_mission_numbers():
    m = kinematics.mission_for_ratio(10)
    assert m['x'] == pytest.approx(4.4999, rel=1e-3)
    assert m['T_years'] == pytest.approx(17.44, rel=1e-3)


# --------------------------------------------------------------------------- Kerr
def test_kerr_ut_formulas_agree():
    for chi in [0.0, 0.5, 0.9, 0.99]:
        r = kerr.r_isco(chi) * 1.1
        assert kerr.ut_from_metric(r, chi) == pytest.approx(kerr.ut_bpt(r, chi), rel=1e-12)
    assert kerr.ut_bpt(6.0, 0.0) == pytest.approx(math.sqrt(2), rel=1e-12)


def test_kerr_chi_for_Z10():
    chi = kerr.chi_for_isco_dilation(10.0)
    assert chi == pytest.approx(0.99752, abs=2e-5)


@pytest.mark.slow
def test_marck_tidal_matches_riemann():
    for (r, chi) in [(8.0, 0.0), (3.0, 0.9), (1.3, 0.997)]:
        a = np.sort(kerr.tidal_eigenvalues_marck(r, chi))
        b = kerr.kerr_tidal_numeric(r, chi)
        assert np.allclose(a, b, rtol=1e-6, atol=1e-10)


# --------------------------------------------------------------------------- static spherical
def test_einstein_equations_symbolic():
    assert all(sp.simplify(x) == 0 for x in ss.einstein_tensor_check())


def test_cavity_bound_random_profiles():
    tight, case = ss.random_profile_test(800, seed=7)
    assert tight < 1.0


def test_cavity_bound_holds_for_thin_shell():
    for a in [0.9, 0.5, 0.2, 0.1, 0.01]:
        M = ts.shell_mass_for_alpha(a, 1.0)
        assert a >= ss.alpha_lower_bound(M, 1.0)


def test_single_dec_shell_cannot_reach_Z10():
    assert ss.optimise_nested_shells(10.0, 1, n_starts=10) is None
    res = ss.optimise_nested_shells(4.0, 1, n_starts=10)
    assert res['M_over_rin'] == pytest.approx(ts.shell_mass_for_alpha(0.25, 1.0), rel=1e-6)


def test_thin_layer_dec_profile_returns_24_over_25():
    y, g = ss.thin_layer_dec_profile(200001)
    i = np.where((g[1:] < 0) & (g[:-1] >= 0))[0][0]
    assert y[i] == pytest.approx(0.96, abs=1e-4)


# --------------------------------------------------------------------------- charged / BPS
def test_extremal_shell_is_stress_free_and_neutral():
    for R in [1.0, 1.5, 3.0]:
        s = charged.rn_static_shell(0.9, 0.9, R)
        assert abs(s['p']) < 1e-14
        assert s['rest_mass'] == pytest.approx(0.9, rel=1e-12)
        assert s['alpha'] == pytest.approx(1 - 0.9 / R, rel=1e-12)


def test_mistuning_formula():
    acc, _ = charged.mistuning_acceleration(10, 2.0, 1e-7)
    assert acc == pytest.approx(9 * 1e-7 / 2.0, rel=1e-3)


# --------------------------------------------------------------------------- MP cage
def test_mp_cage_centre_lapse():
    P, m = mp_cage.cage(mp_cage.icosahedron(), 10.0)
    U, _ = mp_cage.U_and_grad(P, m, np.zeros((1, 3)))
    assert U[0] == pytest.approx(10.0, rel=1e-12)


def test_mp_multipole_cancellation_exponents():
    I, D, ID = mp_cage.icosahedron(), mp_cage.dodecahedron(), mp_cage.icosidodecahedron()
    w2 = mp_cage.cancelling_weights([I, D], [6])
    w3 = mp_cage.cancelling_weights([I, D, ID], [6, 10])
    assert np.all(w2 > 0) and np.all(w3 > 0)
    assert mp_cage.fit_multipole_law([I], [1 / 12], 10)[1] == pytest.approx(5, abs=0.05)
    assert mp_cage.fit_multipole_law([I, D], w2, 10)[1] == pytest.approx(9, abs=0.05)
    assert mp_cage.fit_multipole_law([I, D, ID], w3, 10)[1] == pytest.approx(11, abs=0.05)


# --------------------------------------------------------------------------- dynamics / NS
def test_breathing_shell_never_beats_its_static_depth():
    res = dynamic_shell.scan_pockets(40, seed=3)
    assert len(res) > 5
    assert all(r['Z_avg'] <= r['Z_at_Rmin'] + 1e-9 for r in res)


def test_causal_neutron_star_compactness():
    mx = neutron_star.max_mass_config()
    # Haensel, Lasota & Zdunik (2000): 2M/R = 0.7081, z_max = 0.851 for p = ρ - ρ0
    assert mx['compactness'] == pytest.approx(0.7081, abs=5e-4)
    assert mx['Z_surface'] - 1 == pytest.approx(0.851, abs=1e-3)


def test_theorem2_key_identity_symbolic():
    """Φ' - [y/(r(1-y)) + y'/(2(1-y))] = 4π r² (p_r - ρ)/(r - 2m)  (≤ 0 iff p_r ≤ ρ)."""
    r = sp.symbols('r', positive=True)
    m = sp.Function('m')(r)
    pr = sp.Function('p_r')(r)
    rho = sp.diff(m, r) / (4 * sp.pi * r**2)
    dPhi = (m + 4 * sp.pi * r**3 * pr) / (r * (r - 2 * m))
    y = 2 * m / r
    rhs = y / (r * (1 - y)) + sp.diff(y, r) / (2 * (1 - y))
    assert sp.simplify(dPhi - rhs - 4 * sp.pi * r**2 * (pr - rho) / (r - 2 * m)) == 0


def test_causal_stability_thresholds():
    # light shells with causal EOS are stable exactly outside the photon sphere (r = 3m)
    assert ss.test_shell_critical_compactness(1.0) == pytest.approx(2 / 3, abs=1e-6)
    assert ss.single_shell_critical_Z(1.0) == pytest.approx(2.5297565, abs=1e-6)
    M, info = ss.stable_family_mass(10.0, 1.0)
    assert info['n'] == pytest.approx(1.0, abs=1e-5)
    assert M == pytest.approx(0.5 * (1 - 1 / 2.5297565**2) * 10 / 2.5297565, rel=1e-5)
