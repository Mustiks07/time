#!/usr/bin/env python3
"""Reproduce every number and figure quoted in reports/ (CHRONOS round 1).

    python scripts/run_all.py            # ~3-5 minutes on 4 cores
Outputs: results/round1_results.json, results/*.csv, results/figures/*.png
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

from chronos import (charged, constants as K, dynamic_shell, kerr, kinematics, mp_cage,  # noqa: E402
                     neutron_star, scales, static_spherical as ss, thin_shell as ts)

ROOT = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(ROOT, 'results')
FIG = os.path.join(RES, 'figures')
os.makedirs(FIG, exist_ok=True)

# reference categorical palette (fixed order) + text tokens
S1, S2, S3, S4, S5, S6 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 2.0,
                     'legend.frameon': False, 'figure.dpi': 130})

OUT: dict = {}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)


def section(title):
    print(f'\n=== {title} ===', flush=True)


# ---------------------------------------------------------------------------------------------
section('1. Kinematic (special-relativistic) bound')
kin = {'Z10_a1g': kinematics.mission_for_ratio(10, K.g0),
       'Z10_a3g': kinematics.mission_for_ratio(10, 3 * K.g0),
       'a_for_Z10_T1yr_in_g': kinematics.a_required(10, K.year) / K.g0,
       'track_radius_gamma10_1g_ly': kinematics.circular_track_radius(10) / K.ly}
num, ana, _ = kinematics.optimal_control_check(2.5, N=160)
kin['optimal_control_check'] = {'x_total': 2.5, 'numerical': num, 'analytic': ana}
OUT['kinematics'] = kin
print(json.dumps(kin, indent=1))

fig, ax = plt.subplots(figsize=(6.4, 3.8))
T = np.geomspace(0.1, 60, 400)
for a_g, col in [(1, S1), (3, S2), (10, S3)]:
    x = a_g * K.g0 * T * K.year / (4 * K.c)
    ax.plot(T, np.sinh(x) / x, color=col, label=f'|a| ≤ {a_g} g')
ax.axhline(10, color=INK2, lw=1, ls='--')
ax.text(0.12, 11.5, 'цель 1:10', color=INK2)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1, 1e3)
ax.set_xlabel('собственное время путешественника T, лет'); ax.set_ylabel('t_база / T (максимум)')
ax.set_title('Кинематический предел: t/T ≤ sinh(x)/x, x = aT/4c', fontsize=10)
ax.legend(loc='upper left')
save(fig, 'fig01_kinematic_bound.png')

# ---------------------------------------------------------------------------------------------
section('2. Thin static shell (Israel junction)')
rows = ts.shell_table()
with open(os.path.join(RES, 'thin_shell_table.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Z', 'R_m', 'M_kg', 'M_jupiter', 'M_over_R_geo', 'sigma_kg_m2', 'surface_tension_N_m',
                'stress_if_wall_R/100_Pa', 'p_over_sigma', 'F_great_circle_c4/G', 'DEC_ok'])
    for r in rows:
        w.writerow([r.Z, r.R_m, f'{r.M_kg:.4e}', f'{r.M_jup:.4g}', f'{r.M_over_R_geo:.6f}', f'{r.sigma_kg_m2:.4e}',
                    f'{r.tension_N_m:.4e}', f'{r.stress_1pct_Pa:.4e}', f'{r.p_over_sigma:.4g}',
                    f'{r.F_over_c4G:.4g}', r.DEC])
OUT['thin_shell_table'] = [r.__dict__ for r in rows]
stab = {Z: ts.stability_beta2_min(1 / Z)[0] for Z in [1.5, 2, 2.53, 3, 5, 10, 100]}
OUT['thin_shell_stability_beta2_min'] = stab
OUT['thin_shell_dec_Zmax'] = ts.max_Z_for_p_over_sigma(1.0)
OUT['thin_shell_Zmax_causal_stable'] = 2.52975649803110   # root of β²_min(Z) = 1 (sympy nsolve, see REPORT_05)
OUT['einstein_shell_Zmax_causal'] = ts.max_Z_for_p_over_sigma(0.5)
for r in rows:
    print(f'Z={r.Z:5} R={r.R_m:5} m  M={r.M_kg:.3e} kg ({r.M_jup:.3g} MJ)  p/σ={r.p_over_sigma:.3g} '
          f'tension={r.tension_N_m:.2e} N/m  DEC={r.DEC}')
print('β²_min:', stab)

fig, ax = plt.subplots(figsize=(6.4, 3.8))
Zs = np.geomspace(1.01, 100, 400)
al = 1 / Zs
ax.plot(Zs, (1 - al) / (4 * al), color=S1, label='p/σ (нужно для равновесия)')
ax.plot(Zs, (1 - al) * (3 * al**2 + 2 * al + 1) / (4 * al**2 * (3 * al + 1)), color=S2,
        label='β²_min (нужно для устойчивости)')
ax.axhline(1, color=INK2, lw=1, ls='--'); ax.text(12, 0.55, 'p = σ (граница DEC);\nβ² = 1 (звук = c)', color=INK2, fontsize=8)
ax.axvline(5, color=S3, lw=1); ax.text(5.2, 0.02, 'Z = 5', color=INK2, fontsize=8)
ax.axvline(2.53, color=S4, lw=1); ax.text(2.6, 0.006, 'Z ≈ 2.53', color=INK2, fontsize=8)
ax.axvline(10, color=INK, lw=1, ls=':'); ax.text(10.5, 0.006, 'цель', color=INK2, fontsize=8)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(3e-3, 3e3)
ax.set_xlabel('коэффициент замедления Z = 1/α'); ax.set_ylabel('безразмерное')
ax.set_title('Тонкая оболочка: равновесие и устойчивость', fontsize=10)
ax.legend(loc='upper left')
save(fig, 'fig02_thin_shell_stress_stability.png')

# ---------------------------------------------------------------------------------------------
section('3. Cavity-redshift mass bound and optimised configurations')
tight, case = ss.random_profile_test(4000, seed=1)
adv = ss.adversarial_bound_search()
nested = {}
for Z in [2.0, 5.0, 10.0, 100.0]:
    best = None
    for Kn in [1, 2, 3, 4, 6, 8]:
        r = ss.optimise_nested_shells(Z, Kn, n_starts=25)
        if r and (best is None or r['M_over_rin'] < best['M_over_rin']):
            best = dict(r, K=Kn)
    nested[Z] = best
    if best:
        print(f'nested DEC shells Z={Z}: M/r_in = {best["M_over_rin"]:.5f} (K={best["K"]})')
cont_rad = ss.optimise_continuum(10.0, N=24, mode='RAD', n_starts=8)
OUT['cavity_bound'] = {
    'statement': 'alpha_cavity >= 1 - 2M/r_in  for rho >= 0, p_r <= rho, no horizon',
    'MC_max_tightness': float(tight), 'MC_worst_case': case, 'adversarial_max_tightness': float(adv),
    'RAD_continuum_Z10_M_over_rin': cont_rad['M_over_rin'] if cont_rad else None,
    'bound_M_over_rin': {Z: ss.min_mass_bound(Z) for Z in [2, 5, 10, 100, 1000]},
    'thin_shell_M_over_rin': {Z: ts.shell_mass_for_alpha(1 / Z, 1.0) for Z in [2, 5, 10, 100, 1000]},
    'extremal_shell_M_over_rin': {Z: 1 - 1 / Z for Z in [2, 5, 10, 100, 1000]},
    'nested_DEC_shells_M_over_rin': {Z: (v['M_over_rin'] if v else None) for Z, v in nested.items()},
}
dec_json = os.path.join(RES, 'dec_control_opt_Z10.json')      # produced separately (E9, ~10 min)
if os.path.exists(dec_json):
    d = json.load(open(dec_json))
    OUT['cavity_bound']['continuum_DEC_control_Z10_near_feasible'] = d['history'][-1]
print('MC tightness', tight, 'adversarial', adv, 'RAD continuum', cont_rad and cont_rad['M_over_rin'])

fig, ax = plt.subplots(figsize=(6.4, 4.0))
x = np.linspace(0, 0.999, 500)          # M / r_in
ax.plot(x, np.clip(1 - 2 * x, 1e-3, None), color=INK2, ls='--', lw=1.5, label='теорема: α ≥ 1 − 2M/r (ρ≥0, p_r≤ρ)')
ax.plot(x[x < 0.5], np.sqrt(1 - 2 * x[x < 0.5]), color=S1, label='тонкая оболочка: α = √(1−2M/R)')
ax.plot(x, 1 - x, color=S2, label='экстремальная (BPS) оболочка: α = 1 − M/R')
pts = [(v['M_over_rin'], 1 / Z) for Z, v in nested.items() if v]
ax.plot(*zip(*pts), 'o', color=S3, ms=8, mec='white', mew=1.5, label='оптимум DEC (вложенные оболочки)')
ax.fill_between(x, 1e-3, np.clip(1 - 2 * x, 1e-3, None), color=GRID, alpha=0.6, lw=0)
ax.text(0.03, 0.05, 'запрещено\nпри ρ ≥ 0, p_r ≤ ρ', color=INK2, fontsize=8)
ax.axhline(0.1, color=INK, lw=1, ls=':'); ax.text(0.52, 0.115, 'Z = 10', color=INK2, fontsize=8)
ax.set_yscale('log'); ax.set_ylim(2e-3, 1.05); ax.set_xlim(0, 1)
ax.set_xlabel('M / r_in   (в единицах c²/G на метр радиуса полости)'); ax.set_ylabel('α = dτ_внутри / dt_∞')
ax.set_title('Глубина «колодца времени» против массы', fontsize=10)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8)
fig.set_size_inches(6.4, 4.8)
save(fig, 'fig03_cavity_bound.png')

# ---------------------------------------------------------------------------------------------
section('4. Mass requirements in SI')
mass_tab = []
for R in [1, 2, 10, 100, 1000]:
    row = {'R_m': R}
    for label, coef in [('bound', ss.min_mass_bound(10)), ('thin_shell_noDEC', ts.shell_mass_for_alpha(0.1, 1)),
                        ('nested_DEC', nested[10.0]['M_over_rin'] if nested[10.0] else float('nan')),
                        ('extremal_BPS', 0.9)]:
        row[label + '_kg'] = coef * R * K.KG_PER_M
        row[label + '_Mjup'] = coef * R * K.KG_PER_M / K.M_jup
    row['mean_density_kg_m3_thin_shell'] = scales.mean_density_required(R, 0.99)
    mass_tab.append(row)
OUT['mass_requirements_Z10'] = mass_tab
for row in mass_tab:
    print({k: f'{v:.3g}' for k, v in row.items()})

# ---------------------------------------------------------------------------------------------
section('5. Kerr ISCO free-fall habitat')
chi_star = kerr.chi_for_isco_dilation(10.0)
ri = kerr.r_isco(chi_star)
lam = kerr.tidal_eigenvalues_marck(ri, chi_star)
esc = kerr.escape_boost(ri, chi_star)
kerr_out = {'chi_star': chi_star, 'one_minus_chi': 1 - chi_star, 'r_isco_M': ri, 'E': kerr.energy_L(ri, chi_star)[0],
            'L': kerr.energy_L(ri, chi_star)[1], 'v_zamo': kerr.zamo_speed(ri, chi_star),
            'tidal_eigs_over_Minv2': lam.tolist(),
            'M_min_tides_1g_2m_Msun': kerr.min_mass_for_tides(max(abs(lam))) / K.M_sun,
            'escape_delta_rapidity': esc[0], 'escape_photon_mass_ratio': esc[1],
            'Z_isco': {chi: kerr.ut_bpt(kerr.r_isco(chi), chi) for chi in [0, 0.5, 0.9, 0.99, 0.998]},
            'r_Z10_chi0.998': kerr.r_for_dilation(10, 0.998)}
OUT['kerr'] = kerr_out
print(json.dumps(kerr_out, indent=1, default=float))

fig, ax = plt.subplots(figsize=(6.4, 3.8))
oc = np.linspace(0, 0.9999, 3000)
ax.plot(1 - oc, [kerr.ut_bpt(kerr.r_isco(c), c) for c in oc], color=S1, label='Z на ISCO (прямая орбита)')
ax.axhline(10, color=INK2, ls='--', lw=1); ax.axvline(1 - chi_star, color=S2, lw=1)
ax.axvline(0.002, color=S3, lw=1)
ax.text(1.5e-2, 14, f'χ* = {chi_star:.5f}\n(Z = 10 на ISCO)', color=INK2, fontsize=8)
ax.text(1.8e-3, 2, 'предел Торна\nχ = 0.998', color=INK2, fontsize=8)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('1 − χ'); ax.set_ylabel('Z = dt/dτ')
ax.invert_xaxis(); ax.set_title('Орбита ISCO вокруг керровской ЧД', fontsize=10)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_yticks([1.5, 2, 3, 5, 10, 20, 30])
ax.legend(loc='upper left')
save(fig, 'fig04_kerr_isco.png')

# ---------------------------------------------------------------------------------------------
section('6. Extremal (BPS) shell and Majumdar–Papapetrou cage')
bps = {'electric_R2m': charged.extremal_shell(10, 2.0, magnetic=False),
       'magnetic_R2m': charged.extremal_shell(10, 2.0, magnetic=True),
       'electric_min_R_below_Schwinger_m': charged.min_radius_below_schwinger(10),
       'mistuning_Rddot_over_delta_geo_R2m': charged.mistuning_acceleration(10, 2.0, 1e-7)[0] / 1e-7}
bps['electric_min_mass_below_Schwinger_Msun'] = K.mass_geo_to_kg(0.9 * bps['electric_min_R_below_Schwinger_m']) / K.M_sun
OUT['bps'] = bps
I, D, ID = mp_cage.icosahedron(), mp_cage.dodecahedron(), mp_cage.icosidodecahedron()
w2 = mp_cage.cancelling_weights([I, D], [6])
w3 = mp_cage.cancelling_weights([I, D, ID], [6, 10])
designs = [('I12', [I], [1 / 12]), ('I+D 32', [I, D], w2), ('I+D+ID 62', [I, D, ID], w3)]
mp_out = {'weights_ID': w2.tolist(), 'weights_IDID': w3.tolist(), 'tesla_per_ms2': mp_cage.TESLA_PER_MS2, 'designs': {}}
xs = np.geomspace(0.02, 0.6, 40)
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for (name, orbs, wts), col in zip(designs, [S1, S2, S3]):
    C, n = mp_cage.fit_multipole_law(orbs, wts, 10)
    P, m = mp_cage.weighted_cage(orbs, wts, 10)
    a = [mp_cage.max_acceleration_on_sphere(P, m, xx, n_dirs=1500)[0] for xx in xs]
    a = np.array(a)
    ok = a > 1e-14                      # below ~1e-14 float64 cancellation dominates
    ax.plot(xs[ok], a[ok], color=col, label=f'{name}: a ∝ x^{n:.0f} (расчёт)')
    ax.plot(xs, C * xs**n, color=col, lw=1, ls=':')
    rec = {'C': C, 'n': n}
    for lab, alim in [('1g', K.g0), ('B<=2T', 2 / mp_cage.TESLA_PER_MS2)]:
        rec[lab] = mp_cage.min_mass_from_law(C, n, 10, 1.0, alim)
        rec[lab]['tolerance_relative_displacement'] = None
    mp_out['designs'][name] = rec
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e-17, 1e-1)
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.text(0.021, 3e-17, 'пунктир — степенной закон (проверен mpmath)', color=INK2, fontsize=8)
ax.set_xlabel('r / ρ₀ (координатный радиус в обитаемой зоне)'); ax.set_ylabel('a · ρ₀ / c²  (= поле B в геом. ед.)')
ax.set_title('Клетка Маджумдара–Папапетру: подавление мультиполей', fontsize=10)
ax.legend(loc='lower right', fontsize=8)
save(fig, 'fig05_mp_cage_multipoles.png')
# continuous-shell limit & positional tolerance estimate (single BH displaced by δ creates a dipole)
for name, rec in mp_out['designs'].items():
    for lab, alim in [('1g', K.g0), ('B<=2T', 2 / mp_cage.TESLA_PER_MS2)]:
        rho0 = rec[lab]['rho0_m']
        N = {'I12': 12, 'I+D 32': 32, 'I+D+ID 62': 62}[name]
        m_i = 9 * rho0 / N
        # |∇U| at centre from a displacement δ of one mass m_i: ≈ 2 m_i δ / ρ0³ (radial) → a = that / Z²
        delta = (alim / K.c**2) * rho0**3 * 100 / (2 * m_i)
        rec[lab]['tolerance_relative_displacement'] = delta
mp_out['continuous_shell_limit_kg'] = K.mass_geo_to_kg(0.9)
OUT['mp_cage'] = mp_out
print(json.dumps(mp_out, indent=1, default=float))

# ---------------------------------------------------------------------------------------------
section('7. Dynamic (breathing) Einstein shells, neutron stars')
pockets = dynamic_shell.scan_pockets(200, seed=0)
OUT['breathing'] = {'n_cases': len(pockets), 'max_Z_avg': max(p['Z_avg'] for p in pockets),
                    'max_Z_at_Rmin': max(p['Z_at_Rmin'] for p in pockets),
                    'violations_Zavg_gt_ZRmin': sum(p['Z_avg'] > p['Z_at_Rmin'] + 1e-9 for p in pockets)}
seq = neutron_star.sequence(60)
mx = neutron_star.max_mass_config()
OUT['neutron_star_causal_max_mass'] = mx
print(OUT['breathing']); print(mx)
fig, ax = plt.subplots(figsize=(6.4, 3.6))
Ms = [d['M_sun'] for d in seq]
ax.plot([d['rho_c_over_rhos'] for d in seq], [d['Z_centre'] for d in seq], color=S1, label='Z в центре')
ax.plot([d['rho_c_over_rhos'] for d in seq], [d['Z_surface'] for d in seq], color=S2, label='Z на поверхности')
ax.axvline(mx['rho_c_over_rhos'], color=INK2, lw=1, ls='--')
ax.text(mx['rho_c_over_rhos'] * 1.05, 12, 'макс. масса\n(дальше — неустойчиво)', color=INK2, fontsize=8)
ax.axhline(10, color=INK, lw=1, ls=':')
ax.set_xscale('log'); ax.set_xlabel('ρ_c / ρ_s'); ax.set_ylabel('Z')
ax.set_title('Самая жёсткая причинная ЭС p = ρ − ρ_s (TOV)', fontsize=10)
ax.legend(loc='upper left')
save(fig, 'fig06_neutron_star.png')

# ---------------------------------------------------------------------------------------------
section('8. Scale comparisons (why ordinary engineering cannot reach 1:10 locally)')
R = 2.0
M_needed = ss.min_mass_bound(10) * R * K.KG_PER_M
sc = {
    'R_m': R, 'M_needed_bound_kg': M_needed,
    'energy_equiv_J': M_needed * K.c**2,
    'energy_in_world_years': M_needed * K.c**2 / scales.WORLD_ENERGY_PER_YEAR_J,
    'energy_in_solar_luminosity_years': M_needed * K.c**2 / (scales.SUN_LUMINOSITY_W * K.year),
    'mean_density_kg_m3': scales.mean_density_required(R, 0.9),
    'density_over_nuclear': scales.mean_density_required(R, 0.9) / K.RHO_NUCLEAR,
    'radius_where_nuclear_density_suffices_km': scales.radius_for_density(K.RHO_NUCLEAR, 0.9) / 1e3,
    'stress_scale_Pa': scales.required_stress_scale(R),
    'graphene_epsilon': scales.specific_strength_ratio(),
    'B_field_needed_T': scales.magnetic_field_for_mass(M_needed, R),
    'B_needed_over_lab_static': scales.magnetic_field_for_mass(M_needed, R) / scales.B_LAB_STATIC_T,
    'B_needed_over_magnetar': scales.magnetic_field_for_mass(M_needed, R) / scales.B_MAGNETAR_T,
    'casimir_u_at_1um_J_m3': scales.casimir_energy_density(1e-6),
    'energy_density_needed_J_m3': M_needed * K.c**2 / (4 / 3 * math.pi * R**3),
    'casimir_gap_for_needed_u_m': scales.casimir_gap_for_energy_density(M_needed * K.c**2 / (4 / 3 * math.pi * R**3)),
    'qei_length_m': scales.qei_length_for_energy_density(M_needed * K.c**2 / (4 / 3 * math.pi * R**3)),
    'semiclassical_ratio_Z10_R2m': scales.semiclassical_ratio(10, R),
    'kinematic_energy_per_kg_gamma10_J': 9 * K.c**2,
    'kinematic_energy_100t_J': 9 * K.c**2 * 1e5,
}
OUT['scales'] = sc
print(json.dumps(sc, indent=1))

with open(os.path.join(RES, 'round1_results.json'), 'w') as f:
    json.dump(OUT, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
print('\nwritten results/round1_results.json and figures', flush=True)
