"""Physical constants (CODATA 2018 / IAU nominal values) and unit helpers."""
import math

G = 6.67430e-11            # m^3 kg^-1 s^-2
c = 299_792_458.0          # m s^-1
hbar = 1.054571817e-34     # J s
e_charge = 1.602176634e-19 # C
m_e = 9.1093837015e-31     # kg
m_p = 1.67262192369e-27    # kg
eps0 = 8.8541878128e-12    # F m^-1
mu0 = 1.25663706212e-6     # N A^-2
alpha_fs = 7.2973525693e-3

M_sun = 1.98841e30         # kg  (IAU nominal GM_sun / G)
M_jup = 1.89813e27         # kg
M_earth = 5.97217e24       # kg
year = 3.15576e7           # s (Julian year)
ly = c * year              # m
g0 = 9.80665               # m s^-2

# --- derived scales -------------------------------------------------------
KG_PER_M = c**2 / G                     # mass corresponding to 1 m of geometric mass  (1.3466e27 kg/m)
F_PLANCK_GR = c**4 / G                  # "c^4/G" force scale, N   (1.2103e44 N)
PA_PER_INV_M2 = c**4 / G                # geometric stress (1/m^2) -> Pa
J_PER_M = c**4 / G                      # geometric energy (m) -> J
E_FIELD_PER_INV_M = c**2 / math.sqrt(4 * math.pi * eps0 * G)   # geometric E-field (1/m) -> V/m
B_FIELD_PER_INV_M = E_FIELD_PER_INV_M / c                      # geometric B-field (1/m) -> T
Q_EXTREMAL_PER_KG = math.sqrt(4 * math.pi * eps0 * G)          # C/kg for Q = M (extremal)
QM_EXTREMAL_PER_KG = math.sqrt(4 * math.pi * G / mu0)          # A m / kg, magnetic charge for extremality

E_SCHWINGER = m_e**2 * c**3 / (e_charge * hbar)                # V/m (1.32e18)
B_SCHWINGER = m_e**2 * c**2 / (e_charge * hbar)                # T   (4.41e9)
M_W_GEV = 80.369
B_ELECTROWEAK = B_SCHWINGER * (M_W_GEV / 0.51099895e-3) ** 2   # ~ m_W^2/e, Ambjorn-Olesen scale (T)

RHO_NUCLEAR = 2.3e17      # kg/m^3, nuclear saturation density (approx. 0.16 fm^-3 * m_n)
l_planck = math.sqrt(hbar * G / c**3)


def mass_geo_to_kg(m_geo_metres: float) -> float:
    return m_geo_metres * KG_PER_M


def mass_kg_to_geo(m_kg: float) -> float:
    return m_kg / KG_PER_M
