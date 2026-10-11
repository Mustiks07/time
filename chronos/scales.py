"""Order-of-magnitude scale comparisons (all SI).  Each function documents its formula."""
from __future__ import annotations

import math

from . import constants as K

# Reference data: commonly quoted orders of magnitude.  Primary sources were NOT re-verified in
# round 1 (see REPORT_05 §12); conclusions depend only on the order of magnitude.
GRAPHENE_STRENGTH_PA = 1.3e11       # intrinsic tensile strength of monolayer graphene (~1e11 Pa)
GRAPHENE_DENSITY = 2.2e3            # kg/m^3 (bulk-equivalent)
WORLD_ENERGY_PER_YEAR_J = 6.3e20    # ~ 620 EJ primary energy per year (2023 order of magnitude)
SUN_LUMINOSITY_W = 3.828e26
B_LAB_STATIC_T = 45.5               # strongest continuous lab field order (hybrid/HTS magnets)
B_LAB_PULSED_T = 1.2e3              # destructive single-turn/EMFC record order (~1200 T)
B_MAGNETAR_T = 1e11


def mean_density_required(R, compactness_2M_over_R):
    """ρ̄ = 3M/(4πR³) with M = (y/2) R c²/G."""
    M = 0.5 * compactness_2M_over_R * R * K.KG_PER_M
    return 3 * M / (4 * math.pi * R**3)


def radius_for_density(rho, compactness_2M_over_R):
    """Inverse of the above: R at which a uniform ball of density ρ reaches compactness y."""
    return math.sqrt(3 * compactness_2M_over_R * K.c**2 / (8 * math.pi * K.G * rho))


def specific_strength_ratio(strength_pa=GRAPHENE_STRENGTH_PA, density=GRAPHENE_DENSITY):
    """ε = (stress)/(ρ c²): what fraction of rest energy a material can hold as stress."""
    return strength_pa / (density * K.c**2)


def magnetic_field_for_mass(M_kg, R):
    """Uniform field filling a ball of radius R whose energy equals M c²: B²/(2μ0)·V = M c²."""
    V = 4 / 3 * math.pi * R**3
    return math.sqrt(2 * K.mu0 * M_kg * K.c**2 / V)


def casimir_energy_density(a):
    """|u| = π² ħ c / (720 a⁴) between ideal plates at separation a."""
    return math.pi**2 * K.hbar * K.c / (720 * a**4)


def casimir_gap_for_energy_density(u):
    return (math.pi**2 * K.hbar * K.c / (720 * u)) ** 0.25


def qei_length_for_energy_density(u):
    """Quantum-inequality scaling (massless fields): a negative energy density of magnitude |u|
    can persist only over regions/times L ≲ (ħ c / |u|)^{1/4} (up to O(1)–O(10²) factors)."""
    return (K.hbar * K.c / u) ** 0.25


def semiclassical_ratio(Z, R):
    """Static RSET near a redshift-Z region ~ Z⁴ ħ c /(2880 π² R⁴) (Boulware-like scaling),
    compared with the classical stress needed ~ c⁴/(G R²).  Ratio ≈ Z⁴ ℓ_P² / (2880 π² R²)."""
    return Z**4 * K.l_planck**2 / (2880 * math.pi**2 * R**2)


def required_stress_scale(R):
    """Characteristic stress of a compact (2M/R ~ 1) configuration: c⁴/(G R²) in Pa."""
    return K.c**4 / (K.G * R**2)
