"""Q6: static thin shell of charged dust, flat interior, Reissner-Nordstrom exterior.

G=c=1, Gaussian-geometrised charge (Q has dimension of length): f_out = 1 - 2M/r + Q^2/r^2.
Same Israel formulas as Q1 (they hold for any f):
   sigma = (1 - sqrt(f(R)))/(4 pi R),  p = (1/8piR)[ (f'/2 + f/R)/sqrt(f) * R ... ] i.e.
   p = (1/8pi)[ (f'(R)/2)/sqrt(f) + sqrt(f)/R - 1/R ].
For RN: f'/2 + f/R = (1 - M/R)/R  (the Q^2 terms cancel), so p = (1/8piR)[(1-M/R)/sqrt(f) - 1].
Dust (p=0)  =>  (1 - M/R)^2 = 1 - 2M/R + Q^2/R^2  =>  Q^2 = M^2: the shell must be EXTREMAL, |Q|=M,
for every radius R (> M). Then sqrt(f) = 1 - M/R, interior lapse alpha = 1 - M/R, Z = R/(R-M).
Shell rest mass m = 4 pi R^2 sigma = R (1 - sqrt f) = M, so charge / rest-mass = 1 (geometrised):
the Majumdar-Papapetrou-type balance e/m = sqrt(G) (Gaussian), i.e. in SI q/m = sqrt(4 pi eps0 G).
(The radial electric force on the shell is balanced by gravity: shell equation of motion for dust is
 m d^2R/dtau^2-type force balance; the static condition is precisely p=0 <=> |Q|=M.)
Electric field measured by static observer at shell (orthonormal frame, RN: F_{t^r^} = Q/R^2):
   E_geom = Q/R^2 (1/m);  E_SI = E_geom * c^2 / sqrt(4 pi eps0 G)  [V/m].
"""
import sympy as sp
import numpy as np
from consts import *

M, R, Q, r = sp.symbols('M R Q r', positive=True)
f = 1 - 2 * M / r + Q**2 / r**2
fp = sp.diff(f, r)
p = (fp / 2 / sp.sqrt(f) + sp.sqrt(f) / r - 1 / r).subs(r, R) / (8 * sp.pi)
sigma = ((1 - sp.sqrt(f)) / (4 * sp.pi * r)).subs(r, R)
print("simplified f'/2 + f/R :", sp.simplify((fp / 2 + f / r).subs(r, R)))
sol = sp.solve(sp.Eq(sp.sqrt(f.subs(r, R)), 1 - M / R), Q)
print("p=0 solutions for Q (with sqrt f = 1-M/R branch, R>M):", sol)
print("check p at Q=M:", sp.simplify(p.subs(Q, M).subs(sp.sqrt((R - M)**2 / R**2), (R - M) / R)))
print("numeric check p(Q=M) for M=0.9,R=2:", float(p.subs({Q: 0.9, M: 0.9, R: 2})), " sigma*4piR^2 =",
      float((sigma * 4 * sp.pi * R**2).subs({Q: 0.9, M: 0.9, R: 2})))

qm = np.sqrt(4 * np.pi * eps0 * G)
print(f"\nextremal charge-to-mass ratio q/m = sqrt(4 pi eps0 G) = {qm:.6e} C/kg  (inverse {1/qm:.4e} kg/C)")
print(f"  compare electron e/m_e = {e_ch/m_e:.4e} C/kg ; proton e/m_p = {e_ch/1.67262192595e-27:.4e} C/kg")
print(f"  => one excess elementary charge per {e_ch/qm/1.66053906892e-27:.4e} u of rest mass "
      f"({e_ch/qm:.4e} kg per e)")

Z, Rm = 10.0, 2.0
al = 1 / Z
Mg = Rm * (1 - al)   # M/R = 1 - 1/Z
Mkg = Mg * KG_PER_M
Qg = Mg
Q_SI = Qg * c**2 / np.sqrt(G / (4 * np.pi * eps0))
E_geom = Qg / Rm**2
E_SI = E_geom * c**2 / qm
E_SI_check = Q_SI / (4 * np.pi * eps0 * Rm**2)
E_schw = m_e**2 * c**3 / (e_ch * hbar)
print(f"\nZ=10, R=2 m: M/R = {Mg/Rm}, M = {Mg} m = {Mkg:.5e} kg = {Mkg/M_J:.4f} M_J = {Mkg/M_sun:.4e} Msun")
print(f"  Q = {Q_SI:.5e} C ; check Q/M = {Q_SI/Mkg:.5e} C/kg")
print(f"  E at shell (static frame) = {E_SI:.5e} V/m (Coulomb-form check {E_SI_check:.5e})")
print(f"  Schwinger field m_e^2 c^3/(e hbar) = {E_schw:.5e} V/m ; ratio E/E_S = {E_SI/E_schw:.4e}")
# general: E = (1-1/Z)/R * c^2/sqrt(4 pi eps0 G)
print(f"  general: E = (1-1/Z) (1 m / R) * {c**2/qm:.5e} V/m")
# radius at which the field drops to Schwinger for Z=10
print(f"  R needed for E = E_S at Z=10: {0.9*c**2/qm/E_schw:.4e} m ; mass then {0.9*0.9*c**2/qm/E_schw*KG_PER_M/M_sun:.4e} Msun")
