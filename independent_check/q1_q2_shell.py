"""Q1 + Q2: static thin shell, flat interior, Schwarzschild exterior.

Derivation (Israel junction, G=c=1).
Metric on each side: ds^2 = -f dt^2 + dr^2/f + r^2 dOmega^2,
  f_in = 1, f_out = 1 - 2M/r.
For a shell r=R(tau) with unit normal pointing outward, extrinsic curvature
  K^theta_theta = sqrt(f + Rdot^2)/R,
  K^tau_tau     = (Rddot + f'/2)/sqrt(f + Rdot^2).
Lanczos: S^i_j = -(1/8pi)([K^i_j] - delta^i_j [K]),  S^tau_tau = -sigma, S^theta_theta = p
  => sigma = -(1/4piR) [sqrt(f+Rdot^2)],
     p     =  (1/8pi) [ (Rddot + f'/2)/sqrt(f+Rdot^2) + sqrt(f+Rdot^2)/R ].
Static (Rdot=Rddot=0):
  sigma = (1 - sqrt(1-2M/R)) / (4 pi R)
  p     = (1/8piR) [ (1-M/R)/sqrt(1-2M/R) - 1 ]
Interior clocks: shell proper time dtau = sqrt(f_out) dt_out = dt_in (interior is flat with
lapse 1), so interior static clocks tick at alpha = sqrt(1-2M/R) relative to infinity.
With M/R = (1-alpha^2)/2:
  sigma = (1-alpha)/(4 pi R),  p = (1-alpha)^2 / (16 pi R alpha),  p/sigma = (1-alpha)/(4 alpha).
DEC |p|<=sigma  <=>  alpha >= 1/5  <=>  Z <= 5  (M/R <= 12/25, R >= 25M/12).
"""
import sympy as sp
import numpy as np
from consts import *

# ---- symbolic check -------------------------------------------------------
M, R, a, Rd, Rdd = sp.symbols('M R alpha Rdot Rddot', positive=True)
r = sp.symbols('r', positive=True)

def KK(f):
    fp = sp.diff(f, r)
    Kth = sp.sqrt(f + Rd**2) / r
    Ktau = (Rdd + fp / 2) / sp.sqrt(f + Rd**2)
    return Ktau.subs(r, R), Kth.subs(r, R)

Kt_out, Kth_out = KK(1 - 2 * M / r)
Kt_in, Kth_in = KK(sp.Integer(1) + 0 * r)
jt, jth = Kt_out - Kt_in, Kth_out - Kth_in
trK = jt + 2 * jth
S_tt = -(jt - trK) / (8 * sp.pi)        # = -sigma
S_thth = -(jth - trK) / (8 * sp.pi)     # = p
sigma = sp.simplify((-S_tt).subs({Rd: 0, Rdd: 0}))
p = sp.simplify(S_thth.subs({Rd: 0, Rdd: 0}))
print("sigma(M,R) =", sigma)
print("p(M,R)     =", p)
sub = {M: R * (1 - a**2) / 2}
sig_a = sp.simplify(sigma.subs(sub))
p_a = sp.simplify(sp.factor(sp.radsimp(p.subs(sub).subs(sp.sqrt(1 - (1 - a**2)), a))))
# careful substitution sqrt(1-2M/R)=alpha:
p_a = sp.simplify(((1 / (8 * sp.pi * R)) * ((1 - (1 - a**2) / 2) / a - 1)))
sig_a = sp.simplify((1 - a) / (4 * sp.pi * R))
# numeric verification that these agree with the general formula
for MM, RR in [(0.3, 1.0), (0.45, 1.0), (0.1, 2.3)]:
    al = np.sqrt(1 - 2 * MM / RR)
    assert abs(float(sigma.subs({M: MM, R: RR})) - float(sig_a.subs({a: al, R: RR}))) < 1e-14
    assert abs(float(p.subs({M: MM, R: RR})) - float(p_a.subs({a: al, R: RR}))) < 1e-14
print("sigma(alpha) =", sp.factor(sig_a), "   p(alpha) =", sp.factor(p_a))
ratio = sp.simplify(p_a / sig_a)
print("p/sigma =", sp.factor(ratio))
amin = sp.solve(sp.Eq(ratio, 1), a)
print("DEC p=sigma at alpha =", amin, " => Z_max =", [1 / x for x in amin])
print("p/sigma at Z=10:", ratio.subs(a, sp.Rational(1, 10)), "=", float(ratio.subs(a, sp.Rational(1, 10))))
print("M/R at Z_max:", (1 - sp.Rational(1, 25)) / 2)

# ---- Q2 table -------------------------------------------------------------
print("\nQ2 table: M = (R c^2/2G)(1 - 1/Z^2); p_SI = p_geom * c^4/G ; F = 2 pi R p = (1-alpha)^2/(8 alpha) c^4/G")
rows = []
for Z in [2, 10, 100, 1000]:
    al = 1.0 / Z
    for RR in [1, 10, 100, 1000]:
        Mgeom = RR * (1 - al**2) / 2
        Mkg = Mgeom * KG_PER_M
        p_geom = (1 - al)**2 / (16 * np.pi * RR * al)   # 1/m
        p_SI = p_geom * c**4 / G                        # N/m
        sig_SI = (1 - al) / (4 * np.pi * RR) * c**4 / G  # J/m^2
        F = 2 * np.pi * RR * p_geom                     # in units of c^4/G
        rows.append((Z, RR, Mkg, Mkg / M_J, Mkg / M_sun, p_SI, sig_SI, F))
        print(f"Z={Z:5d} R={RR:5d} m  M={Mkg:.4e} kg = {Mkg/M_J:.4e} M_J = {Mkg/M_sun:.4e} Msun  "
              f"p={p_SI:.4e} N/m  sigma={sig_SI:.4e} J/m^2  F={F:.6g} c^4/G = {F*c**4/G:.4e} N")
np.save('/home/user/time/independent_check/q2_rows.npy', np.array(rows))
