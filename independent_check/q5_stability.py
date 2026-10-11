"""Q5: radial stability of the static thin shell (flat interior, Schwarzschild exterior).

Dynamic junction (G=c=1): with m(R) = 4 pi R^2 sigma(R) ("rest mass"),
   sqrt(1 + Rdot^2) - sqrt(1 - 2M/R + Rdot^2) = m/R.
Call A = sqrt(1+Rdot^2), B = sqrt(1-2M/R+Rdot^2): A - B = m/R, A^2 - B^2 = 2M/R
   => A + B = 2M/m => A = M/m + m/(2R)
   => Rdot^2 + V(R) = 0,  V(R) = 1 - (M/m(R) + m(R)/(2R))^2.
Energy conservation on the shell (from the tau-tau/theta-theta junction eqs, or the
conservation identity S^{ij}_{|j}=0 since the bulk is vacuum on both sides):
   d(sigma * 4piR^2) + p d(4piR^2) = 0  =>  sigma' = -2(sigma+p)/R,  m' = -8 pi R p,
   m'' = -8 pi p - 8 pi R beta^2 sigma' = -8 pi p + 16 pi beta^2 (sigma + p).
At the equilibrium R0 (static solution of Q1): V=0, V'=0 automatically; stability <=> V''(R0) > 0.
Writing G = M/m + m/(2R): V = 1 - G^2, at equilibrium G=1, G'=0  =>  V'' = -2 G''.
"""
import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

M, R, m, m1, m2, sig, p, b2, al = sp.symbols('M R m m1 m2 sigma p beta2 alpha', positive=True)
Rs = sp.symbols('Rs', positive=True)
mf = sp.Function('m')(Rs)
Gfun = M / mf + mf / (2 * Rs)
V = 1 - Gfun**2
V1 = sp.diff(V, Rs)
V2 = sp.diff(V, Rs, 2)

def at_eq(expr):
    # substitute m'' , m', m with equilibrium values in terms of alpha (R kept symbolic)
    expr = expr.subs(sp.Derivative(mf, (Rs, 2)), m2).subs(sp.Derivative(mf, Rs), m1).subs(mf, m).subs(Rs, R)
    sig0 = (1 - al) / (4 * sp.pi * R)
    p0 = (1 - al)**2 / (16 * sp.pi * R * al)
    sub = {m2: -8 * sp.pi * p0 + 16 * sp.pi * b2 * (sig0 + p0),
           m1: -8 * sp.pi * R * p0,
           m: R * (1 - al),
           M: R * (1 - al**2) / 2}
    return sp.simplify(sp.factor(expr.subs(sub)))

V0, V1e, V2e = at_eq(V), at_eq(V1), at_eq(V2)
print("V(R0) =", V0, "   V'(R0) =", V1e)
V2e = sp.factor(sp.simplify(V2e))
print("V''(R0) =", V2e)
b2crit = sp.solve(sp.Eq(V2e, 0), b2)
assert len(b2crit) == 1
b2c = sp.factor(sp.simplify(b2crit[0]))
print("beta^2_min(alpha) =", b2c)
Z = sp.symbols('Z', positive=True)
b2cZ = sp.factor(sp.simplify(b2c.subs(al, 1 / Z)))
print("beta^2_min(Z)     =", b2cZ)
# sign of dV''/dbeta^2
print("dV''/dbeta^2 =", sp.factor(sp.diff(V2e, b2)), "(>0 => stable for beta^2 > beta^2_min)")
print("Newtonian limit (Z->1, M/R->0):", sp.limit(b2cZ, Z, 1))
for Zv in [sp.Rational(3, 2), 2, 3, 5, 10]:
    print(f"   Z={str(Zv):4s}: beta^2_min = {sp.nsimplify(b2cZ.subs(Z, Zv))} = {float(b2cZ.subs(Z, Zv)):.6f}")
poly = sp.expand(sp.numer(sp.together(b2cZ - 1)))
print("beta^2_min = 1  <=>  ", poly, "= 0")
Zone = [sp.re(z) for z in sp.Poly(poly, Z).nroots(n=30) if abs(sp.im(z)) < 1e-25 and sp.re(z) > 1]
print("beta^2_min = 1 at Z =", [sp.N(z, 12) for z in Zone])
print("  corresponding alpha =", [sp.N(1 / z, 12) for z in Zone], " M/R =", [sp.N((1 - 1 / z**2) / 2, 12) for z in Zone])
print("  at DEC limit Z=5: beta^2_min =", b2cZ.subs(Z, 5))
print("asymptotic Z->inf: beta^2_min ~", sp.limit(b2cZ / Z**2, Z, sp.oo), "* Z^2")

# ---------------- independent numerical check: integrate sigma(R) for a linear EOS and
# finite-difference V(R) around R0 (R0=1, M from Z).
def numeric_V2(Zv, beta2, h=1e-3):
    a = 1.0 / Zv
    R0 = 1.0
    Mv = R0 * (1 - a * a) / 2
    s0 = (1 - a) / (4 * np.pi * R0)
    p0 = (1 - a)**2 / (16 * np.pi * R0 * a)
    eos = lambda s: p0 + beta2 * (s - s0)
    rhs = lambda Rr, y: [-2 * (y[0] + eos(y[0])) / Rr]
    def Vof(Rr):
        if Rr == R0:
            s = s0
        else:
            s = solve_ivp(rhs, [R0, Rr], [s0], rtol=1e-13, atol=1e-16).y[0, -1]
        mm = 4 * np.pi * Rr**2 * s
        return 1 - (Mv / mm + mm / (2 * Rr))**2
    return (Vof(R0 + h) - 2 * Vof(R0) + Vof(R0 - h)) / h**2, Vof(R0), (Vof(R0 + h) - Vof(R0 - h)) / (2 * h)

V2f = sp.lambdify((al, b2, R), V2e)
print("\nnumerical check (R0=1): finite-difference V'' vs formula")
for Zv, bb in [(1.5, 0.1), (2, 0.5), (3, 0.9), (10, 2.0), (10, 3.0)]:
    num, v0, v1 = numeric_V2(Zv, bb)
    print(f"   Z={Zv}, beta2={bb}: FD V''={num:.6f}, formula={V2f(1/Zv, bb, 1.0):.6f}, V(R0)={v0:.1e}, V'(R0)={v1:.1e}")
for Zv in [1.5, 2, 3, 5, 10]:
    root = brentq(lambda bb: numeric_V2(Zv, bb, h=1e-4)[0], -5, 50)
    print(f"   Z={Zv}: numerical beta^2_min = {root:.5f}  (formula {float(b2cZ.subs(Z, Zv)):.5f})")
