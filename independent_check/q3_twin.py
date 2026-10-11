"""Q3: maximal base-time / traveller-time ratio with |proper acceleration| <= a, flat spacetime.

Rapidity eta(tau) (c=1): dt/dtau = cosh eta, dx/dtau = sinh eta, |deta/dtau| = |accel| <= a.
Problem: maximise t = int_0^T cosh(eta) dtau subject to
  eta(0)=0 (leaves inertial base), eta(T)=0 (stops), int_0^T sinh(eta) dtau = 0 (returns).
Since cosh is even and increasing in |eta|, we want |eta| as large as possible pointwise. Return
requires eta to change sign at some tau0; |eta(tau)| <= a*min(tau, |tau-tau0|, T-tau).
Symmetric choice tau0=T/2 saturates this bound on both lobes with int sinh = 0 automatically:
  eta = a tau (0..T/4), a(T/2-tau) (T/4..3T/4), a(tau-T) (3T/4..T)
  => t_base = 4 * int_0^{T/4} cosh(a tau) dtau = (4/a) sinh(aT/4).
Ratio  R(x) = sinh(x/4) / (x/4),  x = a T / c.
(Optimality vs. asymmetric tau0 and arbitrary profiles is verified numerically below.)
"""
import numpy as np
from scipy.optimize import minimize, brentq
from consts import *

def ratio(x):
    y = x / 4.0
    return np.sinh(y) / y if y != 0 else 1.0

# ---------- numerical optimal control -------------------------------------
def optimise(x, N=160, nstart=12, seed=0):
    """Discretise: controls u_k in [-1,1] (accel in units of a), T=1, a=x.
    eta_k at nodes, eta_0=0; trapezoid quadrature for cosh/sinh integrals."""
    a = x
    h = 1.0 / N
    rng = np.random.default_rng(seed)

    def eta_of(u):
        return np.concatenate([[0.0], np.cumsum(u) * a * h])

    w = np.full(N + 1, h); w[0] = w[-1] = h / 2

    def obj(u):
        e = eta_of(u)
        return -np.dot(w, np.cosh(e))

    def obj_grad(u):
        e = eta_of(u)
        g_eta = -w * np.sinh(e)
        # d eta_j / d u_k = a h for j>k
        return a * h * np.cumsum(g_eta[::-1])[::-1][1:]

    def c_ret(u):
        return np.dot(w, np.sinh(eta_of(u)))

    def c_ret_grad(u):
        e = eta_of(u)
        g = w * np.cosh(e)
        return a * h * np.cumsum(g[::-1])[::-1][1:]

    cons = [{'type': 'eq', 'fun': lambda u: np.sum(u), 'jac': lambda u: np.ones_like(u)},
            {'type': 'eq', 'fun': c_ret, 'jac': c_ret_grad}]
    best = None
    for s in range(nstart):
        if s == 0:
            u0 = np.zeros(N)
        else:
            u0 = rng.uniform(-1, 1, N)
            u0 -= u0.mean()
        res = minimize(obj, u0, jac=obj_grad, bounds=[(-1, 1)] * N, constraints=cons,
                       method='SLSQP', options={'maxiter': 2000, 'ftol': 1e-14})
        if res.success and abs(c_ret(res.x)) < 1e-8 and abs(np.sum(res.x)) < 1e-8:
            if best is None or res.fun < best.fun:
                best = res
    return -best.fun, best.x

if __name__ == "__main__":
    print("Analytic R(x) = sinh(x/4)/(x/4)")
    print(f"{'x':>6} {'analytic':>14} {'numerical':>14} {'rel.diff':>10}")
    for x in [1.0, 4.0, 8.0, 12.0, 18.0]:
        num, u = optimise(x)
        an = ratio(x)
        print(f"{x:6.1f} {an:14.8f} {num:14.8f} {(num-an)/an:10.2e}")
    # show the structure of the optimum for x=8
    num, u = optimise(8.0, N=80)
    print("optimal control signs (x=8, N=80):", ''.join('+' if v > 0.5 else ('-' if v < -0.5 else '0') for v in u))

    # asymmetric switch time check: one sign change at tau0 != T/2 cannot beat symmetric
    # (the larger lobe must be shrunk to balance int sinh), checked by the optimiser above.

    # (i), (ii)
    x10 = brentq(lambda x: ratio(x) - 10.0, 1e-6, 100)
    print(f"\nratio=10 at x = aT/c = {x10:.10f}   (x/4 = {x10/4:.10f}, solves sinh(y)=10y)")
    T = x10 * c / g0
    print(f"(i) a=1 g0: T = {T:.6e} s = {T/year:.6f} yr ; base time = {10*T/year:.5f} yr")
    a = x10 * c / year
    print(f"(ii) T=1 yr: a = {a:.6f} m/s^2 = {a/g0:.6f} g0")
    # each leg: peak rapidity and gamma, max distance
    y = x10 / 4
    print(f"    peak rapidity = {y:.6f}, peak gamma = {np.cosh(y):.4f}, peak speed = 1-{1-np.tanh(y):.3e} c")
    # max distance from base (reached at tau=T/2): 2*(c^2/a)(cosh(y)-1)
    print(f"    max distance (a=1g0 case) = {2*(c**2/g0)*(np.cosh(y)-1)/9.4607e15:.4f} ly")
