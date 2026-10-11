"""Oscillating thin shell, flat interior (habitat), Schwarzschild exterior, general elastic EoS
lambda(R) = p/sigma with |lambda| <= 1 (DEC).  m(R) = 4 pi R^2 sigma obeys dm/dR = -2 lambda m / R.
EoM: sqrt(1+Rdot^2) = A(R) = M/m + m/(2R).  Habitat time T, exterior time t:
   dT/dR = A/sqrt(A^2-1),  dt/dR = (A - m/R)/(f sqrt(A^2-1)),  f = 1-2M/R.
<Z> = oint dt / oint dT.  Units R_min = 1; A(1) = 1 fixes m_min = 1-sqrt(f_min), M=(1-f_min)/2."""
import numpy as np, math, sys
from scipy.optimize import minimize

NODES = 8
RCAP = 40.0
Rg = np.geomspace(1, RCAP, 6001)
lnRg = np.log(Rg)
th = (np.arange(3000) + 0.5) / 3000 * math.pi


def evaluate(params):
    fmin = 0.0005 + 0.999 / (1 + math.exp(-params[0]))
    lamn = np.tanh(params[1:])
    M = (1 - fmin) / 2
    mmin = 1 - math.sqrt(fmin)
    lam = np.interp(lnRg, np.linspace(0, math.log(RCAP), NODES), lamn)
    lnm = math.log(mmin) - 2 * np.concatenate(([0], np.cumsum(0.5 * (lam[1:] + lam[:-1]) * np.diff(lnRg))))
    m = np.exp(lnm)
    A = M / m + m / (2 * Rg)
    ok = A > 1 + 1e-13
    if not ok[1:4].all():
        return None
    bad = np.where(~ok[1:])[0]
    if len(bad) == 0:
        return None
    j = bad[0] + 1
    R0, R1 = Rg[j - 1], Rg[j]
    a0, a1 = A[j - 1] - 1, A[j] - 1
    Rmax = R0 + (R1 - R0) * a0 / (a0 - a1)
    if Rmax < 1 + 1e-4:
        return None
    R = 1 + (Rmax - 1) * (1 - np.cos(th)) / 2
    dR = (Rmax - 1) * np.sin(th) / 2
    mm = np.exp(np.interp(np.log(R), lnRg, lnm))
    AA = M / mm + mm / (2 * R)
    if np.any(AA - mm / R <= 0):          # physical branch
        return None
    s = np.sqrt(np.maximum(AA * AA - 1, 1e-30))
    f = 1 - 2 * M / R
    T = np.sum(AA / s * dR)
    t = np.sum((AA - mm / R) / (f * s) * dR)
    return t / T, 1 / math.sqrt(fmin), Rmax, fmin, lamn


rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
bestZ = None
bestr = None
for trial in range(150):
    x0 = np.concatenate(([rng.normal(-1, 2)], rng.normal(0, 1.5, NODES)))
    for key in ('Z', 'ratio'):
        def obj(x):
            r = evaluate(x)
            if r is None:
                return 0.0
            return -(r[0] if key == 'Z' else r[0] / r[1])
        res = minimize(obj, x0, method='Nelder-Mead', options=dict(maxiter=1500, xatol=1e-6, fatol=1e-9))
        r = evaluate(res.x)
        if r is None:
            continue
        if key == 'Z' and (bestZ is None or r[0] > bestZ[0]):
            bestZ = r
        if key == 'ratio' and (bestr is None or r[0] / r[1] > bestr[0] / bestr[1]):
            bestr = r
    if trial % 25 == 24:
        print(trial, 'max<Z>=', bestZ and bestZ[0], 'max ratio=', bestr and bestr[0] / bestr[1], flush=True)
print('max <Z> under DEC:', bestZ[0], ' Z(Rmin)=', bestZ[1], ' Rmax/Rmin=', bestZ[2], ' lam=', np.round(bestZ[4], 3))
print('max <Z>/Z(Rmin):', bestr[0] / bestr[1], ' (<Z>=', bestr[0], ' Z(Rmin)=', bestr[1], ' Rmax=', bestr[2], ')')
print('lambda nodes of max-ratio case:', np.round(bestr[4], 3))
