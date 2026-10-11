import sys, math, numpy as np
sys.dont_write_bytecode=True
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from chronos.static_spherical import _fields_yg,_lnZ_yg
from scipy.optimize import minimize
def opt(Z,mode,N=30,n_starts=8,seed=0,maxiter=1500):
    rng=np.random.default_rng(seed); target=math.log(Z); n_sub=6
    def unpack(x):
        h=np.exp(x[:N]); s=np.concatenate(([0.0],np.cumsum(h)))
        y=np.concatenate(([0.0],x[N:2*N])); g=np.concatenate(([0.0],x[2*N:3*N-1],[0.0])); return s,y,g
    def obj(x):
        s,y,_=unpack(x); return 0.5*y[-1]*math.exp(s[-1])
    def ineq(x):
        s,y,g=unpack(x); F=_fields_yg(s,y,g,n_sub); _,yy,P,rho,pr,pt=F.T
        c=[_lnZ_yg(F,n_sub)-target]
        if mode=='RAD': c=np.concatenate((c,rho,rho-pr))
        if mode=='RADNEC': c=np.concatenate((c,rho,rho-pr,rho+pr,rho+pt))
        return np.concatenate((c,0.995-yy))
    best=None
    for k in range(n_starts):
        h=np.full(N,rng.uniform(0.3,1.2)/N)
        if k%2==0:   # thin-layer-like start: mass concentrated near the wall
            h=np.concatenate((np.full(N//2,0.02/(N//2)),np.full(N-N//2,rng.uniform(0.1,1.0)/(N-N//2))))
        y=np.clip(np.linspace(0.1,0.9,N)*rng.uniform(0.6,1.0),0,0.99)
        g=rng.uniform(-0.2,0.5,N-1)
        x0=np.concatenate((np.log(h),y,g))
        bounds=[(-14.0,1.0)]*N+[(0.0,0.995)]*N+[(-20,20)]*(N-1)
        try:
            res=minimize(obj,x0,method='SLSQP',bounds=bounds,constraints=[{'type':'ineq','fun':ineq}],options=dict(maxiter=maxiter,ftol=1e-12))
        except Exception as e:
            continue
        if np.all(ineq(res.x)>-1e-7) and (best is None or res.fun<best.fun): best=res
    if best is None: return None
    s,y,g=unpack(best.x); F=_fields_yg(s,y,g,n_sub)
    return best.fun, math.exp(_lnZ_yg(F,n_sub)), y.max()
for mode in ['RAD','RADNEC']:
    r=opt(10.0,mode)
    print(mode,'Z=10: min M/r_in =',r,' (theorem bound 0.45)',flush=True)
