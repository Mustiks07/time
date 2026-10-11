"""min M/r_in at Z=10 under: rho>=0, -rho<=p_r<=rho (theorem hyp. + radial NEC), p_t>=-rho (tangential NEC),
p_t unbounded above.  Controls per segment: rho~>0, lambda=p_t/rho in [-1, inf)."""
import sys, math, numpy as np
sys.dont_write_bytecode=True
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from chronos.static_spherical import simulate_controls
from scipy.optimize import minimize
target=math.log(10.0)
def unpack(x,K):
    return np.exp(x[:K]),np.exp(x[K:2*K]),-1+np.exp(x[2*K:3*K])
def solve(x0,K,maxiter=500):
    cache={}
    def c(x):
        k=x.tobytes()
        if k not in cache: cache.clear(); cache[k]=simulate_controls(*unpack(x,K))
        return cache[k]
    obj=lambda x: 5.0 if c(x) is None else c(x)['M']
    def ineq(x):
        r=c(x)
        if r is None: return -np.ones(1+8*K)
        rec=r['rec']; return np.concatenate(([r['lnZ']-target],rec[:,3]-np.abs(rec[:,2])))
    eq=lambda x: np.array([1.0]) if c(x) is None else np.array([c(x)['g_end']])
    bounds=[(-12,1)]*K+[(-30,9)]*K+[(-8,5)]*K
    res=minimize(obj,x0,method='SLSQP',bounds=bounds,constraints=[{'type':'ineq','fun':ineq},{'type':'eq','fun':eq}],options=dict(maxiter=maxiter,ftol=1e-12))
    r=simulate_controls(*unpack(res.x,K),n_rk=32)
    rec=r['rec']
    return r['M'],math.exp(r['lnZ']),r['g_end'],(np.abs(rec[:,2])-rec[:,3]).max(),res.x
rng=np.random.default_rng(0)
best=None
for trial in range(6):
    K=24
    # warm start: thin dense layer (Israel-like shell) + a thicker tail
    nl=10
    h=np.concatenate((np.full(nl,rng.uniform(0.002,0.02)/nl),np.full(K-nl,rng.uniform(0.05,1.0)/(K-nl))))
    rho=np.concatenate((np.full(nl,rng.uniform(0.9,1.0)/h[0]/nl),np.full(K-nl,rng.uniform(0.01,0.5))))
    lam=np.concatenate((np.full(nl,rng.uniform(1.5,4.0)),np.full(K-nl,rng.uniform(0.0,2.0))))
    x0=np.concatenate((np.log(h),np.log(rho),np.log(lam+1)))
    try:
        M,Z,gend,viol,x=solve(x0,K)
    except Exception as e:
        print('fail',e); continue
    print('trial',trial,'M/r_in=%.5f Z=%.5f g_end=%.1e max(|g|-rho)=%.1e'%(M,Z,gend,viol),flush=True)
    if abs(gend)<1e-5 and viol<1e-5 and Z>9.999 and (best is None or M<best): best=M
print('best feasible RAD+NEC M/r_in at Z=10:',best,'  (theorem 0.45, single thin shell 0.495)')
