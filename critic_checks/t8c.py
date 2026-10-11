import sys, math, numpy as np
sys.dont_write_bytecode=True
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from chronos import static_spherical as ss
from scipy.optimize import minimize
x=np.load('t8b_x.npy'); K=len(x)//3
target=math.log(10.0)
def unpack(x): return np.exp(x[:K]),np.exp(x[K:2*K]),np.tanh(x[2*K:])
margin=2e-4
def run(x): return ss.simulate_controls(*unpack(x),n_rk=16)
cache={}
def c(x):
    k=x.tobytes()
    if k not in cache: cache.clear(); cache[k]=run(x)
    return cache[k]
obj=lambda x: 5.0 if c(x) is None else c(x)['M']
def ineq(x):
    r=c(x)
    if r is None: return -np.ones(1+16*K)
    rec=r['rec']; return np.concatenate(([r['lnZ']-target-1e-6],rec[:,3]*(1-margin)-np.abs(rec[:,2])))
eq=lambda x: np.array([1.0]) if c(x) is None else np.array([c(x)['g_end']])
# shrink |lam| slightly to obtain strict tangential DEC too
bounds=[(-12,1)]*K+[(-30,8)]*K+[(-3.0,3.0)]*K   # |lam|<=tanh(3)=0.99505
x0=x.copy(); x0[2*K:]=np.clip(x0[2*K:],-3,3)
res=minimize(obj,x0,method='SLSQP',bounds=bounds,constraints=[{'type':'ineq','fun':ineq},{'type':'eq','fun':eq}],options=dict(maxiter=400,ftol=1e-12))
r=run(res.x)
rec=r['rec']
print('strict: M=',r['M'],'Z=',math.exp(r['lnZ']),'g_end=',r['g_end'],'max(|g|/rho)=',(np.abs(rec[:,2])/rec[:,3]).max(),'max|lam|=',np.abs(unpack(res.x)[2]).max())
# fine check
h,rho,lam=unpack(res.x); f=ss.simulate_controls(h,rho,lam,n_rk=128)
print('fine RK128: M=',f['M'],'Z=',math.exp(f['lnZ']),'g_end=',f['g_end'],'max(|g|/rho)=',(np.abs(f['rec'][:,2])/f['rec'][:,3]).max())
rec=f['rec']
print('s, y, g(=8pi r^2 p_r), rho~  at a few points:')
for i in np.linspace(0,len(rec)-1,25).astype(int): print('%.4f %.4f %+.4f %.4f'%tuple(rec[i]))
np.save('t8c_x.npy',res.x)
