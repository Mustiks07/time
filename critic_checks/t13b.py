import sys, math, numpy as np
sys.dont_write_bytecode=True
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from chronos import static_spherical as ss
from chronos.static_spherical import _fields_yg,_lnZ_yg
from scipy.optimize import minimize
N=30; n_sub=6; target=math.log(10.0)
def unpack(x):
    h=np.exp(x[:N]); s=np.concatenate(([0.0],np.cumsum(h)))
    y=np.concatenate(([0.0],x[N:2*N])); g=np.concatenate(([0.0],x[2*N:3*N-1],[0.0])); return s,y,g
def obj(x):
    s,y,_=unpack(x); return 0.5*y[-1]*math.exp(s[-1])
def mk(mode):
    def ineq(x):
        s,y,g=unpack(x); F=_fields_yg(s,y,g,n_sub); _,yy,P,rho,pr,pt=F.T
        c=[_lnZ_yg(F,n_sub)-target]
        if mode=='RADNEC': c=np.concatenate((c,rho,rho-pr,rho+pr,rho+pt))
        if mode=='DEC': c=np.concatenate((c,rho-pr,rho+pr,rho-pt,rho+pt))
        return np.concatenate((c,0.995-yy))
    return ineq
w=ss.warm_start_shell_plus_cluster(10.0,N=N)
s,y,g=unpack(w); F=_fields_yg(s,y,g,n_sub)
print('warm start M=',obj(w),'Z=',math.exp(_lnZ_yg(F,n_sub)),' min DEC slack=',mk('DEC')(w)[1:].min(),flush=True)
for mode in ['DEC','RADNEC']:
    bounds=[(-14.0,1.0)]*N+[(0.0,0.995)]*N+[(-20,20)]*(N-1)
    res=minimize(obj,w,method='SLSQP',bounds=bounds,constraints=[{'type':'ineq','fun':mk(mode)}],options=dict(maxiter=2000,ftol=1e-12))
    print(mode,'M=',res.fun,'feasible:',mk(mode)(res.x).min(),flush=True)
