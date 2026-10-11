import sys, math, numpy as np
sys.dont_write_bytecode=True
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from chronos import static_spherical as ss
x0,K=ss.warm_controls_shell_cluster(10.0)
r0=ss.simulate_controls(*[np.exp(x0[:K]),np.exp(x0[K:2*K]),np.tanh(x0[2*K:])])
print('warm start: M=',r0['M'],'Z=',math.exp(r0['lnZ']),'g_end=',r0['g_end'],flush=True)
res=ss.optimise_controls_dec(10.0,x0,K,maxiter=300)
print('controls DEC optimum: M/r_in=',res['M_over_rin'],'Z=',res['Z'],'feasible=',res['feasible'],flush=True)
w=ss.warm_start_shell_plus_cluster(10.0,N=20)
r2=ss.optimise_continuum_yg(10.0,N=20,n_starts=4,seed=0,maxiter=800,warm=w,mode='DEC')
print('yg DEC optimum:',None if r2 is None else (r2['M_over_rin'],r2['Z']),flush=True)
