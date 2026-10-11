import sys, math, numpy as np, time
sys.dont_write_bytecode=True
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from chronos import static_spherical as ss
t0=time.time()
x0,K=ss.warm_controls_shell_cluster(10.0)
res=ss.optimise_controls_dec(10.0,x0,K,maxiter=600)
sim=res['sim']
rec=sim['rec']
viol=(np.abs(rec[:,2])-rec[:,3]).max()
print('M=',res['M_over_rin'],'Z=',res['Z'],'g_end=',sim['g_end'],'max(|g|-rho)=',viol,'feasible',res['feasible'],'time',time.time()-t0,flush=True)
np.save('t8b_x.npy',res['x'])
# polish: re-simulate with finer RK to check discretisation
h,rho,lam=res['h'],res['rho'],res['lam']
fine=ss.simulate_controls(h,rho,lam,n_rk=64)
print('fine RK: M=',fine['M'],'Z=',math.exp(fine['lnZ']),'g_end=',fine['g_end'],'max(|g|-rho)=',(np.abs(fine['rec'][:,2])-fine['rec'][:,3]).max())
print('segments (h, rho~, lam):')
for a,b,c in zip(h,rho,lam): print('%.4g %.4g %.4f'%(a,b,c))
