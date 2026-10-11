import numpy as np, math
from scipy.optimize import minimize_scalar, brentq
# Einstein shell: mu(R)=sqrt(1+L^2/R^2) (m0=1). Static stable equilibrium = local min of W=mu-mu^2/(2R), M=W(R0).
best=(0,None)
for L in np.geomspace(0.05,50,4000):
    Rs=np.geomspace(0.01,1e3,20000)
    mu=np.sqrt(1+L*L/Rs**2); W=mu-mu*mu/(2*Rs)
    idx=np.where((W[1:-1]<W[:-2])&(W[1:-1]<W[2:]))[0]+1
    for k in idx:
        R0=Rs[k]; M=W[k]; mu0=mu[k]
        if mu0*mu0>=2*M*R0: continue      # unphysical branch
        if 2*M>=R0: continue
        Z=1/math.sqrt(1-2*M/R0)
        v=L/math.sqrt(R0**2+L**2)
        if Z>best[0]: best=(Z,(L,R0,M,v))
print('max Z of a stable static Einstein (particle) shell: %.4f  at L=%.3f R0=%.4f M=%.4f v=%.4f c'%(best[0],*best[1]))
