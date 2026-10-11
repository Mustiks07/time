import numpy as np, math
from scipy.optimize import minimize, brentq
# controls u_k in [-1,1] (a=c=1), phi_k = cumulative sum; constraints phi_N=0, sum sinh = 0
def check(x,N=100,seeds=2):
    T=4*x; dt=T/N; best=-1
    for sd in range(seeds):
        rng=np.random.default_rng(sd)
        u0=rng.uniform(-1,1,N)
        def phi(u): return np.concatenate(([0],np.cumsum(u)*dt))
        def mid(u): p=phi(u); return 0.5*(p[1:]+p[:-1])
        obj=lambda u: -np.sum(np.cosh(mid(u)))*dt/T
        cons=[{'type':'eq','fun':lambda u: phi(u)[-1]},{'type':'eq','fun':lambda u: np.sum(np.sinh(mid(u)))*dt}]
        r=minimize(obj,u0,method='SLSQP',bounds=[(-1,1)]*N,constraints=cons,options=dict(maxiter=3000,ftol=1e-13))
        if r.success: best=max(best,-r.fun)
    return best, math.sinh(x)/x
for x in [2.0,3.0]:
    print(x,check(x))
xs=brentq(lambda x: math.sinh(x)/x-10,1,10)
c=299792458.0;g=9.80665;yr=3.15576e7;ly=c*yr
T=4*xs*c/g
print('x=%.6f T=%.3f yr  t=%.2f yr gamma=%.3f d=%.2f ly a(T=1yr)=%.3f g  e^{4x}=%.3e  ring r=%.2f ly'%(xs,T/yr,10*T/yr,math.cosh(xs),2*c*c/g*(math.cosh(xs)-1)/ly,4*xs*c/yr/g,math.exp(4*xs),100*(1-1/100)*c*c/g/ly))
