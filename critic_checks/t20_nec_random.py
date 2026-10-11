# Random smooth profiles rho>=0, p_r = w(r) rho with |w|<=1, p_r -> 0 at both ends; compute p_t from conservation;
# keep only NEC-satisfying ones (rho+p_t>=0, rho+p_r>=0 automatically). Compare ln Z with thin-shell value -1/2 ln(1-2M/r_in)
# and with theorem value -ln(1-2M/r_in).
import numpy as np, math
from scipy.optimize import minimize
PI=math.pi
r=np.linspace(1,6,3001)
def evalp(x,nb=3):
    x=np.clip(x,-12,12)
    rho=np.zeros_like(r); 
    for k in range(nb):
        a,c,w=math.exp(x[3*k]),1+5/(1+math.exp(-x[3*k+1])),math.exp(x[3*k+2])
        rho+=a*np.exp(-0.5*((r-c)/w)**2)
    # make rho vanish smoothly at the inner edge
    rho*=(1-np.exp(-((r-1)/0.002)**2))
    wfun=np.tanh(x[3*nb]+x[3*nb+1]*(r-1)+x[3*nb+2]*(r-1)**2)
    mt=np.trapezoid(4*PI*r**2*rho,r)
    frac=0.499/(1+math.exp(-x[-1]))
    rho*=frac/mt
    m=np.concatenate(([0],np.cumsum(0.5*(4*PI*r[1:]**2*rho[1:]+4*PI*r[:-1]**2*rho[:-1])*np.diff(r))))
    if np.any(2*m>=r*0.999): return None
    pr=wfun*rho
    dPhi=(m+4*PI*r**3*pr)/(r*(r-2*m))
    dpr=np.gradient(pr,r)
    pt=pr+r/2*(dpr+(rho+pr)*dPhi)
    nec=np.min((rho+pt)/(rho.max()))
    if not np.all(np.isfinite(pt)) or m[-1]<1e-6: return None
    M=m[-1]; lnZ=np.trapezoid(dPhi,r)-0.5*math.log(1-2*M/r[-1])
    Y=2*M
    return lnZ/(-0.5*math.log(1-Y)), lnZ/(-math.log(1-Y)), nec, M
rng=np.random.default_rng(0); best=(0,None)
for t in range(120):
    x0=rng.normal(0,1,3*3+4)
    def obj(x):
        v=evalp(x)
        if v is None: return 0
        pen=max(0,-v[2])*1e3
        return -v[0]+pen
    res=minimize(obj,x0,method='Nelder-Mead',options=dict(maxiter=3000,xatol=1e-6,fatol=1e-9))
    v=evalp(res.x)
    if v and v[2]>=-1e-9 and v[0]>best[0]: best=(v[0],v)
print('max over NEC-satisfying smooth profiles of lnZ / lnZ_thin_shell(same M/r_in):',best)
