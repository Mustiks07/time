# Check alpha(r) >= (1-2M/r)/sqrt(1-2m(r)/r) for random monotone-r profiles with rho>=0, p_r<=rho
import numpy as np, math
PI=math.pi
rng=np.random.default_rng(5); worst=0
for trial in range(3000):
    r=np.linspace(1,1+rng.uniform(0.1,15),3000)
    rho=np.zeros_like(r)
    for k in range(rng.integers(1,5)):
        rho+=rng.uniform(0,1)*np.exp(-0.5*((r-rng.uniform(r[0],r[-1]))/rng.uniform(0.01,2))**2)
    mt=np.trapezoid(4*PI*r**2*rho,r); rho*=rng.uniform(0.05,0.999)*0.49*r[-1]/mt
    m=np.concatenate(([0],np.cumsum(0.5*(4*PI*r[1:]**2*rho[1:]+4*PI*r[:-1]**2*rho[:-1])*np.diff(r))))
    if np.any(2*m>=r): continue
    pr=rho*rng.uniform(-3,1)
    dPhi=(m+4*PI*r**3*pr)/(r*(r-2*m))
    M=m[-1]; Phi_end=0.5*math.log(1-2*M/r[-1])
    Phi=Phi_end-(np.trapezoid(dPhi,r)-np.concatenate(([0],np.cumsum(0.5*(dPhi[1:]+dPhi[:-1])*np.diff(r)))))
    bound=np.log(np.maximum(1-2*M/r,1e-300))-0.5*np.log(1-2*m/r)
    ok=(1-2*M/r)>0
    worst=max(worst,np.max((bound-Phi)[ok]))
print('max over profiles/radii of [ln bound - ln alpha] (should be <= ~0):',worst)
