"""Family satisfying rho>=0, |p_r|<=rho, p_t>=-rho (NEC), p_t unbounded above:
 (i) thin inner layer y:0->y1 raising p_r to g1<=y1 (needs large p_t, allowed), no lapse jump;
 (ii) thick region with rho~=y=y1 (y const), p_t=-rho (fastest NEC-allowed decay of p_r) until g=0;
 (iii) Schwarzschild.  Optimise over (y1, g1/y1, optional extra plateau length with g=g1)."""
import numpy as np, math
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
def config(y,gfrac,Lp):
    g1=gfrac*y
    P1=(y+g1)/(2*(1-y))
    # plateau with g=g1 const requires 2pt~=(rho~+g)P => pt>=0 fine (NEC ok); y const needs rho~=y
    lnZ=P1*Lp; s=Lp
    def f(s,v):
        g=v[0]; P=(y+g)/(2*(1-y)); return [-2*y-(y+g)*P, P]
    ev=lambda s,v: v[0]; ev.terminal=True; ev.direction=-1
    sol=solve_ivp(f,(0,10),[g1,0.0],events=ev,rtol=1e-11,atol=1e-13)
    s+=sol.t_events[0][0]; lnZ+=sol.y_events[0][0][1]
    lnZ+=-0.5*math.log(1-y)
    return 0.5*y*math.exp(s), math.exp(lnZ)
def solve_for_Z(Z):
    best=None
    for y in np.linspace(0.80,0.989,190):
        for gf in np.linspace(0,1,41):
            M0,Z0=config(y,gf,0.0)
            if Z0>=Z:
                if best is None or M0<best[0]: best=(M0,y,gf,0.0,Z0)
                continue
            # add plateau length to reach Z
            P1=(y+gf*y)/(2*(1-y))
            Lp=(math.log(Z)-math.log(Z0))/P1
            M1,Z1=config(y,gf,Lp)
            if best is None or M1<best[0]: best=(M1,y,gf,Lp,Z1)
    return best
for Z in [5.0,10.0,100.0]:
    b=solve_for_Z(Z)
    print('Z=%g: best M/r_in=%.5f (y=%.4f, p_r/rho at start=%.3f, plateau=%.4f, Z check=%.4f) | single shell %.5f | theorem %.5f'%(Z,b[0],b[1],b[2],b[3],b[4],(1-1/Z**2)/2,(1-1/Z)/2))
