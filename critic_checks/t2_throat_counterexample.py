"""Counterexample to 'no horizons => bound': static, spherically symmetric, asymptotically flat
(exact Schwarzschild for r>=r2), regular centre, flat vacuum cavity, rho>=0 and p_r<=rho everywhere,
lapse finite and nonzero everywhere (no horizon), but areal radius r(l) is NOT monotone
(maximal sphere + minimal sphere/throat, so 2m=r there).  alpha_cav << 1-2M/r_in."""
import numpy as np, math
PI=math.pi
def smooth(x):  # C^2 smoothstep 0->1 on [0,1]
    x=np.clip(x,0,1); return x**3*(10-15*x+6*x*x)
def dsmooth(x):
    m=(x>0)&(x<1); x=np.clip(x,0,1); return m*30*x*x*(1-x)**2

r_in=1.0; D=3.0; hmin=-0.9; K=None
# ---- inner part on l in [0, 1+D]: r_l = h(l)
N=400001
l=np.linspace(0,r_in+D,N)
x=(l-r_in)/D
h=np.where(l<r_in,1.0, 1.0+(hmin-1.0)*smooth(x))
hp=np.where(l<r_in,0.0,(hmin-1.0)*dsmooth(x)/D)
r=np.concatenate(([0.0],np.cumsum(0.5*(h[1:]+h[:-1])*np.diff(l))))
r2=r[-1]; M=r2*(1-hmin**2)/2
print(f'r_in={r_in}, max r (equator)={r.max():.4f}, r at end of matter layer r2={r2:.4f}, ADM M={M:.5f}, 2M/r_in={2*M/r_in:.4f}')
print('Theorem-2 bound: alpha_cav >= 1-2M/r_in =',1-2*M/r_in)
# lapse: Phi_l = 0 where h>0 ; Phi_l = Kbump >=0 where h<0 (inside the layer)
neg=h<0
bump=np.where(neg, np.sin(PI*np.clip((l-l[neg].min())/(l[neg].max()-l[neg].min()),0,1))**2,0.0)
# ---- outer part: left Schwarzschild sheet r from r2 down to 2M (r_l=-sqrt(1-2M/r)), throat, right sheet out to r_big
# On the Schwarzschild sheets we keep Phi_l=0 except on the right sheet where we ramp up to exact Schwarzschild lapse.
rS=2*M; r_ramp0=2.5*M; r_ramp1=6*M
# Phi on right sheet as function of r: Phi_r = s(r)*M/(r^2(1-2M/r)), Phi -> 1/2 ln(1-2M/r) for r>=r_ramp1
rr=np.linspace(rS*(1+1e-9),r_ramp1,200001)
s=smooth((rr-r_ramp0)/(r_ramp1-r_ramp0))
PhiS_r=M/(rr**2*(1-2*M/rr))
Phi_r=s*PhiS_r
# integrate inward from r_ramp1 where Phi = 1/2 ln(1-2M/r)
Phi_right=0.5*math.log(1-2*M/r_ramp1)-np.concatenate((np.cumsum((0.5*(Phi_r[1:]+Phi_r[:-1])*np.diff(rr))[::-1])[::-1],[0.0]))
Phi_throat=Phi_right[0]
print(f'Phi at throat = {Phi_throat:.5f}  (finite: no horizon); lapse at throat = {math.exp(Phi_throat):.4f}')
# choose K so that alpha_cav = 0.01  (Phi at end of layer = Phi_throat since Phi const on left sheet)
I=np.sum(0.5*(bump[1:]+bump[:-1])*np.diff(l))
target=math.log(0.01)
K=(Phi_throat-target)/I
Phil=K*bump
Phi=Phi_throat-(np.sum(0.5*(Phil[1:]+Phil[:-1])*np.diff(l))-np.concatenate(([0.0],np.cumsum(0.5*(Phil[1:]+Phil[:-1])*np.diff(l)))))
alpha_cav=math.exp(Phi[0])
print(f'K={K:.3f}; alpha_cav = {alpha_cav:.5f}   vs bound {1-2*M/r_in:.4f}  ->  VIOLATED by factor {(1-2*M/r_in)/alpha_cav:.1f}')
# ---- check conditions in the inner part (l>r_in)
m=l>r_in*1.0000001
rho=((1-h**2)/r**2-2*hp/r)/(8*PI)
pr=(-(1-h**2)/r**2+2*h*Phil/r)/(8*PI)
print('inner layer: min rho =',rho[m].min(),' max(p_r-rho) =',(pr-rho)[m].max(),
      ' min(rho+p_r) =',(rho+pr)[m].min(),' (NEC violated where <0)')
print('inner layer: min 2m/r =',(1-h[m]**2).min(), ' max 2m/r =',(1-h[m]**2).max())
# ---- checks on the Schwarzschild sheets (rho=0 exactly; p_r from formula)
# left sheet: h=-sqrt(1-2M/r), Phi_l=0 -> 8pi p_r = -2M/r^3 <= 0 = rho
rl=np.linspace(rS*(1+1e-9),r2,10001)
print('left sheet: max p_r =',(-2*M/rl**3/(8*PI)).max(),'(<=0=rho)')
# right sheet: h=+sqrt(1-2M/r), Phi_l = h*Phi_r -> 8pi p_r = -2M/r^3 + 2 h^2 Phi_r / r
pr_right=(-2*M/rr**3+2*(1-2*M/rr)*Phi_r/rr)/(8*PI)
print('right sheet: max p_r =',pr_right.max(),' (<= 0 = rho; exactly 0 beyond r=%.3f: pure vacuum Schwarzschild)'%r_ramp1)
print('throat: 8pi(rho+p_r) =',-1/rS**2,' (NEC violated at throat, as it must be)')
