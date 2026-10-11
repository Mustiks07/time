import numpy as np, math
from scipy.integrate import solve_ivp
PI=math.pi
# thin layer at R=1 of thickness eps with p_r = rho (theorem's extreme case); total mass M
def run(Y,eps,shape='sin2'):
    M=Y/2; R=1.0
    # rho profile ~ sin^2 on [R, R+eps], normalised to total mass M
    def prof(r):
        x=(r-R)/eps
        return np.where((x>0)&(x<1),np.sin(PI*x)**2,0.0)
    rr=np.linspace(R,R+eps,20001)
    norm=np.trapezoid(4*PI*rr**2*prof(rr),rr)
    A=M/norm
    def f(r,yv):
        m,Phi=yv
        rho=A*prof(r)
        pr=rho
        return [4*PI*r*r*rho,(m+4*PI*r**3*pr)/(r*(r-2*m))]
    sol=solve_ivp(f,(R,R+eps),[0,0],rtol=1e-11,atol=1e-14,max_step=eps/2000)
    m_end,dPhi=sol.y[:,-1]
    lnalpha=0.5*math.log(1-2*m_end/(R+eps))-dPhi
    # tangential pressure needed: p_t = p_r + r/2 (p_r' + (rho+p_r) Phi')
    r=sol.t; m=sol.y[0]; rho=A*prof(r); dPhi_=(m+4*PI*r**3*rho)/(r*(r-2*m))
    drho=np.gradient(rho,r)
    pt=rho+r/2*(drho+2*rho*dPhi_)
    return math.exp(lnalpha), 1-2*M/R, np.abs(pt).max(), pt.min(), rho.max()
for Y in (0.9,):
  for eps in (1e-1,1e-2,1e-3,1e-4):
    a,b,ptmax,ptmin,rhomax=run(Y,eps)
    print(f'2M/R={Y} eps={eps:7.0e}: alpha_cav={a:.6f}  bound 1-2M/R={b:.6f}  ratio ln-tightness={math.log(a)/math.log(b):.5f}  max|p_t|/max rho={ptmax/rhomax:9.3g}  min p_t={ptmin:9.3g}')
