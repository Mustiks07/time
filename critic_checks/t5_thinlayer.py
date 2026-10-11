import numpy as np, math
from scipy.integrate import solve_ivp
# full (not thin-limit) equations in s=ln(r/r_in): y_s = rho~ - y ; g_s = 2 pt~ - (rho~+g)P ; P=(y+g)/(2(1-y))
# layer with rho~ = c (constant, large), pt~ = lam*rho~ ; stop when g returns to 0.
def layer(c,lam=1.0):
    def f(s,v):
        y,g,lnZ=v; P=(y+g)/(2*(1-y)); return [c-y, 2*lam*c-(c+g)*P, P]
    def ev(s,v): return v[1]
    ev.terminal=True; ev.direction=-1
    sol=solve_ivp(f,(0,50/c+10),[0,0,0],events=ev,rtol=1e-12,atol=1e-14,max_step=0.01/c)
    y,g,_=sol.y_events[0][0]
    gmax=sol.y[1].max()
    return y, sol.t_events[0][0], gmax/c
for c in [10,100,1000,10000]:
    y,s,gr=layer(c)
    print(f'rho~={c:6d}: layer thickness ds={s:.3e}, y at p_r=0 return = {y:.6f} (24/25=0.96), max p_r/rho = {gr:.3e}')
# closed form check: g/u = 6-5u-1/u
import sympy as sp
yy=sp.symbols('y'); u=sp.sqrt(1-yy); g=u*(6-5*u-1/u)
print('ODE residual:',sp.simplify(sp.diff(g,yy)+(yy+g)/(2*(1-yy))-2),' g(0)=',g.subs(yy,0))
print('roots of g:',sp.solve(sp.Eq(6-5*sp.Symbol('u')-1/sp.Symbol('u'),0)))
