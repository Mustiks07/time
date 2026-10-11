# Shell between Schwarzschild(M-) inside and Schwarzschild(M+) outside; barotropic EoS beta2=dp/dsigma.
# V = 1 - 2M-/R - (dM/m + m/(2R))^2,  m=4piR^2 sigma, m'=-8piRp, m''=-8pi p+16pi beta2 (sigma+p).
import sympy as sp, numpy as np, math
R,Mm,Mp,b2,x=sp.symbols('R Mm Mp beta2 x',positive=True)
m0,m1,m2=sp.symbols('m0 m1 m2')
mf=m0+m1*(x-R)+m2*(x-R)**2/2
V=1-2*Mm/x-((Mp-Mm)/mf+mf/(2*x))**2
fm=1-2*Mm/R; fp=1-2*Mp/R
sig=(sp.sqrt(fm)-sp.sqrt(fp))/(4*sp.pi*R)
p=((1-Mp/R)/sp.sqrt(fp)-(1-Mm/R)/sp.sqrt(fm))/(8*sp.pi*R)
subs={m0:4*sp.pi*R**2*sig,m1:-8*sp.pi*R*p,m2:-8*sp.pi*p+16*sp.pi*b2*(sig+p)}
V0=sp.lambdify((R,Mm,Mp),V.subs(x,R).subs(subs))
V1=sp.lambdify((R,Mm,Mp),sp.diff(V,x).subs(x,R).subs(subs))
V2=sp.diff(V,x,2).subs(x,R).subs(subs)
c0=sp.lambdify((R,Mm,Mp),V2.subs(b2,0)); c1=sp.lambdify((R,Mm,Mp),sp.diff(V2,b2))
pf=sp.lambdify((R,Mm,Mp),p); sf=sp.lambdify((R,Mm,Mp),sig)
print('consistency V(R0),V\'(R0) at R=2.6,Mm=0.3:',V0(2.6,0.3,1.0),V1(2.6,0.3,1.0))
for mm in [0.0,0.01,0.1,0.3,0.5,0.7,0.9,0.99]:
    def b2min(Rv): return -c0(Rv,mm,1.0)/c1(Rv,mm,1.0)
    Rs=np.linspace(2.0005,8,60000)
    vals=np.array([b2min(r) for r in Rs])
    ok=Rs[vals<=1.0]
    print('M-/M+=%.2f: critical R for beta2<=1: %.4f M+   (photon sphere 3; sigma>2p at crit: %s)'%(mm,ok.min(),sf(ok.min(),mm,1)>2*pf(ok.min(),mm,1)))
