import sympy as sp, math
R,M,b2=sp.symbols('R M beta2',positive=True)
# independent derivation: general static shell between f_in=1 and f_out=1-2M/R with EoS p(sigma).
# Use mass function m(R) with m(R0)=4piR0^2 sigma0, m'(R0)=-8piR0 p0, m''(R0)=-8pi p0+16 pi b2 (sigma0+p0).
m0,m1,m2=sp.symbols('m0 m1 m2')
x=sp.symbols('x')
mfun=m0+m1*(x-R)+m2*(x-R)**2/2
V=1-(M/mfun+mfun/(2*x))**2
Vpp=sp.diff(V,x,2).subs(x,R)
a=sp.symbols('alpha',positive=True)
Mval=R*(1-a**2)/2
sig=(1-a)/(4*sp.pi*R); p=(1-a)**2/(16*sp.pi*R*a)
subs={M:Mval,m0:4*sp.pi*R**2*sig,m1:-8*sp.pi*R*p,m2:-8*sp.pi*p+16*sp.pi*b2*(sig+p)}
V0=sp.simplify((1-(M/mfun+mfun/(2*x))**2).subs(x,R).subs(subs))
V1=sp.simplify(sp.diff(V,x).subs(x,R).subs(subs))
print('V(R0)=',V0,' V\'(R0)=',V1)
Vpp_s=sp.simplify(Vpp.subs(subs))
sol=sp.solve(sp.Eq(Vpp_s,0),b2)[0]
claim=(1-a)*(3*a**2+2*a+1)/(4*a**2*(1+3*a))
print('threshold:',sp.factor(sol),'   difference from report:',sp.simplify(sol-claim))
print('d Vpp/d beta2 =',sp.simplify(sp.diff(Vpp_s,b2)),' (>0 => stable for beta2 above threshold)')
# root beta2=1
aroot=sp.nsolve(claim-1,a,0.4)
Z=1/aroot; RoverM=2/(1-aroot**2)
print('beta2_min=1 at alpha=%.6f Z=%.5f  R/M=%.4f  (photon sphere R/M=3; Buchdahl R/M=9/4=2.25)'%(aroot,Z,RoverM))
for Zv in [1.5,2,2.53,3,5,10,100]:
    print(Zv, float(claim.subs(a,1/sp.Float(Zv))))
print('beta2_min at photon sphere R=3M: ',float(claim.subs(a,sp.sqrt(sp.Rational(1,3)))))
# DEC at threshold: p/sigma
print('p/sigma at Z=2.53:',float((1-aroot)/(4*aroot)))
