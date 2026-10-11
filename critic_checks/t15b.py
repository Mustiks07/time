import sympy as sp
R,L=sp.symbols('R L',positive=True)
mu=sp.sqrt(1+L**2/R**2); W=mu-mu**2/(2*R)
W1=sp.diff(W,R); W2=sp.diff(W,R,2)
sol=sp.nsolve([W1,W2],[R,L],[4.0,1.9])
Rv,Lv=sol
M=W.subs({R:Rv,L:Lv})
Z=1/sp.sqrt(1-2*M/Rv); v=Lv/sp.sqrt(Rv**2+Lv**2)
print('marginal stability: R0=%.5f L=%.5f M=%.5f Z=%.5f v=%.5f c  p/sigma=v^2/2=%.4f'%(Rv,Lv,M,Z,v,v**2/2))
