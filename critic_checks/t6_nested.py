import numpy as np, math, sys
PI=math.pi
# analytic structure: shell at r_in=1 (y: 0->0.96), vacuum to r_g=1.2 (y=0.8), plateau y=0.8 (p_r=0) to r_out
y0,yp=0.96,0.8
a=math.sqrt(1-y0)
sigma=(1-a)/(4*PI); p=(1-a)**2/(16*PI*a)
print('inner shell p/sigma =',p/sigma)
rg=y0/yp
Z=10; L=(math.log(Z)-math.log(5))/2
rout=rg*math.exp(L)
M=yp*rout/2
print('M/r_in =',M,' 0.48*sqrt(Z/5)=',0.48*math.sqrt(Z/5))
# direct integration of Phi through the structure
r=np.linspace(rg,rout,200001)
m=yp*r/2; rho=yp/(8*PI*r**2); pr=0*r
dPhi=(m+4*PI*r**3*pr)/(r*(r-2*m))
Phi_out=0.5*math.log(1-2*M/rout)
Phi_g=Phi_out-np.trapezoid(dPhi,r)
Phi_in=Phi_g-0.5*(math.log(1-yp)-math.log(1-y0))   # vacuum Schwarzschild gap (m const)
print('Z from integration =',math.exp(-Phi_in))
# p_t on plateau from conservation (p_r=0, p_r'=0): p_t = r/2 * rho * Phi'
pt=r/2*rho*dPhi
print('plateau p_t/rho: min %.6f max %.6f'%((pt/rho).min(),(pt/rho).max()))
# Einstein cluster: p_t/rho = v^2/2 where v = orbital speed wrt static observers; circular geodesic speed v^2 = m/(r-2m)
v2=(m/(r-2*m))
print('Einstein-cluster (geodesic particles) at y=0.8 would need v^2 =',v2[0],' i.e. v =',math.sqrt(v2[0]),'c  (superluminal); p_t/rho =',v2[0]/2)
print('Einstein cluster max y (v<1  <=>  r>3m):  y<2/3')
# family optimum for a genuine Einstein cluster (p_t <= rho/2 => plateau y <= 2/3, n=y/(2(1-y))=1)
for yp2 in [2/3]:
    n=yp2/(2*(1-yp2))
    best=None
    for y0b in np.linspace(yp2,0.96,2000):
        L2=(math.log(Z)+0.5*math.log(1-y0b))/n
        if L2<0: continue
        Mb=0.5*y0b*math.exp(L2)
        if best is None or Mb<best[0]: best=(Mb,y0b)
    print('same construction with a physical Einstein cluster plateau (y=2/3) and a DEC inner shell: min M/r_in = %.4f at y0=%.3f'%best)
