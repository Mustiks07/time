import numpy as np, math
from scipy.optimize import brentq
def traj(qmin,R1):
    M=qmin*(1-qmin/2); Zmin=1/(1-qmin)
    def m(R): return qmin*R**-2 if R<R1 else qmin*R1**-2*(R/R1)**2
    A=lambda R: M/m(R)+m(R)/(2*R)
    Rs=np.linspace(1.0001,40,200000); Av=np.array([A(r) for r in Rs])
    j=np.where(Av<=1)[0][0]
    Rmax=brentq(lambda R: A(R)-1,Rs[j-1],Rs[j])
    th=(np.arange(200000)+0.5)/200000*math.pi
    R=1+(Rmax-1)*(1-np.cos(th))/2; dR=(Rmax-1)*np.sin(th)/2
    mm=np.where(R<R1,qmin*R**-2,qmin*R1**-2*(R/R1)**2); AA=M/mm+mm/(2*R)
    s=np.sqrt(np.maximum(AA**2-1,1e-30)); f=1-2*M/R
    T=np.sum(AA/s*dR); t=np.sum((AA-mm/R)/(f*s)*dR)
    inst=((AA-mm/R)/(AA*f)).max()
    return t/T, Zmin, inst/Zmin, Rmax
print('q_min=0.34,R1=1.1786: <Z>, Z(Rmin), max inst/Z(Rmin), Rmax =',traj(0.34,1.1786))
