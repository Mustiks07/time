"""Oscillating thin shell, flat interior, Schwarzschild exterior; general elastic EoS given by
lambda(R)=p/sigma (|lambda|<=1 = DEC).  m(R)=4piR^2 sigma = m_min exp(-2 int lambda/R dR).
Motion: sqrt(1+Rdot^2)=A(R)=M/m+m/(2R).  Interior (habitat) time T, exterior time t:
dT/dR = A/sqrt(A^2-1), dt/dR = (A-m/R)/(f sqrt(A^2-1)).  <Z> = oint dt / oint dT."""
import numpy as np, math
from scipy.optimize import minimize, brentq
from scipy.integrate import quad
def setup(fmin, lam_nodes, Rcap):
    M=(1-fmin)/2; mmin=1-math.sqrt(fmin)
    Rn=np.geomspace(1,Rcap,len(lam_nodes))
    lam=lambda R: np.interp(R,Rn,lam_nodes)
    # integrate ln m on fine grid
    Rg=np.geomspace(1,Rcap,4001)
    lg=lam(Rg)
    lnm=math.log(mmin)-2*np.concatenate(([0],np.cumsum(0.5*(lg[1:]/Rg[1:]+lg[:-1]/Rg[:-1])*np.diff(Rg))))
    mg=np.exp(lnm)
    A=M/mg+mg/(2*Rg)
    return M,Rg,mg,A
def zavg(params,Rcap=30.0,verbose=False):
    fmin=1/(1+math.exp(-params[0]))*0.999+0.0005
    lam_nodes=np.tanh(params[1:])
    M,Rg,mg,A=setup(fmin,lam_nodes,Rcap)
    # need A>1 just above R=1, then return to 1
    above=A>1+1e-12
    if not above[1:5].all(): return None
    idx=np.where(~above[1:])[0]
    if len(idx)==0: return None
    j=idx[0]+1
    Rmax=Rg[j]
    if Rmax<1.001: return None
    mi=lambda R: np.interp(R,Rg,mg)
    Ai=lambda R: M/mi(R)+mi(R)/(2*R)
    if np.any(Ai(Rg[:j])-mg[:j]/Rg[:j]<=0): return None   # physical branch sqrt(f+Rdot^2)>0
    def integ(th,which):
        R=1+(Rmax-1)*(1-math.cos(th))/2
        dR=(Rmax-1)*math.sin(th)/2
        a=Ai(R); s=math.sqrt(max(a*a-1,1e-300))
        if which=='T': return a/s*dR
        f=1-2*M/R; return (a-mi(R)/R)/(f*s)*dR
    T=quad(integ,1e-6,math.pi-1e-6,args=('T',),limit=200)[0]
    t=quad(integ,1e-6,math.pi-1e-6,args=('t',),limit=200)[0]
    Zmin=1/math.sqrt(fmin)
    return t/T, Zmin, Rmax, fmin, lam_nodes
rng=np.random.default_rng(1)
best=None; worst_ratio=0; worst=None
nn=7
for trial in range(400):
    x0=np.concatenate(([rng.normal(0,2)],rng.normal(0,1.5,nn)))
    def obj(x):
        r=zavg(x)
        return 0.0 if r is None else -r[0]
    def obj2(x):
        r=zavg(x)
        return 0.0 if r is None else -r[0]/r[1]
    for o,lab in [(obj,'Z'),(obj2,'ratio')]:
        try:
            res=minimize(o,x0,method='Nelder-Mead',options=dict(maxiter=600,xatol=1e-5,fatol=1e-7))
        except Exception as e:
            continue
        r=zavg(res.x)
        if r is None: continue
        if lab=='Z' and (best is None or r[0]>best[0]): best=r
        if lab=='ratio' and r[0]/r[1]>worst_ratio: worst_ratio=r[0]/r[1]; worst=r
print('max <Z> found (DEC, |p/sigma|<=1):',best[0],' its Z(R_min)=',best[1],' R_max/R_min=',best[2])
print('max <Z>/Z(R_min) found:',worst_ratio,' case: <Z>=',worst[0],'Z(Rmin)=',worst[1],'Rmax=',worst[2])
