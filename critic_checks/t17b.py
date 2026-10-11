import numpy as np, math
# explicit trajectory: R_min=1, lambda=+1 on [1,R1], lambda=-1 beyond (DEC-saturating), find pointwise max of dt/dT / Z(R_min)
best=(0,None)
for qmin in np.linspace(0.05,0.79,75):
    M=qmin*(1-qmin/2); Zmin=1/(1-qmin)
    for R1 in np.linspace(1.01,3,60):
        R=np.linspace(1,40,400001)
        lnm=np.where(R<R1,math.log(qmin)-2*np.log(R),math.log(qmin)-2*math.log(R1)+2*np.log(R/R1))
        m=np.exp(lnm); A=M/m+m/(2*R)
        if not A[1]>1: continue
        idx=np.where(A[1:]<=1)[0]
        if len(idx)==0: continue
        j=idx[0]+1
        A_=A[1:j]; R_=R[1:j]; m_=m[1:j]
        if np.any(A_-m_/R_<=0): continue
        f=1-2*M/R_
        r=((A_-m_/R_)/(A_*f)).max()/Zmin
        # period average
        if r>best[0]: best=(r,(qmin,R1,R[j]))
print('max pointwise (dt/dT)/Z(R_min) on DEC trajectories:',best)
