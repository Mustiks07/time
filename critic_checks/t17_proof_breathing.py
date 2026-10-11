# Pointwise bound: dt/dT = sqrt(f+A^2-1)/(A f) is increasing in A at fixed R; under |p/sigma|<=w,
# m >= m_min (R/Rmin)^(-2w) so A <= A_max(R) = M R^{2w}/m_min + m_min/(2 R^{1+2w}) (R_min=1; physical branch).
# If F(R)=sqrt(f+Amax^2-1)/(Amax f) <= Z_min = 1/(1-q_min) for all R>=1, then <Z> <= Z(R_min) rigorously.
import numpy as np, math
worst=-1
for w in [0.25,0.5,1.0,2.0,5.0]:
    for qmin in np.linspace(0.001,0.999,999):
        M=qmin*(1-qmin/2); fmin=(1-qmin)**2
        # static DEC-type feasibility at R_min requires p/sigma(R_min) > static value (Z-1)/4 <= w
        Zmin=1/(1-qmin)
        if (Zmin-1)/4>=w: continue
        R=np.geomspace(1,1e3,200001)[1:]
        f=1-2*M/R
        Am=M*R**(2*w)/qmin+qmin/(2*R**(1+2*w))
        F=np.sqrt(f+Am**2-1)/(Am*f)
        worst=max(worst,(F/Zmin).max())
    print('w=',w,' max_R,q_min F(R)/Z(R_min) =',worst)
