import sys, math, numpy as np, sympy as sp
sys.dont_write_bytecode=True
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# 1. symbolic: static observer acceleration and orthonormal-frame field in MP metric
x,y,z,t=sp.symbols('x y z t',real=True)
U=sp.Function('U')(x,y,z)
X=[t,x,y,z]
g=sp.diag(-U**-2,U**2,U**2,U**2); gi=g.inv()
# acceleration of static observer a_mu = d_mu ln N, N=1/U ; |a|^2 = g^{ij} a_i a_j
N=1/U
a_cov=[sp.diff(sp.log(N),v) for v in X]
a2=sp.simplify(sum(gi[i,i]*a_cov[i]**2 for i in range(4)))
gradU2=sum(sp.diff(U,v)**2 for v in (x,y,z))
print('|a|^2 - |gradU|^2/U^4 =',sp.simplify(a2-gradU2/U**4))
# field: A_t = 1/U, F_{it}=d_i A_t ; orthonormal E_i = F_{it}/(sqrt(g_ii) sqrt(-g_tt))
E2=sum((sp.diff(1/U,v)/(U*U**-1))**2 for v in (x,y,z))
print('|E|^2 - |gradU|^2/U^4 =',sp.simplify(E2-gradU2/U**4))
# check Maxwell+Einstein normalisation: for single extremal BH U=1+m/rho, at large rho E ~ m/r^2 (Coulomb with Q=m)
G=6.67430e-11;c=299792458.0;eps0=8.8541878128e-12
print('B[T] per a[m/s^2] =',1/(c*math.sqrt(4*math.pi*eps0*G)), ' -> at 1 g:',9.80665/(c*math.sqrt(4*math.pi*eps0*G)),'T')
print('E[V/m] per a[m/s^2] =',1/math.sqrt(4*math.pi*eps0*G), ' -> at 1 g: %.3e V/m'%(9.80665/math.sqrt(4*math.pi*eps0*G)))
from chronos import mp_cage as mc
I,D,ID=mc.icosahedron(),mc.dodecahedron(),mc.icosidodecahedron()
print('orbit sizes',len(I),len(D),len(ID))
w2=mc.cancelling_weights([I,D],[6]); w3=mc.cancelling_weights([I,D,ID],[6,10])
print('weights I+D:',w2,' I+D+ID:',w3)
# check that invariant-coefficient cancellation is direction independent: evaluate multipole sums on many probe dirs
rng=np.random.default_rng(0)
def multipole_power(orbits,weights,l,ndir=500):
    dirs=rng.normal(size=(ndir,3)); dirs/=np.linalg.norm(dirs,axis=1)[:,None]
    tot=np.zeros(ndir)
    for o,w in zip(orbits,weights):
        tot+=w*np.sum(mc.legendre(l,dirs@o.T),axis=1)
    return np.abs(tot).max()
for l in range(1,15):
    print('l=%2d  I:%.2e  I+D:%.2e  I+D+ID:%.2e'%(l,multipole_power([I],[1/12],l),multipole_power([I,D],w2,l),multipole_power([I,D,ID],w3,l)))
for name,orbs,wts in [('I12',[I],[1/12]),('I+D',[I,D],w2),('I+D+ID',[I,D,ID],w3)]:
    C,n=mc.fit_multipole_law(orbs,wts,10)
    r1=mc.min_mass_from_law(C,n,10,1.0,9.80665); r2=mc.min_mass_from_law(C,n,10,1.0,2/mc.TESLA_PER_MS2)
    print(name,'n=%.4f'%n,' 1g: %.3g Msun = %.3g MJ ; 2T: %.3g Msun = %.3g MJ'%(r1['M_sun'],r1['M_jup'],r2['M_sun'],r2['M_jup']), ' rho0(1g)=%.3g m'%r1['rho0_m'])
