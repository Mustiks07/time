import sympy as sp, math, mpmath as mp
R,M,Q,d=sp.symbols('R M Q delta',positive=True)
f=1-2*M/R+Q**2/R**2
# static junction (flat interior): p = (1/8piR)[ (f'/2 R + f)/sqrt f  - 1 ]  general formula
p=( (sp.diff(f,R)*R/2+f)/sp.sqrt(f) - 1)/(8*sp.pi*R)
print('p - report formula:',sp.simplify(p-((1-M/R)/sp.sqrt(f)-1)/(8*sp.pi*R)))
print('p at Q=M:',sp.simplify(p.subs(Q,M)))
# mistuning: exact R'' for dust shell momentarily at rest at R with Q=M(1+d)
mu=sp.symbols('mu',positive=True)
A=((2*M-Q**2/R)/mu+mu/R)/2
musol=R-sp.sqrt(R**2-2*M*R+Q**2)   # A(R)=1 root with mu<R
Rdd=sp.diff(A,R)*A   # R'' = -V'/2 = A A'
Rdd0=sp.simplify(Rdd.subs(mu,musol))
ser=sp.series(Rdd0.subs(Q,M*(1+d)),d,0,2).removeO()
a=sp.symbols('alpha',positive=True)
print('R\'\' to O(delta):',sp.simplify(ser.subs(M,R*(1-a))),'   claim: delta*(Z-1)/R =',sp.simplify((1/a-1)*d/R))
# SI numbers
G=6.67430e-11;c=299792458.0;eps0=8.8541878128e-12;me=9.1093837015e-31;e=1.602176634e-19;hbar=1.054571817e-34
Eunit=c**2/math.sqrt(4*math.pi*eps0*G); ES=me**2*c**3/(e*hbar)
Rm=2.0;Z=10;al=1/Z
E=(1-al)/Rm*Eunit
print('E=%.3e V/m  E/E_S=%.3e  B=%.3e T'%(E,E/ES,E/c))
Rmin=(1-al)*Eunit/ES; print('R_min below Schwinger=%.3e m  M=%.3e Msun'%(Rmin,0.9*Rmin*c**2/G/1.98841e30))
BS=me**2*c**2/(e*hbar); BW=BS*(80.369/0.51099895e-3)**2
print('B_S=%.3e  B_W=%.3e  B/B_W=%.3e'%(BS,BW,E/c/BW))
# tolerance: drift 5 cm in 10 yr
for label,t in [('10 yr (shell proper time)',10*3.15576e7),('10 yr exterior -> 1 yr shell time',3.15576e7)]:
    acc=2*0.05/t**2
    delta=acc*Rm/((Z-1)*c**2)
    print(label,': required |delta| <~ %.2e'%delta)
# density
Mkg=0.9*Rm*c**2/G; print('M=%.3e kg = %.3f M_J ; rho(thickness R/10)=%.3e'%(Mkg,Mkg/1.89813e27,Mkg/(4*math.pi*Rm**3*0.1)))
# extremal magnetic charge of Dirac monopole: mass needed for q_m = M (geometric)
alpha_fs=7.2973525693e-3
mpl=math.sqrt(hbar*c/G)
# Dirac monopole g = e/(2 alpha) in Gaussian-like units; extremal mass = g/sqrt(G) (Gaussian) = mpl*sqrt(alpha)/(2 alpha)
m_ext=mpl*math.sqrt(alpha_fs)/(2*alpha_fs)
print('extremal mass for one Dirac monopole = %.3e kg = %.2e GeV'%(m_ext,m_ext*c**2/1.602176634e-10))
