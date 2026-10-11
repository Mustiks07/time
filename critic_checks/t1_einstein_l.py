# Einstein tensor for ds^2 = -e^{2Phi(l)} dt^2 + dl^2 + r(l)^2 dOmega^2 (proper-length gauge)
import sympy as sp
t,l,th,ph=sp.symbols('t l theta phi',real=True)
Phi=sp.Function('Phi')(l); r=sp.Function('r')(l)
X=[t,l,th,ph]
g=sp.diag(-sp.exp(2*Phi),1,r**2,r**2*sp.sin(th)**2)
gi=g.inv(); n=4
Gam=[[[sp.simplify(sum(gi[i,k]*(sp.diff(g[k,j],X[m])+sp.diff(g[k,m],X[j])-sp.diff(g[j,m],X[k])) for k in range(n))/2) for m in range(n)] for j in range(n)] for i in range(n)]
Ric=sp.zeros(n)
for j in range(n):
  for m in range(n):
    Ric[j,m]=sp.simplify(sum(sp.diff(Gam[i][j][m],X[i])-sp.diff(Gam[i][j][i],X[m])+sum(Gam[i][i][k]*Gam[k][j][m]-Gam[i][m][k]*Gam[k][j][i] for k in range(n)) for i in range(n)))
R=sp.simplify(sum(gi[i,j]*Ric[i,j] for i in range(n) for j in range(n)))
G=sp.simplify(gi*Ric-sp.eye(n)*R/2)
rl=sp.diff(r,l); rll=sp.diff(r,l,2); Pl=sp.diff(Phi,l)
rho=-G[0,0]/(8*sp.pi); pr=G[1,1]/(8*sp.pi); pt=G[2,2]/(8*sp.pi)
print('8pi rho  - [(1-r_l^2)/r^2 - 2 r_ll/r] =',sp.simplify(8*sp.pi*rho-((1-rl**2)/r**2-2*rll/r)))
print('8pi p_r  - [-(1-r_l^2)/r^2 + 2 r_l Phi_l/r] =',sp.simplify(8*sp.pi*pr-(-(1-rl**2)/r**2+2*rl*Pl/r)))
print('8pi p_t  =',sp.simplify(8*sp.pi*pt))
# flat-slice metric check of section 9: r=l
a=sp.Function('alpha')(l)
sub={r:l}
pr9=sp.simplify((8*sp.pi*pr).subs(r,l).doit().subs(Phi,sp.log(a)).doit())
pt9=sp.simplify((8*sp.pi*pt).subs(r,l).doit().subs(Phi,sp.log(a)).doit())
print('sec9 p_r*4pi r alpha/alpha\' :',sp.simplify(pr9/(8*sp.pi)*4*sp.pi*l*a/sp.diff(a,l)))
print('sec9 8pi p_t - (r a\')\'/(r a):',sp.simplify(pt9-sp.diff(l*sp.diff(a,l),l)/(l*a)))
