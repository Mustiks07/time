import mpmath as mp
mp.mp.dps=40
def risco(a):
    Z1=1+(1-a**2)**(mp.mpf(1)/3)*((1+a)**(mp.mpf(1)/3)+(1-a)**(mp.mpf(1)/3))
    Z2=mp.sqrt(3*a**2+Z1**2)
    return 3+Z2-mp.sqrt((3-Z1)*(3+Z1+2*Z2))
def ut(r,a): return (r**1.5+a)/(r**0.75*mp.sqrt(r**1.5-3*mp.sqrt(r)+2*a))
def ut_metric(r,a):
    gtt=-(1-2/r); gtp=-2*a/r; gpp=r*r+a*a+2*a*a/r; W=1/(r**1.5+a)
    return 1/mp.sqrt(-(gtt+2*gtp*W+gpp*W*W))
chi=mp.findroot(lambda a: ut(risco(a),a)-10, (mp.mpf('0.99'),mp.mpf('0.9999')),solver='anderson')
r=risco(chi)
print('chi* =',mp.nstr(chi,10),' 1-chi*=',mp.nstr(1-chi,6),' r_isco=',mp.nstr(r,8),' ut(BPT)-ut(metric)=',mp.nstr(ut(r,chi)-ut_metric(r,chi),3))
den=r**0.75*mp.sqrt(r**1.5-3*mp.sqrt(r)+chi)
den=r**0.75*mp.sqrt(r**1.5-3*mp.sqrt(r)+2*chi)
E=(r**1.5-2*mp.sqrt(r)+chi)/den; L=(r**2-2*chi*mp.sqrt(r)+chi**2)/den
print('E=',mp.nstr(E,6),' L=',mp.nstr(L,6))
for a in ['0.998','0.99','0.9','0']:
    a=mp.mpf(a); print('chi=',a,' Z_isco=',mp.nstr(ut(risco(a),a),6))
# ZAMO speed
gtt=-(1-2/r); gtp=-2*chi/r; gpp=r*r+chi*chi+2*chi*chi/r
om=-gtp/gpp; vp=mp.sqrt(gpp); N=mp.sqrt(gtp**2/gpp-gtt); W=1/(r**1.5+chi)
v=vp*(W-om)/N
print('v_ZAMO=',mp.nstr(v,6), ' lapse N=',mp.nstr(N,6), ' gamma_ZAMO=',mp.nstr(1/mp.sqrt(1-v*v),6))
# Marck tidal eigenvalues
Kc=(L-chi*E)**2
lam=[(1+3*Kc/r**2)/r**3,-(2+3*Kc/r**2)/r**3,1/r**3]
print('tidal eigenvalues (1/M^2):',[mp.nstr(x,6) for x in lam])
# tide mass
G=6.67430e-11;c=299792458.0;Msun=1.98841e30
Mgeo=mp.sqrt(abs(lam[1])*c**2*2/9.80665)
print('M_min tides (1 g over 2 m, full length) = %.4g Msun ; (±1 m about CoM) = %.4g Msun'%(float(Mgeo*c**2/G/Msun),float(Mgeo*c**2/G/Msun/mp.sqrt(2))))
# escape boost
def Eofv(vv): return (N+om*vp*vv)/mp.sqrt(1-vv*vv)
v1=mp.findroot(lambda vv: Eofv(vv)-1,(v,mp.mpf('0.99999')),solver='bisect')
dphi=mp.atanh(v1)-mp.atanh(v)
print('escape: v1=',mp.nstr(v1,6),' dphi=',mp.nstr(dphi,6),' mass ratio=',mp.nstr(mp.e**dphi,6),' static Z=10 escape ratio gamma(1+v)=',mp.nstr(10*(1+mp.sqrt(1-mp.mpf(1)/100)),6))
# marginal stability: d^2V_eff/dr^2 at ISCO = 0. Z just outside ISCO at chi=0.998 :
a=mp.mpf('0.998'); ri=risco(a)
rz=mp.findroot(lambda rr: ut(rr,a)-10,(ri*1.0000001,ri*2),solver='bisect')
print('chi=0.998: r_isco=',mp.nstr(ri,7),' r(Z=10)=',mp.nstr(rz,7),' (r-r_isco)/M=',mp.nstr(rz-ri,4))
# Orbital period (coordinate) at ISCO for M=1.13e5 Msun
Mgeo_m=float(Mgeo); Omega=1/(r**1.5+chi)
P=2*mp.pi/Omega*Mgeo_m/c
print('orbital period at infinity (s):',mp.nstr(P,5),' proper period:',mp.nstr(P/10,5))
