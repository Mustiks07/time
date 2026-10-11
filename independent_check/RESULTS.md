# Independent verification: results

All values were computed independently from first principles. No files under `chronos/` or `reports/` were read.
Units are G=c=1 internally. SI conversions use G=6.67430e-11, c=299792458, M_sun=1.98841e30 kg,
M_J=1.89813e27 kg, yr=3.15576e7 s, g0=9.80665 m/s². Q6 also uses CODATA-2022 ε0, e, m_e and ħ.
Scripts are `q*.py` and their outputs are `q*_output.txt` in this directory.

## Q1: Static thin shell (flat interior, Schwarzschild exterior); `q1_q2_shell.py`

Israel junction conditions (derived symbolically and checked numerically):
- σ = (1 − √(1−2M/R)) / (4πR)
- p = (1/(8πR)) [ (1 − M/R)/√(1−2M/R) − 1 ]

Interior static clocks tick at α = √(1 − 2M/R) relative to infinity, so M/R = (1−α²)/2. Substituting:
- σ = (1−α)/(4πR)
- p = (1−α)²/(16πRα)
- **p/σ = (1−α)/(4α) = (Z−1)/4**

DEC (|p| ≤ σ) requires α ≥ 1/5, so **Z_max = 5** (M/R ≤ 12/25 = 0.48, R ≥ 25M/12).
**At Z = 10: p/σ = 9/4 = 2.25**, which violates the DEC.

## Q2: Thin-shell table; `q1_q2_shell.py`

Formulas used:
- M = (Rc²/2G)(1 − 1/Z²)
- p_SI = p_geom·c⁴/G (N/m)
- F = 2πR·p = (1−α)²/(8α) in units of c⁴/G. F does not depend on R.

| Z | R (m) | M (kg) | M (M_J) | p (N/m) | F (c⁴/G) |
|---|---|---|---|---|---|
| 2 | 1 | 5.0497e26 | 0.26604 | 1.2039e42 | 0.0625 |
| 2 | 10 | 5.0497e27 | 2.6604 | 1.2039e41 | 0.0625 |
| 2 | 100 | 5.0497e28 | 26.604 | 1.2039e40 | 0.0625 |
| 2 | 1000 | 5.0497e29 | 266.04 | 1.2039e39 | 0.0625 |
| 10 | 1 | 6.6656e26 | 0.35117 | 1.9503e43 | 1.0125 |
| 10 | 10 | 6.6656e27 | 3.5117 | 1.9503e42 | 1.0125 |
| 10 | 100 | 6.6656e28 | 35.117 | 1.9503e41 | 1.0125 |
| 10 | 1000 | 6.6656e29 | 351.17 | 1.9503e40 | 1.0125 |
| 100 | 1 | 6.7323e26 | 0.35468 | 2.3598e44 | 12.25125 |
| 100 | 10 | 6.7323e27 | 3.5468 | 2.3598e43 | 12.25125 |
| 100 | 100 | 6.7323e28 | 35.468 | 2.3598e42 | 12.25125 |
| 100 | 1000 | 6.7323e29 | 354.68 | 2.3598e41 | 12.25125 |
| 1000 | 1 | 6.7329e26 | 0.35471 | 2.4029e45 | 124.750125 |
| 1000 | 10 | 6.7329e27 | 3.5471 | 2.4029e44 | 124.750125 |
| 1000 | 100 | 6.7329e28 | 35.471 | 2.4029e43 | 124.750125 |
| 1000 | 1000 | 6.7329e29 | 354.71 | 2.4029e42 | 124.750125 |

In SI, F = 7.564e42 N, 1.2254e44 N, 1.4827e45 N and 1.5098e46 N for Z = 2, 10, 100 and 1000.
For comparison, c⁴/(4G) = 0.25 c⁴/G. The surface energy density is σ_SI = (1−α)c⁴/(4πGR), for example 8.668e42 J/m² at Z=10, R=1 m.

## Q3: Bounded-acceleration twin; `q3_twin.py`, `q3_convergence.py`

**Ratio_max(x) = sinh(x/4)/(x/4)**, with x = a_max·T/c.

The optimal profile is bang-bang:
1. Accelerate for T/4.
2. Decelerate for T/2 (turning around at the midpoint).
3. Accelerate for T/4.

Numerical optimal control (SLSQP, rapidity discretised with N=80–160, 12 random starts each) converges to exactly this
profile. Its objective equals the same-discretisation value of the analytic profile to 1e-15. The remaining
offset from the continuum formula is trapezoid quadrature bias, which shrinks as h² (10.0112, 10.0013, 10.00071 → 10.00067).

Ratio = 10 at x = 17.99965599 (x/4 = 4.4999140, the root of sinh y = 10y).
- (i) a = 1 g0: **T = 5.5026e8 s = 17.4365 yr** of traveller time; the base experiences 174.365 yr. Peak γ = 45.01, and the maximum distance is 85.27 ly.
- (ii) T = 1 yr: **a = 170.994 m/s² = 17.4365 g0**. The numbers in (i) and (ii) coincide by construction, because both equal x·c/(g0·yr).

## Q4: Kerr prograde circular equatorial orbits; `q4_kerr.py`

u^t from the metric (−1/(g_tt+2g_tφΩ+g_φφΩ²))^½ agrees with the BPT closed form (r^{3/2}+a)/(r^{3/4}√(r^{3/2}−3r^{1/2}+2a)) to the 18 digits printed.
The ZAMO speed v = (Ω−ω)√g_φφ/lapse agrees with the BPT formula. Also, u^t = γ_ZAMO/lapse.

| χ | r_isco/M | u^t(ISCO) | v_ZAMO/c | λ (1/M²): azim, vert, radial | M_min (2 m, 1 g0) |
|---|---|---|---|---|---|
| **0.997519536166** (u^t=10) | **1.257735233** | **10.000** | **0.568685** | 0.502611, 1.005223, −1.507834 | **1.1259e5 M_sun** |
| 0 | 6 | 1.414214 (√2) | 0.5 | 0.0046296, 0.0092593, −0.0138889 | 1.0805e4 M_sun |
| 0.9 | 2.320883 | 2.696378 | 0.624549 | 0.079991, 0.159982, −0.239973 | 4.4914e4 M_sun |
| 0.99 | 1.454498 | 6.081170 | 0.596932 | 0.324983, 0.649966, −0.974949 | 9.0531e4 M_sun |
| 0.998 | 1.236971 | 10.790943 | 0.564750 | 0.528350, 1.056700, −1.585050 | 1.1543e5 M_sun |

Notes:
- For u^t = 10, 1−χ = 2.48046e-3. At that orbit E = 0.685527, L = 1.409980 M, r_+ = 1.070390 M, lapse = 0.121572 and γ_ZAMO = 1.215724. The orbit is inside the ergosphere, so there are no static observers there and the speed is given relative to ZAMOs.
- The tidal tensor is E_ij = R_{0i0j} in the orbiter's orthonormal frame; the tidal acceleration is −E_ij ξ^j. It was computed from the full symbolic Kerr Riemann tensor, evaluated with mpmath at 50 digits. It reproduces the requested Schwarzschild check M/r³·{1, r/(r−3M), −(2r−3M)/(r−3M)} at r = 4, 6 and 10M, and Marck's closed form M/r³·{1, 1+3K/r², −(2+3K/r²)} with K=(L−aE)².
- At the ISCO, 3K/r² = 1 for every spin tested, so the eigenvalues are exactly (M/r³)·{1, 2, −3}. The radial (stretching) eigenvalue 3M/r_isco³ is the largest.
- Minimum mass condition: 3M/r³ · c² · (2 m) ≤ g0, so M_geom ≥ √(λ̂·2 m·c²/g0). If "over 2 m" means ±1 m about the centre (a 1 m lever arm), the masses fall by √2: 7.961e4 M_sun at χ for u^t=10, and 7.641e3, 3.176e4, 6.401e4, 8.162e4 M_sun for χ = 0, 0.9, 0.99, 0.998.
- Near-extremal scaling of u^t(ISCO): 13.77 at 1−χ=1e-3, 30.5 at 1e-4, 66.7 at 1e-5, 144.6 at 1e-6, and 674 at 1e-8. It diverges as χ→1.

## Q5: Radial stability of the Q1 shell; `q5_stability.py`

Equation of motion: Ṙ² + V(R) = 0, with V(R) = 1 − (M/m(R) + m(R)/(2R))² and m = 4πR²σ (from √(1+Ṙ²) − √(1−2M/R+Ṙ²) = m/R).

Shell energy conservation gives σ' = −2(σ+p)/R, so m' = −8πRp and m'' = −8πp + 16πβ²(σ+p).

At R0, V = V' = 0 identically, and

V''(R0) = [ (12α³+4α²)β² + 3α³ − α² − α − 1 ] / (2R0²α²).

**Stability (V''>0) requires β² > β²_min = (1−α)(1+2α+3α²)/(4α²(1+3α)) = (Z−1)(Z²+2Z+3)/(4(Z+3)).**

| Z | 1.5 | 2 | 3 | 5 | 10 |
|---|---|---|---|---|---|
| β²_min | 11/48 = 0.22917 | 11/20 = 0.55 | 3/2 = 1.5 | 19/4 = 4.75 | 1107/52 = 21.288 |

- The formula was checked by directly integrating σ(R) for a linear EOS and finite-differencing V(R). The roots agree to 1e-5 (Z=10: 21.2887 numerical vs 21.2885 exact).
- Limits: β²_min → 0 as Z→1, and β²_min ~ Z²/4 at large Z.
- **β²_min ≤ 1 only for Z < 2.529756** (the root of Z³+Z²−3Z−15=0), which corresponds to M/R < 0.42187 and α > 0.39529.
- So stability with β² ≤ 1 is compatible only with Z ≲ 2.53. At the DEC limit Z=5 it needs β² > 4.75, and at Z=10 it needs β² > 21.3. A Z=10 shell therefore violates the DEC (p/σ=2.25) and also needs a superluminal β.

## Q6: Charged dust shell, RN exterior; `q6_charged.py`

For any f, p = (1/8π)[(f'/2)/√f + √f/R − 1/R]. For RN, f'/2 + f/R = (1−M/R)/R, because the Q² terms cancel.

**Static dust (p=0) requires (1−M/R)² = f, i.e. |Q| = M (extremal), at any radius R > M.**

The interior clock rate is **α = 1 − M/R** (Z = R/(R−M)). The shell rest mass is 4πR²σ = M, so charge/rest mass = 1 in Gaussian geometrised units.

- **Extremal q/m = √(4πε0 G) = 8.6175e-11 C/kg** (1.1604e10 kg/C, or about one excess elementary charge per 1.86e-9 kg).
- For Z=10, R=2 m: M = 1.8 m (geometric) = 2.4239e27 kg = 1.277 M_J, and Q = 2.0888e17 C.
- **The field at the shell, in the static orthonormal frame, is E = Q/R² → 4.6932e26 V/m.**
- The Schwinger field m_e²c³/(eħ) is 1.3233e18 V/m, so **E/E_S ≈ 3.55e8**.
- In general E = (1−1/Z)(1 m/R)·1.0429e27 V/m. Reaching E ≤ E_S at Z=10 would need R ≳ 7.1e8 m, i.e. M ≈ 4.3e5 M_sun.

## Q7: Majumdar–Papapetrou static observers; `q7_mp.py`, `q7_crit.py`, `q7_check_outer.py`

a = |∇U|/U². With masses m = 9ρ0/N (U(0)=10), the total ADM mass is 9ρ0.

The maximum over the ball |x| ≤ r is computed on 12 radial shells (6000 random directions plus all BH directions, then Nelder–Mead refinement). In every case the maximum was on the outer boundary.

a_max over the ball, in units of c²/ρ0:

| r/ρ0 | ico (N=12) | Fib N=100 | Fib N=1000 |
|---|---|---|---|
| 0 | 0 (exactly) | 2.1787e-5 | 1.2000e-6 |
| 0.01 | 2.3760e-11 | 2.2007e-5 | 1.2045e-6 |
| 0.02 | 7.6032e-10 | 2.2497e-5 | 1.2132e-6 |
| 0.05 | 7.4250e-8 | 2.5711e-5 | 1.2727e-6 |
| 0.1 | 2.3761e-6 | 3.5345e-5 | 1.4772e-6 |
| 0.2 | 7.6096e-5 | 6.5557e-5 | 2.2455e-6 |
| 0.3 | 5.8003e-4 | 1.1464e-4 | 3.5899e-6 |
| 0.4 | 2.4719e-3 | 2.0376e-4 | 5.9748e-6 |
| 0.5 | 7.7506e-3 | 3.9604e-4 | 1.0671e-5 |
| 0.6 | 2.0419e-2 | 9.0900e-4 | 2.1504e-5 |
| 0.7 | 4.9342e-2 | 2.5526e-3 | 5.3564e-5 |
| 0.8 | 1.1878e-1 | 8.0036e-3 | 2.0614e-4 |
| 0.9 | 3.2156e-1 | 5.9351e-2 | 2.2995e-3 |

- **Icosahedron.** It is a spherical 5-design, so multipoles l=1..5 vanish and **a_max ≈ 0.2376 (r/ρ0)⁵ c²/ρ0** along the vertex directions.
  - The coefficient is exact: 6·m·S6/U² with S6(vertex) = 2 + 10 P6(1/√5) = 5.28, giving 0.75·6·5.28/100 = 0.2376.
  - This was verified with 40-digit mpmath, which was needed because double-precision cancellation fails at r/ρ0 ≲ 1e-2.
- **Fibonacci sphere** (convention z_i = 1−(2i+1)/N, golden-angle φ). These sets are not balanced: |Σn̂_i| = 0.02421 (N=100) and 0.01333 (N=1000). This leaves a constant gradient at the centre, a(0) = 0.09|Σn̂|/(N ρ0), which dominates for r/ρ0 ≲ 0.1.

Minimum total mass for a habitat with proper radius 1 m (coordinate r = 1 m/U(0) = 0.1 m) and a ≤ 1 g0 (g0/c² = 1.0911e-16 m⁻¹):

| N | ρ0_min | M_total = 9ρ0c²/G |
|---|---|---|
| **12 (icosahedron)** | **52.844 m** (each BH m = 39.63 m = 5.34e28 kg) | **6.404e29 kg = 0.3221 M_sun = 337.4 M_J** |
| **100 (Fibonacci, origin-centred)** | 1.9967e11 m | **2.420e39 kg = 1.217e9 M_sun** |
| **1000 (Fibonacci, origin-centred)** | 1.0998e10 m | **1.333e38 kg = 6.703e7 M_sun** |

For the icosahedron, ρ0 = (0.2376 r⁵c²/g0)^{1/6}.

**Caveat on the Fibonacci numbers.** They are set entirely by the residual dipole of the particular point set, so they are not intrinsic to N.
- With the other common convention, z_i = 1−2i/(N−1), |Σn̂| is 0.157 (N=100) and 0.0250 (N=1000), and the masses would be about 6.5× and 1.9× larger.
- If the habitat is instead centred on the nearest true critical point of U (∇U=0), that point lies far off-centre (0.52ρ0 for N=100, 0.69ρ0 for N=1000). There a ∝ r (Hessian), giving M_total ≈ 2.17e3 M_sun (N=100) and 351 M_sun (N=1000); see `q7_crit_output.txt`.
- A truly balanced design (a spherical t-design) would give a ∝ r^t. The icosahedron (t=5) is the best of the configurations computed here.

## Uncertainties and assumptions
- Q4(c): "largest tidal acceleration over 2 m" is read as the differential acceleration across a 2 m separation; the ±1 m reading is also given above. Only the ISCO orbits were evaluated for tidal forces.
- Q7: "proper radius ≈ U(0)·r" was used as instructed. U is constant to 1e-15 over the icosahedral habitat, so this is exact there. The Fibonacci results depend strongly on the point-set convention, as explained above.
- Q5: β² = dp/dσ is treated as a "sound speed" only heuristically; the β² ≤ 1 criterion is the conventional reading.
