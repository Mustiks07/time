# CRITIC — Round 1 (Agent 09, scientific critic)

**Target:** `reports/REPORT_05_Mathematical_Verification.md` and the code in `chronos/` (static_spherical, thin_shell, charged, mp_cage, kerr, kinematics, dynamic_shell).
**Method:** each claim was re-derived independently (SymPy, high-precision mpmath, or my own ODE/optimisation code), and I then tried to build counterexamples. All checks ran from a scratch directory outside the repository. Where the authors' code was used, it was imported read-only (`python -B`). No repository file except this one was written.
**Units:** G = c = 1 unless stated. y ≡ 2m/r. In the "tilde" variables, ρ̃ = 8πr²ρ and g = 8πr²p_r.

## 0. Verdict summary

| # | Claim (REPORT_05 section) | Verdict |
|---|---|---|
| 1a | Theorem 2 proof, under its literal hypothesis 2m(r) < r | **CONFIRMED WITH CAVEATS** |
| 1b | "No horizons" treated as equivalent to 2m < r; Corollary 4 ("can be circumvented only by p_r > ρ, non-staticity, non-sphericity or modified gravity") | **ERROR FOUND**: explicit counterexample |
| 1c | Sharpness of α ≥ 1 − 2M/r_in | **CONFIRMED** (analytic family). **Caveat:** saturation needs p_t → −∞ (NEC violated). The NEC version is **UNRESOLVED**; the evidence points to the thin-shell value |
| 1d | Corollary 2 (all EM fields obey p_r ≤ ρ) | CONFIRMED |
| 1e | Corollary 3 (ρ < 0 allowed, bound with m_max) | CONFIRMED WITH CAVEATS. Notably, it is the m_max form that survives the counterexample in 1b |
| 1f | "Probably new" | **UNRESOLVED** (novelty not established) |
| 2 | §2.3 β²_min(α) and Z < 2.530 | Formula **CONFIRMED** (independent derivation). The BLP/photon-sphere sentence contradicts the report's own number: **ERROR FOUND (interpretation) / UNRESOLVED (vs. BLP)** |
| 2' | §2.4 stable Einstein shell "Z ≤ 1.36 at v = 0.42c" | **ERROR FOUND (minor)**: supremum is Z = 1.372 at v = 0.431c |
| 3 | §2.2 thin-layer pointwise-DEC ODE, closed form, 24/25 | CONFIRMED |
| 4 | §4 nested DEC shells, 0.680 at Z = 10, formula 0.48·√(Z/5) | Mathematics **CONFIRMED**. The "Einstein plateau" identification is an **ERROR FOUND** (p_t = ρ at y = 0.8 is not an Einstein cluster). "Optimum" holds only within the family: a DEC continuum solution with M/r_in = 0.6682 exists |
| 5 | §5 BPS shell, mistuning, Schwinger/EW numbers | CONFIRMED WITH CAVEATS |
| 6 | §6 MP cage identity, 38.7 T/(m s⁻²), multipole cancellation, mass table, tolerance | CONFIRMED WITH CAVEATS |
| 7 | §7 Kerr: χ* = 0.997520, 1.13·10⁵ M_⊙, escape boost | CONFIRMED WITH CAVEATS |
| 8a | §8 breathing shell does not beat a static one | Claim **CONFIRMED** numerically, even for general elastic DEC shells. The **[ДОКАЗАНО] label is not justified**: the key step is unproven and its pointwise form is false |
| 8b | §9 NEC theorem for flat-slice metrics; Statement 5 | CONFIRMED |
| 8c | §11 exit-energy statements | CONFIRMED |
| 8d | §1 kinematic bound sinh(x)/x and its numbers | CONFIRMED |
| 9 | Category errors and overclaims | See §9 (six items) |

---

## 1. Theorem 2 (§3): α_cav ≥ 1 − 2M/r_in

### 1a. Line-by-line check of the proof — CONFIRMED WITH CAVEATS

* **Field equation.** Φ' = (m + 4πr³p_r)/(r(r − 2m)) is correct. I re-derived it in an independent gauge, ds² = −e^{2Φ(l)}dt² + dl² + r(l)²dΩ², with SymPy. This gives 8πρ = (1−r_l²)/r² − 2r_ll/r and 8πp_r = −(1−r_l²)/r² + 2r_lΦ_l/r. Using 1 − r_l² = 2m/r, it reproduces the report's formula wherever r_l > 0.
* **The inequality.** p_r ≤ ρ = m'/(4πr²) gives Φ' ≤ (m + rm')/(r²(1−y)) = y/(r(1−y)) + y'/(2(1−y)). The algebra is correct (m = yr/2, m' = (y + ry')/2).
* **Total-derivative term.** ∫ y'/(2(1−y)) dr = [−½ln(1−y)] vanishes between y(r_in) = 0 and y(∞) = 0. This is legitimate for continuous y with y < 1.
* **Thin shells (jumps in y).** "The same antiderivative" is justified by regularisation. An Israel shell with σ > 0 is the limit of thin smooth layers in which ρ ~ σ/ε while p_r = O(1), so p_r ≤ ρ holds pointwise and the integrated inequality is exact. **Caveat:** the step is *false* for downward jumps (σ < 0). There the left side is 0 (Φ is continuous) and the right side is negative. Such shells are excluded only because any regularisation of a σ < 0 Israel shell violates p_r ≤ ρ. The proof should say this explicitly, because Corollary 3 explicitly allows ρ < 0.
* **m ≤ M.** This follows from ρ ≥ 0, and also m ≥ 0. Monotonicity of y/(1−y) is fine. The final integral is finite only if 2M < r_in. If 2M ≥ r_in the claim is trivially true, so this costs nothing, but it should be stated.
* **Very negative p_r.** Inside the hypothesis 2m < r this is harmless: it only lowers Φ'. It is **not** harmless once 2m < r is dropped (see 1b).
* **Free strengthening.** The same proof run from r to ∞ gives, at every radius, α(r) ≥ (1 − 2M/r)/√(1 − 2m(r)/r). I checked this on 3000 random profiles with ρ ≥ 0 and p_r ∈ [−3ρ, ρ]: the maximum violation was 1.5·10⁻⁵, which is discretisation error.

### 1b. "No horizons" ≠ "2m < r"; Corollary 4 is false — ERROR FOUND

The boxed statement assumes 2m(r) < r and calls this "без горизонтов". In a static spacetime, a sphere with 2m = r is a **minimal or maximal sphere** of the t = const slice. It is a horizon only if the lapse vanishes there. Where the areal radius *decreases* outward (r_l < 0), the hypothesis p_r ≤ ρ turns into r_lΦ_l ≤ (1−r_l²)/r − r_ll. That is a **lower** bound on Φ_l, so radial tension can raise the lapse outward without limit.

**Explicit counterexample.** It is static, spherically symmetric, asymptotically flat (exactly vacuum Schwarzschild for r ≥ 6M), has a regular centre and a flat vacuum cavity, satisfies ρ ≥ 0 and p_r ≤ ρ everywhere (in fact p_r ≤ 0 everywhere), and has a finite, nonzero lapse everywhere, so there is no horizon. With r_in = 1:

1. 0 ≤ l ≤ 1: r = l, Φ = Φ_c (flat cavity).
2. 1 ≤ l ≤ 4: r_l = h(l) decreases smoothly (C² smoothstep) from +1 to −0.9. This passes through a maximal sphere at r_eq = 2.056, where 2m = r. Set Φ_l = 0 where h > 0 and Φ_l = K·bump(l) ≥ 0 where h < 0. Here ρ ≥ 0 automatically, because h' ≤ 0 and |h| ≤ 1.
3. Left Schwarzschild sheet: r_l = −√(1−2M/r) with M = r₂(1−0.81)/2 = 0.10925 (r₂ = 1.15), ρ = 0, Φ constant, so 8πp_r = −2M/r³ < 0.
4. A throat (minimal sphere) at r = 2M = 0.2185. Then the right sheet r_l = +√(1−2M/r), with Φ_r = s(r)·M/(r²(1−2M/r)), where s is a smoothstep from 0 (r ≤ 2.5M) to 1 (r ≥ 6M). This gives 8πp_r = (2M/r³)(s − 1) ≤ 0, and exact Schwarzschild beyond 6M.

Numbers: lapse at the throat 0.715. K = 5.86 gives **α_cav = 0.0100**, while Theorem 2 demands **α_cav ≥ 1 − 2M/r_in = 0.7815**. That is violated by a factor of 78. Numerical checks: min ρ ≥ 0; max(p_r − ρ) = −1.7·10⁻¹¹ < 0. NEC fails, as it must: ρ + p_r < 0 where the lapse rises, and 8π(ρ + p_r) = −1/r_th² at the throat. Letting h_min → −1 sends M/r_in → 0 with α_cav arbitrary.

**Consequences and corrections**

* The theorem is correct *only with 2m(r) < r as an independent hypothesis*. The phrase "без горизонтов" must not be read as equivalent. The same applies to the docstring of `static_spherical.alpha_lower_bound`.
* A cleaner physical hypothesis is **radial NEC (ρ + p_r ≥ 0) plus a lapse bounded away from 0**. At a minimal sphere r_l = 0 and r_ll ≥ 0, so 8π(ρ + p_r) = −2r_ll/r ≤ 0. In the degenerate cases (r_ll = 0, or an inflection), NEC forces Φ → −∞, which is a horizon. Hence, with a regular centre and asymptotic flatness, radial NEC plus no horizon implies 2m < r everywhere. Under NEC, the authors' REPORT_10 statement that "p_r > ρ is the only loophole" is then consistent.
* **Corollary 4 is wrong as written.** Add: "or radial NEC violation (p_r < −ρ, allowed by the hypotheses ρ ≥ 0, p_r ≤ ρ), which allows a non-monotone areal radius: a 'bag of gold' behind a throat."
* **What survives.** The m_max form of Corollary 3 is *unconditional*. If y < 1 everywhere, the proof applies. If y = 1 at some r* > r_in, then m_max ≥ r*/2 > r_in/2 and the bound is trivial. In the counterexample m_max = r_eq/2 = 1.03 r_in. So the physical message "positive Misner–Sharp mass ≳ (1 − 1/Z)r_in/2 must exist *somewhere*" survives. Only the ADM-mass form fails: a throat can hide the mass from a distant observer even with ρ ≥ 0 everywhere. I recommend stating Theorem 2 primarily in the m_max form.

### 1c. Is the bound sharp? — CONFIRMED (analytic), with a physically important caveat

The report supports sharpness only numerically (0.462 vs 0.45). An exact saturating family exists. Take a thin layer of thickness ε at R with p_r = ρ. Then Φ' = d[−½ln(1−y)]/dr + 2m/(r(r−2m)), so the lapse *jumps* by −½ln(1−Y) across the layer, and α_cav → 1 − 2M/R. Numerically, with 2M/R = 0.9: α_cav = 0.10638 (ε = 10⁻²), 0.100646 (10⁻³), 0.100065 (10⁻⁴), against 0.1.

**Caveat.** In this family max|p_t|/ρ ≈ 2/ε and min p_t ≈ −7·10⁶ for ε = 10⁻⁴, so **tangential NEC and WEC are violated**. A thin-layer analysis shows this is unavoidable. With ρ + p_t ≥ 0, conservation gives p_r' ≥ −(ρ + p_r)(Φ' + 2/r). A radial pressure of order 1/ε, which is what produces the lapse jump, then cannot return to zero inside the layer: p_r ≥ 1/(1/p₀ + cΔr).

* With NEC added (ρ ≥ 0, |p_r| ≤ ρ, p_t ≥ −ρ, p_t unbounded above), my optimal-control run at Z = 10 converged to **M/r_in = 0.49507**. That is the single-thin-shell value (1 − Z⁻²)/2 = 0.495, not 0.45.
* A structured NEC family (inner Israel layer plus a thick layer that drops p_r at the NEC limit) never beat the thin shell at Z = 5, 10 or 100.
* 120 Nelder–Mead searches over smooth NEC profiles reached at most 0.90 of the thin-shell ln Z.
* The authors' own E19 ratios (1.58, 1.26, 1.12, 1.06 at Z = 1.7, 3.8, 8.4, 18) equal the thin-shell/theorem ratio (1 − Z⁻²)/(1 − Z⁻¹) = 1.588, 1.263, 1.119, 1.056 to the quoted precision.
* **Proven special case.** If p_r ≤ 0 everywhere, then Φ' ≤ y/(2r(1−y)) ≤ (M/r²)/(1 − 2M/r), and so **α_cav ≥ √(1 − 2M/r_in)**. The thin shell is optimal.

**Conjecture for round 2.** For ρ ≥ 0, p_r ≤ ρ, NEC and 2m < r, α_cav ≥ √(1 − 2M/r_in), i.e. M ≥ (1 − Z⁻²)r_in/2 = 0.495 r_in at Z = 10. This is UNRESOLVED. It would be 10% stronger than Theorem 2 at Z = 10 and is the physically relevant version.

### 1d. Corollary 2 (EM fields) — CONFIRMED

The Maxwell stress tensor obeys DEC, so every eigenvalue satisfies |p_i| ≤ ρ_EM. For a radial field p_r = −ρ_EM; for a (random-direction) tangential field p_r = +ρ_EM. A *coherent* purely tangential field cannot be spherically symmetric (hairy-ball theorem), but that does not affect the conclusion.

### 1e. Corollary 3 (ρ < 0 somewhere, bound with m_max) — CONFIRMED WITH CAVEATS

This needs 2m < r (or the unconditional argument in 1b) and the explicit exclusion of σ < 0 Israel shells, which violate p_r ≤ ρ in any regularisation (see 1a).

### 1f. Novelty — UNRESOLVED

The proof is two lines and is close in spirit to Bondi (1964), Guven–Ó Murchadha (1999) and Narlikar–Das (1976, central versus surface redshift). The authors should check these and Karageorgis–Stalker (2008) before calling it new. Nothing I found contradicts it.

---

## 2. Thin-shell stability (§2.3) and the Einstein shell (§2.4)

**Formula — CONFIRMED.** I independently Taylor-expanded m(x) = m₀ + m₁(x−R) + m₂(x−R)²/2, with m₁ = −8πRp and m₂ = −8πp + 16πβ²(σ + p), in V = 1 − (M/m + m/(2x))². This gives V(R₀) = V'(R₀) = 0 and ∂V''/∂β² = 2(3α+1)/R² > 0, so stability holds for β² *above* the threshold. The threshold is exactly β²_min = (1−α)(3α²+2α+1)/(4α²(1+3α)). The table values are correct, and β²_min = 1 at α = 0.395295, Z = 2.52976.

**ERROR (interpretation) / UNRESOLVED.** Z = 2.53 corresponds to **R = 2.370M**, which is *inside* the photon sphere 3M. β²_min at R = 3M is 0.366 < 1. The sentence "согласуется с BLP 1991: … устойчивость лишь вне радиуса, большего радиуса фотонной сферы" therefore contradicts the report's own result.

I generalised the computation to a Schwarzschild interior of mass M₋. The causal-stability radius is 2.370, 2.443, 2.523, 2.653, 2.861 and 2.985 M₊ for M₋/M₊ = 0, 0.3, 0.5, 0.7, 0.9 and 0.99. It stays below 3M₊ and reaches it only in the test-shell limit. The repository's literature note quotes BLP's abstract as "always larger than the radius of the circular photon orbit". Either BLP mean the photon orbit of the inner black hole (3M₋), with which the present results agree, or there is a genuine conflict. **Action:** read BLP and fix the sentence. The causal-stability statement should also say it applies to barotropic EoS and radial perturbations only.

**§2.4 minor error.** For a particle (Einstein) shell, μ = m₀√(1 + L²/R²). The stable branch ends at marginal stability, W' = W'' = 0, where R₀ = 4.086, L = 1.954, M = 0.958, **Z = 1.3723, v = 0.4314c**. The report quotes "Z ≤ 1.36 at v = 0.42c". It should be Z < 1.372 (supremum, not attained).

---

## 3. Thin-layer pointwise-DEC ODE (§2.2) — CONFIRMED

* **Derivation.** In a layer of thickness ε, ρ̃ ~ 1/ε while g = O(1). Dividing the conservation law by dy/dr = ρ̃/R gives dg/dy = 2λ − (y + g)/(2(1−y)), with λ = p_t/ρ. This is exactly the report's ODE at λ = 1.
* **Closed form.** g/√(1−y) = 6 − 5√(1−y) − 1/√(1−y) satisfies the ODE and g(0) = 0 (SymPy). Its zeros are at u = √(1−y) = 1 and 1/5, so y = 24/25.
* **Finite thickness.** Full integration with constant ρ̃ = 10, 10², 10³, 10⁴ returns g = 0 at y = 0.95108, 0.95914, 0.95992, 0.95999, approaching 24/25 from below. Max |p_r|/ρ = 0.08/ρ̃ → 0.
* **Optimality.** For λ < 1 the zero of g occurs earlier, by comparison, so 24/25 is the thin-layer maximum.
* **Caveat (scope).** "Поверхностное DEC — правильный предел поточечного DEC" is true only for *thin* walls. Thick walls beat Z = 5 (see §4).

---

## 4. Nested DEC shells and the "Einstein plateau" (§4)

**Mathematics — CONFIRMED.**

* The structure is: a DEC shell at y: 0 → 0.96 (p/σ = 1.000), a vacuum gap to r = 1.2 r_in (y = 0.8), a plateau with y = 0.8, p_r = 0, p_t/ρ = 1.000000 (checked from conservation), out to r_out = 1.2√2, then Schwarzschild. Integrating Φ directly gives Z = 10.0000 and M/r_in = 0.678823 = 0.48√(Z/5).
* Junctions are legitimate: p_r = 0 on both sides of each density jump, so no surface layer is needed.
* Within the family the optimum is as claimed. With M/r_in = (y₀/2)·exp[(ln Z + ½ln(1−y₀))/n_p] and n_p = y_p/(2(1−y_p)), the optimum is at y_p = 0.8. As a function of y₀ this has a *maximum* at y₀ = 0.8 and decreases on [0.8, 0.96], so the endpoint y₀ = 0.96 is optimal.
* The authors' optimiser with K = 2, 3, 4, 6 gives 0.7200, 0.6880, 0.6828, 0.6802, approaching 0.6788 from above. The K = 8 solution contains degenerate zero-mass shells (σ ~ 10⁻¹⁵ with p/σ = 5.9); these are not real DEC violations, but the absolute constraint tolerance can hide them.

**ERROR FOUND — "Einstein plateau" is not an Einstein cluster.** In a real Einstein cluster (particles on circular geodesics), p_t = ρv²/2 with v² = m/(r − 2m). That requires y < 2/3 (photon sphere) and p_t ≤ ρ/2. At y = 0.8 one needs v² = 2, i.e. **v = √2·c**. The plateau is therefore DEC-saturated anisotropic matter (p_t = ρ) with no known microphysical realisation. The same is true of the p = σ inner shell, since particle shells have p/σ ≤ 1/2, i.e. Z < 3.

With genuinely particulate matter (inner shell p/σ ≤ 1/2, so y₀ ≤ 8/9; cluster y ≤ 2/3, n = 1), the same construction needs **M/r_in ≈ 1.48** at Z = 10, not 0.68. The sentence "анизотропное давление обходит предел Z ≤ 5 без нарушения DEC" is correct mathematically but should not suggest that a known form of matter does it.

**Optimum.** Starting from the authors' own warm start, I ran `optimise_controls_dec` and polished it with strict margins (|p_t|/ρ ≤ 0.995 and |p_r| ≤ ρ(1 − 2·10⁻⁴)). Re-checked with RK at 256 sub-steps, it gives **M/r_in = 0.66824 at Z = 10.00004**, with residuals g_end = −7·10⁻⁷ and max(|g| − ρ̃) = 1.7·10⁻⁷. Those residuals are far too small to explain the 1.6% gap. The structure is a DEC layer to y = 0.959, a short vacuum gap, then a thick layer with **radial tension** (p_r ≈ −0.3 to −0.4 ρ), p_t ≈ ρ and y falling from 0.93 to 0.65.

So the true DEC infimum is ≤ 0.668, and REPORT_06 E9's "not strictly feasible" caveat can be lifted at about the 10⁻⁶ level. Relabel 0.680 as the "nested-shell / p_r = 0 family optimum". The "×1.37" cost becomes about ×1.35.

**Further caveat.** The DEC-saturated inner shell at Z = 5 needs β² ≥ 4.75 for radial stability (§2.3), so the composite configurations are radially unstable for causal EoS unless the thick layer stabilises them. This is not discussed in §4.

---

## 5. BPS shell (§5) — CONFIRMED WITH CAVEATS

* **Junction.** p = (1/8πR)[(Rf'/2 + f)/√f − 1] equals the report's formula identically. p = 0 ⇔ Q = ±M, giving α = 1 − M/R and rest mass 4πR²σ = M (SymPy).
* **Mistuning.** For a dust shell momentarily at rest with Q = M(1+δ), the exact R̈ = AA' expanded to first order is R̈ = δ(1−α)/(Rα) = **δ(Z−1)/R**.
* **Numbers.** E = 4.693·10²⁶ V/m, E/E_S = 3.55·10⁸, R_min = 7.09·10⁸ m, M = 4.32·10⁵ M_⊙, B = 1.565·10¹⁸ T, B_W = 1.092·10²⁰ T (B/B_W = 0.014), M = 2.42·10²⁷ kg = 1.28 M_J, ρ̄ = 2.4·10²⁶ kg/m³. All agree.
* **Caveat (time base).** "5 cm in 10 years" gives |δ| ≲ 2.5·10⁻³⁶ if the 10 years are shell or interior proper time, and 2.5·10⁻³⁴ if exterior time. State which.
* **Caveat (Schwinger).** E_S is not a sharp threshold: the rate goes as exp(−πE_S/E). For macroscopic volumes and long times, discharge is significant at E ~ E_S/10–E_S/30, so R ≥ 7·10⁸ m is optimistic by one to two orders of magnitude.
* **Caveat (monopoles).** For the magnetic version, the extremal mass of one Dirac monopole is 1.27·10⁻⁷ kg ≈ 7·10¹⁹ GeV. GUT monopoles (10¹⁶–10¹⁷ GeV) are super-extremal, so the shell must be built from bound "grains" of monopoles plus neutral ballast, each grain with q = m. Otherwise it is not stress-free.
* **Note.** A neutral payload enters the p = 0 condition only at O(m_h²) once Q is tuned to the total ADM mass, so the payload is not a problem here (it is for §6).

---

## 6. Majumdar–Papapetrou cage (§6) — CONFIRMED WITH CAVEATS

* **Identity.** For the static observer in ds² = −U⁻²dt² + U²dx², |a|² = |∇U|²/U⁴ and the orthonormal-frame field |E|² = |∇U|²/U⁴ (SymPy, exact).
* **Conversion.** B/a = 1/(c√(4πε₀G)) = **38.708 T per m/s²**, so 379.6 T at 1 g.
* **Multipoles.** For centrosymmetric icosahedral orbits, the invariant harmonics are ℓ = 0, 6, 10, 12, 16, … (odd ℓ, including ℓ = 15, vanish), and each invariant space is one-dimensional for ℓ < 30. That makes single-probe-direction cancellation legitimate. I checked over 500 random directions: the residual is ≤ 7·10⁻¹⁶ for ℓ = 6 (I+D) and for ℓ = 6, 10 (I+D+ID).
* **Weights.** All positive: (0.02976, 0.03214) and (0.02255, 0.00877, 0.01847).
* **Fitted exponents.** 5.0148, 8.9994, 11.0003.
* **Mass table.** Fully reproduced: 0.318 / 0.762 M_⊙; 27.6 / 46.6 M_J; 14.7 / 22.8 M_J.
* **Tolerance formula.** δ = (a_lim/c²)ρ₀³Z²/(2m_i) is correct for a radial displacement of one mass.

**Caveats**

1. **Must be magnetic.** For an *electric* cage the habitat field at 1 g is **1.14·10¹¹ V/m** (lethal; 4–5 orders above air breakdown), and sub-solar extremal electric black holes discharge (Gibbons). Say so explicitly. Like §5, this needs magnetic monopoles.
2. **Neutral payload breaks equilibrium.** A neutral payload (human plus habitat) inside the cage breaks the MP no-force equilibrium, and black holes cannot be made super-extremal to compensate (unlike the BPS shell, §5). Either the payload must itself be extremally charged (same sign), or the black holes must be actively held. A Newtonian estimate puts the time for an off-centre payload to push black-hole positions beyond the quoted 10⁻¹³ m tolerance at well under a second.
3. **Tolerance meaning.** The tolerance is a *coordinate* displacement and an instantaneous criterion. MP equilibria are neutral, so any velocity error drifts linearly; the requirement is really on velocity, e.g. ≲ 3·10⁻²² m/s over 10 years.

---

## 7. Kerr ISCO (§7) — CONFIRMED WITH CAVEATS

Checked with 40-digit mpmath:

* χ* = 0.9975195 (1 − χ* = 2.4805·10⁻³), r_ISCO = 1.25774M, E = 0.68553, L = 1.40998.
* The BPT closed form for u^t matches the metric to 10⁻³⁸.
* v_ZAMO = 0.56869c, lapse N = 0.12157 (so Z = γ/N = 1.2157/0.12157 = 10).
* Z_ISCO = 10.7909 at χ = 0.998 and 6.0812 at χ = 0.99.
* Marck eigenvalues (1.00522, −1.50783, 0.50261)/M², giving M_min = 1.126·10⁵ M_⊙.
* Escape: Δφ = 0.27632, mass ratio 1.3183; static escape at Z = 10 needs γ(1+v) = 19.950.

**Caveats**

1. **Marginal stability.** At the ISCO itself the orbit is marginally stable, and any inward perturbation leads to a plunge. At χ = 0.998 the Z = 10 orbit sits at r = 1.25491M, only 0.018M (Boyer–Lindquist) outside the ISCO = 1.23697M. Say that station-keeping is required.
2. **Tidal convention.** The criterion uses the full 2 m head-to-foot difference. A ±1 m about the centre-of-mass convention gives 8.0·10⁴ M_⊙. State the convention.
3. For M = 1.13·10⁵ M_⊙ the orbital period is 8.4 s at infinity (0.84 s proper). This is harmless, but worth noting.

---

## 8. Other claims

### 8a. §8 breathing shells — claim CONFIRMED numerically; "[ДОКАЗАНО]" NOT justified

The report's argument shows p(R_min) > p_static(R_min). That part is correct: at A = 1 with A' > 0, −m' exceeds its static value because M/m² − 1/(2R) > 0 ⇔ m < R. But the report then *asserts* ⟨Z⟩ ≤ Z(R_min) without proof, and the pointwise version of that inequality is **false**.

Two facts:

* At fixed R, dt/dT = √(f + Ṙ²)/(f√(1 + Ṙ²)) ≥ 1/√f(R). Motion *increases* the instantaneous dilation.
* On DEC trajectories (λ = p/σ = +1 on [1, 1.179], −1 beyond, q_min = 0.34) the instantaneous dt/dT reaches **1.057·Z(R_min)**. The period average ⟨Z⟩ = 1.484 is still below Z(R_min) = 1.515.

**Adversarial test** over *general elastic* shells, which is broader than the report's particle shells: p/σ = λ(R), |λ| ≤ 1, eight free nodes, plus f_min; 150 × 2 Nelder–Mead runs.

* Max ⟨Z⟩ = **4.986 < 5**, with Z(R_min) = 4.992.
* Max ⟨Z⟩/Z(R_min) = 1 − 1.3·10⁻⁹, approached only by small oscillations.

So the conclusion holds and is in fact stronger than stated, but the label should be [ПРОВЕРЕНО ЧИСЛЕННО] until a proof is supplied. The particle-shell numbers (max ⟨Z⟩ = 1.34) are consistent with the static stable supremum of 1.372 found in §2.

### 8b. §9 — CONFIRMED

For ds² = −α²dt² + dr² + r²dΩ², SymPy gives p_r = α'/(4πrα) and 8πp_t = (rα')'/(rα) exactly. The proof is valid: rα' ≥ 0 vanishes at both ends, so (rα')' < 0 somewhere and ρ + p_t < 0. It also covers distributional jumps (a negative surface p_t with σ = 0).

Statement 5: with s = √(1−2x), 16πR(σ₂ + p₂) = (3s + 1)(s − 1)/s < 0. This is correct; σ₂ < 0 already violates WEC on its own.

### 8c. §11 — CONFIRMED

E = αmc² (Killing energy). Ballistic escape needs γα ≥ 1, i.e. γ ≥ Z. Lift work is (1 − α)mc². "Dissolving the well" with BPS requires the shell to move from R ≈ 1.11M to R ≫ M, so the installation changes size by orders of magnitude; worth stating.

### 8d. §1 — CONFIRMED

An independent control-parametrised SLSQP run (controls u ∈ [−1, 1], N = 100) gives 1.81295 versus sinh 2/2 = 1.81343 and 3.33729 versus sinh 3/3 = 3.33929; the small shortfall is discretisation error. The other numbers check: x = 4.499914, T = 17.437 yr, t = 174.37 yr, γ = 45.010, d = 85.27 ly, a = 17.437 g, e^{4x} = 6.56·10⁷, ring radius 95.90 ly. The Pontryagin sketch is informal but the result is standard.

### 8e. §12 — minor numerical inconsistencies (text vs. code)

* "R ≳ 27 km": the code gives 25.08 km.
* "средняя плотность ~10²⁶ kg/m³": the code gives 3.6·10²⁵.
* "Casimir gap ~10⁻¹⁷ m": the computation gives 3.4·10⁻¹⁸ m.

All are within order of magnitude, but the text should match `round1_results.json`.

---

## 9. Category errors and overclaims

1. **"No horizons" used for "monotone areal radius"** (Theorem 2, Corollary 4, the code docstring). See 1b. This is the most important logical gap.
2. **"Einstein cluster" for p_t = ρ matter at y = 0.8.** This makes DEC-saturated, physically unrealised matter sound like ordinary particle dynamics (§4).
3. **"[ДОКАЗАНО]" for Statement 3** (§8), where the decisive inequality is unproven and its pointwise form fails.
4. **Internal contradiction in the BLP comparison** (§2.3).
5. **Safety criteria.** The B ≤ 2 T criterion is applied without noting that an electric realisation gives ~10¹¹ V/m at 1 g (§6). Also, ISCO marginal stability (§7) and neutral-equilibrium drift (§§5–6) are not tied back to the "safe" verdicts.
6. **Clock bookkeeping.** I found no coordinate-time/proper-time confusion. Z = u^t for orbits, Z = U(0) for MP, Z = 1/√f for static shells, and ⟨Z⟩ = ∮dt/∮dT are all invariant ratios against static clocks at infinity. The only ambiguity is the time base of the BPS tolerance (§5).

## 10. ERROR FOUND and UNRESOLVED, concretely

* **E1 (Theorem 2 scope / Corollary 4).** There is an explicit static, horizon-free, asymptotically flat counterexample with ρ ≥ 0 and p_r ≤ ρ (indeed p_r ≤ 0) in which α_cav = 0.01 while 1 − 2M/r_in = 0.78. The ADM mass can be made arbitrarily small. Fix: either keep 2m < r explicitly or add radial NEC, and state the result in the unconditional m_max form.
* **E2 (§2.3).** The threshold Z = 2.53 sits at R = 2.37M < 3M, contradicting the claimed agreement with BLP's "outside the photon orbit". The formula itself is right.
* **E3 (§2.4).** The stable Einstein-shell supremum is Z = 1.372 at v = 0.431c, not 1.36 at 0.42c.
* **E4 (§4).** The y = 0.8, p_t = ρ plateau is not an Einstein cluster; it would need v = √2·c. With real particle matter the same construction needs about 1.48 r_in.
* **E5 (§8).** "[ДОКАЗАНО]" is unjustified: the step ⟨Z⟩ ≤ Z(R_min) is not proven, and instantaneous dt/dT can exceed Z(R_min) by about 6%. The conclusion itself survives numerically.
* **U1.** The NEC-constrained sharp constant is open. The evidence (optimiser 0.49507; E19 data; a proof for p_r ≤ 0) points to α_cav ≥ √(1 − 2M/r_in), i.e. M ≥ 0.495 r_in at Z = 10.
* **U2.** The BLP statement needs checking against the paper (3M₊ or 3M₋?).
* **U3.** Novelty of Theorem 2 is unverified (check Bondi 1964; Narlikar–Das 1976; Karageorgis–Stalker 2008).
* **U4 (resolved for the record, not an error).** The DEC infimum is ≤ 0.66824 at Z = 10 (feasible to about 10⁻⁶), below the 0.680 "optimum", so the E9 caveat can be lifted. The exact infimum is still open.

## Appendix: reproduction notes

All scripts are in the session scratchpad (not in the repo), under `critic/`:

* `t1` SymPy Einstein tensor in proper-length gauge and the §9 check
* `t2` throat counterexample
* `t3` saturating family
* `t4`, `t21` stability, including the M₋ generalisation
* `t5` thin layer
* `t6`, `t7` nested shells
* `t8b`, `t8c` DEC continuum polish
* `t9` BPS
* `t10` MP
* `t11` Kerr
* `t12b`, `t17*` breathing shells
* `t14` kinematics
* `t15b` Einstein shell
* `t16` the generalised r-bound
* `t18`, `t19`, `t20` NEC-constrained searches

The essential formulas for the counterexample are given in §1b, so it can be rebuilt without these files.
