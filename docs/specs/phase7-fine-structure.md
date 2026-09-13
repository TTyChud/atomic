# Phase 7 — Fine structure, Dirac, Zeeman, Stark, hyperfine (spec)

## What and why
Beyond the gross Bohr ladder, real hydrogen-like levels carry structure the
coarse model hides. This phase adds five closed-form refinement modules, each
carrying its own provenance tier, and makes them composable in one levels
response:

- **Fine structure** (`analytic/fine_structure.py`): the Pauli α² combination
  of spin-orbit + relativistic kinetic + Darwin terms, `ΔE = -(μ'Z⁴α²/2n⁴)(n/(j+½) - 3/4)`.
  `APPROXIMATION` at real α; `COUNTERFACTUAL` when α is altered (the seam the
  constants lab threads).
- **Dirac** (`analytic/dirac.py`): the exact closed-form Dirac-Coulomb energy
  `E(n, j)`, rest energy subtracted, cancellation-free evaluation. `EXACT` for
  its model, with the omitted physics (Lamb/QED, hyperfine, finite nucleus,
  recoil) named in assumptions and their Lamb-dominated scale as the error
  estimate. Its teaching payload: energy depends on `(n, j)` only, so
  `2s₁/₂ = 2p₁/₂` exactly — reality splits them (the Lamb shift), and the model
  says so. Supercritical `Zα ≥ j+½` is rejected, never returned as NaN.
- **Zeeman** (`analytic/zeeman.py`): the full Breit-Rabi crossover. For each
  `(l, m_j)` with both j = l±½ present, a 2×2 block (fine-structure or Dirac
  diagonal + linear-Zeeman coupling through ⟨S_z⟩) solved in closed form —
  no eigensolver, zero numerical error. Stretched states and l=0 are exactly
  linear. Each sublevel carries both the low-field label (j) and the high-field
  label (m_l, m_s), so the good-quantum-number handoff is shown, not asserted.
  `APPROXIMATION` (diamagnetic B² and gₛ−2 quantified in the error estimate).
- **Stark** (`analytic/stark.py`): parabolic (n₁, n₂, m) sublevels with the
  electric quantum number k = n₁ − n₂, through second order: linear
  `(3/2)nkF/(Zμ)` (hydrogen's l-degeneracy signature) + quadratic
  `-(n⁴/16)(17n² - 3k² - 9m² + 19)F²/(Z⁴μ³)`. `APPROXIMATION` always, no α
  dependence (honest: a non-relativistic gross-structure effect), error scale
  ∝ F³ via F/F_ion. n² sublevels per shell.
- **Hyperfine** (`analytic/hyperfine.py`): Fermi contact, s-states only,
  `A = (2/3) g_e g_I (m_e/m_p) α² (μ/m_e)³ Z³/n³` with the **fixed** proton mass
  in the nuclear magneton (a per-nucleus mass is a factor-of-2 bug for
  deuterium; tests lock it out). Nuclei are keyed by preset (h, d, t) with
  CODATA moments; He-4 (I=0) is honestly unsplit; ps and mu-h are honestly
  unavailable with a reason; generic Z has no identified nucleus.

## Integration
`/api/levels` gains `fine_structure`, `dirac`, `b_field` (Tesla), `e_field`
(MV/m), `hyperfine`, and `alpha` (the counterfactual α seam). Fine levels are
`(n, l, j)` entries; Zeeman sublevels ride on their fine level, Stark sublevels
on the gross level, hyperfine F-shells at response level. Screened systems
return `ScreenedLevelsModel` before any of this, as before.

## Deliberately neglected
Dirac radial wavefunctions and field-split clouds/planes (energies only);
spectrum line-splitting for any of these effects; the diamagnetic B² term
(built into the error estimate); Stark third order and field-ionization
resonances; the l > 0 hyperfine dipolar channel; simultaneous B and E fields.

## Validation
- `test_fine_structure`: 2p splitting vs measured 10.969 GHz within the g≠2
  scale; Z⁴ scaling; α-quadratic COUNTERFACTUAL behaviour.
- `test_dirac`: α→0 recovers Bohr; published 1s₁/₂ leading behaviour
  −1/2 − α²/8; O(α⁴) agreement with the perturbative module with the residual
  shrinking 16× when α halves; exact (n,j) degeneracy; supercritical rejected.
- `test_zeeman`: B=0 recovery bit-for-bit; Landé low-field slope g_J μ_B m_j;
  Paschen-Back integer high-field slopes; trace invariance; stretched states
  exactly linear; sublevel counts 4l+2 (l≥1) / 2 (l=0).
- `test_stark`: zero field recovers Bohr; n² count; parabolic constraint;
  linear slope (3/2)nk; traceless linear fan; n=1 polarizability 9/2 a.u.;
  ±m degeneracy; Z⁴ and μ³ quadratic scaling (not μ⁴).
- `test_hyperfine`: five experimental anchors — H 1s 1420.405751 MHz (21 cm),
  H 2s 177.556, D 327.384, T 1516.701, He-3 −8665.65; centroid theorem; F
  ranges for I = ½ and 1; negative-moment order inversion.
- Server: dirac EXACT + degeneracy through the route, 422s for supercritical/
  negative fields/bad α, screened systems ignore the new params, hyperfine
  split ≈ 5.874e-6 eV.
- Web: URL round-trip `fs=1&dirac=1&b=2.5&ef=30&hf=1`; setters clear cached
  levels; `getLevels` threads all controls.
