# Phase 8 — Transitions, Wigner, NIST spectra, SpectrumView (spec)

## What and why
The level ladder gives wavelengths; this phase gives the lines their strengths
and checks both against measurements, without ever querying anything live.

- **Transition strengths** (`analytic/transitions.py`): the radial dipole
  matrix element `R = ∫ R'·r·R·r² dr` by high-order Gauss quadrature over the
  EXACT closed-form radials, with the error estimated by node doubling
  (`NUMERICAL`). From R: absorption oscillator strength
  `f = (2/3) dE (l_max/(2l+1)) |R|²`, Einstein A
  `A = (4/3) α³ dE³ (l_max/(2l'+1)) |R|² / t_au`, and radiative lifetimes
  `τ = 1/ΣA`. The formulas are extracted once (`f_from_radial_dipole`,
  `A_from_radial_dipole`) so the analytic and numerical paths share one copy.
  Selection rules are structural: `Δl ≠ ±1` returns a disclosed zero.
- **Wigner 6j** (`analytic/wigner.py`): the Racah formula with half-integer
  arguments carried as doubled integers, so no floating-point comparison ever
  decides a triangle condition. Returns a plain `float` — the module's one
  stated provenance exception, since a 6j is an algebraic constant like π.
  This closes the fine-structure gap: j-resolved rates
  `A(n'l'j' → nlj) = (4/3) α³ dE³ (2j+1) {6j}² l_max |R|²`, validated by the
  sum rule `Σ_j (2j+1) {6j}² = 1/(2l'+1)`, which ties every fine-structure
  multiplet back to its gross rate.
- **Numerical dipole** (`numerics/dipole.py`): the dipole over solved radials,
  `∫u_a·u_b·r dr`. Both states on **one common grid** (`r_max` sized by the
  largest n of the whole line list, box radius `10(n+1)²`, solved channels
  cached) — the single thing most likely to be silently wrong, and the test
  that catches it feeds a pair whose natural boxes differ.
- **Line lists** (`spectra.py`): `SpectralLine`/`LineList` with optional
  `einstein_a` and `oscillator_strength`, `transition_lines` (hydrogenic,
  gross or j-resolved) and `screened_transition_lines` (GSZ orbitals, strengths
  over the numerical radials, `APPROXIMATION` because the model error
  dominates the grid error — labelling it `NUMERICAL` would be a lie about the
  dominant term). Lines carry vacuum wavelengths in nm and energies in eV.
- **Vendored NIST** (`data/nist_{h,d,he,li,na}_i.json`): measured wavelengths
  with citations and retrieval dates. `load_reference`/`compare_lines` match
  computed lines to references with two separate scales: a coarse association
  window (does this transition exist at all) and the disclosed pass bar
  (3e-5 gross / 1e-5 fine for hydrogenic; 5% with a 25% window for screened,
  whose valence lines sit percent-scale off a fitted model).

## Server
`GET /api/spectrum` (system, `n_max` ∈ [2,10] for hydrogenic, `fine_structure`,
`intensities`). `LineModel` carries wavelength/energy plus optional A and f
with provenance; `ComparisonModel` carries each matched reference residual;
`intensity_note` names any case where strengths were asked for and withheld.

## Web
Full component port from the reference: log-λ axis (`lib/axis.ts`), series
names/colors (`lib/spectrum.ts`), NIST residual panel with a ppm/percent
residual axis and a headline count (`lib/explain.ts`), log-compressed
intensity bars with the stated liberty ("bar height is the spontaneous
emission *rate* A, not a population-weighted brightness" — `lib/liberties.ts`),
pointer hover readout (`PlotHover`), domain-based zoom with a follower
residual panel (`PlotZoom`, `lib/zoom.ts`), ASCII→typography notation
(`lib/mathText.tsx`). The within-n fine-structure components (microwave group)
stay in the data and the count of axis-hidden lines is disclosed, with a
toggle to include them; the two-regime axis problem otherwise stands, exactly
as the reference left it. Absorption, profiles and thermal populations are
**not** ported (they are the thermal-light phase); the `emissivity` field
exists and stays null.

## Deliberately neglected
Population modelling (Boltzmann/Saha), Voigt profiles, optical depth, curve of
growth, absorption (the whole thermal-light layer); hyperfine-resolved line
strengths; Dirac-corrected rates; live NIST queries.

## Validation
- `test_transitions`: f(1s→2p) = 0.4162 (Bethe-Salpeter); A(2p→1s) = 6.27e8
  s⁻¹ (NIST); τ(2p) = 1.60 ns; selection-rule zeros; monotone f(1s→np).
- `test_transitions_fine`: Σ_j A over the lower j returns the gross rate; the
  D-line doublet ratio A(3p₃/₂→3s)/A(3p₁/₂→3s) = 2; τ(2p₁/₂) = τ(2p₃/₂).
- `test_wigner`/`test_wigner_3j`: the zero-argument closed form `{½ 1 ³⁄₂; 1 ½ 0}² = 1/6`,
  column symmetries, exact zeros on triangle violations, the sum rule.
- `test_numeric_dipole`: a pure Coulomb potential reproduces the exact
  closed-form dipole integrals (⟨2p|r|1s⟩ = 1.290266 bohr to ~1e-3), the
  common-grid pair check, and grid-halving convergence.
- `test_spectra`/`test_spectra_intensities`: line structure, selection rules,
  the strongest-line anchors (Lyman-α overall, Balmer-α among Balmer), A>0 on
  every served line, intensities off leaves wavelengths unchanged.
- `test_screened_intensities`: hydrogenic-limit anchor through the numerical
  engine; alkali strengths in the order-unity band (not percent digits, which
  GSZ cannot support); Na 3p→3s strongest among lines above the closed core.
- `test_server_intensities`: the flag honoured, provenance tiers surviving the
  boundary, positronium's μ=½ halving A relative to hydrogen through the API.
- Server spectrum tests: NIST citation + all-within comparison for H, honest
  absence for systems without vendored data.
