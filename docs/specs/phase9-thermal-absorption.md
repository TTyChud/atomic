# Phase 9 — Thermal light and absorption (spec)

## What and why
Phase 8 gave the lines their strengths; this phase gives the gas its
populations and the lines their shapes, so a spectrum stops being a list of
rates and becomes something an instrument would actually see.

- **Populations** (`populations.py`): level and ion populations from
  Boltzmann factors and the Saha equation. The ionization balance is solved
  with `scipy.optimize.brentq` on the log of the Saha ratio (a bracket on the
  log keeps the root find away from overflow at extreme temperatures).
  Partition functions truncate the hydrogenic level ladder at a density-
  dependent limit — a bound level inside a plasma must be smaller than the
  mean inter-particle spacing (the Planck-Larkin / occupation-probability
  idea, here in its simplest disclosed form). Provenance is `MODEL`: the
  physics is standard LTE, but the truncation is a modelling choice.
- **Line profiles** (`broadening.py`): Gaussian (Doppler, from the thermal
  velocity of the emitting species), Lorentzian (natural width ΣA plus an
  optional van der Waals/stark term kept as a hook), and the Voigt
  pseudo-profile (the convention of using the Lorentz FWHM where the
  Voigt-Hjerting function wants the Gaussian HWHM, applied once, in one
  place, and said so). Normalization is by unit area everywhere; the caller
  scales by what the line actually radiates.
- **Optical depth and transfer** (`transfer.py`): column density from the
  absorbing-state population, τ(ν) = N·f·φ(ν) with the standard
  πe²/(m_e c) normalization, the emergent intensity I = B·(1 − e^−τ)
  against a continuum (the slab solution with no scattering), and the
  curve of growth — equivalent width W(τ₀) through the linear,
  flat and square-root regimes, computed by numerical integration of the
  profile rather than the classical asymptotes.

## Spectra integration
`spectra.py` regains the thermal machinery stripped in Phase 8:
`transition_lines(..., thermal=...)` populates `SpectralLine.emissivity` from
the Boltzmann/Saha state (`_thermal_state` for hydrogenic lists,
`_screened_thermal` over GSZ orbitals), and `lower_fraction` — the population
fraction in the lower state of each line — which the absorption path needs.
`LineList.thermal` records the conditions the lines were computed under, so a
spectrum carries its own state description.

## Server
- `GET /api/spectrum` grows the thermal parameters: `thermal` (LTE toggle),
  temperature, electron density, and the profile choice with its resolving
  power. The response carries `ThermalInfo` (the populations' summary),
  `LineWidthInfo` (per-profile widths at a reference line) and `ProfileInfo`
  (the synthesized full-range curve) when thermal is on.
- `GET /api/absorption`: same gas against a continuum — per-line optical
  depths, the synthesized absorption profile, and the regime each line sits
  in.
- `GET /api/curve-of-growth`: equivalent width against column density over a
  user-supplied range, with the regime classification per point.

## Web
- `SpectrumView.tsx` regains the cut blocks: the LTE toggle with
  temperature/electron-density sliders (`describeTemperature`,
  `describeDensity` readouts), the bar-quantity switch (rate vs emissivity —
  `BarQuantity`, `intensityScale(lines, quantity)`), the log-compressed
  full-range trace (`profileScale`, the `SPECTRUM_PROFILE_LIBERTY`
  disclosure), the zoomed profile panel with its own linear axis, and the
  curve-of-growth toggle. The `tourId` props the reference adds are dropped:
  the tour system is a later phase.
- `AbsorptionView.tsx` and `CurveOfGrowthView.tsx` ported whole, sharing the
  regime colors/labels (`REGIME_COLOR`/`REGIME_LABEL`) so both views speak
  about linear/flat/saturated the same way.
- Types mirror the schemas; `getAbsorption`/`getCurveOfGrowth` clients; store
  gains `thermal`, `profile`, `absorption`, `curveOfGrowth` state with
  `absorptionData`/`curveOfGrowth` in `INVALIDATED`.

## Deliberately neglected
The tour system (`tourId`, TourSpotlight/TourMenu/TourPanel), later-phase
store drift in the reference (`triangles`, HF flags), non-LTE statistical
equilibrium, Stark/van der Waals broadening laws (the hooks exist; the
physics arrives with the density-compare phase), and live anything.

## Validation
- `test_populations`: Saha equilibrium sanity (hydrogen half-ionized at
  ~Bethe's 15800 K under the truncation model — a MODEL number, tested
  as monotone + bracketed, not to digits); Boltzmann population of n=2 vs n=1
  at 10000 K matching the exact factor; partition-function truncation
  shrinking the ladder as density rises; populations summing to one.
- `test_broadening`: unit area on each profile to integration error;
  Doppler width at a stated temperature matches the closed form; natural
  width ΣA(2p) matches the lifetime; Voigt between its Gaussian and Lorentz
  limits.
- `test_transfer`: τ from column density reproduces the closed form in the
  linear regime; emergent intensity I = B(1−e^−τ) at τ = 1 gives
  1 − 1/e of the continuum; curve-of-growth W grows linearly, flattens, then
  √τ — tested as regime classification, not asymptote digits.
- `test_absorption_engine`: hydrogen Lyman-α dominates the optical depth at
  LTE 10000 K; a fully-ionized gas has no absorption (population × fraction
  = 0); the absorption profile's peaks sit on the line wavelengths.
- Server: thermal on/off changes the response shape (thermal/profile blocks
  present iff asked); absorption and curve-of-growth endpoints validate
  `n_max`, return regime labels and monotone W(N); provenance tiers survive
  the boundary.
- Web: `SpectrumView.test.ts` extended with the quantity-switch and
  profile-scale cases from the reference (154-line file); all prior tests
  hold.
