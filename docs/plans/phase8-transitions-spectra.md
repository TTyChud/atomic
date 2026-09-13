# Phase 8 — Transitions, Wigner, NIST spectra, SpectrumView (plan)

Ported stage-for-stage from `../AtomSim` (its agent.md Phase 9 = specs
2026-07-24-phase13-transition-strengths, 2026-07-25-phase14-line-intensities,
phase15-fine-structure-line-strengths, phase16-screened-line-strengths),
`s/atomsim/atomic/`. The user asked for the *same level as AtomSim at the
same stage*, and chose the full SpectrumView component port with the
absorption/profile/thermal parts stubbed until the thermal-light phase.

Status at completion (2026-09-10): engine, data, server and web in; full gate
green — `ruff check .` clean, `pytest` 520 passed, `cd web && npm test`
169 passed (22 files), `npm run build` passes.

1. **Engine**: copy `analytic/wigner.py`, `analytic/transitions.py`,
   `numerics/dipole.py`, `spectra.py`; `s/atomsim/atomic/`. Strip the thermal
   machinery from `spectra.py` (the `atomic.populations` import block,
   `_thermal_state`/`_hydrogenic_thermal`/`_screened_thermal`, the `thermal`
   parameter and the emissivity wiring): the `SpectralLine.emissivity`,
   `lower_fraction` and `LineList.thermal` fields stay as `None` placeholders
   so the thermal-light phase can populate them without reshaping the schema.
   Fix `load_reference` to read `atomic.data`. Vendor the five NIST JSONs +
   `data/__init__.py`.
2. **Screened path**: `screened_atom.py` has evolved in the reference since
   this repo's Phase-6 snapshot (the dipole channel cache,
   `screened_dipole_integral`, the one-box-per-list grid sizing, the density
   fixes). Copy it wholesale; the Phase-6 tests validate the deltas.
3. **Engine tests**: port `test_wigner`, `test_wigner_3j`, `test_transitions`,
   `test_transitions_fine`, `test_numeric_dipole`, `test_spectra`,
   `test_spectra_intensities`, `test_screened_intensities` (imports only).
   `test_spectra_thermal` is NOT ported (thermal-light phase).
4. **Schemas + route**: `LineModel` (+ `emissivity` placeholder) and
   `ComparisonModel` into `server/schemas.py`; `SpectrumResponse` beside the
   other response models in `app.py`; `GET /api/spectrum` with the hydrogenic
   branch (`n_max` ∈ [2,10], tolerance 3e-5 gross / 1e-5 fine) and the
   screened branch (5% pass bar, 25% association window, strengths over the
   numerical radials).
5. **Server tests**: port `test_server_intensities.py` wholesale and the two
   reference spectrum endpoint tests (+ one n_max-validation test) into
   `test_server.py`.
6. **Web**: copy `lib/{axis,hover,zoom,spectrum,liberties,explain,mathText,
   nucleus}.ts(x)` and their tests; extend `lib/classical.ts` to the
   reference's superset (trajectory helpers used by its test). Copy
   `components/{PlotHover,PlotZoom}.tsx`. Port `SpectrumView.tsx` with the
   absorption/profile/thermal UI blocks cut (ZoomPanel, profile controls,
   curve-of-growth, absorption controls, LTE toggle/sliders all deferred);
   keep intensity bars, hover, zoom + follower, NIST residual panel, the
   within-n axis toggle and the caveat disclosures. Extend `Field.tsx`
   (Toggle `why`/`disabled`/`disabledReason`, Slider `anchor`/`disabled`/
   `atRest`, ControlGroup `tone`) and `Disclosure` (`tone`) with optional
   props rather than swapping in the reference's restyled controls, so the
   existing views keep their look. Types mirror the schemas; `getSpectrum`
   client; store `spectrum` + `intensities` (default on) in `INVALIDATED`;
   `spectrum` ViewMode in urlState (`int=0` URL param, default on so it is
   only written when turned off), TopBar, Controls and App wiring.
7. **Gate**: `ruff check . && pytest` green (520). `cd web && npm test &&
   npm run build` green at 155 tests before the last additions
   (`SpectrumView.test.ts`, the store spectrum stub + one loader test);
   re-run to confirm.
8. **Docs**: this spec + plan; index in `docs/README.md`; agent.md status.

Delta to carry forward: the reference repo's copy of this phase's specs
records two things found while building — the two-regime fine-structure axis
(deferred there, deferred here identically, disclosed in the view) and the
screened resonance-line anchor correction (A ∝ dE³ means keV-scale core
transitions beat valence lines; the test asserts Na 3p→3s strongest only among
lines ending above the closed core).
