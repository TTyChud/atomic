# Phase 9 — Thermal light and absorption (plan)

Closes the parity gap deliberately left in Phase 8: port the reference's
thermal-light layer (its specs 2026-07-26-phase17-population-modelling,
2026-07-26-phase18-line-profiles, 2026-07-26-phase19-optical-depth,
2026-07-27-phase20-absorption-spectrum) wholesale with `s/atomsim/atomic/`,
restore the stripped thermal machinery in `spectra.py`, and re-grow the
SpectrumView blocks cut in the Phase 8 port.

Status at completion (2026-09-10): engine, server and web in; full gate
green — `ruff check src tests` clean, `pytest` 665 passed,
`cd web && npm test` 172 passed (22 files), `npm run build` passes.

1. **Engine**: copy `populations.py`, `broadening.py`, `transfer.py`
   wholesale with `s/atomsim/atomic/`. Their only intra-package imports are
   `provenance` and `spectra`; the scipy dependency (`brentq`,
   `special.voigt_profile`) is already in the project.
2. **Thermal restore in `spectra.py`**: re-copy the reference's thermal
   machinery (the `atomic.populations` import block, `_thermal_state`,
   `_hydrogenic_thermal`, `_screened_thermal`, the `thermal` parameter, the
   emissivity/lower_fraction wiring and `LineList.thermal`). The Phase 8
   diff showed no other divergence in the file, so the restore is wholesale.
3. **Engine tests**: port `test_populations.py`, `test_broadening.py`,
   `test_transfer.py`, `test_absorption.py` with imports only; they pass
   unmodified.
4. **Server (deltas, not wholesale)**: atomic's `schemas.py` and `app.py`
   are phase-appropriate subsets, so port only the thermal additions:
   `ThermalModel`, `LineWidthModel`, `ProfileModel`, `AbsorptionResponse`
   models, `CurveOfGrowthModel` into `schemas.py`; the `thermal`/`profile`
   parameters on the spectrum endpoint, `_resolve_thermal`, `emitter_mass`
   (imported from `atomic.systems` — the reference keeps it in its spectra
   module) and the `/api/absorption` + `/api/curve-of-growth` routes into
   `app.py`.
5. **Server tests**: port `test_server_absorption.py` wholesale; the
   spectrum endpoint tests in `test_server.py` updated to the thermal
   response shape.
6. **Web**: extend `api/types.ts` with `ThermalInfo`, `LineWidthInfo`,
   `ProfileInfo`, `AbsorptionInfo`/`AbsorbingLineInfo`,
   `CurveOfGrowthInfo`/`GrowthRegime`; `getSpectrum` gains the thermal
   params, `getAbsorption`/`getCurveOfGrowth` clients added. Copy
   `components/AbsorptionView.tsx` and `CurveOfGrowthView.tsx` whole.
   Restore `SpectrumView.tsx` to the reference version (1045 lines) with
   the reference's `tourId` props dropped — the tour system is later-phase.
   Store: `thermal`/`profile`/`absorption`/`curveOfGrowth` state, loaders,
   and `absorptionData`/`curveOfGrowth`/`profileZoom` in `INVALIDATED`.
   Do NOT copy the reference store wholesale (it carries later-phase drift:
   isosurface `triangles`, HF flags).
7. **Gate**: `ruff check src tests && pytest` green (665). `cd web && npm
   test && npm run build` green at 172 tests (22 files).
8. **Docs**: this spec + plan; index in `docs/README.md`; agent.md status.

Delta to carry forward: the reference's `Toggle`/store carry `tourId` wiring
into everything they touch; we strip the prop but keep the placement (the
props sit exactly where the tour anchors will go), so the tour phase can add
them back without re-reading the reference.
