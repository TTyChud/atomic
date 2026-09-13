# Phase 12 — Isosurfaces (plan)

Port of `../AtomSim` Phase 25 (+ Phase-26 HF-3D seam for `hf_isosurface` over
HTTP), `s/atomsim/atomic/`. Engine → schemas → route → view, with that
phase's AtomSim tests ported nearly verbatim.

## Prerequisite check (done, all green before this phase)

- `hf_atom.py` has `evaluate_hf_state(z, N, n, l, m, pos, basis, config,
  exchange, pauli)` with tier read off the solve; `hf_radial` same flags;
  `sampling.py:sample_hf_density`, `plane.py:hf_plane_grid` same shape.
- `screened_atom.py:evaluate_screened_state`, `analytic/wavefunction.py:
  evaluate_state` exist with `(pos)` evaluator protocol returning
  `.values` + `.provenance.assumptions`.
- `server/app.py` has `ManyElectronRequest` (model/config/exchange/pauli +
  422 validator), `SampleRequest`/`PlaneRequest`, `_many_electron_target`,
  `_hf_view_target` (occupancy + pauli + antisymmetry refusals),
  `is_atom_key`/`_screened_element`/`_resolve_system` dispatch,
  `job_systems`/`job_models` maps, `_dispatch`, `_finished_result`.
- `tests/test_hf_views.py` has 4 `@needs_isosurface` tests importing
  `atomic.isosurface.hf_isosurface` with skip
  "atomic.isosurface arrives in Phase 12"; `tests/test_server_hf_views.py`
  has 1 `@pytest.mark.skip` for the iso counterfactual badge.
- Web has `api/client.ts` (`createSampleJob`/`createPlaneJob`,
  `decodeFloats`/`decodePositions`), `api/types.ts` (`SampleMeta`/
  `PlaneMeta`/`JobMeta`), `lib/colormap.ts:phaseColor`, `state/store.ts`
  with model/config/exchange/pauli threading from Phase 11.

## Part 1 — Engine (no shape changes elsewhere)

1. `src/atomic/numerics/marching_tets.py` (new): copy reference verbatim
   (no `atomsim` imports inside — pure numpy). Covers `_kuhn_tetrahedra`,
   `_TET_EDGES`/`_EDGE_OF`, `_tet_cases`, `Mesh`, `marching_tets`,
   `_unflatten`, `_inside_centroid`, `enclosed_volume`, `surface_area`,
   `edge_use_counts`, `connected_components`.
2. `src/atomic/isosurface.py` (new): copy reference with
   `s/atomsim/atomic/` on the 5 imports (`analytic.angular`,
   `analytic.hydrogen`, `analytic.wavefunction`, `atoms`, `hf_atom`,
   `numerics.marching_tets`, `provenance`, `screened_atom`). Keeps
   `GRID_SIZES`, `_BOX_CAPTURE`, `_TAIL_SAMPLES`, `_EVAL_CHUNK`,
   `Isosurface`, `default_half_width`, `_density_on_grid`, `_cell_volume`,
   `_captured`, `radial_mass`, `_fit_box`, `solve_level`, `fraction_above`,
   `_phase_at`, `_build`, `isosurface`, `screened_isosurface`,
   `hf_isosurface`.

## Part 2 — Server (inside `create_app`, atomic conventions)

3. Imports: `from atomic.isosurface import (GRID_SIZES, Isosurface,
   hf_isosurface, isosurface, screened_isosurface)`.
4. `IsoRequest(ManyElectronRequest)` after `PlaneRequest`: `n/l/m`,
   `fraction = PydanticField(default=0.9, gt=0.0, lt=1.0)`,
   `resolution: int = 96`, `basis`, `system`, plus `_resolution_is_offered`
   validator against `GRID_SIZES`.
5. `IsoMetaModel` after `PlaneMetaModel`: kind/vertex/triangle_count,
   channels, target/enclosed/outside/level/escaped/mesh_volume/voxel_volume/
   area, components, half_width, resolution, axis_unit, n/l/m/basis/system/
   model (`str = "hydrogenic"` to match this tree's sample/plane default —
   reference uses `"gsz"`; keep tree-consistent, tests assert geometry not
   the default string), label, provenance.
6. `POST /api/jobs/isosurface`: `_validate_state`, then
   `hf_target = _hf_view_target(req) if req.model == "hf" else None`
   (reuses occupancy/pauli/antisymmetry 422s); `jobs.create()`,
   `job_systems[job.id] = req.system`, `job_models[job.id]` =
   `"hf"`/`"screened"`/`"hydrogenic"` following the sample/plane pattern
   in this tree; three `work(progress)` branches calling the three engine
   functions with `target_fraction=req.fraction`, `resolution`,
   `basis`, `progress` (+ `config`/`exchange`/`pauli` on HF,
   `Z`/`mu_ratio` from `_resolve_system` on hydrogenic).
7. `_iso_meta(surf, system_key, model_key)` + `_iso_channel_payload`
   (vertices float32 default, triangles uint32, phase float32; unknown →
   422 "it has vertices, triangles, phase").
8. `job_meta` response_model += `IsoMetaModel`, dispatch on
   `isinstance(res, Isosurface)`; `job_data` dispatch on same via
   `_iso_channel_payload`.

## Part 3 — Web

9. `web/src/lib/frame.ts` (new): `PHYSICS_TO_SCREEN`.
10. `web/src/lib/isoSurface.ts` (new) + `isoSurface.test.ts` (new): port
    reference verbatim (`buildSurfaceColors`, `enclosedCaption`,
    `surfaceExtent`, `componentsCaption`).
11. `web/src/components/IsoSurface.tsx` (new): port verbatim (three.js
    `BufferGeometry`, vertex colors, `computeVertexNormals`,
    `DoubleSide`, `PHYSICS_TO_SCREEN`).
12. `web/src/api/types.ts`: add `IsoMeta` (mirror `IsoMetaModel`), extend
    `JobMeta = SampleMeta | PlaneMeta | IsoMeta | HFLevels`.
13. `web/src/api/client.ts`: add `IsoParams`/`createIsoJob`
    (`{ resolution: 96, ...params }` → `/api/jobs/isosurface`),
    `getIndexChannel`/`decodeIndices` (multiple-of-12 guard). Full 3D
    store/view wiring (surfaceMode, CloudView both-mode) stays deferred —
    this phase lands engine + route + geometry component + client, matching
    the Phase-11 build record ("deferred to Phase 12: hf_isosurface and the
    isosurface route/view").

## Part 4 — Tests (verbatim, `s/atomsim/atomic/`)

14. `tests/test_marching_tets.py` (new, 214 ln): table-derivation checks,
    sphere volume/area convergence, refinement, interpolation, plane exact,
    watertight, outward orientation, welding, two-ball components, empty
    meshes, non-3D refusal, chunk invariance.
15. `tests/test_isosurface.py` (new, 393 ln): level solve, closed-form 1s,
    sampler cross-check, box growth/refusal/escape disclosure, mesh-vs-voxel,
    convergence, watertight, topology + sign/phase, provenance + dual error
    bars, grid-size offers, screened + HF surfaces.
16. `tests/test_server_iso.py` (new, 168 ln): end-to-end 1s geometry,
    default channel bytes, unknown-channel 422, provenance survival,
    screened tier, muonic scale, refusal matrix, fraction monotonicity.
17. Unskip: the 4 `@needs_isosurface` tests in `test_hf_views.py` and the 1
    skipped iso-badge test in `test_server_hf_views.py` go green on arrival
    (remove skip markers — import now succeeds; keep the `try/except` guard
    harmless or drop it).

## Gates

`ruff check .`; `pytest tests/test_marching_tets.py
tests/test_isosurface.py tests/test_server_iso.py tests/test_hf_views.py
tests/test_server_hf_views.py` green; `cd web && npm test && npm run build`
green; `server/schemas` ↔ `api/types` in sync for the three iso channels.
