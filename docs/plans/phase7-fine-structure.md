# Phase 7 — Fine structure, Dirac, Zeeman, Stark, hyperfine (plan)

Ported stage-for-stage from `../AtomSim` (its agent.md Phase 8 = specs
2026-07-23-phase9-dirac, phase10-zeeman, phase11-stark,
phase12-hyperfine), `s/atomsim/atomic/`, in the reference's own order.
One engine slice at a time, suite green after each.

1. **Engine**: copy the five analytic modules with import renames. Reword
   `fine_structure.py`'s `refinement` (it cited "the planned Phase 3 flagship";
   here Dirac already exists at `analytic/dirac.py`). Point each module's
   docstring at `docs/specs/phase7-fine-structure.md`. No `System` change:
   hyperfine carries its `_NUCLEI`/`_UNAVAILABLE` tables keyed by preset.
   `constants.py` already has `B0_TESLA` and `E0_V_PER_M`.
2. **Engine tests**: port `test_fine_structure.py`, `test_dirac.py`,
   `test_zeeman.py`, `test_stark.py`, `test_hyperfine.py` verbatim (imports
   only). Run `pytest tests/test_{fine_structure,dirac,zeeman,stark,hyperfine}.py`.
3. **Schemas + route** (`server/app.py`; this repo keeps the levels models
   beside the endpoint, as the reference does): replace `LevelEntry` with
   `GrossLevelModel` (+ optional Stark `sublevels`), add `FineLevelModel`
   (+ optional Zeeman `sublevels`), `StarkSublevelModel`,
   `ZeemanSublevelModel`, `HyperfineLevelModel`, `HyperfineShellModel`, and
   the `LevelsResponse` echo fields. Rewrite the endpoint: `alpha` validation
   (0, 0.5], `b_field`/`e_field` ≥ 0 → 422; dirac `ValueError` → 422;
   hyperfine shells via `hyperfine_report`, one honest unavailable reason at
   n=1, not n_max copies. Also: `_resolve_system` now maps a bad generic-Z
   (`z0`) to 422, matching the reference contract.
4. **Server tests**: port the reference `/api/levels` tests (gross shape,
   fine ordering, dirac degeneracy + supercritical 422, altered-α
   counterfactual echo, zeeman fan + negative-field 422 + screened inert,
   stark fan + independence + negative-field 422 + screened inert, hyperfine
   21 cm split + flag absence + spin-0 + unavailable + screened inert).
   Repo delta: atomic accepts any generic Z, so `z99` is a valid 200 and
   supercriticality is reached directly via `z200&dirac=true` → 422.
5. **Web**: `api/types.ts` mirrors the schemas (GrossLevel/FineLevel/
   ZeemanSublevel/StarkSublevel/HyperfineShell + echoed params); `client.ts`
   `getLevels(system, nMax, fineStructure, alpha?, dirac, bField, eField,
   hyperfine)`; `state/store.ts` gains the five controls, each clearing
   cached levels (`setFineStructure(false)` also resets `dirac`);
   `lib/urlState.ts` serializes `fs`, `dirac` (only with fs on), `b` (only
   with fs on), `ef` (not `e`: charge-multiplier collision), `hf`;
   `components/LevelsView.tsx` renders the controls, the ladder over
   `gross`, a µeV FineFan for the selected shell (Dirac caption naming the
   Lamb shift), ZeemanFan per fine level under a field, a meV StarkFan, and
   the hyperfine 21 cm caption.
6. **Docs**: spec + plan (this file); index in `docs/README.md`; update the
   agent.md status line.
7. **Gate**: `ruff check . && pytest`, `cd web && npm test && npm run build`.
