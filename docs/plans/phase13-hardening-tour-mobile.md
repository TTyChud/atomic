# Phase 13 — Server hardening, tour, mobile (plan)

Port of `../AtomSim` Phases 15–16, `s/atomsim/atomic/`, env `ATOMIC_`
prefix. Engine → schemas → route → view, AtomSim tests ported nearly
verbatim (adapted only where this tree's names differ: env prefix, logger
name, store key names, colormap signature).

## Prerequisite check

- `matplotlib` in `pyproject.toml` dependencies ✓ (thumbnails renderer).
- `plane_grid` / `get_system` / `ATOM_KEYS` / `atom_for_key` /
  `aufbau_configuration` / `hf_valence_ionization_energy` /
  `solve_hartree_fock` / `transition_lines(system, n_max,
  fine_structure)` — signatures match the reference; `tour_claims.py`
  ports with import renames only.
- `server/app.py` already has `_configure_logging`, `_web_dist`
  (`ATOMIC_WEB_DIST`), `_job_worker_count`, mount + both log lines,
  `is_atom_key` / `_screened_element` / `_resolve_system`,
  `_many_electron_target` / `_hf_view_target`, `job_systems`/`job_models`
  maps. Missing: `_lifespan`, `_build_rate_limiter`, `_client_key`,
  middleware, thumbnail route.
- Web has `Badge`, `Legend`, `Notation` (Unicode-based, no KaTeX needed),
  `stateLabel`, `subshellAvailable`, `THUMBNAIL_LIBERTY`,
  `lib/nucleus.ts` (`NucleusMode`, `nucleusSphere`, `nucleusCaption`),
  `phaseColor(phase, brightness)`, `resolveModel`, `URL_DEFAULTS`,
  `currentUrlState`, `parseAppUrl`/`serializeAppUrl` + tests.
- Missing web: `tours/`, `Tour*.tsx`, `NarrowNotice`, `GalleryStrip`,
  `gallery.ts`, `systemKind.ts`, `thumbnailUrl`, tour/store/URL keys.

## Part 1 — Backend hardening

1. `src/atomic/server/ratelimit.py` (new): verbatim copy (no imports to
   rename — stdlib only).
2. `src/atomic/server/thumbnails.py` (new): `s/atomsim/atomic/` on the
   `plane` + `systems` imports; keep `matplotlib.use("Agg")` +
   `lru_cache(512)` + `GAMMA = 0.5`.
3. `src/atomic/tour_claims.py` (new): `s/atomsim/atomic/`; `_TOUR_DIR`
   resolves to this repo's `web/src/tours` via `parents[2]` (same
   layout). `CLAIM_KINDS` identical tuple.
4. `server/app.py`:
   - imports: `asyncio`, `math`, `asynccontextmanager`, `JSONResponse`,
     `TokenBucket`/`DEFAULT_CAPACITY`/`DEFAULT_PERIOD`,
     `render_thumbnail`.
   - `_lifespan` (new) + `FastAPI(..., lifespan=_lifespan)`.
   - `_build_rate_limiter` (new, `ATOMIC_RATE_LIMIT*` env) +
     `_client_key` (new, `ATOMIC_CLIENT_IP_HEADER`) + HTTP middleware
     (POST `/api/jobs/*` only → 429 + `Retry-After` + warning with key).
   - `GET /api/thumbnail/{n}/{l}/{m}` (`size ∈ [32,256]`, PNG +
     `Cache-Control: public, max-age=86400`).
5. `tests/conftest.py` (new): session autouse fixture setting
   `ATOMIC_RATE_LIMIT=off` (adapted env name, same docstring intent).
6. Tests (verbatim + env/logger renames): `test_ratelimit.py`,
   `test_server_ratelimit.py` (`ATOMSIM_`→`ATOMIC_`,
   logger `atomic.server.app`), `test_server_static.py`
   (`ATOMSIM_WEB_DIST`→`ATOMIC_WEB_DIST`, same logger rename),
   `test_thumbnails.py`, `test_tour_claims.py`.

## Part 2 — Test-debt sweep (earlier phases, impl already present)

Port verbatim (`s/atomsim/atomic/`): `test_import.py`, `test_oscillator.py`,
`test_display_window.py`, `test_screened_total_density.py`,
`test_hf_view_performance.py`, `test_server_curve_of_growth.py`,
`test_server_profile.py`, `test_server_thermal.py`,
`test_spectra_thermal.py`. Any failure is a real finding about this tree,
not a port artifact — fix the tree, not the test.

## Part 3 — URL state (round-trip contract)

Extend `lib/urlState.ts` (`UrlState`, `URL_DEFAULTS`, `currentUrlState`,
`parseAppUrl`, `serializeAppUrl`) with: `thermal` (`lte`), `temperatureK`
(`tk`, [1e2,1e6]), `logNe` (`ne`, [4,22]), `profile` (`prof`),
`logResolvingPower` (`rp`, [2,7]), `profileZoom` (`zoom` lo,hi),
`absorption` (`abs`), `logColumn` (`col`, [14,26]), `nucleusMode`
(`nucleus` enum), `surfaceMode` (`surf` enum), `isoFraction` (`iso`
open interval), `tour`/`step` (step only alongside tour). Defaults mirror
the store (`temperatureK 10000`, `logNe 13`, `logColumn 20`,
`nucleusMode "marker"`, `surfaceMode "cloud"`, `isoFraction 0.9`,
`tour null`, `step 0`). Existing keys untouched; existing round-trip
tests must stay green.

## Part 4 — Store

- `export INVALIDATED` (add one keyword) for `tours/apply.ts`.
- New state: `nucleusMode` (+`setNucleusMode`), `surfaceMode`/
  `isoFraction`/`iso`/`isoStatus`/`isoProgress` (+`setSurfaceMode`,
  `setIsoFraction`, `loadIso` over `createIsoJob`/`getJobMeta`/
  `getChannel`/`getIndexChannel`), tour slice (`tourId`, `stepIndex`,
  `savedState: UrlState | null`, `inviteOpen`, `completedTours`,
  `startTour`, `goToStep`, `exitTour`, `finishTour`, `dismissInvite`)
  with `startingMemory = readMemory()` seeding.
- `ISO_FRACTIONS = [0.5, 0.75, 0.9, 0.95, 0.99]` exported (reference
  lives in store).
- `loadIso` payload threads `model/config/exchange/pauli` like
  sample/plane; `setSurfaceMode`/`setIsoFraction` reset `iso`/`isoStatus`
  (fraction stays a question to re-ask of the next orbital).

## Part 5 — Web views

7. `web/src/tours/` (new): `types.ts`, `registry.ts`, `step.ts`,
   `seen.ts`, `spotlight.ts` verbatim; `apply.ts` adapted to this tree's
   store keys (`hfLevels`, `ghost`/`ghostStatus`, `forceLaw`/
   `forceStatus`, `whatif`/`whatifStatus` cleared explicitly);
   4 JSONs verbatim.
8. `TourInvite`/`TourMenu`/`TourPanel`/`TourSpotlight` verbatim
   (`Notation` resolves to this tree's Unicode renderer).
9. `NarrowNotice.tsx` verbatim (`MIN_WIDTH = 900`) + brand rename
   `atomsim`→`atomic` in the notice copy.
10. `GalleryStrip.tsx` (adapted: `hf`→`hfLevels`), `lib/gallery.ts` +
    `lib/systemKind.ts` verbatim, `thumbnailUrl` in `api/client.ts`.
11. `Controls.tsx`: `nucleus-picker` (new `Choice<NucleusMode>` over
    `NUCLEUS_MODES`) + `data-tour` anchors on system/model/compare/
    exchange/pauli/view/l/basis pickers.
12. `CloudView.tsx`: nucleus sphere + caption (via `lib/nucleus.ts`,
    `systems` table), `surface-controls` block (`data-tour`), iso fetch
    effect, `IsoSurface` render + enclosed/level/escaped/components HUD;
    sample only when the cloud shows.
13. `InfoPanel` (`state-card`), `LevelsView` (`dirac-toggle`,
    fine-structure anchor), `SpectrumView` (`spectrum-options`,
    `curve-of-growth-toggle`), `WhatIfView` (`const-sliders`),
    `ForceLawView` (`force-preset`): anchor attributes only.
14. `TopBar`: `TourMenu` entry. `App`: narrow gate + `TourInvite` +
    `GalleryStrip` + `TourPanel` + `TourSpotlight` in this tree's layout
    classes. `main.tsx`: `?tour=&step=` microtask through `startTour`.
15. `index.css`: tour/gallery/thumb/narrow/surface styles in this tree's
    idiom (no new `text-transform: uppercase`).
16. Docs index: `docs/README.md` + `agent.md` Phase 13 (DONE + counts).

## Part 6 — Web tests

`tours/{anchors,apply,prose,registry,seen,spotlight,step}.test.ts`
(`apply.test.ts` adapted to atomic `tourReset` keys; rest verbatim),
`NarrowNotice.test.ts`, `lib/{gallery,systemKind,frame}.test.ts`
(`frame.test.ts` new — `frame.ts` landed in Phase 12 untested),
`uppercaseSafety.test.ts` (adapted `ALLOWED` to this tree's two
word-only legends). `api/client.test.ts` untouched.

## Gates

`ruff check .`; full `pytest` green (incl. 429s with default-off
conftest + isolated limiter tests); `npm test && npm run build` green;
`IsoMeta`/`IsoRequest` unchanged; every tour claim holds; every step
round-trips the URL; every spotlight anchor resolves in component source.
