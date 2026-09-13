# Phase 13 — Server hardening, tour, copy, mobile (spec)

Adapted from `../AtomSim` Phases 15–16 (rate limiting, thumbnails/static
hosting, guided tours with claim checks, mobile truth-telling), plus the
thumbnail gallery strip that consumes the thumbnail endpoint.
`s/atomsim/atomic/`, with `ATOMIC_` env prefix per this tree's
`ATOMIC_WEB_DIST` convention. Code is current truth; this explains what
and why.

## What and why

Three hardening problems and one honesty problem, all about the boundary
between the instrument and the world. The job API is a compute oracle: one
POST buys seconds of pinned CPU, which is nobody's problem on a laptop and
the whole attack surface on a public URL. The gallery needs pictures that
cost nothing. The tours quote numbers in prose, and prose rots. A phone
cannot hold the instrument, and squeezing it until the plots lie is worse
than refusing.

## Engine deltas

- `server/ratelimit.py` (new, 113 ln): per-client fractional token bucket,
  clock-injected, `DEFAULT_CAPACITY = 80` / `DEFAULT_PERIOD = 240.0`
  (1/3 job/s sustained; burst sized off the widest honest click-storm: 36
  tiles × 2 jobs + 1 HF solve, pinned by
  `test_default_burst_covers_widest_click_through`), `DEFAULT_MAX_CLIENTS
  = 4096` with refill-based pruning. No distributed state: correct for the
  single-machine deploy, wrong the moment there are two.
- `server/thumbnails.py` (new, 38 ln): `render_thumbnail` —
  `lru_cache(512)` inferno PNGs of hydrogenic plane density at
  `GAMMA = 0.5`, row-flipped so +z is up. Navigation aids, not measurement
  surfaces; mirrors `densityColor` gamma and the `inferno` LUT the browser
  uses. Screened atoms 422: a screened plane costs seconds even at 96 px
  and the strip asks for up to 36 at once.
- `tour_claims.py` (new, 134 ln): `CLAIM_KINDS = ("energy_eV", "mean_r_pm",
  "wavelength_nm", "ionization_eV")`, `load_tours` (reads the same JSON the
  browser bundles — single source of truth), `iter_claims` (step-state
  inheritance restricted to resolver keys), narrow resolvers pinned to
  closed forms. Adding a kind is one function + one `CLAIM_KINDS` entry on
  each side, each with its own test.

## Server (`server/app.py`, inside `create_app`)

- `_lifespan`: shuts the job executor down on exit instead of leaving
  worker threads behind the event loop.
- `_build_rate_limiter`: on by default (`ATOMIC_RATE_LIMIT=off/0/false`
  disables; `ATOMIC_RATE_LIMIT_BURST`/`ATOMIC_RATE_LIMIT_PERIOD`
  override). On-by-default because forgetting it on a public host is worse
  than tripping it on a laptop.
- `_client_key`: charges the rightmost `X-Forwarded-For`-style entry under
  an explicitly named `ATOMIC_CLIENT_IP_HEADER`; unnamed headers ignored
  (a caller-chosen string must never buy a fresh bucket). Falls back to
  `request.client.host`.
- HTTP middleware meters only `POST /api/jobs/*` (reads and static stay
  free). Refusal is 429 with integer `Retry-After` and a detail naming the
  charged key in the warning log — the line that tells popularity apart
  from a collapsed-behind-proxy bucket.
- `GET /api/thumbnail/{n}/{l}/{m}`: hydrogenic plane PNG, `size ∈ [32,
  256]`, `Cache-Control: public, max-age=86400`. Validates state, system,
  basis; screened systems 422 via the shared `_screened_element` refusal.
- `tests/conftest.py`: suite-wide session fixture switching the limiter
  off (`ATOMIC_RATE_LIMIT=off`), so the suite's own POST rate never
  throttles itself into physics-looking failures.

## Web

- `tours/` (data + logic, JSON verbatim): `types.ts` (`CLAIM_KINDS`
  mirrors the Python tuple — the structural test asserts the match),
  `registry.ts` (4 tours, `FLAGSHIP_TOUR_ID = "hydrogen-honestly"`),
  `step.ts` (`stepState` resets from `URL_DEFAULTS`, never patches),
  `apply.ts` (`tourReset`: step state + `resolveModel` + `INVALIDATED` +
  explicit clears for payloads outside it), `seen.ts` (dismissed vs
  completed memory in localStorage, fail-silent), `spotlight.ts`
  (zero-area anchors ring nothing).
- `TourInvite` (one-time bar, never a modal), `TourMenu` (top bar picker
  with done marks), `TourPanel` (narration docked under the stage, never
  over the evidence), `TourSpotlight` (fixed ring, re-synced every render
  + ResizeObserver; never intercepts clicks).
- Store tour slice: `tourId/stepIndex/savedState` (entry snapshot,
  restored on exit), `inviteOpen/completedTours` (seeded once from
  `readMemory`), `startTour/goToStep/exitTour/finishTour/dismissInvite`.
  Tour deep links (`?tour=id&step=k`) run through `startTour`, never a raw
  state landing.
- 17 spotlight anchors as `data-tour` literals on existing controls
  (system/model/compare/exchange/pauli/view/n/l/basis/fine-structure/
  dirac/nucleus/const-sliders/force-preset/spectrum-options/
  curve-of-growth/state-card/surface-controls), pinned by
  `tours/anchors.test.ts` source scan.
- `NarrowNotice` (`MIN_WIDTH = 900`, JS-gated so a phone never builds a
  WebGL context): the squeezed-plot refusal stated plainly, with the live
  width and the five fidelity tiers.
- `GalleryStrip` + `lib/gallery.ts` + `lib/systemKind.ts` + `thumbnailUrl`:
  the n² tile row under the stage; hydrogenic tiles show cached PNGs,
  screened atoms keep working buttons with the picture missing; HF tiles
  disable where the configuration holds no orbital. Three-state kind check
  (`hydrogenic`/`screened`/unknown-yet) so the first render never fires a
  request the gate exists to prevent.
- URL growth for the round-trip contract (`registry.test.ts` runs every
  step's state through `serializeAppUrl`/`parseAppUrl`): `lte/tk/ne`,
  `prof/rp/zoom`, `abs/col`, `nucleus`, `surf`, `iso`, `tour/step`.
  Defaults omitted; junk dropped, never thrown.
- Iso view completion (deferred from Phase 12): `surfaceMode`/
  `isoFraction`/`iso`/`isoStatus` in store + URL, `loadIso` over the
  Phase-12 job channels, `surface-controls` block + `IsoSurface` render +
  enclosed/level/escaped/components disclosures in `CloudView`.
- Nucleus completion: `nucleusMode` in store + URL, `nucleus-picker` in
  `Controls`, marker/true-scale sphere + caption in `CloudView` via the
  existing `lib/nucleus.ts`.
- `uppercaseSafety.test.ts`: `text-transform: uppercase` stays restricted
  to word-only selectors (CSS uppercasing maps α→A, ²→2).

## Deliberately out

`GhostOverlay` (classical 3D overlay — no tour anchor needs it),
`ShowPhysics`/`PhysicsBody`/`physics/content.ts` (per-view KaTeX layer —
this tree's `Notation` is Unicode-based and the views carry their own
leads), `analytics.ts`/`startup.ts` (deploy phase), `luts.ts`/`gen_luts.py`
(deploy phase), `plot.ts`/`levels.ts`/`spark.ts` (earlier-phase view
internals this tree reimplemented), component tests for diverged views.

## Expected physics

Unchanged engine numbers; new pins are arithmetic and wiring: bucket exact
cases on a fake clock, click-storm burst coverage, sustained-rate ratio,
429 + integer `Retry-After`, reads never charged, proxy-header trust rules,
static mount override + both disclosures + logging-guard behaviour, PNG
magic + cache hits + per-state bytes + full 422 matrix, resolver kinds
against Bohr/closed-form/HF-Koopmans values, and every tour claim holding
against the engine that draws it.
