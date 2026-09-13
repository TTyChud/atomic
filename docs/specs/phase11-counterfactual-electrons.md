# Phase 11 — Counterfactual electrons (spec)

## What and why

Phase 10 built Hartree-Fock and deliberately stripped the counterfactual
machinery that grew on top of it in the reference (`../AtomSim` Phases 22 and
24 = our Phase 11, per its agent.md Phase 12). This phase restores it. The
phase exists because Phase 10 put a real non-local exchange term in the model
for the first time: before it, "turn exchange off" had nothing to turn off —
GSZ is a fitted central field with no exchange term anywhere in it. Now the
toggle removes something, the difference is a number, and the number is the
exchange energy.

Two toggles, two tiers of disclosure, and the distinction between them is the
teaching payload:

- **Exchange off (the Hartree model, `COUNTERFACTUAL`)**: electrons that repel
  but are distinguishable. The wavefunction is a product instead of an
  antisymmetrized determinant. The Pauli principle is *not* switched off —
  occupancies stay capped at `2(2l+1)` and the configuration is untouched —
  and the disclosure says so in as many words, because a badge reading
  `COUNTERFACTUAL` without naming *which* counterfactual is decoration. The
  `(q_a - 1)` self-interaction factor stays: it is classical electrostatics,
  true in either model, and folding it into the exchange bucket would put a
  self-interaction error inside a number reported as exchange energy.
- **Pauli off (configuration collapse, `COUNTERFACTUAL`)**: the occupancy cap
  is lifted and every electron falls into the 1s. The atom stops having
  shells, so it stops having chemistry, and the number on the screen says by
  how much. Pauli off *forces* exchange off — antisymmetry is what the
  exclusion principle is, so `pauli=False, exchange=True` is refused with an
  explanation, never silently corrected, never computed. Term structure is
  *replaced* rather than omitted in the disclosure: `subshell_terms` counts
  distinct spin-orbital assignments, which is Pauli's own combinatorics, so
  `1s^N` spans no terms rather than many.

## Engine deltas (port of the reference's post-Phase-21 machinery)

All of this already exists in the reference and is restored wholesale,
`s/atomsim/atomic/`, minus the reference's later drift (HF 3D threading
arrives with this phase only where Phase 10's stripped code has the seams; the
3D view wiring itself is the same phase in the reference but our Phase 11
combines its Phases 12 and 13, so 3D is planned separately below).

- `numerics/hf_terms.py`: unchanged — `exchange_operator`,
  `exchange_apply`, the coefficients all exist. The `exchange=False` branch
  is expressed as an empty `ExchangeOperator(terms=())`, a shared frozen
  `_NO_EXCHANGE`, not a branch inside `matvec`.
- `numerics/hartree_fock.py`: `exchange: bool = True` on `fock_operator`,
  `_fock_parts`, `solve_channel`, `orbital_energy`, `total_energy_direct`,
  `kinetic_and_potential`, `scf`. The `(q_a - 1)` factor in
  `direct_potential` does not move. The two-route energy agreement check in
  `hf_atom.solve_hartree_fock` catches a half-applied flag for free: exchange
  lives in the functional for route 1 and in the operator for route 2, so a
  mismatch raises `HFConvergenceError` on every atom.
- `numerics/hf_atom.py`:
  - `solve_hartree_fock(z, n_electrons, config, exchange=True, pauli=True)`,
    all in the `lru_cache` key; the result `key` names the calculation
    (`-nopauli`/`-nox` suffixes) so the three models never share a cached
    solve. `HFResult` gains `exchange: bool` and `pauli: bool` as fields, not
    prose to be parsed.
  - `pauli=False` validates with `validate_config(config, pauli)` (the cap
    check drops; the `n > l` check stays — that is what makes `(n, l)` name a
    radial function at all), resolves the configuration to `1s^N` via
    `aufbau_configuration(n_electrons, pauli=False)` when none is given.
  - `_energy_assumptions(config, z, exchange, pauli)`: conditional
    disclosures. The alteration line leads the assumption list
    (`COUNTERFACTUAL: ...`), the Pauli-intact line follows for the Hartree
    model, and the multi-term/open-shell lines are *replaced* under collapse
    rather than dropped. Fidelity is read off the flags
    (`COUNTERFACTUAL` unless both true), never asserted as a literal below
    the solve — the Phase-26 lesson from the reference: a tier hardcoded one
    layer below the solve produces exactly the badge-lie the flag exists to
    prevent.
  - `hf_exchange_energy(z, n_electrons, config)`: solves both models itself,
    on the same mesh, and returns `E_HF - E_Hartree` as a `Quantity`.
    Negative (stabilizing), exactly zero for helium and any closed single-s
    shell — bit-exact, not approximately, because the exchange branches are
    empty loops. `COUNTERFACTUAL` provenance with the mesh spread as the
    error bar, stated as arithmetic, not truth-distance.
  - `collapsed_variational_energy(z, n_electrons)`: the closed-form check.
    `zeta* = Z - (5/16)(N-1)`,
    `E(zeta) = N(zeta²/2 - Zζ) + [N(N-1)/2](5ζ/8)`. The SCF optimizes the
    whole radial function, so `E_SCF <= E(zeta*)` and close. `COUNTERFACTUAL`
    despite being closed form — the tier is truth-distance, not arithmetic
    precision.
  - `pauli_collapse(z, n_electrons)`: solves the real atom and the collapsed
    one on the same mesh, returns `PauliCollapse` (binding change, both mean
    radii, radius ratio, the variational bound). Both solves stay here — a
    client free to difference two jobs is free to difference a warm solve
    against a cold one.
  - `hf_mean_radius` already exists; gains the no-error-bar rule (the solve
    estimates spread in hartree; a length is not that).
- `atoms.py`: `aufbau_configuration`, `validate_config`, `is_ground` already
  carry the `pauli: bool = True` parameter (landed in Phase 10's port of
  `atoms.py`). Only `subshell_terms`' behaviour above capacity is exercised by
  new tests; it already raises.

## Expected physics, i.e. what the tests assert

- Helium's exchange energy is exactly zero (`== 0.0`, bit-exact): no
  same-spin pair, so `exchange_operator` builds no terms. The sharpest test
  in the phase — any leak of `(q_a - 1)` into the exchange bucket shows up as
  a spurious few hartree.
- Beryllium's is not (`< -0.01` hartree); exchange is stabilizing for every
  atom with any (`E_HF <= E_Hartree`), monotone in Z over the pinned set
  (measured in the reference: He 0, Li 0.0203, Be 0.0641, C 0.390, Ne 2.14,
  Ar 7.38 hartree). Note what is deliberately *not* asserted: that the
  Hartree energy bounds the exact energy — a product wavefunction is not an
  admissible fermionic trial function, so the variational theorem says
  nothing about it.
- Both models satisfy the virial theorem (`-V/T = 2` to 1e-4): Hartree is a
  legitimate variational model, not a broken HF.
- Exchange changes the orbitals, not only the energy; half-applying the flag
  makes the two energy routes disagree (exercised, not asserted in prose).
- The collapsed atom: far more bound (Ne lands near −264.3 hartree against
  the real −128.5), `E_SCF <= E(zeta*)` within a few percent for every
  collapsed atom, exactly one ladder rung, helium a bit-exact no-op.
  The size payoff as an inequality: collapsed `<r>` falls monotonically
  while the real one does not (Be over He rises — a period boundary inside
  the sample, so the test would fail if periodicity died).
- Refusal: `pauli=False, exchange=True` raises with a message naming
  antisymmetry and saying what to pass; not flipped automatically, because a
  caller that asked for both asked for something that does not exist.
- The three models never share a cache key; a collapsed solve is
  `COUNTERFACTUAL` everywhere it reports (energy, orbital energies, orbital
  shapes); disclosures pinned (alteration leads, Pauli-intact vs cap-gone
  contradiction, forced exchange, term structure replaced, refinement
  promises no better calculation).

## Server

`HFRequest` gains `exchange: bool = True`, `pauli: bool = True` and a
model-validator refusing `pauli=false` with `exchange=true` as a 422 — a
validation refusal, not a handler check: 400 is the server declining a
well-posed request, and this one is not well posed. `_parse_config_or_422`
and `_validate_hf_request` take `pauli` (the capacity check goes with it).
The job handler passes both flags to `solve_hartree_fock`; on a
`pauli=False` ground-configuration solve it additionally computes
`pauli_collapse`, and on an `exchange=False` ground solve it computes
`hf_exchange_energy` — both server-side so the client never subtracts two
numbers fetched separately. `HFResultModel` gains `exchange`, `pauli`,
`exchange_energy`/`exchange_energy_ev` (null unless computed), and a nested
`PauliCollapseModel` (binding change, real energy/config, both radii, ratio,
variational ζ and E). `HFJobResult` wraps the result with optional extras;
`None` when the job ran only the real model.

## Web

- `urlState.ts`: `exchange` as `nox=1` and `pauli` as `nopauli=1` — negative
  polarity so absence means real physics. `nopauli=1` implies `nox` in the
  parser, enforced rather than trusted to the query string.
- `store.ts`: `exchange`, `pauli`, `hf: HFLevels | null`, `hfStatus`,
  `loadHF`, `ensureHF` (used by the Levels view when `model === "hf"`;
  waits for the systems table first, since Z and N live there).
  `setExchange`/`setPauli` clear the solve and clear `config` so the solve
  uses the ground configuration of the rule now in force; both directions
  couple (exchange back on restores the cap, and vice versa), because a store
  that can hold `pauli=false, exchange=true` will eventually send it.
- `Controls.tsx`: two checkboxes under the HF model selector, live only when
  `model === "hf"`; the exchange one disables (stays ticked, not hidden) when
  Pauli is off. Hints say which counterfactual each is.
- `LevelsView.tsx`: `HFLadder` — the log-binding-energy ladder (ionization
  limit marked off-scale, `HF_LADDER_AXIS_LIBERTY`), virial ratio as a
  convergence diagnostic, the exchange-energy caption (with the zero case as
  the answer, not a missing number), and `PauliComparison` (cost in eV,
  radius ratio, the closed-form disclosure). Caption wording: "stationary for
  this model, but not a bound on the real atom" under the Hartree branch —
  the variational sentence inherited from HF is false there and must not be.

## Deliberately stripped still / deferred

HF 3D views (cloud/plane/surface under `model=hf`), the configuration
threading through `evaluate_hf_state`, `sample_hf_density`, `hf_plane_grid`
and the `compare` density overlay — the reference's Phases 26/27/29 = our
Phase 12 — and is not part of this phase. The tour system stays later-phase.
The `subshell_terms` raise above capacity is already in place from Phase 1.

## Validation

Ported nearly verbatim from the reference (they encode the physics anchors):
`test_hf_exchange.py` (He zero bit-exact, stabilizing inequality, virial,
tier + disclosure pins, cache-key separation, the route-disagreement
exercise, `hf_exchange_energy` provenance), `test_hf_pauli.py` (configuration
combinatorics, the refusal and its message, the textbook −2.84765625 variational
helium anchor, SCF-below-bound, one-rung ladder, He no-op bit-exact, mean
radius 3/2 bohr and its no-error-bar rule, the monotonicity-vs-periodicity
inequality, provenance pins), and the HF-request subset of
`tests/test_server_hf.py` (422 on the impossible combination, flags through
the job, `exchange_energy`/`collapse` present iff asked, refusal of the
comparison for a hand-written non-ground collapsed configuration), plus
web tests for the URL polarity round-trip, the store coupling both ways, and
the payload carrying the flags.

## Build record (2026-09-11)

Most of the counterfactual engine above was already in the tree on arrival
(verified, not assumed: `test_hf_exchange.py` 25 passed incl. the bit-exact
He zero, `test_hf_pauli.py` 32 passed). New this session:

- `sampling.py`: `sample_hf_density`; `plane.py`: `hf_plane_grid`
  (tier read off the evaluated psi, as the spec requires).
- `density_compare.py`: the full module — common log grid, half-L1
  displaced charge, 4-term bar, 1e-8 maxima-only floor, inside-out shell
  matching, counterfactual inheritance. Measured Ne 0.0232, Ar 0.0599±0.0024,
  matching the reference figures exactly.
- Server: `ManyElectronRequest` (model/config/exchange/pauli + the 422
  schema validator), HF branches on the sample/plane jobs, `exchange`/`pauli`
  on `HFRequest`, radial `total_density`/`density_comparison` + `compare`,
  `_many_electron_target` split out of the orbital-only refusal, S/Cl listed
  with `has_gsz: false` and the redirect wording on the 400.
- Web: `config`/`nox`/`nopauli`/`compare` URL params (nopauli implies nox in
  the parser), store fields with two-way switch coupling, `loadHF`/`ensureHF`,
  job payload threading, Controls checkboxes + config input + subshell
  greying, RadialView D(r) + dashed overlay + displaced-charge readout with
  bar-relative decimals + shell table.
- Deferred to Phase 12, as planned: `hf_isosurface` and the isosurface
  route/view. Four tests in `test_hf_views.py` and one in
  `test_server_hf_views.py` skip until then, each naming Phase 12.
- The reference's HFLadder/Levels-view HF branch was not built: this tree's
  web never wired HF levels into the Levels view, and that predates this
  phase.
