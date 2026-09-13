# Phase 11 — Counterfactual electrons (plan)

Ported from `../AtomSim` (its agent.md Phase 12 = its spec/plans phases 22 and
24), `s/atomsim/atomic/`. Phase 10 stripped exactly these parts on the way in
(`exchange`/`pauli` flag threading, `hf_exchange_energy`, `pauli_collapse`,
the collapsed-atom schemas and server fields, the `test_hf_exchange` /
`test_hf_pauli` test groups); this phase restores them wholesale. The
reference's Phase 26 (HF 3D) wiring is *not* restored here — its Phases 26/27/29
are our Phase 12.

**Prerequisite check (done 2026-09-11, all green):**
- `numerics/hf_terms.py` already carries `ExchangeOperator`,
  `exchange_operator`, `exchange_apply`, `same_shell_coefficient`,
  `exchange_coefficient`, `direct_potential` with the `(q_a - 1)` factor —
  the reference's `exchange=False` branch is just `ExchangeOperator(terms=())`,
  so the terms layer needs no changes at all.
- `atoms.py` already has `pauli: bool = True` on `aufbau_configuration`,
  `validate_config`, `is_ground` (AST-identical to the reference), and
  `subshell_terms` already raises above capacity. `parse_config` /
  `format_config` exist.
- `hf_atom.py` currently has the Phase-10 shape: `solve_hartree_fock(z, n,
  config)` with `lru_cache(maxsize=8)`, `HFResult` without `exchange`/`pauli`
  fields, `_energy_assumptions(config, z)`, two-route agreement at
  `_ROUTE_AGREEMENT = 1e-6`. The diff is threading, not restructuring.
- `numerics/hartree_fock.py` currently has no `exchange` kwarg on
  `fock_operator`/`_fock_parts`/`solve_channel`/`orbital_energy`/
  `total_energy_direct`/`kinetic_and_potential`/`scf`; the reference diffs are
  flag threading plus the `_NO_EXCHANGE` constant and updated docstrings.
- Server: `HFRequest` in `server/app.py` has `z`, `n_electrons`, `config`;
  `HFResultModel` in `schemas.py` lacks `exchange`/`pauli`/`exchange_energy`/
  `collapse`. `_parse_config_or_422`/`_validate_hf_request` lack `pauli`.
- Web: store has `model` + `setModel` + `resolveModel` + `gszAvailable`
  (Phase 10), `urlState.ts` has the `model` key but no `nox`/`nopauli`;
  `LevelsView.tsx` has the screened and hydrogenic ladders but no
  `HFLadder`; `liberties.ts` lacks `HF_LADDER_AXIS_LIBERTY`.

## Part 1 — The numerics layer

1. **`numerics/hartree_fock.py`** (ref diff: ~60 ln): thread
   `exchange: bool = True` keyword-only through `fock_operator`,
   `_fock_parts`, `solve_channel`, `orbital_energy`, `total_energy_direct`,
   `kinetic_and_potential`, `scf`. Add the module-level
   `_NO_EXCHANGE = ExchangeOperator(terms=())` and the
   `exchange_op = exchange_operator(...) if exchange else _NO_EXCHANGE`
   line in `_fock_parts` (the empty operator beats a branch inside
   `matvec` — its comment ports verbatim). Port the reference's docstring
   blocks: the module docstring's "three places exchange enters, and what
   does NOT move (the `(q_a - 1)` factor)" passage, `solve_channel`'s
   residual-floor notes for exchange-on vs exchange-off channels, and
   `_interaction_energy`'s k>0/G_k conditional.
2. **`tests/test_hf_channel.py`** additions (ref): the exchange-off channel
   still converges (3e-10 attainable, vs the ~6e-6 exchange-on floor),
   orthogonality holds on the Hartree operator, and the flag defaults keep
   existing callers bit-identical (rerun the existing suite — nothing else
   may move).

## Part 2 — The atom-level counterfactuals

3. **`hf_atom.py`** (ref diff: ~450 ln across the file):
   - `HFResult` gains `exchange: bool` and `pauli: bool`; the result `key`
     becomes `z{z}n{n}` + `-nopauli` when `not pauli` + `-nox` when
     `pauli and not exchange`. `is_ground(config, pauli)`.
   - `solve_hartree_fock(z, n_electrons, config, exchange=True, pauli=True)`:
     raise `ValueError(_NO_PAULI_WITH_EXCHANGE)` on the impossible
     combination; `validate_config(config, pauli)`; thread the flag through
     `_solve_on_grid(..., exchange=exchange)` (which passes it to `scf`,
     `orbital_energy`, `total_energy_direct`) and
     `kinetic_and_potential(..., exchange=exchange)`; pick method/refinement
     from `_NO_PAULI_METHOD`/`_HARTREE_METHOD`/`_TOTAL_ENERGY_METHOD`;
     fidelity `COUNTERFACTUAL` unless both flags true, shared by
     `energy_prov` and `shape_prov` (read off the solve, never a literal
     downstream).
   - `_energy_assumptions(config, z, exchange, pauli)`: port the four
     `_HARTREE_*` constants (`_ALTERATION`, `_PAULI_INTACT`,
     `_SELF_INTERACTION`, `_REFINEMENT`) and the four `_NO_PAULI_*`
     constants (`_METHOD`, `_ALTERATION`, `_IMPLIES_NO_EXCHANGE`,
     `_NO_TERMS`, `_REFINEMENT`); the alteration block prepends with
     `out[:0]`; under collapse the open-shell/multi-term lines are replaced
     by `_NO_PAULI_NO_TERMS`, not dropped.
   - **`hf_exchange_energy(z, n_electrons, config)`**: both solves inside,
     `max` of the two mesh error estimates, `COUNTERFACTUAL`, "not an
     observable" assumption, zero-case assumption.
   - **`collapsed_variational_energy(z, n_electrons)`**: the ζ* formula and
     `E(zeta*)`, `COUNTERFACTUAL`, the "upper bound on the collapsed atom"
     assumption.
   - **`PauliCollapse` dataclass + `pauli_collapse(z, n_electrons=None)`**:
     real + collapsed solves (ground configs for their own rules), mean
     radii via `hf_mean_radius`, `binding_change` (sum of both mesh
     spreads), `radius_ratio` (error bar deliberately `None` — dimension
     rule), the variational pair carried.
   - `hf_mean_radius` gains its docstring rule: no error estimate, a
     hartree bar on a length is the wrong dimension.
4. **`tests/test_hf_exchange.py`** (ref: 224 ln) — port verbatim. Gate:
   He `== 0.0` exactly, `hf_exchange_energy` `< -0.01` for Be, stabilizing
   across He/Li/Be/C/Ne, monotone growth, virial 2 for the Hartree model,
   orbital-level difference for Li/Be, tier + disclosure pins
   ("distinguishable", "pauli principle is not switched off", "2(2l+1)",
   "does not repel itself", "correlation"), no-error-bar-against-truth pin,
   cache separation, and the route-disagreement exercise.
5. **`tests/test_hf_pauli.py`** (ref: 366 ln) — port verbatim. Gate:
   `1s^N` aufbau, ground-under-its-own-rule, validate keeps `n > l`,
   `subshell_terms` raises, refusal message names antisymmetry +
   `exchange=False`, ζ* = 1.6875 / −2.84765625 anchor, SCF below bound
   within 5% (He/Be/Ne), one rung, He bit-exact no-op, `<r>`(H) = 1.5,
   no wrong-dimension bar, Ne collapse ≈ −264, monotone-vs-periodicity,
   three cache keys, COUNTERFACTUAL everywhere, all five disclosure pins.
6. **`tests/test_hf_performance.py`**: extend the wall-time guard with the
   worst Hartree case (the reference measured convergence, not accuracy, is
   what the toggle costs — more coarse iterations on closed shells). Guard
   the iteration counts, not a new wall-time budget, unless measurements
   say otherwise; record what fired in the agent.md phase note.

## Part 3 — Server

7. **`schemas.py`**: `HFResultModel` gains `exchange: bool = True`,
   `pauli: bool = True`, `exchange_energy: QuantityModel | None = None`,
   `exchange_energy_ev`, and `collapse: PauliCollapseModel | None = None`.
   New `PauliCollapseModel` (binding_change(+ev), real_total_energy(+ev),
   real_config, real_radius, collapsed_radius, radius_ratio,
   variational_zeta, variational_energy(+ev)) with the reference's
   field docstrings. Mirror in `web/src/api/types.ts` (`HFLevels`).
8. **`server/app.py`**:
   - `HFRequest` gains `exchange: bool = True`, `pauli: bool = True` and a
     pydantic `model_validator(mode="after")` refusing
     `pauli=false` with `exchange=true` as a 422 with the antisymmetry
     message.
   - `_parse_config_or_422(text, pauli=True)` (validate with the rule in
     force) and `_validate_hf_request(z, n, config, pauli=True)`.
   - `create_hf_job`: `aufbau_configuration(n_electrons, req.pauli)`, pass
     both flags to `solve_hartree_fock`, and — ported verbatim, including
     the "nothing honest to report between 0 and 1" comment — compute
     `hf_exchange_energy` when `not req.exchange and req.pauli` and
     `pauli_collapse` when `not req.pauli` and the config *is* the collapsed
     ground configuration (`comparable` guard: a hand-written `1s5 2s3` has
     no twin and gets no comparison). Wrap in `HFJobResult(result, delta,
     collapse)`; `_hf_result_model` grows the three optional fields.
9. **`tests/test_server_hf.py`** additions (the Phase-11 subset of the ref's
   `test_server_hf.py`): 422 on the impossible combination (validator, not
   handler), flags survive into the meta (`exchange`/`pauli` echoed),
   `exchange_energy` present iff `exchange=false` ground solve, `collapse`
   present iff `pauli=false` ground solve and null for a hand-written
   collapsed config, 422 texts name antisymmetry. Existing job-lifecycle and
   provenance-tier tests stay green untouched.

## Part 4 — Web

10. **`urlState.ts`**: `exchange: boolean` (serializes `nox=1`, negative
    polarity, absence = real physics — port the rationale comment) and
    `pauli: boolean` (`nopauli=1`); the parser makes `nopauli=1` imply
    `nox` rather than trusting the string. Serialize in
    `serializeAppUrl`, defaults `true`/`true` in `URL_DEFAULTS`.
    Port the reference's urlState tests for both keys and the implication
    rule.
11. **`store.ts`**: `exchange`, `pauli`, `hf: HFLevels | null`,
    `hfStatus: LoadStatus`, `config: string | null` (needed by the
    comparable guard deep-link case only if exposed; if the config picker
    is not part of this phase, keep `config` internal and always null —
    decide at implementation, record either way), `loadHF`, `ensureHF`
    (table first: `systems.length === 0` → `loadSystems()`; resolves Z/N
    from the table, never from the key), the `setExchange`/`setPauli`
    coupling (both directions, `config: null`, `hf: null`,
    `hfStatus: "idle"`), and `setSystem` resetting both flags. The Levels
    view calls `ensureHF` when `model === "hf"` before rendering the HF
    branch. Extend `store.test.ts`: coupling both ways, invalidation of the
    solve but not of unrelated payloads, payload carries the flags.
12. **`lib/hfModel.ts`**: export `HF_LADDER_AXIS_LIBERTY`'s home if it
    belongs in `liberties.ts` instead (match the reference: `liberties.ts`);
    port that constant verbatim (log-binding-energy axis disclosure).
13. **`Controls.tsx`**: under the existing model `Choice`, when
    `model === "hf"`, render the two checkboxes with the reference's hint
    copy (exchange disabled-and-ticked while Pauli is off; the three-state
    hint strings port verbatim). The model hint sentence gains the
    counterfactual-aware wording only where the reference's Phase 26 copy
    (3D) is required — skip those sentences, they are Phase 12's.
14. **`LevelsView.tsx`**: port `HFLadder` (log |ε| axis, off-scale
    ionization mark, `OffWindowMarks`-equivalent or simplify if the
    component does not exist here — record the simplification), the virial
    readout labelled a diagnostic, the total-energy caption with the
    variational-vs-stationary branch, the exchange-energy caption with its
    zero case, `PauliComparison`, and the disclosure block. Wire the
    `wantHF` branch: `hfStatus === "error"` → reason; `hf === null` →
    "Solving Hartree-Fock, this takes a few seconds…".
15. **Gate**: `ruff check src tests && pytest` green (888 + new − none
    removed). `cd web && npm test && npm run build` green.
16. **Docs**: `docs/README.md` index; `agent.md` Phase 11 marked DONE with
    the what-building-it-changed notes (expect at least: the store coupling
    running both directions, the caption theorem trap the reference hit,
    the bit-exact helium zeros, the convergence-not-accuracy cost — record
    only what fires here).

## Order and gates

Parts are gated in order: Part 1's flag default keeps every existing test
bit-identical (run the full suite after step 1); Part 2's tests are the
physics gate; Parts 3–4 are surface. Commit per logical change: numerics
threading, hf_atom counterfactuals + tests, server + tests, urlState +
store, views + tests, docs.

## What the build will likely change (predict, verify at close)

The reference's own build notes predict: the store toggle does *not* live in
`INVALIDATED`-style blanket clearing for the HF solve alone (it clears `hf`
explicitly — an HF solve is derived from (Z, N, config, flags), not from
(n, l, m), so n-clicks must not throw away seconds of solve); the Hartree
caption inherits a variational claim that does not cover it and must be
rewritten per-branch; both helium zeros (energy and radius) are bit-exact and
worth more than tolerances; Hartree takes *more* SCF iterations than HF on
closed shells (shell competition, not the missing exchange — the collapsed
solve converges as fast as the real atom); and the collapse comparison must
compare whole-atom `<r>` with whole-atom `<r>`, never with a single orbital's
radius, because for Be those two questions have opposite answers. Each
becomes a note in the agent.md phase entry if it fires here.

## Build record (2026-09-11)

- Parts 1–2 (numerics + atom-level counterfactuals): already in the tree;
  validated by porting `test_hf_exchange.py` and `test_hf_pauli.py` verbatim
  (imports only) — green on arrival, no engine changes needed.
- HF 3D (cloud/plane/radial) + `density_compare.py` + server + web, per the
  plan's Slices A–C. One adaptation: the request `model` vocabulary stays
  `"gsz"` (this tree's job metas echo `"screened"`/`"hydrogenic"`, and the
  ported view tests were adapted to that rather than renaming the server).
- `test_density_compare.py` needed one fix on port: the reference keeps a
  mid-file import block separating fast unit tests from slow integration
  tests; merged into the top block (E402).
- Gate: `ruff check .` clean, `pytest` 1037 passed + 5 Phase-12 skips,
  `npm test` 191 passed, `npm run build` green.
