# Phase 10 — Hartree-Fock core (plan)

Ported from `../AtomSim` (its agent.md Phase 11 = specs/plans
2026-07-27-phase21-hartree-fock), `s/atomsim/atomic/`. The reference's HF
files have evolved past its Phase 21 with the counterfactual machinery
(exchange flag, Pauli collapse, HF 3D) — its Phases 12/13 = our Phase 11.
Those parts are stripped on port, per the Phase 8 thermal strip-out
precedent, and restored next phase.

**Prerequisite check (done 2026-09-10, all green):**
- `analytic/wigner.py` already carries `wigner_3j` — diff vs reference is
  IDENTICAL (landed with Phase 8's fine-structure work).
- `atoms.py` AST-identical to the reference (incl. `subshell_terms`,
  `pauli=False` in `aufbau_configuration`/`validate_config`,
  `NO_GSZ_PARAMETERS = {16, 17}`); only docstrings differ.
- `numerics/mesh.py` at API parity: `RadialMesh` with `integrate`/
  `cumulative`/`to_s`/`to_p`/`normalized`/`hamiltonian_bands`,
  `exponential_mesh`, `mesh_for_atom{,_at_step}`, `display_window`.
  Note: the reference's copy of `mesh.py` is currently syntactically broken
  (a docstring closes early in `display_window`); ours is the cleaned,
  parsing version. Port nothing from the ref's tail; treat ours as truth.
- `numerics/radial_solver.py`, `numerics/screening.py` — AST-identical;
  `screening.py`'s refinement string already points at `hf_atom.py`.
- `data/hf_reference_energies.json` already vendored with all five Bunge
  energies (He, Be, Ne, Mg, Ar) and the full transcription trail.

## Part 1 — Quadrature and algebra (no SCF, fast tests)
1. **`numerics/slater.py`** (ref: 133 ln): copy wholesale with
   `s/atomsim/atomic/`. `pair_potential` (cumulative trapezoid, O(N)),
   `slater_f`, `slater_g`. Port `tests/test_slater.py` (ref: 81 ln) —
   5Z/8 closed form, `U_0 -> 1/r` past the charge, argument symmetry.
2. **`numerics/hf_terms.py`** (ref: 184 ln): copy wholesale. The module
   docstring must carry the average-of-configuration derivation verbatim
   (it does in the reference). `Subshell`, `direct_potential`,
   `exchange_apply`, `exchange_coefficient`, `same_shell_coefficient`.
   Port `tests/test_hf_terms.py` (ref: 167 ln) — the four coefficient
   anchors (H no self-interaction, He one unit of `U_0`, closed-shell
   reduction, Be `4J − 2K`). **This part is the gate**: if the anchors do
   not pin, nothing past it is worth building.
3. **`tests/test_wigner_3j.py`** already exists from Phase 8 — verify it
   still covers the ref's cases (ref: 85 ln vs ours), extend only if a case
   is missing.

## Part 2 — Channel solve and SCF
4. **`numerics/hartree_fock.py`** (ref: 676 ln) — copy, then strip the
   Phase-11 counterfactuals (`exchange=` parameter on `fock_operator` /
   `_fock_parts` / `solve_channel` / `scf`; `HFJobResult`-adjacent extras).
   Keep: `local_hamiltonian_bands` (mesh-variable form), `fock_operator`,
   `_preconditioner` (banded Cholesky on the shifted local bands),
   `solve_channel` (LOBPCG, gate on achieved residual NOT
   `len(history) >= maxiter`, no guard vectors), `ChannelSolution`,
   `HFConvergenceError`, `one_electron_integral`, `orbital_energy`,
   `_interaction_energy`, `total_energy_direct`,
   `total_energy_from_orbitals`, `kinetic_and_potential`, `scf`
   (α = 0.65-style tuned value as the ref settled on, recorded in the
   docstring with the sweep), `SCFSolution`.
   Port `tests/test_hf_channel.py` (ref: 220 ln, minus any exchange-flag
   cases) and `tests/test_hartree_fock.py` (ref: 209 ln): H exactly −0.5,
   two energy routes agree to 1e-8, virial = 2, non-convergence raises.
5. Verify mesh-variable consistency: the ref's engine solves in S, exchange
   lives in P (`mesh.to_s`/`to_p` conversions at the operator boundary) —
   the tests above must pass on the exponential mesh, not a uniform grid.

## Part 3 — The atom, end to end
6. **`hf_atom.py`** (ref: 1238 ln) — the big port. Copy, then strip
   Phase-11 machinery: `hf_exchange_energy`, `collapsed_variational_energy`,
   `PauliCollapse`, `pauli_collapse`, and every `exchange`/`pauli`
   parameter threading. Keep: `HFOrbital`, `HFResult`, `hf_mesh`,
   `_start_potential`, `_guess_from_central_field`, `_refine`,
   `_relativistic_scale`, `_energy_assumptions`, `_solve_on_grid`,
   `solve_hartree_fock` (two-route energy cross-check at 1e-6,
   grid-halving error estimate = spread + 4e-6·|E| bracket, K+ and
   Ar-like-ion convergence, `_HF_MAX_Z = 36` as "as far as tested"),
   `hf_valence_ionization_energy` (Koopmans, own provenance),
   `hf_mean_radius`, `hf_total_radial_density`, `_occupied_orbital`,
   `hf_radial`, `evaluate_hf_state`, plus the `lru_cache` solve cache.
   Port `tests/test_hf_atom.py` (ref: 176 ln): five vendored benchmarks at
   1e-4 relative, variational bound, provenance tiers
   (`total_energy` APPROXIMATION vs `virial_ratio` NUMERICAL),
   convergence record. Port `tests/test_hf_reference_data.py` (ref: 84 ln)
   — mostly a guard that our already-vendored JSON stays sane.
7. **Open shells** (ref `tests/test_hf_open_shell.py`, 122 ln): Li→P plus
   **S and Cl, which GSZ cannot do at all** — the visible payoff. The
   averaged functional handles fractional occupancy natively; the work is
   conditional provenance (configuration-average assumption only when a
   subshell spans more than one term — `atoms.subshell_terms` already
   exists) and per-atom α if alkalis oscillate. Also port the
   `tests/test_hf_atom_api.py` (ref: 181 ln) Koopmans-vs-NIST cross-model
   checks, with the ref's helium exception pinned as explained.
8. **`tests/test_hf_performance.py`** (ref: 165 ln): wall-time guard on the
   worst case — the ref measured Cl 7.2 s / Na 5.5 s / Ar 4.7 s, so guard
   chlorine, not just argon. Cache-hit test; iteration-count bound.

## Part 4 — Surface: server, web, docs
9. **Server**: `HFOrbitalModel`/`HFResultModel` in `schemas.py` (ref fields
   minus `exchange`/`pauli`/`collapse`), `POST /api/jobs/hf` through
   `jobs.py` with `HFRequest` (z, n_electrons), results on the job
   data/meta route. Port the Phase-10 subset of ref
   `tests/test_server_hf.py` (ref: 390 ln — cut the exchange/pauli/3D test
   groups, keep job lifecycle, provenance tiers, Z-refusal-with-reason).
   Unsupported Z answers with an explanation, mirroring
   `NO_GSZ_PARAMETERS` honesty.
10. **Web**: `model: 'gsz' | 'hf'` in store + `urlState` (`model` query
    key, default `gsz`, in `INVALIDATED` — switching models changes the
    physics); client fn for the HF job; model selector on the
    screened-atom views; badge shows the fidelity split and the
    configuration-average disclosure for open shells; virial ratio as a
    convergence readout labelled a diagnostic. Port the ref's urlState
    model-selection tests. **Caution**: the ref store has drifted further
    (isosurface triangles, HF 3D flags) — port deltas only, do not copy
    wholesale.
11. **Gate**: `ruff check src tests && pytest` green (665 + new).
    `cd web && npm test && npm run build` green.
12. **Docs**: this spec + plan (done); `docs/README.md` index; `agent.md`
    Phase 10 marked DONE with the strip-out delta note; repo map already
    lists `hf_atom.py`/`hf_reference.py`/`slater.py` targets.

## What the build will likely change (record at close)
The ref's own build notes predict: mesh beats uniform by ~10× on wall time;
the error bar needs the shared-`r_min` spread + floor bracket; no guard
vectors in LOBPCG; gating on achieved residual not history length; open
shells need two separate disclosures (restricted radial vs term average);
the performance worst case is an open-shell third-row atom, not argon; and
Koopmans loses to GSZ on helium specifically. Each becomes a note in the
agent.md phase entry if it fires here.
