# Phase 10 — Hartree-Fock core (spec)

## What and why
`numerics/screening.py` has carried a promise in its provenance since Phase 6:
self-consistent Hartree-Fock removes the model error. This phase pays that
off. GSZ is fast and honest about being approximate, but it needs a parameter
table (S and Cl have no entry and are refused), its total energy ignores e-e
double counting, and its error has no sign. HF is variational —
`E_HF >= E_exact`, always — so an unsigned model error is traded for a
one-signed, bounded one, which is a statement the provenance can make and a
test can check.

Restricted HF, average of configuration: one radial function per `(n, l)`
subshell shared by all its electrons; open shells handled by averaging the
energy functional over microstates. Consequence, disclosed not buried: the
result is one energy per *configuration*, not per *term* — carbon gets the
centroid of `3P`/`1D`/`1S` and cannot split them. `HFResult` carries the
assumption string; the badge repeats it for open shells.

- **Slater integrals** (`numerics/slater.py`): the pair potential
  `U_k[a,b]` by two cumulative trapezoid passes, O(N), no ODE solve; `F^k`
  and `G^k` from it. Self-checking: `U_0 -> 1/r` past the charge (asserted at
  the box edge), and `U_k` is symmetric in its orbital arguments.
- **Angular coefficients** (`numerics/hf_terms.py`): derived by varying the
  average-of-configuration functional, derivation in the module docstring.
  Pinned four ways before anything else is trusted: hydrogen has no
  self-interaction at all (the `(q_a - 1)` factor), helium sees exactly one
  unit of `U_0`, the averaged coefficients reduce to the closed-shell ones at
  full occupancy, beryllium reproduces the textbook `4J − 2K`. A wrong
  coefficient does not crash — it converges smoothly to the wrong energy,
  which is exactly the failure mode the prime directive exists to prevent.
- **Channel solve** (`numerics/hartree_fock.py`): exchange is non-local, so
  the per-`l` eigenproblem stops being tridiagonal. Matrix-free LOBPCG on a
  `LinearOperator`, preconditioned by the still-tridiagonal local part
  (`M = (H_local − σI)^-1`, banded Cholesky, σ below the lowest sought
  eigenvalue). Warm start from GSZ where parameters exist, hydrogenic `Z_eff`
  guess otherwise; iteration counts returned so the preconditioner claim is
  falsifiable. Non-convergence raises `HFConvergenceError`, never returns a
  result with `converged=False`.
- **SCF loop**: damped linear mixing (α tuned against measured iteration
  counts, worst case not average; recorded in the docstring). Converged when
  max |Δε| < 1e-8 hartree.
- **Total energy, three ways**: direct assembly; the orbital identity
  `E = ½ Σ q_a (I_a + ε_a)` (shares no code with route 1, so a coefficient
  error shows as disagreement); and the virial ratio `-V/T = 2` at
  convergence. Routes 1–2 agree to 1e-8 (algebraically identical), virial to
  1e-4 relative.
- **Grid**: the exponential mesh of `numerics/mesh.py` — the same
  constant-relative-resolution argument the ref recorded (argon: ~8 s on a
  few thousand mesh points vs about an hour on 72k uniform points). The mesh
  module is already in the tree at API parity.

## Fidelity model
Two errors kept apart: solving the equations imperfectly (`NUMERICAL`:
grid-halving, box, SCF residual) and HF not being the atom (`APPROXIMATION`,
signed: no correlation, variational upper bound, configuration average if
open shell). `total_energy` is `APPROXIMATION` while `kinetic`/`potential`/
`virial_ratio` are `NUMERICAL` — the virial ratio says whether the equations
were solved; the total energy says something about an atom. The refinement
string becomes "CI or MBPT would recover the correlation energy".

## Benchmarks
Vendored **Bunge, Barrientos, Bunge (1993)** Roothaan-Hartree-Fock total
energies for He, Be, Ne, Mg, Ar
(`data/hf_reference_energies.json`, already transcribed with citation and
retrieval trail). Finite-basis values sit slightly above the grid limit, so
the tolerance is relative and grid-dominated (1e-4). Anchors in ladder order:
H exactly −0.5 with zero direct and exchange; He one unit of `U_0`; Be; Ne
(k > 0 exchange); Mg, Ar against the vendored energies. Cross-model: HF
valence IE vs GSZ and vs vendored NIST for He/Li/Na — agreement with GSZ is a
check on the implementation (Szydlik-Green fitted to HF), NIST is the column
with external weight; helium is a pinned explained exception (Koopmans
freezes the ion's orbitals; GSZ's fit absorbs part of that).

## Deliberately stripped (Phase 11)
The counterfactual machinery the reference grew after this phase:
`exchange=` flag threading through Fock/SCF, `hf_exchange_energy`,
`pauli_collapse`/`PauliCollapse`/`collapsed_variational_energy`, the
`collapse`/`exchange`/`pauli` schema fields and server-job fields, and the
`test_hf_exchange`/`test_hf_pauli`/HF-3D-view tests. Same pattern as the
Phase 8 thermal strip-out: fields keep natural defaults where the schema
shape would otherwise change, and Phase 11 restores the machinery wholesale.

## Deliberately neglected
DIIS (linear mixing first), term energies (multiconfiguration HF), correlation,
switching the spectroscopy stack's default to HF (the adapter ships so HF
orbitals *can* drive `spectra.py`/`transfer.py`), relativistic corrections.

## Server and web
HF runs as a background job through `jobs.py` — an SCF loop is seconds of
work, which is what the job protocol exists for. `HFOrbitalModel`/
`HFResultModel` in schemas (mirroring the screened-atom model plus `kinetic`,
`potential`, `virial_ratio`, `iterations`, `converged`); provenance survives
to the browser as always. Unsupported Z is refused with a reason, never
silently. Web: a model selector (`gsz | hf`) on the screened-atom views with
`model=gsz` default so deep links keep resolving; the badge shows the
fidelity split and, for open shells, the configuration-average disclosure;
the virial ratio renders as a convergence readout labelled a diagnostic, not
physics. `model` lives in the store's `INVALIDATED` block — switching models
changes the physics.

## Validation
`test_slater` (5Z/8 closed form, 1/r tail, symmetry), `test_hf_terms` (the
four coefficient anchors), `test_hf_channel` (hydrogenic levels, ortho-
normality, operator symmetry, warm start falsifiable), `test_hartree_fock`
(H exactly −0.5, three-way energy, virial, non-convergence raises),
`test_hf_atom` (five vendored benchmarks at 1e-4, variational bound,
provenance tiers, grid-halving), `test_hf_atom_api` (Koopmans IE vs NIST,
radial/state API mirroring screened_atom), `test_hf_open_shell` (Li→P, S and
Cl converge — the visible payoff), `test_hf_performance` (worst-case wall
time guarded, iteration counts bounded, cache hit), `test_hf_reference_data`,
and the Phase-10 subset of the ref's server HF tests.
