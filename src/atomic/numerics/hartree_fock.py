
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_solve_banded, cholesky_banded, eigh_tridiagonal
from scipy.sparse.linalg import LinearOperator, lobpcg

from atomic.analytic.wigner import wigner_3j
from atomic.numerics.hf_terms import (
    ExchangeOperator,
    Subshell,
    direct_potential,
    exchange_apply,
    exchange_operator,
)
from atomic.numerics.mesh import RadialMesh
from atomic.numerics.slater import slater_f, slater_g

__all__ = [
    "ChannelSolution",
    "HFConvergenceError",
    "SCFSolution",
    "fock_operator",
    "kinetic_and_potential",
    "local_expectation",
    "local_hamiltonian_bands",
    "one_electron_integral",
    "orbital_energy",
    "scf",
    "solve_channel",
    "total_energy_direct",
    "total_energy_from_orbitals",
]

_NO_EXCHANGE = ExchangeOperator(terms=())

@dataclass(frozen=True)
class ChannelSolution:

    energies: np.ndarray
    orbitals: np.ndarray
    iterations: int
    residual: float

class HFConvergenceError(RuntimeError):
    pass

def local_hamiltonian_bands(
    v_local: np.ndarray, l: int, r: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if r.ndim != 1 or r.size < 3:
        raise ValueError(f"the radial grid must be 1-D with at least 3 points, got {r.shape}")
    h = float(r[1] - r[0])
    if h <= 0:
        raise ValueError(f"the radial grid must be increasing, got h={h!r}")
    if not np.allclose(np.diff(r), h, rtol=1e-9, atol=0.0):
        raise ValueError(
            "a uniform radial grid is required here; the 3-point stencil in "
            "this module assumes a constant step"
        )
    if abs(float(r[0]) - h) > 1e-9 * h:
        raise ValueError(
            f"the radial grid must have r[0] == h, got r[0]={float(r[0])!r} with "
            f"h={h!r}. The Dirichlet condition u = 0 sits one step below "
            f"r[0], so any other start silently moves the origin: r[0]=1e-5 on "
            f"an h=2e-3 grid costs 4.6% on hydrogen's ground state. Use "
            f"h * np.arange(1, N + 1), as radial_solver.solve_radial does."
        )

    inv2m = 0.5
    v_eff = v_local + l * (l + 1) * inv2m / r**2
    diag = 2.0 * inv2m / h**2 + v_eff
    offdiag = np.full(r.size - 1, -inv2m / h**2)
    return diag, offdiag

def fock_operator(
    subshells: tuple[Subshell, ...],
    a_index: int,
    v_nuclear: Callable[[np.ndarray], np.ndarray],
    l: int,
    mesh: RadialMesh,
    *,
    exchange: bool = True,
) -> LinearOperator:
    return _fock_parts(
        subshells, a_index, v_nuclear, l, mesh, exchange=exchange
    )[0]

def _fock_parts(
    subshells: tuple[Subshell, ...],
    a_index: int,
    v_nuclear: Callable[[np.ndarray], np.ndarray],
    l: int,
    mesh: RadialMesh,
    *,
    exchange: bool = True,
) -> tuple[LinearOperator, np.ndarray, np.ndarray]:
    r = mesh.r
    v_local = np.asarray(v_nuclear(r), dtype=float) + direct_potential(
        subshells, a_index, r
    )
    diag, offdiag = mesh.hamiltonian_bands(v_local, l)
    scale = np.sqrt(mesh.step * mesh.jacobian)
    exchange_op = (
        exchange_operator(subshells, a_index, r) if exchange else _NO_EXCHANGE
    )

    def matvec(s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float).ravel()
        out = diag * s
        out[:-1] += offdiag * s[1:]
        out[1:] += offdiag * s[:-1]
        return out - scale * exchange_op.apply(s / scale)

    n = mesh.points
    op = LinearOperator((n, n), matvec=matvec, rmatvec=matvec, dtype=float)
    return op, diag, offdiag

def _preconditioner(
    diag: np.ndarray, offdiag: np.ndarray, lowest: float
) -> LinearOperator:
    sigma = lowest - 1.0
    ab = np.zeros((2, diag.size))
    ab[0, 1:] = offdiag
    ab[1, :] = diag - sigma
    factor = cholesky_banded(ab, lower=False)

    def apply(x: np.ndarray) -> np.ndarray:
        return cho_solve_banded((factor, False), np.asarray(x, dtype=float).ravel())

    return LinearOperator((diag.size, diag.size), matvec=apply, dtype=float)

def local_expectation(
    p: np.ndarray, v_local: np.ndarray, l: int, mesh: RadialMesh
) -> float:
    diag, offdiag = mesh.hamiltonian_bands(v_local, l)
    s = mesh.to_s(p)
    return float(s @ (diag * s) + 2.0 * float(offdiag @ (s[:-1] * s[1:])))

def solve_channel(
    subshells: tuple[Subshell, ...],
    a_index: int,
    v_nuclear: Callable[[np.ndarray], np.ndarray],
    l: int,
    mesh: RadialMesh,
    n_states: int,
    guess: np.ndarray | None = None,
    tol: float = 1e-6,
    residual_ceiling: float = 1e-3,
    maxiter: int = 150,
    *,
    exchange: bool = True,
) -> ChannelSolution:
    r = mesh.r
    op, diag, offdiag = _fock_parts(
        subshells, a_index, v_nuclear, l, mesh, exchange=exchange
    )
    lowest = float(
        eigh_tridiagonal(
            diag, offdiag, select="i", select_range=(0, 0), eigvals_only=True
        )[0]
    )
    precond = _preconditioner(diag, offdiag, lowest)

    energy_scale = max(1.0, abs(lowest))
    scaled_tol = tol * energy_scale

    block = min(n_states, r.size)

    def cold_block() -> np.ndarray:
        return eigh_tridiagonal(
            diag, offdiag, select="i", select_range=(0, block - 1)
        )[1]

    def warm_block() -> np.ndarray:
        x = mesh.to_s(np.asarray(guess, dtype=float).reshape(-1, r.size)).T
        if x.shape[1] >= block:
            return x[:, :block]

        local = eigh_tridiagonal(
            diag, offdiag, select="i", select_range=(0, block - 1)
        )[1]
        taken: set[int] = set()
        for column in range(x.shape[1]):
            overlaps = np.abs(local.T @ x[:, column])
            for candidate in np.argsort(overlaps)[::-1]:
                if int(candidate) not in taken:
                    taken.add(int(candidate))
                    local[:, int(candidate)] = x[:, column]
                    break
        return local

    def attempt(x: np.ndarray, budget: int):
        w, v, history = lobpcg(
            op, np.linalg.qr(x)[0], M=precond, tol=scaled_tol, maxiter=budget,
            largest=False, retResidualNormsHistory=True,
        )
        w = np.atleast_1d(w)
        kept = np.argsort(w)[:n_states]
        res = float(np.max(np.atleast_1d(history[-1])[kept]))
        limit = residual_ceiling * max(1.0, float(np.max(np.abs(w[kept]))))
        return w, v, history, res, limit

    starts = (
        [(warm_block, maxiter), (cold_block, 2 * maxiter)]
        if guess is not None
        else [(cold_block, 2 * maxiter)]
    )
    for attempt_index, (build, budget) in enumerate(starts):
        eigenvalues, eigenvectors, history, residual, limit = attempt(
            build(), budget
        )
        if np.isfinite(residual) and residual <= limit:
            break
        if attempt_index == len(starts) - 1:
            raise HFConvergenceError(
                f"LOBPCG did not converge for l={l} from "
                f"{'either a warm or a cold start' if guess is not None else 'a cold start'}"
                f": the achieved residual, {residual:.3e}, exceeds the ceiling "
                f"{limit:.3e} ({residual_ceiling:.3e} scaled by the channel's "
                f"energy magnitude); tolerance {scaled_tol:.3e} was requested on "
                f"a budget of {budget} iterations. Returning the eigenvalue "
                f"anyway would be a plausible-looking wrong number."
            )

    order = np.argsort(eigenvalues)[:n_states]
    out = mesh.to_p(eigenvectors.T[order])

    ortho = []
    for u in out:
        for v in ortho:
            u = u - v * np.trapezoid(u * v, r)
        u = mesh.normalized(u)
        first = np.argmax(np.abs(u) > 0.01 * np.abs(u).max())
        if u[first] < 0:
            u = -u
        ortho.append(u)

    return ChannelSolution(
        energies=eigenvalues[order],
        orbitals=np.array(ortho),
        iterations=len(history),
        residual=residual,
    )

@dataclass(frozen=True)
class SCFSolution:
    subshells: tuple[Subshell, ...]
    energies: tuple[float, ...]
    iterations: int
    residual_history: tuple[float, ...]

def one_electron_integral(subshell: Subshell, z: float, mesh: RadialMesh) -> float:
    return local_expectation(subshell.p, -z / mesh.r, subshell.l, mesh)

def orbital_energy(
    subshells: tuple[Subshell, ...],
    a_index: int,
    z: int,
    mesh: RadialMesh,
    v_nuclear: Callable[[np.ndarray], np.ndarray] | None = None,
    *,
    exchange: bool = True,
) -> float:
    r = mesh.r
    a = subshells[a_index]
    v_nuc = (-z / r) if v_nuclear is None else np.asarray(v_nuclear(r), dtype=float)
    one = local_expectation(a.p, v_nuc, a.l, mesh)
    direct = float(np.trapezoid(a.p**2 * direct_potential(subshells, a_index, r), r))
    k_term = (
        float(np.trapezoid(a.p * exchange_apply(subshells, a_index, a.p, r), r))
        if exchange
        else 0.0
    )
    return one + direct - k_term

def _interaction_energy(
    subshells: tuple[Subshell, ...], r: np.ndarray, *, exchange: bool = True
) -> float:
    total = 0.0
    for i, a in enumerate(subshells):
        total += (a.q * (a.q - 1) / 2.0) * slater_f(a.p, a.p, r, 0)
        if exchange:
            for k in range(2, 2 * a.l + 1, 2):
                tj = wigner_3j(a.l, k, a.l, 0, 0, 0)
                coeff = ((2 * a.l + 1) / (4 * a.l + 1)) * tj * tj
                total -= (a.q * (a.q - 1) / 2.0) * coeff * slater_f(a.p, a.p, r, k)
        for b in subshells[i + 1:]:
            total += a.q * b.q * slater_f(a.p, b.p, r, 0)
            if exchange:
                for k in range(abs(a.l - b.l), a.l + b.l + 1):
                    tj = wigner_3j(a.l, k, b.l, 0, 0, 0)
                    total -= 0.5 * a.q * b.q * tj * tj * slater_g(a.p, b.p, r, k)
    return float(total)

def total_energy_direct(
    z: int, subshells: tuple[Subshell, ...], mesh: RadialMesh,
    *, exchange: bool = True,
) -> float:
    one = sum(a.q * one_electron_integral(a, z, mesh) for a in subshells)
    return float(
        one + _interaction_energy(subshells, mesh.r, exchange=exchange)
    )

def total_energy_from_orbitals(
    subshells: tuple[Subshell, ...],
    energies: tuple[float, ...],
    z: int,
    mesh: RadialMesh,
) -> float:
    return float(
        0.5
        * sum(
            a.q * (one_electron_integral(a, z, mesh) + e)
            for a, e in zip(subshells, energies, strict=True)
        )
    )

def kinetic_and_potential(
    z: int, subshells: tuple[Subshell, ...], mesh: RadialMesh,
    *, exchange: bool = True,
) -> tuple[float, float]:
    zero = np.zeros_like(mesh.r)
    kinetic = 0.0
    nuclear = 0.0
    for a in subshells:
        free = local_expectation(a.p, zero, a.l, mesh)
        kinetic += a.q * free
        nuclear += a.q * (one_electron_integral(a, z, mesh) - free)
    return float(kinetic), float(
        nuclear + _interaction_energy(subshells, mesh.r, exchange=exchange)
    )

def scf(
    z: int,
    subshells: tuple[Subshell, ...],
    v_nuclear: Callable[[np.ndarray], np.ndarray],
    mesh: RadialMesh,
    alpha: float = 0.65,
    max_iterations: int = 200,
    tol: float = 1e-8,
    *,
    exchange: bool = True,
) -> SCFSolution:
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"the mixing parameter must be in (0, 1], got {alpha}")

    current = tuple(subshells)
    energies = tuple(0.0 for _ in current)
    residuals: list[float] = []

    for iteration in range(1, max_iterations + 1):
        updated: list[Subshell] = []
        new_energies: list[float] = []
        for index, a in enumerate(current):
            k = a.n - a.l - 1
            channel = solve_channel(
                current, index, v_nuclear, a.l, mesh, n_states=k + 1,
                guess=a.p[None, :], exchange=exchange,
            )
            mixed = (1.0 - alpha) * a.p + alpha * channel.orbitals[k]
            mixed = mesh.normalized(mixed)
            updated.append(Subshell(n=a.n, l=a.l, q=a.q, p=mixed))
            new_energies.append(float(channel.energies[k]))

        residual = max(
            abs(new - old) for new, old in zip(new_energies, energies, strict=True)
        )
        residuals.append(residual)
        current, energies = tuple(updated), tuple(new_energies)
        if residual < tol and iteration > 1:
            return SCFSolution(current, energies, iteration, tuple(residuals))

    raise HFConvergenceError(
        f"the SCF did not converge in {max_iterations} iterations for Z={z}; "
        f"the last orbital-energy change was {residuals[-1]:.3e} hartree"
    )
