
import dataclasses
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from atomic.analytic.angular import spherical_harmonic
from atomic.analytic.wavefunction import WavefunctionValues
from atomic.atoms import Configuration, aufbau_configuration, is_ground
from atomic.numerics.dipole import (
    dipole_box_radius,
    dipole_from_solutions,
    grid_points_for,
)
from atomic.numerics.mesh import display_window
from atomic.numerics.radial_solver import RadialSolution, solve_radial, solve_radial_with_error
from atomic.numerics.screening import screened_potential, screening_provenance
from atomic.provenance import Fidelity, Field, Provenance, Quantity

_SCREENED_EVAL_POINTS = 4096

@dataclass(frozen=True)
class Orbital:
    n: int
    l: int
    occupancy: int
    energy: Quantity

@dataclass(frozen=True)
class ScreenedAtomResult:
    key: str
    z: int
    n_electrons: int
    config: Configuration
    is_ground: bool
    orbitals: tuple[Orbital, ...]
    total_energy: Quantity
    provenance: Provenance

def _r_max(z: int, n_electrons: int, n_top: int) -> float:
    z_net = z - n_electrons + 1
    return 40.0 * (n_top + 1) ** 2 / z_net

def _solve_energies(z: int, n_electrons: int, l: int, n_states: int) -> tuple[Quantity, ...]:
    potential = screened_potential(z, n_electrons)
    r_max = _r_max(z, n_electrons, n_states + l)
    sol = solve_radial_with_error(
        potential, l=l, mu_ratio=1.0, r_max=r_max, n_states=n_states
    )
    prov_model = screening_provenance(z, n_electrons)
    out = []
    for e in sol.energies:
        merged = Provenance(
            fidelity=Fidelity.APPROXIMATION,
            method=f"{prov_model.method}; radial Schrodinger equation solved numerically",
            assumptions=prov_model.assumptions + e.provenance.assumptions,
            error_estimate=e.provenance.error_estimate,
            refinement=prov_model.refinement,
        )
        out.append(dataclasses.replace(e, provenance=merged))
    return tuple(out)

def solve_screened_atom(
    z: int, n_electrons: int, config: Configuration,
    l_max: int = 2, n_states_per_l: int = 4,
) -> ScreenedAtomResult:
    occ = {nl: c for nl, c in config}
    l_top = max((l for (_, l), _ in config), default=0)
    n_top = max((n for (n, _), _ in config), default=1)
    l_max = max(l_max, l_top)
    n_states = max(n_states_per_l, n_top)

    orbitals: list[Orbital] = []
    for l in range(l_max + 1):
        energies = _solve_energies(z, n_electrons, l, n_states)
        for k, e in enumerate(energies):
            n = k + l + 1
            orbitals.append(Orbital(n=n, l=l, occupancy=occ.get((n, l), 0), energy=e))
    orbitals.sort(key=lambda o: (o.energy.value, o.n, o.l))

    total = sum(o.occupancy * o.energy.value for o in orbitals)
    total_prov = Provenance(
        fidelity=Fidelity.APPROXIMATION,
        method="summed the occupancy-weighted independent-particle orbital energies",
        assumptions=(
            "this is not a variational total energy: e-e double counting is ignored",
            f"uses a {'ground' if is_ground(config) else 'non-ground'} configuration",
        ),
    )
    return ScreenedAtomResult(
        key=f"z{z}n{n_electrons}",
        z=z, n_electrons=n_electrons, config=config, is_ground=is_ground(config),
        orbitals=tuple(orbitals),
        total_energy=Quantity(total, "hartree", "E_total", total_prov),
        provenance=screening_provenance(z, n_electrons),
    )

def valence_ionization_energy(result: ScreenedAtomResult) -> Quantity:
    occupied = [o for o in result.orbitals if o.occupancy > 0]
    if not occupied:
        raise ValueError("no occupied orbitals")
    valence = max(occupied, key=lambda o: o.energy.value)
    prov = dataclasses.replace(
        valence.energy.provenance,
        method=(
            valence.energy.provenance.method
            + "; the ionization energy is taken as -epsilon_valence"
        ),
    )
    return Quantity(-valence.energy.value, "hartree", "IE_valence", prov)

def screened_radial(
    z: int, n_electrons: int, n: int, l: int, points: int = 400,
) -> tuple[Field, Field]:
    if n <= l:
        raise ValueError(f"n must be > l, got n={n}, l={l}")
    k = n - l - 1
    potential = screened_potential(z, n_electrons)
    r_max = _r_max(z, n_electrons, n)
    sol = solve_radial_with_error(
        potential, l=l, mu_ratio=1.0, r_max=r_max, n_states=k + 1
    )
    r_solver = sol.r
    R = sol.u[k] / r_solver
    r_out = display_window(r_solver, r_solver**2 * R**2)
    grid = np.linspace(r_solver[0], r_out, points)
    R_i = np.interp(grid, r_solver, R)
    prov = Provenance(
        fidelity=Fidelity.APPROXIMATION,
        method=(
            f"{screening_provenance(z, n_electrons).method}; R_nl = u/r taken "
            f"numerically and drawn to r={grid[-1]:.3g} bohr, windowed from a "
            f"solve box of {r_solver[-1]:.3g}"
        ),
        assumptions=screening_provenance(z, n_electrons).assumptions,
        error_estimate=None,
    )
    r_field = Field(values=R_i, grid=grid, unit="bohr^-3/2", grid_unit="bohr",
                    label=f"R_{n},{l}(r)", provenance=prov)
    p_field = Field(values=grid**2 * R_i**2, grid=grid, unit="bohr^-1",
                    grid_unit="bohr", label=f"P_{n},{l}(r) = r^2 R^2", provenance=prov)
    return r_field, p_field

def _density_grid(z: int, n_electrons: int, n_top: int) -> tuple[float, int]:
    r_max = 4.0 * (n_top + 1) ** 2 / (z - n_electrons + 1)
    return r_max, int(np.ceil(r_max * 40.0 * z))

def screened_total_radial_density(
    z: int, n_electrons: int, config: Configuration | None = None,
    points: int = 400,
) -> Field:
    cfg = aufbau_configuration(n_electrons) if config is None else config
    occupied = [(nl, q) for nl, q in cfg if q > 0]
    if not occupied:
        raise ValueError("no occupied subshells, so there is no density to give")
    n_top = max(n for (n, _), _ in occupied)
    l_max = max(l for (_, l), _ in occupied)
    r_max, n_points = _density_grid(z, n_electrons, n_top)

    potential = screened_potential(z, n_electrons)
    channels = {
        l: solve_radial(
            potential, l=l, mu_ratio=1.0, r_max=r_max,
            n_points=n_points, n_states=n_top - l,
        )
        for l in range(l_max + 1)
    }

    solver_r = channels[0].r
    grid = np.geomspace(solver_r[0], solver_r[-1], points)
    values = np.zeros_like(grid)
    for (n, l), q in occupied:
        if n <= l:
            raise ValueError(f"n must be > l for a real subshell, got n={n}, l={l}")
        u = channels[l].u[n - l - 1]
        values += q * np.interp(grid, channels[l].r, u) ** 2

    n_total = sum(q for _, q in occupied)
    residual = abs(float(np.trapezoid(values, grid)) - float(n_total))
    model = screening_provenance(z, n_electrons)
    return Field(
        values=values, grid=grid, unit="electrons/bohr", grid_unit="bohr",
        label=f"D(r) = sum_a q_a u_a(r)^2 (N = {n_total})",
        provenance=Provenance(
            fidelity=Fidelity.APPROXIMATION,
            method=(
                f"{model.method}; summed the squared radial functions weighted "
                f"by occupancy, then resampled onto a logarithmic grid"
            ),
            assumptions=model.assumptions + (
                "GSZ was fitted to a potential, not to a density, so this shape "
                "is further from the model's own data than its energies",
                "one central field serves every electron, so the shells share a "
                "potential rather than each seeing the others",
                f"solved in a {r_max:.0f} bohr box at h = "
                f"{r_max / n_points:.1e} bohr, sized to hold the valence and "
                f"resolve the core",
            ),
            error_estimate=residual,
            refinement=(
                "an orbital-dependent potential instead, i.e. Hartree-Fock, "
                "which this engine also solves and which needs no fitted "
                "parameters"
            ),
        ),
    )

_DIPOLE_CHANNEL_CACHE = 16

@lru_cache(maxsize=_DIPOLE_CHANNEL_CACHE)
def _dipole_channel(
    z: int, n_electrons: int, l: int, r_max: float, n_points: int, n_states: int
) -> RadialSolution:
    return solve_radial(
        screened_potential(z, n_electrons), l=l, mu_ratio=1.0,
        r_max=r_max, n_points=n_points, n_states=n_states,
    )

def screened_dipole_integral(
    z: int, n_electrons: int, n_a: int, l_a: int, n_b: int, l_b: int,
    n_box: int | None = None,
) -> Quantity:
    for n, l, name in ((n_a, l_a, "a"), (n_b, l_b, "b")):
        if n <= l:
            raise ValueError(f"state {name}: n must be > l, got n={n}, l={l}")
    n_top = max(n_a, n_b, n_box or 0)
    z_net = z - n_electrons + 1
    r_max = dipole_box_radius(n_top, z_net)
    n_points = grid_points_for(r_max)

    def overlap(points: int) -> float:
        sol_a = _dipole_channel(z, n_electrons, l_a, r_max, points, n_top - l_a)
        sol_b = _dipole_channel(z, n_electrons, l_b, r_max, points, n_top - l_b)
        return dipole_from_solutions(sol_a, n_a - l_a - 1, sol_b, n_b - l_b - 1)

    coarse = overlap(n_points)
    fine = overlap(2 * n_points)
    model = screening_provenance(z, n_electrons)
    return Quantity(
        value=fine,
        unit="bohr",
        label=f"<{n_b},{l_b}|r|{n_a},{l_a}> (Z={z}, N={n_electrons})",
        provenance=Provenance(
            fidelity=Fidelity.APPROXIMATION,
            method=(
                f"{model.method}; the dipole is integral u_a u_b r dr over the "
                "numerically solved radials, both states on one grid"
            ),
            assumptions=model.assumptions
            + (
                f"one shared uniform grid: r_max={r_max:g} bohr, N={2 * n_points}",
                "GSZ model error dominates the grid-halving figure quoted here",
                "independent-particle: no correlation, no core polarization",
            ),
            error_estimate=abs(fine - coarse),
            refinement=model.refinement,
        ),
    )

def evaluate_screened_state(
    z: int,
    n_electrons: int,
    n: int,
    l: int,
    m: int,
    positions: np.ndarray,
    *,
    basis: str = "complex",
) -> WavefunctionValues:
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {pos.shape}")

    r = np.linalg.norm(pos, axis=1)
    safe_r = np.where(r > 0.0, r, 1.0)
    theta = np.arccos(np.clip(pos[:, 2] / safe_r, -1.0, 1.0))
    theta = np.where(r > 0.0, theta, 0.0)
    phi = np.arctan2(pos[:, 1], pos[:, 0])

    r_field, _ = screened_radial(z, n_electrons, n, l, points=_SCREENED_EVAL_POINTS)
    R = np.interp(r, r_field.grid, r_field.values, left=r_field.values[0], right=0.0)
    angular = spherical_harmonic(l, m, theta, phi, basis=basis)
    values = R * angular.values

    base = screening_provenance(z, n_electrons)
    prov = Provenance(
        fidelity=Fidelity.APPROXIMATION,
        method=(
            f"psi_nlm built as numerical screened R_nl (u/r) x "
            f"{angular.provenance.method}; {base.method}"
        ),
        assumptions=base.assumptions
        + angular.provenance.assumptions
        + ("values are in bohr^-3/2 at Cartesian positions in bohr",),
        error_estimate=r_field.provenance.error_estimate,
    )
    return WavefunctionValues(
        values=values, positions=pos, n=n, l=l, m=m, Z=z, mu_ratio=1.0,
        basis=basis, provenance=prov,
    )
