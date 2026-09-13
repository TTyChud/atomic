
from collections.abc import Callable

import numpy as np

from atomic.numerics.radial_solver import RadialSolution, solve_radial
from atomic.provenance import Fidelity, Provenance, Quantity

__all__ = [
    "dipole_matrix_element",
    "dipole_box_radius",
    "dipole_from_solutions",
    "grid_points_for",
]

_H_TARGET = 0.01

def dipole_box_radius(n_top: int, z_net: float = 1.0) -> float:
    return 10.0 * (n_top + 1) ** 2 / z_net

def grid_points_for(r_max: float, h_target: float = _H_TARGET) -> int:
    return int(np.ceil(r_max / h_target))

def dipole_from_solutions(
    sol_a: RadialSolution, k_a: int, sol_b: RadialSolution, k_b: int
) -> float:
    if sol_a.r.shape != sol_b.r.shape or not np.array_equal(sol_a.r, sol_b.r):
        raise ValueError(
            "the two states must be solved on one grid; got "
            f"{sol_a.r.size} and {sol_b.r.size} points"
        )
    r = sol_a.r
    return float(np.trapezoid(sol_a.u[k_a] * sol_b.u[k_b] * r, r))

def _overlap(
    potential: Callable[[np.ndarray], np.ndarray],
    l_a: int, k_a: int, l_b: int, k_b: int,
    r_max: float, n_points: int, mu_ratio: float,
) -> float:
    same_l = l_b == l_a
    states_a = max(k_a, k_b) + 1 if same_l else k_a + 1
    sol_a = solve_radial(
        potential, l=l_a, mu_ratio=mu_ratio, r_max=r_max,
        n_points=n_points, n_states=states_a,
    )
    sol_b = sol_a if same_l else solve_radial(
        potential, l=l_b, mu_ratio=mu_ratio, r_max=r_max,
        n_points=n_points, n_states=k_b + 1,
    )
    return dipole_from_solutions(sol_a, k_a, sol_b, k_b)

def dipole_matrix_element(
    potential: Callable[[np.ndarray], np.ndarray],
    l_a: int, k_a: int, l_b: int, k_b: int,
    n_top: int,
    z_net: float = 1.0,
    n_points: int | None = None,
    mu_ratio: float = 1.0,
) -> Quantity:
    if k_a < 0 or k_b < 0:
        raise ValueError(f"node indices must be >= 0, got k_a={k_a}, k_b={k_b}")
    if l_a < 0 or l_b < 0:
        raise ValueError(f"l must be >= 0, got l_a={l_a}, l_b={l_b}")
    r_max = dipole_box_radius(n_top, z_net)
    if n_points is None:
        n_points = grid_points_for(r_max)
    coarse = _overlap(potential, l_a, k_a, l_b, k_b, r_max, n_points, mu_ratio)
    fine = _overlap(potential, l_a, k_a, l_b, k_b, r_max, 2 * n_points, mu_ratio)
    return Quantity(
        value=fine,
        unit="bohr",
        label=f"<l={l_b},k={k_b}|r|l={l_a},k={k_a}>",
        provenance=Provenance(
            fidelity=Fidelity.NUMERICAL,
            method=(
                "overlap integral of u = rR from the finite-difference radial "
                "solver: <b|r|a> = integral u_a u_b r dr (both states on one grid)"
            ),
            assumptions=(
                f"both states on one uniform grid: r_max={r_max:g} bohr, N={2 * n_points}",
                "trapezoid quadrature on that solver grid",
                "only box-converged bound states mean anything here",
            ),
            error_estimate=abs(fine - coarse),
            refinement="raise n_points or r_max; this was estimated by grid-halving",
        ),
    )
