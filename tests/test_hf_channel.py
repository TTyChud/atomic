
import numpy as np
import pytest
from scipy.linalg import eigh_tridiagonal

from atomic.numerics.hartree_fock import (
    HFConvergenceError,
    fock_operator,
    local_hamiltonian_bands,
    solve_channel,
)
from atomic.numerics.hf_terms import Subshell
from atomic.numerics.mesh import uniform_mesh


@pytest.fixture
def mesh():
    return uniform_mesh(40.0, 20000)

def coulomb(z):
    return lambda r: -z / r

def hydrogenic_1s(r, z):
    return 2.0 * z**1.5 * r * np.exp(-z * r)

def test_one_electron_channel_reproduces_hydrogen(mesh):
    shells = (Subshell(n=1, l=0, q=1, p=np.zeros_like(mesh.r)),)
    sol = solve_channel(shells, 0, coulomb(1.0), l=0, mesh=mesh, n_states=3,
                        guess=None)
    assert sol.energies[0] == pytest.approx(-0.5, rel=2e-4)
    assert sol.energies[1] == pytest.approx(-0.125, rel=2e-4)
    assert sol.energies[2] == pytest.approx(-1.0 / 18.0, rel=2e-3)

def test_one_electron_p_channel_reproduces_hydrogen(mesh):
    shells = (Subshell(n=2, l=1, q=1, p=np.zeros_like(mesh.r)),)
    sol = solve_channel(shells, 0, coulomb(1.0), l=1, mesh=mesh, n_states=2,
                        guess=None)
    assert sol.energies[0] == pytest.approx(-0.125, rel=2e-4)
    assert sol.energies[1] == pytest.approx(-1.0 / 18.0, rel=2e-3)

def test_returned_orbitals_are_normalized(mesh):
    shells = (Subshell(n=1, l=0, q=1, p=np.zeros_like(mesh.r)),)
    sol = solve_channel(shells, 0, coulomb(1.0), l=0, mesh=mesh, n_states=3,
                        guess=None)
    for u in sol.orbitals:
        assert np.trapezoid(u**2, mesh.r) == pytest.approx(1.0, rel=1e-8)

def test_returned_orbitals_are_orthogonal(mesh):
    shells = (Subshell(n=1, l=0, q=1, p=np.zeros_like(mesh.r)),)
    sol = solve_channel(shells, 0, coulomb(1.0), l=0, mesh=mesh, n_states=3,
                        guess=None)
    for i in range(3):
        for j in range(i + 1, 3):
            overlap = np.trapezoid(sol.orbitals[i] * sol.orbitals[j], mesh.r)
            assert abs(overlap) < 1e-8

def test_fock_operator_is_symmetric_on_random_vectors(mesh):
    rng = np.random.default_rng(0)
    p = mesh.r * np.exp(-mesh.r)
    p /= np.sqrt(np.trapezoid(p**2, mesh.r))
    shells = (Subshell(n=1, l=0, q=2, p=p),)
    op = fock_operator(shells, 0, coulomb(2.0), l=0, mesh=mesh)
    x = rng.standard_normal(mesh.points)
    y = rng.standard_normal(mesh.points)
    left = float(x @ op.matvec(y))
    right = float(op.matvec(x) @ y)
    assert left == pytest.approx(right, rel=1e-6)

def test_helium_direct_term_changes_the_answer(mesh):
    p = hydrogenic_1s(mesh.r, 2.0)
    shells = (Subshell(n=1, l=0, q=2, p=p),)
    sol = solve_channel(shells, 0, coulomb(2.0), l=0, mesh=mesh, n_states=1,
                        guess=None)
    assert -2.0 < sol.energies[0] < -0.5

def test_exchange_shifts_the_eigenvalue(mesh):
    p1 = hydrogenic_1s(mesh.r, 4.0)
    p2 = np.sqrt(2.0) * mesh.r * (1.0 - mesh.r) * np.exp(-mesh.r)
    p2 /= np.sqrt(np.trapezoid(p2**2, mesh.r))
    shells = (Subshell(n=1, l=0, q=2, p=p1), Subshell(n=2, l=0, q=2, p=p2))

    full = solve_channel(shells, 0, coulomb(4.0), l=0, mesh=mesh, n_states=1,
                         guess=None)

    from atomic.numerics.hf_terms import direct_potential

    v_local = coulomb(4.0)(mesh.r) + direct_potential(shells, 0, mesh.r)
    diag, offdiag = local_hamiltonian_bands(v_local, 0, mesh.r)
    local_only = eigh_tridiagonal(diag, offdiag, select="i",
                                  select_range=(0, 0), eigvals_only=True)[0]

    assert abs(full.energies[0] - local_only) > 1e-3
    assert full.energies[0] < local_only

def test_warm_start_does_not_cost_more_iterations(mesh):
    p = hydrogenic_1s(mesh.r, 2.0)
    shells = (Subshell(n=1, l=0, q=2, p=p),)
    cold = solve_channel(shells, 0, coulomb(2.0), l=0, mesh=mesh, n_states=1,
                         guess=None)
    warm = solve_channel(shells, 0, coulomb(2.0), l=0, mesh=mesh, n_states=1,
                         guess=cold.orbitals)
    assert warm.iterations <= cold.iterations
    assert warm.energies[0] == pytest.approx(cold.energies[0], rel=1e-8)

def test_iteration_count_shows_the_preconditioner_is_working(mesh):
    p = hydrogenic_1s(mesh.r, 2.0)
    shells = (Subshell(n=1, l=0, q=2, p=p),)
    sol = solve_channel(shells, 0, coulomb(2.0), l=0, mesh=mesh, n_states=1,
                        guess=None)
    assert sol.iterations < 50

def test_solution_reports_the_residual_it_achieved(mesh):
    p = hydrogenic_1s(mesh.r, 2.0)
    shells = (Subshell(n=1, l=0, q=2, p=p),)
    sol = solve_channel(shells, 0, coulomb(2.0), l=0, mesh=mesh, n_states=1,
                        guess=None)
    assert 0.0 < sol.residual < 1e-4

def test_unreachable_accuracy_raises_instead_of_returning_a_number(mesh):
    p = hydrogenic_1s(mesh.r, 2.0)
    shells = (Subshell(n=1, l=0, q=2, p=p),)
    with pytest.raises(HFConvergenceError, match="residual"):
        solve_channel(shells, 0, coulomb(2.0), l=0, mesh=mesh, n_states=1,
                      guess=None, tol=1e-30, residual_ceiling=1e-30,
                      maxiter=10)

def test_exchange_channel_converges_within_the_ceiling(mesh):
    p1 = hydrogenic_1s(mesh.r, 4.0)
    p2 = np.sqrt(2.0) * mesh.r * (1.0 - mesh.r) * np.exp(-mesh.r)
    p2 /= np.sqrt(np.trapezoid(p2**2, mesh.r))
    shells = (Subshell(n=1, l=0, q=2, p=p1), Subshell(n=2, l=0, q=2, p=p2))
    sol = solve_channel(shells, 0, coulomb(4.0), l=0, mesh=mesh, n_states=1,
                        guess=None)
    assert sol.residual < 1e-4

def test_local_bands_reject_a_grid_whose_origin_is_misplaced():
    bad = np.linspace(1e-5, 40.0, 20000)
    with pytest.raises(ValueError, match="r\\[0\\]"):
        local_hamiltonian_bands(-1.0 / bad, 0, bad)

def test_local_bands_reject_a_non_uniform_grid():
    bad = np.geomspace(1e-3, 40.0, 20000)
    with pytest.raises(ValueError, match="uniform"):
        local_hamiltonian_bands(-1.0 / bad, 0, bad)

def test_local_bands_match_the_radial_solver_discretization(mesh):
    from atomic.numerics.radial_solver import solve_radial

    diag, offdiag = local_hamiltonian_bands(coulomb(1.0)(mesh.r), 0, mesh.r)
    bands = eigh_tridiagonal(diag, offdiag, select="i", select_range=(0, 2),
                             eigvals_only=True)
    reference = solve_radial(coulomb(1.0), l=0, r_max=40.0,
                             n_points=mesh.points, n_states=3)
    assert np.allclose(bands, [e.value for e in reference.energies], rtol=1e-12)
