
import numpy as np
import pytest

from atomic.numerics.hartree_fock import (
    HFConvergenceError,
    kinetic_and_potential,
    orbital_energy,
    scf,
    solve_channel,
    total_energy_direct,
    total_energy_from_orbitals,
)
from atomic.numerics.hf_terms import Subshell
from atomic.numerics.mesh import uniform_mesh


def coulomb(z):
    return lambda r: -z / r

def start(mesh, z, shells):
    p = 2.0 * z**1.5 * mesh.r * np.exp(-z * mesh.r)
    p /= np.sqrt(np.trapezoid(p**2, mesh.r))
    return tuple(Subshell(n=n, l=l, q=q, p=p.copy()) for n, l, q in shells)

@pytest.fixture(scope="module")
def mesh():
    return uniform_mesh(30.0, 30000)

@pytest.fixture(scope="module")
def hydrogen(mesh):
    return scf(1, start(mesh, 1.0, [(1, 0, 1)]), coulomb(1.0), mesh)

@pytest.fixture(scope="module")
def helium(mesh):
    return scf(2, start(mesh, 1.7, [(1, 0, 2)]), coulomb(2.0), mesh)

@pytest.fixture(scope="module")
def beryllium(mesh):
    return scf(4, start(mesh, 3.0, [(1, 0, 2), (2, 0, 2)]), coulomb(4.0), mesh)

def quadrature_energies(sol, z, mesh):
    return tuple(
        orbital_energy(sol.subshells, i, z, mesh)
        for i in range(len(sol.subshells))
    )

def test_hydrogen_is_exactly_minus_one_half(hydrogen, mesh):
    assert total_energy_direct(1, hydrogen.subshells, mesh) == pytest.approx(
        -0.5, rel=2e-4
    )
    assert hydrogen.energies[0] == pytest.approx(-0.5, rel=2e-4)

def test_helium_total_energy_is_physical(helium, mesh):
    e = total_energy_direct(2, helium.subshells, mesh)
    assert -4.0 < e < -2.0

def test_the_two_energy_routes_agree(helium, mesh):
    direct = total_energy_direct(2, helium.subshells, mesh)
    identity = total_energy_from_orbitals(
        helium.subshells, quadrature_energies(helium, 2, mesh), 2, mesh
    )
    assert direct == pytest.approx(identity, abs=1e-8)

def test_the_two_energy_routes_agree_for_beryllium(beryllium, mesh):
    direct = total_energy_direct(4, beryllium.subshells, mesh)
    identity = total_energy_from_orbitals(
        beryllium.subshells, quadrature_energies(beryllium, 4, mesh), 4, mesh
    )
    assert direct == pytest.approx(identity, abs=1e-8)

def test_eigenvalue_and_quadrature_orbital_energies_agree(helium, mesh):
    eig = helium.energies[0]
    quad = orbital_energy(helium.subshells, 0, 2, mesh)
    assert quad == pytest.approx(eig, abs=1e-4)
    assert quad != eig

def test_the_eigenvalue_gap_is_quadrature_not_convergence(helium, mesh):
    loose = solve_channel(
        helium.subshells, 0, coulomb(2.0), l=0, mesh=mesh, n_states=1,
        guess=helium.subshells[0].p[None, :], tol=1e-4,
    )
    tight = solve_channel(
        helium.subshells, 0, coulomb(2.0), l=0, mesh=mesh, n_states=1,
        guess=helium.subshells[0].p[None, :], tol=1e-10,
    )
    assert tight.residual < 0.1 * loose.residual

    quad = orbital_energy(helium.subshells, 0, 2, mesh)
    assert (quad - tight.energies[0]) == pytest.approx(
        quad - loose.energies[0], rel=1e-3
    )

def test_virial_ratio_is_two(helium, mesh):
    t, v = kinetic_and_potential(2, helium.subshells, mesh)
    assert -v / t == pytest.approx(2.0, rel=1e-3)

def test_energy_equals_minus_kinetic_at_convergence(helium, mesh):
    t, _ = kinetic_and_potential(2, helium.subshells, mesh)
    e = total_energy_direct(2, helium.subshells, mesh)
    assert e == pytest.approx(-t, rel=1e-3)

def test_beryllium_converges_and_orders_its_shells(beryllium):
    assert beryllium.energies[0] < beryllium.energies[1] < 0.0

def test_beryllium_orbital_energies_match_the_published_ones(beryllium):
    assert beryllium.energies[0] == pytest.approx(-4.7326699, rel=1e-4)
    assert beryllium.energies[1] == pytest.approx(-0.3092695, rel=1e-4)

def test_beryllium_shells_come_out_orthogonal(beryllium, mesh):
    overlap = np.trapezoid(
        beryllium.subshells[0].p * beryllium.subshells[1].p, mesh.r
    )
    assert abs(overlap) < 1e-5

def test_residual_history_is_monotone_enough_to_show_convergence(helium):
    assert helium.residual_history[-1] < helium.residual_history[0]
    assert helium.residual_history[-1] < 1e-8

def test_non_convergence_raises_rather_than_returning(mesh):
    with pytest.raises(HFConvergenceError, match="SCF did not converge"):
        scf(2, start(mesh, 1.7, [(1, 0, 2)]), coulomb(2.0), mesh,
            max_iterations=1, tol=1e-14)

def test_mixing_parameter_is_validated(mesh):
    with pytest.raises(ValueError, match="mixing parameter"):
        scf(2, start(mesh, 1.7, [(1, 0, 2)]), coulomb(2.0), mesh, alpha=0.0)
