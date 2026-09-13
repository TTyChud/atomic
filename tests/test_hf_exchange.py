
import numpy as np
import pytest

from atomic.atoms import aufbau_configuration, element_by_symbol
from atomic.hf_atom import (
    _ROUTE_AGREEMENT,
    hf_exchange_energy,
    hf_mesh,
    solve_hartree_fock,
)
from atomic.numerics.hartree_fock import (
    orbital_energy,
    total_energy_direct,
    total_energy_from_orbitals,
)
from atomic.numerics.hf_terms import Subshell
from atomic.provenance import Fidelity

ATOMS = ["He", "Li", "Be", "C", "Ne"]

def _solve_both(symbol: str):
    z = element_by_symbol(symbol).z
    config = aufbau_configuration(z)
    return (
        solve_hartree_fock(z, z, config, True),
        solve_hartree_fock(z, z, config, False),
    )

def test_helium_exchange_energy_is_exactly_zero():
    assert hf_exchange_energy(2, 2, aufbau_configuration(2)).value == 0.0

def test_helium_is_the_same_atom_in_both_models():
    hf, hartree = _solve_both("He")
    assert hf.total_energy.value == hartree.total_energy.value
    np.testing.assert_array_equal(hf.orbitals[0].P.values, hartree.orbitals[0].P.values)

def test_beryllium_exchange_energy_is_not_zero():
    e_x = hf_exchange_energy(4, 4, aufbau_configuration(4)).value
    assert e_x < -0.01

@pytest.mark.parametrize("symbol", ATOMS)
def test_exchange_is_stabilizing(symbol):
    hf, hartree = _solve_both(symbol)
    assert hf.total_energy.value <= hartree.total_energy.value

def test_exchange_energy_grows_with_z():
    magnitudes = [
        abs(hf_exchange_energy(z, z, aufbau_configuration(z)).value)
        for z in (2, 3, 4, 6, 10)
    ]
    assert magnitudes == sorted(magnitudes)
    assert magnitudes[0] == 0.0

@pytest.mark.parametrize("symbol", ATOMS)
def test_both_models_satisfy_the_virial_theorem(symbol):
    _, hartree = _solve_both(symbol)
    assert hartree.virial_ratio.value == pytest.approx(2.0, abs=1e-4)

@pytest.mark.parametrize("symbol", ["Li", "Be"])
def test_exchange_changes_the_orbitals_not_only_the_energy(symbol):
    hf, hartree = _solve_both(symbol)
    valence_hf = hf.orbitals[-1].P.values
    valence_hartree = hartree.orbitals[-1].P.values
    assert not np.allclose(valence_hf, valence_hartree, atol=1e-6)

@pytest.mark.parametrize("symbol", ATOMS)
def test_a_hartree_solve_is_counterfactual_not_approximation(symbol):
    hf, hartree = _solve_both(symbol)
    assert hf.total_energy.provenance.fidelity is Fidelity.APPROXIMATION
    assert hartree.total_energy.provenance.fidelity is Fidelity.COUNTERFACTUAL
    assert hartree.orbitals[-1].P.provenance.fidelity is Fidelity.COUNTERFACTUAL

def test_the_disclosure_says_which_counterfactual_this_is():
    _, hartree = _solve_both("Ne")
    disclosure = " ".join(hartree.total_energy.provenance.assumptions).lower()
    assert "distinguishable" in disclosure
    assert "pauli principle is not switched off" in disclosure
    assert "2(2l+1)" in disclosure
    assert "does not repel itself" in disclosure
    assert "correlation" in disclosure

def test_the_exchange_energy_carries_no_error_bar_against_the_real_atom():
    q = hf_exchange_energy(4, 4, aufbau_configuration(4))
    assert q.provenance.fidelity is Fidelity.COUNTERFACTUAL
    assert "not an observable" in " ".join(q.provenance.assumptions)
    assert q.provenance.error_estimate is not None
    assert 0 < q.provenance.error_estimate < 0.01 * abs(q.value)

def test_the_two_models_do_not_share_a_cached_solve():
    config = aufbau_configuration(4)
    hf = solve_hartree_fock(4, 4, config, True)
    hartree = solve_hartree_fock(4, 4, config, False)
    assert hf is not hartree
    assert hf.key != hartree.key
    assert hf.exchange is True
    assert hartree.exchange is False

def test_half_applying_the_toggle_makes_the_two_energy_routes_disagree():
    hartree = solve_hartree_fock(4, 4, aufbau_configuration(4), False)
    mesh = hf_mesh(4, 4, n_top=2, refinement=2)
    subshells = tuple(
        Subshell(n=o.n, l=o.l, q=o.occupancy, p=o.P.values)
        for o in hartree.orbitals
    )

    mixed_energies = tuple(
        orbital_energy(subshells, i, 4, mesh, exchange=True)
        for i in range(len(subshells))
    )
    route_1 = total_energy_direct(4, subshells, mesh, exchange=False)
    route_2 = total_energy_from_orbitals(subshells, mixed_energies, 4, mesh)
    assert abs(route_1 - route_2) > 1000 * _ROUTE_AGREEMENT

    honest_energies = tuple(
        orbital_energy(subshells, i, 4, mesh, exchange=False)
        for i in range(len(subshells))
    )
    assert total_energy_from_orbitals(
        subshells, honest_energies, 4, mesh
    ) == pytest.approx(route_1, abs=_ROUTE_AGREEMENT)
