
import pytest

from atomic.atoms import (
    NO_GSZ_PARAMETERS,
    aufbau_configuration,
    is_single_term,
)
from atomic.hf_atom import solve_hartree_fock

SINGLE_TERM = [("Li", 3), ("B", 5), ("F", 9), ("Na", 11), ("Al", 13), ("Cl", 17)]
MULTI_TERM = [("C", 6), ("N", 7), ("O", 8), ("Si", 14), ("P", 15), ("S", 16)]
OPEN_SHELL = SINGLE_TERM + MULTI_TERM
NO_GSZ = [("S", 16), ("Cl", 17)]

_NEEDED = sorted({z for _, z in OPEN_SHELL} | {z - 1 for _, z in OPEN_SHELL} | {10})

@pytest.fixture(scope="module")
def solved():
    return {z: solve_hartree_fock(z, z, aufbau_configuration(z)) for z in _NEEDED}

@pytest.mark.parametrize("symbol,z", OPEN_SHELL)
def test_open_shell_atoms_converge(solved, symbol, z):
    assert solved[z].converged
    assert solved[z].total_energy.value < 0.0

@pytest.mark.parametrize("symbol,z", NO_GSZ)
def test_atoms_gsz_cannot_do_now_work(solved, symbol, z):
    assert z in NO_GSZ_PARAMETERS
    assert solved[z].total_energy.value < 0.0

@pytest.mark.parametrize("symbol,z", OPEN_SHELL)
def test_total_energy_decreases_monotonically_with_z(solved, symbol, z):
    assert solved[z].total_energy.value < solved[z - 1].total_energy.value

@pytest.mark.parametrize("symbol,z", MULTI_TERM)
def test_multi_term_atoms_disclose_the_configuration_average(solved, symbol, z):
    joined = " ".join(solved[z].total_energy.provenance.assumptions)
    assert "average of configuration" in joined

@pytest.mark.parametrize("symbol,z", OPEN_SHELL)
def test_open_shells_disclose_the_spin_restriction(solved, symbol, z):
    joined = " ".join(solved[z].total_energy.provenance.assumptions)
    assert "spin-polarize" in joined

@pytest.mark.parametrize("symbol,z", SINGLE_TERM)
def test_single_term_atoms_do_not_claim_a_term_average_they_do_not_make(
    solved, symbol, z
):
    assert is_single_term(aufbau_configuration(z))
    joined = " ".join(solved[z].total_energy.provenance.assumptions)
    assert "not per term" not in joined

def test_closed_shell_does_not_claim_a_term_limitation_it_does_not_have(solved):
    joined = " ".join(solved[10].total_energy.provenance.assumptions)
    assert "not per term" not in joined
    assert "spin-polarize" not in joined

def test_the_orbital_shape_carries_the_same_disclosure_as_the_energy(solved):
    for orbital in solved[6].orbitals:
        joined = " ".join(orbital.P.provenance.assumptions)
        assert "average of configuration" in joined

def test_virial_ratio_holds_for_open_shells(solved):
    assert solved[7].virial_ratio.value == pytest.approx(2.0, rel=5e-3)

def test_half_filled_shell_energies_are_ordered_by_shell(solved):
    energies = {(o.n, o.l): o.energy.value for o in solved[7].orbitals}
    assert energies[(1, 0)] < energies[(2, 0)] < energies[(2, 1)] < 0.0
