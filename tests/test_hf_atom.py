
import pytest

from atomic.atoms import aufbau_configuration
from atomic.hf_atom import solve_hartree_fock
from atomic.hf_reference import load_hf_reference
from atomic.provenance import Fidelity

CLOSED_SHELL = [("He", 2), ("Be", 4), ("Ne", 10), ("Mg", 12), ("Ar", 18)]

@pytest.fixture(scope="module")
def solved():
    return {
        symbol: solve_hartree_fock(z, z, aufbau_configuration(z))
        for symbol, z in CLOSED_SHELL
    }

@pytest.mark.parametrize("symbol,z", CLOSED_SHELL)
def test_total_energy_matches_the_vendored_reference(solved, symbol, z):
    reference = load_hf_reference(symbol)["total_energy_hartree"]
    if reference is None:
        pytest.skip("reference energies not yet transcribed from the source")
    got = solved[symbol].total_energy.value
    assert got == pytest.approx(reference, rel=1e-4)

@pytest.mark.parametrize("symbol,z", CLOSED_SHELL)
def test_energy_is_a_variational_upper_bound(solved, symbol, z):
    reference = load_hf_reference(symbol)["total_energy_hartree"]
    if reference is None:
        pytest.skip("reference energies not yet transcribed from the source")
    assert solved[symbol].total_energy.value > reference * (1.0 + 1e-4)

@pytest.mark.parametrize("symbol,z", CLOSED_SHELL)
def test_the_quoted_error_estimate_actually_brackets_the_error(solved, symbol, z):
    reference = load_hf_reference(symbol)["total_energy_hartree"]
    if reference is None:
        pytest.skip("reference energies not yet transcribed from the source")
    result = solved[symbol]
    deviation = abs(result.total_energy.value - reference)
    assert deviation < result.total_energy.provenance.error_estimate

@pytest.mark.parametrize("symbol,z", CLOSED_SHELL)
def test_virial_ratio_is_near_two(solved, symbol, z):
    assert solved[symbol].virial_ratio.value == pytest.approx(2.0, rel=2e-3)

@pytest.mark.parametrize("symbol,z", CLOSED_SHELL)
def test_orbital_energies_are_ordered_by_binding(solved, symbol, z):
    energies = [orbital.energy.value for orbital in solved[symbol].orbitals]
    assert energies == sorted(energies)
    assert all(e < 0.0 for e in energies)

@pytest.mark.parametrize("symbol,z", CLOSED_SHELL)
def test_total_energy_is_approximation_with_a_numerical_sub_scale(solved, symbol, z):
    energy = solved[symbol].total_energy
    assert energy.provenance.fidelity is Fidelity.APPROXIMATION
    assert energy.provenance.error_estimate is not None
    joined = " ".join(energy.provenance.assumptions)
    assert "correlation" in joined
    assert "variational" in joined

@pytest.mark.parametrize("symbol,z", CLOSED_SHELL)
def test_virial_ratio_is_numerical_not_approximation(solved, symbol, z):
    assert solved[symbol].virial_ratio.provenance.fidelity is Fidelity.NUMERICAL

@pytest.mark.parametrize("symbol,z", [(s, z) for s, z in CLOSED_SHELL if z > 2])
def test_each_orbital_carries_its_own_error_bar_not_the_totals(solved, symbol, z):
    result = solved[symbol]
    bars = [orbital.energy.provenance.error_estimate for orbital in result.orbitals]
    assert all(bar is not None and bar > 0.0 for bar in bars)
    assert len(set(bars)) == len(bars)
    assert all(bar < result.total_energy.provenance.error_estimate for bar in bars)

def test_beryllium_orbital_error_bars_bracket_the_published_energies(solved):
    got = [orbital.energy.value for orbital in solved["Be"].orbitals]
    bars = [orbital.energy.provenance.error_estimate for orbital in solved["Be"].orbitals]
    for value, bar, published in zip(got, bars, [-4.7326699, -0.3092695], strict=True):
        assert abs(value - published) < bar

@pytest.mark.parametrize("symbol,z", CLOSED_SHELL)
def test_the_amplitude_field_carries_no_energy_error_bar(solved, symbol, z):
    for orbital in solved[symbol].orbitals:
        assert orbital.P.unit == "bohr^-1/2"
        assert orbital.P.provenance.error_estimate is None
        assert orbital.P.provenance.fidelity is Fidelity.APPROXIMATION

def test_hydrogen_is_exact_to_the_grid():
    result = solve_hartree_fock(1, 1, aufbau_configuration(1))
    assert result.total_energy.value == pytest.approx(-0.5, rel=1e-4)

def test_hydrogen_error_estimate_brackets_the_exact_answer():
    result = solve_hartree_fock(1, 1, aufbau_configuration(1))
    residual = abs(result.total_energy.value - (-0.5))
    assert residual < result.total_energy.provenance.error_estimate

def test_result_records_its_convergence(solved):
    result = solved["He"]
    assert result.converged is True
    assert result.iterations > 1
    assert len(result.residual_history) == result.iterations

def test_configuration_must_match_the_electron_count():
    with pytest.raises(ValueError, match="configuration holds"):
        solve_hartree_fock(4, 4, aufbau_configuration(2))
