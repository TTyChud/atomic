
import numpy as np
import pytest

from atomic.atoms import aufbau_configuration
from atomic.hf_atom import (
    evaluate_hf_state,
    hf_radial,
    hf_valence_ionization_energy,
    solve_hartree_fock,
)
from atomic.provenance import Fidelity
from atomic.screened_atom import solve_screened_atom, valence_ionization_energy

HARTREE_TO_EV = 27.211386245981

NIST_IE_EV = {"He": (2, 24.587), "Li": (3, 5.392), "Na": (11, 5.139)}

ALKALIS = ["Li", "Na"]

def koopmans_ev(symbol):
    z, _ = NIST_IE_EV[symbol]
    result = solve_hartree_fock(z, z, aufbau_configuration(z))
    return hf_valence_ionization_energy(result).value * HARTREE_TO_EV

def gsz_ev(symbol):
    z, _ = NIST_IE_EV[symbol]
    result = solve_screened_atom(z, z, aufbau_configuration(z))
    return valence_ionization_energy(result).value * HARTREE_TO_EV

@pytest.mark.parametrize("symbol", list(NIST_IE_EV))
def test_ionization_energies_land_near_nist(symbol):
    assert abs(koopmans_ev(symbol) - NIST_IE_EV[symbol][1]) < 0.4

@pytest.mark.parametrize("symbol", ALKALIS)
def test_hartree_fock_beats_gsz_on_the_alkalis(symbol):
    reference = NIST_IE_EV[symbol][1]
    assert abs(koopmans_ev(symbol) - reference) < abs(gsz_ev(symbol) - reference)

def test_helium_is_the_case_where_gsz_wins_and_that_is_expected():
    reference = NIST_IE_EV["He"][1]
    koopmans_error = abs(koopmans_ev("He") - reference)
    gsz_error = abs(gsz_ev("He") - reference)
    assert koopmans_error > gsz_error
    assert koopmans_ev("He") > reference
    assert koopmans_error < 0.5

def test_the_ionization_energy_discloses_that_it_froze_the_orbitals():
    result = solve_hartree_fock(2, 2, aufbau_configuration(2))
    ie = hf_valence_ionization_energy(result)
    assert ie.provenance.fidelity is Fidelity.APPROXIMATION
    joined = " ".join(ie.provenance.assumptions) + " " + ie.provenance.method
    assert "Koopmans" in joined
    assert "relax" in joined

def test_hf_radial_returns_fields_with_matching_grids():
    r_field, p_field = hf_radial(2, 2, 1, 0)
    assert r_field.grid.shape == p_field.grid.shape
    assert np.allclose(p_field.values, r_field.grid**2 * r_field.values**2)

def test_radial_density_integrates_to_one():
    r_field, _ = hf_radial(2, 2, 1, 0)
    norm = np.trapezoid((r_field.grid * r_field.values) ** 2, r_field.grid)
    assert norm == pytest.approx(1.0, rel=1e-3)

def test_hf_radial_rejects_n_not_greater_than_l():
    with pytest.raises(ValueError):
        hf_radial(10, 10, 1, 1)

def test_hf_radial_rejects_a_subshell_the_configuration_does_not_hold():
    with pytest.raises(ValueError, match="not occupied"):
        hf_radial(2, 2, 3, 1)

def test_hf_radial_matches_the_solvers_own_orbital():
    result = solve_hartree_fock(4, 4, aufbau_configuration(4))
    orbital = next(o for o in result.orbitals if (o.n, o.l) == (2, 0))
    r_field, _ = hf_radial(4, 4, 2, 0)
    direct = np.interp(r_field.grid, orbital.P.grid, orbital.P.values)
    assert np.allclose(r_field.values * r_field.grid, direct, atol=1e-5)
    assert np.any(r_field.values > 0) and np.any(r_field.values < 0)

def test_evaluate_hf_state_shape_and_provenance():
    positions = np.array([[0.5, 0.0, 0.0], [0.0, 1.0, 0.0]])
    values = evaluate_hf_state(10, 10, 2, 1, 0, positions)
    assert values.values.shape == (2,)
    assert "Hartree-Fock" in values.provenance.method

def test_evaluate_hf_state_rejects_bad_positions():
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        evaluate_hf_state(2, 2, 1, 0, 0, np.array([0.5, 0.0, 0.0]))

def test_evaluate_hf_state_is_finite_at_the_origin():
    values = evaluate_hf_state(2, 2, 1, 0, 0, np.zeros((1, 3)))
    assert np.all(np.isfinite(values.values))

def test_screening_refinement_now_points_at_the_implementation():
    from atomic.numerics.screening import screening_provenance

    refinement = screening_provenance(10, 10).refinement
    assert "hf_atom" in refinement
    assert "a later phase" not in refinement
