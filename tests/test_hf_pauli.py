
import pytest

from atomic.atoms import (
    aufbau_configuration,
    element_by_symbol,
    is_ground,
    subshell_terms,
    validate_config,
)
from atomic.hf_atom import (
    collapsed_variational_energy,
    hf_mean_radius,
    pauli_collapse,
    solve_hartree_fock,
)
from atomic.provenance import Fidelity

COLLAPSE_ATOMS = ["He", "Be", "Ne"]

def _collapsed_config(n_electrons: int):
    return aufbau_configuration(n_electrons, pauli=False)

@pytest.mark.parametrize("n_electrons", [1, 2, 3, 10, 18])
def test_aufbau_without_pauli_is_one_orbital(n_electrons):
    assert _collapsed_config(n_electrons) == (((1, 0), n_electrons),)

def test_collapsed_configuration_is_ground_only_under_its_own_rule():
    collapsed = _collapsed_config(10)
    assert is_ground(collapsed, pauli=False)
    assert not is_ground(collapsed, pauli=True)

def test_validate_drops_the_cap_but_keeps_n_greater_than_l():
    validate_config((((1, 0), 10),), pauli=False)
    with pytest.raises(ValueError, match="exceeds capacity"):
        validate_config((((1, 0), 10),), pauli=True)
    with pytest.raises(ValueError, match="n must be > l"):
        validate_config((((1, 1), 4),), pauli=False)

def test_term_structure_raises_above_capacity_rather_than_answering():
    with pytest.raises(ValueError, match="out of range"):
        subshell_terms(0, 10)

def test_pauli_off_with_exchange_on_is_refused_not_silently_corrected():
    with pytest.raises(ValueError, match="not a model"):
        solve_hartree_fock(4, 4, _collapsed_config(4), exchange=True, pauli=False)

def test_the_refusal_says_why_and_says_what_to_pass():
    message = str(
        pytest.raises(
            ValueError,
            solve_hartree_fock,
            4, 4, _collapsed_config(4), True, False,
        ).value
    )
    assert "antisymmetry" in message
    assert "exchange=False" in message

def test_variational_formula_reproduces_the_textbook_helium_number():
    zeta, energy = collapsed_variational_energy(2, 2)
    assert zeta.value == pytest.approx(1.6875, abs=1e-12)
    assert energy.value == pytest.approx(-2.84765625, abs=1e-12)

def test_variational_formula_is_counterfactual_despite_being_closed_form():
    zeta, energy = collapsed_variational_energy(10, 10)
    assert zeta.provenance.fidelity is Fidelity.COUNTERFACTUAL
    assert energy.provenance.fidelity is Fidelity.COUNTERFACTUAL
    assert zeta.value == pytest.approx(7.1875, abs=1e-12)
    assert energy.value == pytest.approx(-258.30078125, abs=1e-9)

@pytest.mark.parametrize("symbol", COLLAPSE_ATOMS)
def test_collapsed_scf_sits_below_the_exponential_bound_and_near_it(symbol):
    z = element_by_symbol(symbol).z
    collapsed = solve_hartree_fock(
        z, z, _collapsed_config(z), exchange=False, pauli=False
    )
    _, bound = collapsed_variational_energy(z, z)
    assert collapsed.total_energy.value <= bound.value
    assert collapsed.total_energy.value == pytest.approx(bound.value, rel=0.05)

@pytest.mark.parametrize("symbol", COLLAPSE_ATOMS)
def test_the_collapsed_ladder_has_exactly_one_rung(symbol):
    z = element_by_symbol(symbol).z
    collapsed = solve_hartree_fock(
        z, z, _collapsed_config(z), exchange=False, pauli=False
    )
    assert len(collapsed.orbitals) == 1
    only = collapsed.orbitals[0]
    assert (only.n, only.l, only.occupancy) == (1, 0, z)

def test_helium_is_the_case_where_switching_pauli_off_changes_nothing():
    hartree = solve_hartree_fock(2, 2, aufbau_configuration(2), exchange=False)
    collapsed = solve_hartree_fock(
        2, 2, _collapsed_config(2), exchange=False, pauli=False
    )
    assert collapsed.total_energy.value == hartree.total_energy.value
    assert hf_mean_radius(collapsed).value == hf_mean_radius(hartree).value

def test_hydrogen_mean_radius_is_the_analytic_three_halves():
    hydrogen = solve_hartree_fock(1, 1, aufbau_configuration(1))
    assert hf_mean_radius(hydrogen).value == pytest.approx(1.5, rel=2e-4)

def test_mean_radius_carries_no_error_bar_in_the_wrong_dimension():
    radius = hf_mean_radius(solve_hartree_fock(4, 4, aufbau_configuration(4)))
    assert radius.unit == "bohr"
    assert radius.provenance.error_estimate is None

def test_the_collapsed_atom_is_far_more_bound():
    collapse = pauli_collapse(10)
    assert collapse.binding_change.value < 0
    assert collapse.collapsed.total_energy.value < 2 * collapse.real.total_energy.value

def test_collapsed_size_falls_monotonically_while_the_real_one_does_not():
    collapses = [pauli_collapse(z) for z in (2, 4, 10)]
    collapsed_radii = [c.collapsed_radius.value for c in collapses]
    real_radii = [c.real_radius.value for c in collapses]

    assert collapsed_radii == sorted(collapsed_radii, reverse=True)
    assert real_radii != sorted(real_radii, reverse=True)
    for collapse in collapses[1:]:
        assert collapse.radius_ratio.value < 1.0

def test_collapse_compares_two_solves_of_the_same_atom():
    collapse = pauli_collapse(4)
    assert collapse.real.z == collapse.collapsed.z == 4
    assert collapse.real.n_electrons == collapse.collapsed.n_electrons == 4
    assert collapse.real.exchange and collapse.real.pauli
    assert not collapse.collapsed.exchange and not collapse.collapsed.pauli

def test_the_three_models_never_share_a_cache_key():
    config = aufbau_configuration(4)
    keys = {
        solve_hartree_fock(4, 4, config).key,
        solve_hartree_fock(4, 4, config, exchange=False).key,
        solve_hartree_fock(4, 4, _collapsed_config(4), False, False).key,
    }
    assert len(keys) == 3

def test_collapsed_solve_is_counterfactual_everywhere_it_reports():
    z = 4
    collapsed = solve_hartree_fock(
        z, z, _collapsed_config(z), exchange=False, pauli=False
    )
    assert collapsed.total_energy.provenance.fidelity is Fidelity.COUNTERFACTUAL
    for orbital in collapsed.orbitals:
        assert orbital.energy.provenance.fidelity is Fidelity.COUNTERFACTUAL
        assert orbital.P.provenance.fidelity is Fidelity.COUNTERFACTUAL

def test_the_alteration_leads_the_assumption_list():
    collapsed = solve_hartree_fock(4, 4, _collapsed_config(4), False, False)
    first = collapsed.total_energy.provenance.assumptions[0]
    assert first.startswith("COUNTERFACTUAL:")
    assert "Pauli exclusion principle is switched off" in first

def test_disclosure_contradicts_the_weaker_counterfactual_it_supersedes():
    hartree = solve_hartree_fock(4, 4, aufbau_configuration(4), exchange=False)
    collapsed = solve_hartree_fock(4, 4, _collapsed_config(4), False, False)

    hartree_text = " ".join(hartree.total_energy.provenance.assumptions)
    collapsed_text = " ".join(collapsed.total_energy.provenance.assumptions)

    assert "the Pauli principle is NOT switched off" in hartree_text
    assert "the Pauli principle is NOT switched off" not in collapsed_text
    assert "occupancy cap is gone" in collapsed_text

def test_disclosure_says_exchange_was_forced_off_rather_than_chosen():
    collapsed = solve_hartree_fock(4, 4, _collapsed_config(4), False, False)
    text = " ".join(collapsed.total_energy.provenance.assumptions)
    assert "not a separate choice" in text

def test_term_structure_is_replaced_rather_than_omitted():
    collapsed = solve_hartree_fock(4, 4, _collapsed_config(4), False, False)
    text = " ".join(collapsed.total_energy.provenance.assumptions)
    assert "term structure is undefined" in text
    assert "average of configuration" not in text

def test_refinement_does_not_promise_a_better_calculation():
    collapsed = solve_hartree_fock(4, 4, _collapsed_config(4), False, False)
    refinement = collapsed.total_energy.provenance.refinement
    assert "turn the exclusion principle back on" in refinement

def test_the_comparison_itself_is_counterfactual_and_says_it_is_not_observable():
    collapse = pauli_collapse(4)
    prov = collapse.binding_change.provenance
    assert prov.fidelity is Fidelity.COUNTERFACTUAL
    assert any("not an observable" in a for a in prov.assumptions)
    assert prov.error_estimate > 0
    assert collapse.radius_ratio.provenance.error_estimate is None
