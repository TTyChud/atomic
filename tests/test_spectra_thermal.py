
import pytest

from atomic.atoms import aufbau_configuration
from atomic.populations import ThermalConditions
from atomic.provenance import Fidelity
from atomic.screened_atom import solve_screened_atom
from atomic.spectra import screened_transition_lines, transition_lines
from atomic.systems import get_system

H = get_system("h")
PHOTOSPHERE = 1e13

def _find(lines, n_up, n_low):
    for ln in lines:
        if (ln.n_upper, ln.n_lower) == (n_up, n_low):
            return ln
    raise AssertionError(f"no line {n_up} -> {n_low}")

def _lines(t, ne=PHOTOSPHERE, n_max=6, **kw):
    return transition_lines(H, n_max=n_max, thermal=ThermalConditions(t, ne), **kw)

def test_emissivity_is_off_unless_conditions_are_given():
    ll = transition_lines(H, n_max=4, intensities=True)
    assert all(ln.emissivity is None for ln in ll.lines)
    assert ll.thermal is None

def test_thermal_implies_intensities():
    ll = _lines(10000.0, n_max=4)
    assert all(ln.einstein_a is not None for ln in ll.lines)
    assert all(ln.emissivity is not None for ln in ll.lines)

def test_the_line_list_carries_the_conditions_that_produced_it():
    ll = _lines(9000.0, ne=1e12)
    assert ll.thermal.conditions.temperature_k == 9000.0
    assert ll.thermal.conditions.electron_density_cm3 == 1e12
    assert 0.0 <= ll.thermal.ionized_fraction.value <= 1.0
    assert ll.thermal.partition_function.value >= 2.0

def test_balmer_gains_on_lyman_as_the_gas_heats():
    ratios = []
    for t in (4000.0, 8000.0, 15000.0, 30000.0):
        ll = _lines(t)
        ratios.append(
            _find(ll.lines, 3, 2).emissivity.value
            / _find(ll.lines, 2, 1).emissivity.value
        )
    assert all(a < b for a, b in zip(ratios, ratios[1:], strict=False)), ratios

def test_the_spectrum_dims_once_the_gas_is_ionized():
    def total(t):
        return sum(ln.emissivity.value for ln in _lines(t).lines)

    assert total(9000.0) > total(4000.0), "should brighten before the knee"
    assert total(50000.0) < total(9000.0), "should dim after it"

def test_a_denser_gas_stays_neutral_and_therefore_brighter():
    hot = 12000.0
    thin = _lines(hot, ne=1e10)
    thick = _lines(hot, ne=1e16)
    assert thick.thermal.ionized_fraction.value < thin.thermal.ionized_fraction.value
    assert (
        _find(thick.lines, 2, 1).emissivity.value
        > _find(thin.lines, 2, 1).emissivity.value
    )

def test_emissivity_ordering_is_not_just_the_A_ordering():
    ll = _lines(6000.0)
    by_a = [id(x) for x in sorted(ll.lines, key=lambda x: -x.einstein_a.value)]
    by_eps = [id(x) for x in sorted(ll.lines, key=lambda x: -x.emissivity.value)]
    assert by_a != by_eps

def test_wavelengths_and_rates_do_not_move_when_conditions_are_added():
    plain = transition_lines(H, n_max=5, intensities=True)
    warm = _lines(10000.0, n_max=5)
    assert [x.wavelength.value for x in plain.lines] == [
        x.wavelength.value for x in warm.lines
    ]
    assert [x.einstein_a.value for x in plain.lines] == [
        x.einstein_a.value for x in warm.lines
    ]

def test_emissivity_provenance_names_LTE_and_optical_thinness():
    eps = _find(_lines(10000.0, n_max=4).lines, 2, 1).emissivity
    assert eps.provenance.fidelity is Fidelity.APPROXIMATION
    assert any("LTE" in a for a in eps.provenance.assumptions)
    assert any("optically thin" in a.lower() for a in eps.provenance.assumptions)
    assert eps.unit == "eV/s per atom"

def test_the_ionized_fraction_inherits_the_partition_function_cutoff():
    ll = _lines(10000.0, n_max=6)
    assumptions = ll.thermal.ionized_fraction.provenance.assumptions
    assert any("truncat" in a.lower() and "n_max=6" in a for a in assumptions)

def test_the_cutoff_actually_moves_the_ionization_at_high_temperature():
    hot = 40000.0
    shallow = transition_lines(
        H, n_max=3, thermal=ThermalConditions(hot, 1e19)
    ).thermal.ionized_fraction.value
    deep = transition_lines(
        H, n_max=10, thermal=ThermalConditions(hot, 1e19)
    ).thermal.ionized_fraction.value
    assert deep < shallow

def test_fine_structure_components_carry_their_own_emissivity():
    ll = transition_lines(
        H, n_max=3, fine_structure=True,
        thermal=ThermalConditions(10000.0, PHOTOSPHERE),
    )
    assert ll.lines and all(x.emissivity is not None for x in ll.lines)
    assert all(x.emissivity.value >= 0.0 for x in ll.lines)

def test_the_within_n_microwave_lines_are_negligible_when_weighted():
    ll = transition_lines(
        H, n_max=4, fine_structure=True,
        thermal=ThermalConditions(10000.0, PHOTOSPHERE),
    )
    within = [x for x in ll.lines if x.n_upper == x.n_lower]
    across = [x for x in ll.lines if x.n_upper != x.n_lower]
    assert within and across
    assert max(x.emissivity.value for x in within) < 1e-6 * max(
        x.emissivity.value for x in across
    )

def _screened(key_z, t, ne=PHOTOSPHERE):
    config = aufbau_configuration(key_z)
    from atomic.atoms import total_electrons

    result = solve_screened_atom(key_z, total_electrons(config), config)
    return screened_transition_lines(result, thermal=ThermalConditions(t, ne))

def test_screened_atoms_get_emissivity_too():
    ll = _screened(3, 8000.0)
    assert ll.lines and all(x.emissivity is not None for x in ll.lines)
    assert ll.thermal is not None

def test_screened_ionization_discloses_the_koopmans_estimate():
    ll = _screened(3, 8000.0)
    assumptions = ll.thermal.ionized_fraction.provenance.assumptions
    assert any("Koopmans" in a for a in assumptions)
    assert any("relaxation" in a for a in assumptions)

def test_lithium_ionizes_more_easily_than_hydrogen():
    t, ne = 6000.0, PHOTOSPHERE
    li = _screened(3, t, ne).thermal.ionized_fraction.value
    h = transition_lines(
        H, n_max=6, thermal=ThermalConditions(t, ne)
    ).thermal.ionized_fraction.value
    assert li > h
    assert li > 0.5 > h

def test_rejects_impossible_conditions():
    for bad in (ThermalConditions(0.0, 1e13), ThermalConditions(1e4, 0.0)):
        with pytest.raises(ValueError):
            transition_lines(H, n_max=3, thermal=bad)
