
import pytest

from atomic.analytic.transitions import lifetime
from atomic.atoms import aufbau_configuration
from atomic.provenance import Fidelity
from atomic.screened_atom import solve_screened_atom
from atomic.spectra import screened_transition_lines, transition_lines
from atomic.systems import get_system

H = get_system("h")

def _find(lines, n_up, l_up, n_low, l_low):
    for ln in lines:
        if (ln.n_upper, ln.l_upper, ln.n_lower, ln.l_lower) == (n_up, l_up, n_low, l_low):
            return ln
    raise AssertionError(f"no line {n_up}{l_up} -> {n_low}{l_low} in list")

def test_intensities_are_off_by_default():
    ll = transition_lines(H, n_max=4)
    assert all(ln.einstein_a is None for ln in ll.lines)
    assert all(ln.oscillator_strength is None for ln in ll.lines)

def test_intensities_do_not_disturb_wavelengths():
    plain = transition_lines(H, n_max=5)
    loud = transition_lines(H, n_max=5, intensities=True)
    assert [ln.wavelength.value for ln in plain.lines] == [
        ln.wavelength.value for ln in loud.lines
    ]

def test_lyman_alpha_rate_matches_nist():
    ll = transition_lines(H, n_max=4, intensities=True)
    lya = _find(ll.lines, 2, 1, 1, 0)
    assert lya.einstein_a.value == pytest.approx(6.27e8, rel=3e-3)
    assert lya.einstein_a.unit == "s^-1"

def test_lyman_alpha_is_the_strongest_line():
    ll = transition_lines(H, n_max=6, intensities=True)
    strongest = max(ll.lines, key=lambda ln: ln.einstein_a.value)
    assert (strongest.n_upper, strongest.l_upper) == (2, 1)
    assert (strongest.n_lower, strongest.l_lower) == (1, 0)

def test_h_alpha_is_the_strongest_balmer_line():
    ll = transition_lines(H, n_max=6, intensities=True)
    balmer = [ln for ln in ll.lines if ln.n_lower == 2]
    assert balmer, "expected Balmer lines at n_max = 6"
    assert max(balmer, key=lambda ln: ln.einstein_a.value).n_upper == 3

def test_every_listed_line_has_a_positive_rate():
    ll = transition_lines(H, n_max=6, intensities=True)
    assert all(ln.einstein_a.value > 0.0 for ln in ll.lines)
    assert all(ln.oscillator_strength.value > 0.0 for ln in ll.lines)

def test_rates_sum_to_the_independently_computed_lifetime():
    ll = transition_lines(H, n_max=6, intensities=True)
    mu = H.mu_ratio.value
    for n_up, l_up in [(2, 1), (3, 1), (3, 2), (4, 3), (5, 2)]:
        total = sum(
            ln.einstein_a.value
            for ln in ll.lines
            if (ln.n_upper, ln.l_upper) == (n_up, l_up)
        )
        assert total == pytest.approx(1.0 / lifetime(n_up, l_up, mu_ratio=mu).value, rel=1e-9)

def test_intensity_provenance_is_numerical():
    ll = transition_lines(H, n_max=3, intensities=True)
    a = ll.lines[0].einstein_a
    assert a.provenance.fidelity is Fidelity.NUMERICAL
    assert a.provenance.error_estimate is not None

def test_fine_structure_lines_now_carry_j_resolved_strengths():
    ll = transition_lines(H, n_max=4, fine_structure=True, intensities=True)
    assert ll.lines and all(ln.einstein_a is not None for ln in ll.lines)
    assert all(ln.einstein_a.value > 0.0 for ln in ll.lines)
    assert ll.intensity_note is None

def test_fine_structure_components_sum_to_the_gross_line_rate():
    gross = transition_lines(H, n_max=4, intensities=True)
    fine = transition_lines(H, n_max=4, fine_structure=True, intensities=True)
    for g in gross.lines:
        key = (g.n_upper, g.l_upper, g.n_lower, g.l_lower)
        by_upper_j: dict[float, float] = {}
        for f in fine.lines:
            if (f.n_upper, f.l_upper, f.n_lower, f.l_lower) == key:
                by_upper_j[f.j_upper] = by_upper_j.get(f.j_upper, 0.0) + f.einstein_a.value
        assert by_upper_j, f"no fine components for {key}"
        for total in by_upper_j.values():
            assert total == pytest.approx(g.einstein_a.value, rel=1e-3)

def test_gross_structure_intensities_carry_no_apology_note():
    ll = transition_lines(H, n_max=4, intensities=True)
    assert ll.intensity_note is None

def test_screened_lines_withhold_intensities_until_asked():
    result = solve_screened_atom(z=2, n_electrons=2, config=aufbau_configuration(2))
    ll = screened_transition_lines(result)
    assert all(ln.einstein_a is None for ln in ll.lines)
    assert ll.intensity_note is None

def test_intensities_track_the_isotope_through_reduced_mass():
    d = get_system("d")
    a_h = _find(transition_lines(H, n_max=3, intensities=True).lines, 2, 1, 1, 0)
    a_d = _find(transition_lines(d, n_max=3, intensities=True).lines, 2, 1, 1, 0)
    assert a_d.einstein_a.value > a_h.einstein_a.value
    assert a_d.einstein_a.value / a_h.einstein_a.value == pytest.approx(1.0, abs=1e-3)
