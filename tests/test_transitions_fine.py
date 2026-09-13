
import pytest

from atomic.analytic.transitions import (
    einstein_A,
    einstein_A_fine,
    lifetime,
    lifetime_fine,
    oscillator_strength_fine,
)
from atomic.provenance import Fidelity


def test_sums_to_the_gross_rate():
    for n_up, l_up, n_low, l_low in [
        (2, 1, 1, 0), (3, 1, 1, 0), (3, 2, 2, 1), (4, 3, 3, 2), (5, 1, 3, 2),
    ]:
        gross = einstein_A(n_up, l_up, n_low, l_low).value
        for j_up in ([l_up - 0.5, l_up + 0.5] if l_up > 0 else [0.5]):
            js = [l_low - 0.5, l_low + 0.5] if l_low > 0 else [0.5]
            total = sum(
                einstein_A_fine(n_up, l_up, j_up, n_low, l_low, j).value for j in js
            )
            assert total == pytest.approx(gross, rel=1e-9)

def test_the_two_p_components_have_equal_rates_not_a_two_to_one_ratio():
    upper_3_2 = einstein_A_fine(2, 1, 1.5, 1, 0, 0.5).value
    upper_1_2 = einstein_A_fine(2, 1, 0.5, 1, 0, 0.5).value
    assert upper_3_2 == pytest.approx(upper_1_2, rel=1e-9)

def test_degeneracy_weighted_rates_do_show_the_two_to_one_ratio():
    strong = 4 * einstein_A_fine(2, 1, 1.5, 1, 0, 0.5).value
    weak = 2 * einstein_A_fine(2, 1, 0.5, 1, 0, 0.5).value
    assert strong / weak == pytest.approx(2.0, rel=1e-9)

def test_d_to_p_branching_ratio_is_five_to_one():
    to_p_half = einstein_A_fine(3, 2, 1.5, 2, 1, 0.5).value
    to_p_three_halves = einstein_A_fine(3, 2, 1.5, 2, 1, 1.5).value
    assert to_p_half / to_p_three_halves == pytest.approx(5.0, rel=1e-9)

def test_f_to_d_branching_ratio_is_fourteen_to_one():
    favoured = einstein_A_fine(4, 3, 2.5, 3, 2, 1.5).value
    other = einstein_A_fine(4, 3, 2.5, 3, 2, 2.5).value
    assert favoured / other == pytest.approx(14.0, rel=1e-9)

def test_both_2p_fine_levels_keep_the_gross_lifetime():
    gross = lifetime(2, 1).value
    for j in (0.5, 1.5):
        assert lifetime_fine(2, 1, j).value == pytest.approx(gross, rel=1e-9)

def test_delta_j_selection_rule_gives_exact_zeros():
    assert einstein_A_fine(3, 2, 2.5, 2, 0, 0.5).value == 0.0
    assert einstein_A_fine(3, 0, 0.5, 2, 0, 0.5).value == 0.0

def test_delta_j_zero_is_allowed_when_l_changes():
    assert einstein_A_fine(2, 1, 0.5, 1, 0, 0.5).value > 0.0

def test_rejects_a_j_that_does_not_belong_to_its_l():
    with pytest.raises(ValueError):
        einstein_A_fine(2, 1, 2.5, 1, 0, 0.5)
    with pytest.raises(ValueError):
        einstein_A_fine(2, 1, 1.5, 1, 0, 1.5)

def test_fine_oscillator_strengths_sum_over_upper_j_to_the_gross_value():
    from atomic.analytic.transitions import oscillator_strength

    gross = oscillator_strength(1, 0, 2, 1).value
    parts = [oscillator_strength_fine(1, 0, 0.5, 2, 1, j).value for j in (0.5, 1.5)]
    assert all(p > 0 for p in parts)
    assert sum(parts) == pytest.approx(gross, rel=1e-9)

def test_provenance_names_the_6j_and_stays_numerical():
    a = einstein_A_fine(2, 1, 1.5, 1, 0, 0.5)
    assert a.provenance.fidelity is Fidelity.NUMERICAL
    assert "6j" in a.provenance.method
    assert a.unit == "s^-1"

def test_every_allowed_component_is_positive_and_finite():
    import math

    for n_up, l_up, n_low, l_low in [(3, 1, 2, 0), (4, 2, 2, 1), (5, 3, 4, 2)]:
        for j_up in (l_up - 0.5, l_up + 0.5):
            for j_low in ([l_low - 0.5, l_low + 0.5] if l_low > 0 else [0.5]):
                v = einstein_A_fine(n_up, l_up, j_up, n_low, l_low, j_low).value
                assert v >= 0.0 and math.isfinite(v)
