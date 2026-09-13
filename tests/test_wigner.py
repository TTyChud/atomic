
import math

import pytest

from atomic.analytic.wigner import triangular, wigner_6j


def test_zero_argument_closed_form():
    for j2 in (0.5, 1.5):
        assert wigner_6j(0.5, 1, j2, 1, 0.5, 0) ** 2 == pytest.approx(1 / 6, rel=1e-12)

def test_zero_argument_closed_form_over_a_range():
    for a in (0.5, 1.0, 1.5, 2.0, 2.5):
        for b in (1.0, 2.0):
            for c in (abs(a - b), a + b):
                expected = ((-1) ** (a + b + c)) / math.sqrt((2 * a + 1) * (2 * b + 1))
                assert wigner_6j(a, b, c, b, a, 0) == pytest.approx(expected, rel=1e-10)

def test_symmetric_under_permuting_columns():
    args = (1.0, 2.0, 2.0, 1.5, 1.5, 2.5)
    j1, j2, j3, j4, j5, j6 = args
    base = wigner_6j(*args)
    assert base != 0.0
    assert wigner_6j(j2, j1, j3, j5, j4, j6) == pytest.approx(base, rel=1e-12)
    assert wigner_6j(j3, j2, j1, j6, j5, j4) == pytest.approx(base, rel=1e-12)
    assert wigner_6j(j1, j3, j2, j4, j6, j5) == pytest.approx(base, rel=1e-12)

def test_symmetric_under_swapping_upper_and_lower_in_two_columns():
    j1, j2, j3, j4, j5, j6 = 1.0, 2.0, 2.0, 1.5, 1.5, 2.5
    base = wigner_6j(j1, j2, j3, j4, j5, j6)
    assert wigner_6j(j4, j5, j3, j1, j2, j6) == pytest.approx(base, rel=1e-12)
    assert wigner_6j(j1, j5, j6, j4, j2, j3) == pytest.approx(base, rel=1e-12)

def test_triangle_violations_are_exactly_zero():
    assert triangular(1, 1, 5) is False
    assert wigner_6j(1, 1, 5, 1, 1, 1) == 0.0
    assert wigner_6j(1, 1, 1, 1, 1, 9) == 0.0
    assert wigner_6j(0.5, 0.5, 0.5, 1, 1, 1) == 0.0

def test_rejects_negative_and_non_half_integer_arguments():
    with pytest.raises(ValueError):
        wigner_6j(-1, 1, 1, 1, 1, 1)
    with pytest.raises(ValueError):
        wigner_6j(0.3, 1, 1, 1, 1, 1)

@pytest.mark.parametrize("l_up", [1, 2, 3, 4])
def test_the_sum_rule_the_line_strengths_rest_on(l_up):
    for l_low in (l_up - 1, l_up + 1):
        if l_low < 0:
            continue
        for j_up in ([l_up - 0.5, l_up + 0.5] if l_up > 0 else [0.5]):
            js = [l_low - 0.5, l_low + 0.5] if l_low > 0 else [0.5]
            total = sum(
                (2 * j + 1) * wigner_6j(j, 1, j_up, l_up, 0.5, l_low) ** 2 for j in js
            )
            assert total == pytest.approx(1.0 / (2 * l_up + 1), rel=1e-10)

def test_known_tabulated_value():
    assert wigner_6j(1, 1, 1, 1, 1, 1) == pytest.approx(1 / 6, rel=1e-12)

def _sum_over_x(a, b, c, d, y, yp):
    total = 0.0
    x = abs(a - b)
    while x <= a + b + 1e-9:
        total += (
            (2 * x + 1) * wigner_6j(a, b, x, c, d, y) * wigner_6j(a, b, x, c, d, yp)
        )
        x += 1
    return total

@pytest.mark.parametrize(
    ("a", "b", "c", "d", "y"),
    [(1, 1, 1, 1, 1), (2, 1, 2, 1, 1), (1.5, 1.5, 0.5, 0.5, 1)],
)
def test_orthogonality_diagonal_term_is_one(a, b, c, d, y):
    assert (2 * y + 1) * _sum_over_x(a, b, c, d, y, y) == pytest.approx(1.0, rel=1e-9)

def test_orthogonality_off_diagonal_term_vanishes():
    assert _sum_over_x(1, 1, 1, 1, 1, 2) == pytest.approx(0.0, abs=1e-12)
    assert _sum_over_x(2, 1, 2, 1, 1, 2) == pytest.approx(0.0, abs=1e-12)
