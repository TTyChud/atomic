import math

import pytest

from atomic.analytic.oscillator import oscillator_energy, oscillator_levels
from atomic.provenance import Fidelity


def test_ground_state_is_three_halves_omega():
    q = oscillator_energy(k=0, l=0, omega=0.5)
    assert q.unit == "hartree"
    assert q.provenance.fidelity is Fidelity.EXACT
    assert math.isclose(q.value, 0.5 * 1.5, rel_tol=1e-12)

@pytest.mark.parametrize("k,l,omega,expected", [
    (0, 0, 0.5, 0.75),
    (1, 0, 0.5, 1.75),
    (0, 1, 0.5, 1.25),
    (2, 3, 0.3, 0.3 * (4 + 3 + 1.5)),
])
def test_level_formula(k, l, omega, expected):
    assert math.isclose(oscillator_energy(k, l, omega).value, expected, rel_tol=1e-12)

def test_levels_are_ascending_and_counted():
    levels = oscillator_levels(omega=0.4, l=1, n_states=4)
    assert len(levels) == 4
    values = [q.value for q in levels]
    assert values == sorted(values)
    assert math.isclose(values[0], 0.4 * (0 + 1 + 1.5), rel_tol=1e-12)
