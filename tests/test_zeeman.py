
import pytest

from atomic.analytic.dirac import dirac_energy
from atomic.analytic.fine_structure import level_energy
from atomic.analytic.zeeman import (
    MU_B_PER_TESLA,
    lande_g,
    zeeman_sublevels,
)
from atomic.constants import ALPHA
from atomic.provenance import Fidelity


def _by_mj(subs, m_j, branch):
    return next(s for s in subs if s.m_j == m_j and s.branch == branch)

def test_zero_field_recovers_fine_structure():
    for l, j, branch in [(0, 0.5, "single"), (1, 1.5, "upper"), (1, 0.5, "lower")]:
        subs = zeeman_sublevels(2, l, b_tesla=0.0)
        want = level_energy(2, l, j).value
        for s in (s for s in subs if s.j_label == j and s.branch == branch):
            assert s.energy.value == pytest.approx(want, rel=1e-12)

def test_sublevel_count():
    assert len(zeeman_sublevels(2, 0, b_tesla=1.0)) == 2
    assert len(zeeman_sublevels(2, 1, b_tesla=1.0)) == 6
    assert len(zeeman_sublevels(3, 2, b_tesla=1.0)) == 10

def test_lande_g_values():
    assert lande_g(1, 1.5) == pytest.approx(4.0 / 3.0)
    assert lande_g(1, 0.5) == pytest.approx(2.0 / 3.0)
    assert lande_g(0, 0.5) == pytest.approx(2.0)

def test_low_field_slope_is_lande():
    b = 1e-4
    e0 = {(s.j_label, s.m_j, s.branch): s.energy.value for s in zeeman_sublevels(2, 1, b_tesla=0.0)}
    for s in zeeman_sublevels(2, 1, b_tesla=b):
        slope = (s.energy.value - e0[(s.j_label, s.m_j, s.branch)]) / b
        want = lande_g(1, s.j_label) * MU_B_PER_TESLA * s.m_j
        assert slope == pytest.approx(want, rel=1e-3, abs=1e-16)

def test_high_field_slope_is_paschen_back():
    b1, b2 = 1.0e4, 2.0e4
    e1 = {(s.m_j, s.branch): s.energy.value for s in zeeman_sublevels(2, 1, b_tesla=b1)}
    for s in zeeman_sublevels(2, 1, b_tesla=b2):
        slope = (s.energy.value - e1[(s.m_j, s.branch)]) / (b2 - b1)
        integer = round(slope / MU_B_PER_TESLA)
        assert slope / MU_B_PER_TESLA == pytest.approx(integer, abs=1e-3)
        assert integer in (-2, -1, 0, 1, 2)

def test_trace_invariance():
    n, l = 2, 1
    e_up = level_energy(n, l, l + 0.5).value
    e_dn = level_energy(n, l, l - 0.5).value
    for b in (0.5, 5.0, 50.0):
        subs = zeeman_sublevels(n, l, b_tesla=b)
        for m_j in (0.5, -0.5):
            up = _by_mj(subs, m_j, "upper").energy.value
            lo = _by_mj(subs, m_j, "lower").energy.value
            trace = e_up + e_dn + 2.0 * MU_B_PER_TESLA * b * m_j
            assert up + lo == pytest.approx(trace, rel=1e-12)

def test_stretched_states_linear():
    def stretched(b):
        return _by_mj(zeeman_sublevels(2, 1, b_tesla=b), 1.5, "single").energy.value
    a, mid, c = stretched(0.0), stretched(1.0), stretched(2.0)
    assert (a - 2 * mid + c) == pytest.approx(0.0, abs=1e-18)

def test_dirac_diagonal_zero_field():
    subs = zeeman_sublevels(2, 1, b_tesla=0.0, dirac=True)
    for s in subs:
        assert s.energy.value == pytest.approx(dirac_energy(2, s.j_label).value, rel=1e-12)

def test_provenance_tiers_and_error():
    real = zeeman_sublevels(2, 1, b_tesla=2.0)[0].energy.provenance
    assert real.fidelity is Fidelity.APPROXIMATION
    assert real.error_estimate is not None and real.error_estimate > 0.0
    e_small = zeeman_sublevels(2, 1, b_tesla=1.0)[0].energy.provenance.error_estimate
    e_big = zeeman_sublevels(2, 1, b_tesla=10.0)[0].energy.provenance.error_estimate
    assert e_big > e_small
    altered = zeeman_sublevels(2, 1, b_tesla=2.0, alpha=ALPHA * 1.5)[0].energy.provenance
    assert altered.fidelity is Fidelity.COUNTERFACTUAL

def test_negative_field_rejected():
    with pytest.raises(ValueError):
        zeeman_sublevels(2, 1, b_tesla=-1.0)
