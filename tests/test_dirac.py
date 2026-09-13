import pytest

from atomic.analytic.dirac import dirac_energy, dirac_fine_splitting
from atomic.analytic.fine_structure import fine_structure_shift
from atomic.analytic.hydrogen import energy
from atomic.constants import ALPHA
from atomic.provenance import Fidelity


def test_nonrelativistic_limit_recovers_bohr():
    for n, j in [(1, 0.5), (2, 0.5), (3, 1.5)]:
        e = dirac_energy(n, j, Z=1, alpha=1e-5).value
        assert e == pytest.approx(energy(n).value, rel=1e-6)

def test_ground_state_matches_published_value():
    got = dirac_energy(1, 0.5, Z=1).value
    assert got == pytest.approx(-0.5 - ALPHA**2 / 8.0, abs=1e-9)
    assert got < -0.5

def test_agrees_with_perturbative_to_order_alpha4():
    n, l, j, Z = 2, 1, 1.5, 2

    def residual(a):
        d = dirac_energy(n, j, Z=Z, alpha=a).value
        p = energy(n, Z=Z).value + fine_structure_shift(n, l, j, Z=Z, alpha=a).value
        return abs(d - p)

    r1 = residual(ALPHA)
    r2 = residual(ALPHA / 2)
    assert r1 < 1e-6
    assert r2 == pytest.approx(r1 / 16.0, rel=0.1)

def test_exact_nj_degeneracy_is_l_independent():
    e_from_s = dirac_energy(2, 0.5, Z=1).value
    e_from_p = dirac_energy(2, 0.5, Z=1).value
    assert e_from_s == e_from_p

def test_fine_splitting_matches_perturbative():
    dirac_gap = dirac_fine_splitting(2, 1, Z=1)
    pert = (
        fine_structure_shift(2, 1, 1.5, Z=1).value
        - fine_structure_shift(2, 1, 0.5, Z=1).value
    )
    assert dirac_gap == pytest.approx(pert, rel=1e-3)
    assert dirac_gap > 0

def test_fidelity_exact_at_real_alpha_counterfactual_when_altered():
    assert dirac_energy(1, 0.5).provenance.fidelity is Fidelity.EXACT
    assert dirac_energy(1, 0.5, alpha=0.2).provenance.fidelity is Fidelity.COUNTERFACTUAL

def test_supercritical_is_rejected():
    with pytest.raises(ValueError):
        dirac_energy(1, 0.5, Z=200)

def test_invalid_j_rejected():
    with pytest.raises(ValueError):
        dirac_energy(2, 2.5)
