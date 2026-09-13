import math

import pytest

from atomic.constants import ALPHA, FundamentalConstants
from atomic.constants_lab import ConstantsReport, analyze_constants
from atomic.provenance import Fidelity


def test_real_universe_is_exact_and_unchanged():
    r = analyze_constants()
    assert isinstance(r, ConstantsReport)
    assert r.altered is False
    assert r.alpha.quantity.value == pytest.approx(ALPHA, rel=1e-6)
    assert r.alpha.ratio == pytest.approx(1.0)
    assert r.alpha.changed is False
    assert r.bohr_radius_pm.changed is False
    assert r.hartree_ev.changed is False
    assert r.alpha.quantity.provenance.fidelity is Fidelity.EXACT
    assert r.bohr_radius_pm.quantity.value == pytest.approx(52.9, rel=1e-2)
    assert r.hartree_ev.quantity.value == pytest.approx(27.211, rel=1e-4)

def test_degeneracy_pair_changes_nothing_observable():
    r = analyze_constants(e=2.0, eps0=4.0)
    assert r.altered is True
    assert r.alpha.changed is False
    assert r.bohr_radius_pm.changed is False
    assert r.hartree_ev.changed is False
    assert r.alpha.quantity.provenance.fidelity is Fidelity.COUNTERFACTUAL

def test_electron_mass_scales_size_and_binding():
    r = analyze_constants(m_e=0.5)
    assert r.bohr_radius_pm.ratio == pytest.approx(2.0)
    assert r.hartree_ev.ratio == pytest.approx(0.5)
    assert r.alpha.changed is False

def test_altered_provenance_names_the_multipliers():
    r = analyze_constants(e=2.0, eps0=4.0)
    method = r.alpha.quantity.provenance.method.lower()
    assert "altered" in method
    assert "e" in method and "eps0" in method

def test_derived_alpha_matches_fundamental_constants():
    real = FundamentalConstants.codata()
    altered = FundamentalConstants(
        hbar=real.hbar, e=real.e * 1.5, m_e=real.m_e, eps0=real.eps0, c=real.c
    )
    r = analyze_constants(e=1.5)
    assert r.alpha.quantity.value == pytest.approx(altered.alpha, rel=1e-12)
    assert r.alpha.ratio == pytest.approx(altered.alpha / real.alpha, rel=1e-12)
    assert not math.isclose(r.alpha.ratio, 1.0)
