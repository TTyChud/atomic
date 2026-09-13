import pytest
from scipy import constants as sc

from atomic.analytic.fine_structure import fine_structure_shift, level_energy, validate_j
from atomic.constants import ALPHA, HARTREE_EV
from atomic.provenance import Fidelity

HARTREE_HZ = sc.physical_constants["hartree-hertz relationship"][0]
MU_H = 1836.152673426 / 1837.152673426

def test_shift_is_negative_and_j_ordered():
    s12 = fine_structure_shift(2, 0, 0.5).value
    p12 = fine_structure_shift(2, 1, 0.5).value
    p32 = fine_structure_shift(2, 1, 1.5).value
    assert s12 < 0 and p12 < 0 and p32 < 0
    assert s12 == pytest.approx(p12)
    assert p32 > p12

def test_2p_splitting_matches_measurement_within_g2_scale():
    split_hartree = (
        fine_structure_shift(2, 1, 1.5, mu_ratio=MU_H).value
        - fine_structure_shift(2, 1, 0.5, mu_ratio=MU_H).value
    )
    ghz = split_hartree * HARTREE_HZ / 1e9
    assert ghz == pytest.approx(10.969, rel=5e-3)

def test_1s_shift_magnitude():
    ev = fine_structure_shift(1, 0, 0.5, mu_ratio=MU_H).value * HARTREE_EV
    assert ev == pytest.approx(-1.810e-4, rel=2e-3)

def test_level_energy_composes_bohr_plus_shift():
    e = level_energy(2, 1, 0.5)
    shift = fine_structure_shift(2, 1, 0.5)
    assert e.value == pytest.approx(-0.125 + shift.value)
    assert e.unit == "hartree"
    assert e.provenance.fidelity is Fidelity.APPROXIMATION

def test_provenance_is_honest():
    q = fine_structure_shift(2, 1, 0.5, mu_ratio=0.5, m_over_M=1.0)
    assert q.provenance.fidelity is Fidelity.APPROXIMATION
    assert q.provenance.error_estimate >= abs(q.value)
    joined = " ".join(q.provenance.assumptions).lower()
    assert "darwin" in joined
    assert "g = 2" in joined or "g=2" in joined
    assert "dirac" in (q.provenance.refinement or "").lower()

def test_z_scaling_is_quartic():
    r = fine_structure_shift(2, 1, 1.5, Z=2).value / fine_structure_shift(2, 1, 1.5).value
    assert r == pytest.approx(16.0)

def test_validate_j():
    validate_j(1, 0.5)
    validate_j(1, 1.5)
    with pytest.raises(ValueError):
        validate_j(1, 2.5)
    with pytest.raises(ValueError):
        validate_j(0, -0.5)
    with pytest.raises(ValueError):
        fine_structure_shift(2, 1, 2.5)

def test_alpha_defaults_to_real_and_stays_approximation():
    default = fine_structure_shift(2, 1, 1.5)
    explicit = fine_structure_shift(2, 1, 1.5, alpha=ALPHA)
    assert default.value == explicit.value
    assert default.provenance.fidelity is Fidelity.APPROXIMATION

def test_altered_alpha_scales_shift_quadratically():
    base = fine_structure_shift(2, 1, 1.5).value
    doubled = fine_structure_shift(2, 1, 1.5, alpha=2 * ALPHA).value
    assert doubled / base == pytest.approx(4.0)

def test_altered_alpha_is_counterfactual_and_disclosed():
    q = fine_structure_shift(2, 1, 1.5, alpha=0.05)
    assert q.provenance.fidelity is Fidelity.COUNTERFACTUAL
    assert "altered" in q.provenance.method.lower()
    assert f"{ALPHA:g}" in q.provenance.method
    assert q.provenance.error_estimate == pytest.approx(
        abs(q.value) * ((1 * 0.05) ** 2 + 2 * 0.00116)
    )

def test_level_energy_follows_altered_fidelity():
    assert level_energy(2, 1, 1.5, alpha=0.05).provenance.fidelity is Fidelity.COUNTERFACTUAL
    assert level_energy(2, 1, 1.5).provenance.fidelity is Fidelity.APPROXIMATION
