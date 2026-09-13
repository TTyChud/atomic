
import numpy as np
import pytest
from scipy.signal import find_peaks

from atomic.atoms import aufbau_configuration
from atomic.hf_atom import hf_total_radial_density
from atomic.provenance import Fidelity


def test_density_integrates_to_the_electron_count():
    d = hf_total_radial_density(10, 10)
    total = np.trapezoid(d.values, d.grid)
    assert total == pytest.approx(10.0, rel=1e-3)
    assert d.provenance.error_estimate == pytest.approx(abs(total - 10.0), abs=1e-9)

def test_the_closure_residual_is_quadrature_and_not_a_defect():
    errors = []
    for points in (400, 800, 1600):
        d = hf_total_radial_density(10, 10, points=points)
        errors.append(abs(float(np.trapezoid(d.values, d.grid)) - 10.0))

    for coarse, fine in zip(errors, errors[1:], strict=False):
        assert 3.0 < coarse / fine < 6.0, f"{coarse:.3e} -> {fine:.3e} is not h^2"

def test_argon_has_three_shells_and_neon_has_two():
    for z, expected in ((10, 2), (18, 3)):
        d = hf_total_radial_density(z, z)
        peaks, _ = find_peaks(d.values, prominence=0.05 * d.values.max())
        assert len(peaks) == expected, f"Z={z} gave {len(peaks)} shells"

def test_the_collapsed_atom_has_one_shell():
    collapsed = aufbau_configuration(18, pauli=False)
    d = hf_total_radial_density(
        18, 18, config=collapsed, exchange=False, pauli=False
    )
    peaks, _ = find_peaks(d.values, prominence=0.05 * d.values.max())
    assert len(peaks) == 1
    assert d.provenance.fidelity is Fidelity.COUNTERFACTUAL

def test_the_density_is_the_observable_and_says_so():
    d = hf_total_radial_density(10, 10)
    joined = " ".join(d.provenance.assumptions)
    assert "observable" in joined
    assert d.unit == "electrons/bohr"
    assert d.provenance.fidelity is Fidelity.APPROXIMATION

def test_a_non_aufbau_configuration_changes_the_density():
    from atomic.atoms import parse_config

    ground = hf_total_radial_density(10, 10)
    excited = hf_total_radial_density(10, 10, config=parse_config("1s2 2s2 2p5 3s1"))
    assert not np.allclose(ground.values, excited.values)
