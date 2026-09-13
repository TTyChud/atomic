
import numpy as np
import pytest

from atomic.atoms import aufbau_configuration
from atomic.density_compare import (
    _common_grid,
    _displaced_charge,
    _peaks_with_depth,
    _resample,
    _shell_table,
    _weaker,
    _window_loss,
    compare_total_densities,
)
from atomic.hf_atom import hf_total_radial_density
from atomic.provenance import Fidelity, Field, Provenance
from atomic.screened_atom import screened_total_radial_density


def _field(values, grid, fidelity=Fidelity.APPROXIMATION, error=None):
    return Field(
        values=np.asarray(values, dtype=float),
        grid=np.asarray(grid, dtype=float),
        unit="electrons/bohr",
        grid_unit="bohr",
        label="D(r)",
        provenance=Provenance(
            fidelity=fidelity, method="test fixture", error_estimate=error
        ),
    )

def test_displaced_charge_matches_a_hand_integral():
    r = np.linspace(0.0, 1.0, 101)
    assert _displaced_charge(r, 2 * r, 2 - 2 * r) == pytest.approx(0.5, abs=1e-12)

def test_a_density_is_not_displaced_from_itself():
    r = np.linspace(0.0, 1.0, 101)
    assert _displaced_charge(r, 2 * r, 2 * r) == 0.0

def test_displaced_charge_is_symmetric():
    r = np.linspace(0.0, 1.0, 101)
    a, b = 2 * r, 2 - 2 * r
    assert _displaced_charge(r, a, b) == _displaced_charge(r, b, a)

def test_the_common_grid_is_the_intersection_of_the_two_boxes():
    a = _field(np.ones(50), np.geomspace(1e-4, 60.0, 50))
    b = _field(np.ones(50), np.geomspace(1e-3, 64.0, 50))
    grid = _common_grid(a, b, 200)
    assert grid[0] == pytest.approx(1e-3)
    assert grid[-1] == pytest.approx(60.0)
    assert len(grid) == 200

def test_window_loss_is_the_charge_left_outside():
    f = _field(np.ones(201), np.linspace(0.0, 2.0, 201))
    assert _window_loss(f, np.linspace(0.0, 1.0, 101)) == pytest.approx(1.0, abs=1e-12)

def test_the_weaker_tier_wins():
    assert _weaker(Fidelity.APPROXIMATION, Fidelity.COUNTERFACTUAL) is Fidelity.COUNTERFACTUAL
    assert _weaker(Fidelity.COUNTERFACTUAL, Fidelity.APPROXIMATION) is Fidelity.COUNTERFACTUAL
    assert _weaker(Fidelity.APPROXIMATION, Fidelity.APPROXIMATION) is Fidelity.APPROXIMATION
    assert _weaker(Fidelity.EXACT, Fidelity.NUMERICAL) is Fidelity.NUMERICAL

def test_a_visual_liberty_has_no_place_in_this_comparison():
    with pytest.raises(KeyError):
        _weaker(Fidelity.VISUAL_LIBERTY, Fidelity.APPROXIMATION)

def test_resampling_carries_the_provenance_and_discloses_itself():
    f = _field(np.linspace(1.0, 2.0, 50), np.geomspace(1e-3, 10.0, 50))
    out = _resample(f, np.geomspace(1e-2, 5.0, 30))
    assert out.provenance.fidelity is f.provenance.fidelity
    assert "resampled" in out.provenance.method
    assert len(out.values) == 30
    assert out.unit == f.unit

def test_the_innermost_peak_has_no_separation_to_report():
    r = np.linspace(0.1, 10.0, 500)
    v = np.exp(-((r - 1.0) ** 2) / 0.02)
    peaks = _peaks_with_depth(r, v)
    assert len(peaks) == 1
    assert peaks[0][0] == pytest.approx(1.0, abs=0.05)
    assert peaks[0][1] is None

def test_depth_is_the_relative_drop_into_the_preceding_minimum():
    r = np.linspace(0.0, 10.0, 2001)
    v = np.exp(-((r - 2.0) ** 2) / 0.08) + np.exp(-((r - 6.0) ** 2) / 0.08)
    peaks = _peaks_with_depth(r, v)
    assert len(peaks) == 2
    assert peaks[1][1] == pytest.approx(1.0, abs=0.01)

def test_the_noise_floor_keeps_the_faint_shell_and_drops_the_fainter_wiggle():
    r = np.linspace(0.0, 3.0, 3001)
    tall = np.exp(-((r - 1.0) ** 2) / 0.01)

    real_bump = 0.022 * np.exp(-((r - 2.0) ** 2) / 0.01)
    with_real_shell = _peaks_with_depth(r, tall + real_bump)
    assert len(with_real_shell) == 2, "a 2.2 percent peak is a real shell"

    noise = 1e-30 * np.sign(np.sin(r * 5000.0))
    with_noise = _peaks_with_depth(r, tall + noise)
    assert len(with_noise) == 1, "a 1e-30 wiggle is noise, not a second shell"

def test_gsz_does_not_resolve_sodiums_third_shell_and_hartree_fock_barely_does():
    c = compare_total_densities(11, 11)
    assert [s.label for s in c.shells] == ["K", "L", "M"]
    k, ell, m = c.shells
    assert k.gsz_radius is not None and k.hf_radius is not None
    assert ell.gsz_radius is not None and ell.hf_radius is not None
    assert m.gsz_radius is None, "GSZ is not expected to resolve sodium's M shell"
    assert m.hf_radius == pytest.approx(3.16, rel=0.05)
    assert m.hf_depth == pytest.approx(0.003, abs=0.002)

def test_magnesium_is_the_same_case_with_a_deeper_dimple():
    c = compare_total_densities(12, 12)
    m = c.shells[2]
    assert m.label == "M"
    assert m.gsz_radius is None
    assert m.hf_radius == pytest.approx(2.43, rel=0.05)
    assert m.hf_depth == pytest.approx(0.015, abs=0.005)

@pytest.mark.parametrize("z", [13, 14, 18])
def test_three_clean_shells_under_both_models(z):
    c = compare_total_densities(z, z)
    assert len(c.shells) == 3
    for s in c.shells:
        assert s.gsz_radius is not None
        assert s.hf_radius is not None

def test_the_shell_count_comes_from_the_configuration_not_from_the_peaks():
    c = compare_total_densities(11, 11)
    n_shells = len({n for (n, _), _ in aufbau_configuration(11)})
    assert len(c.shells) == n_shells == 3

def test_more_maxima_than_shells_raises_with_enough_to_diagnose_it():
    r = np.linspace(0.0, 10.0, 2001)
    v = (
        np.exp(-((r - 1.0) ** 2) / 0.01)
        + 0.8 * np.exp(-((r - 3.0) ** 2) / 0.01)
        + 0.6 * np.exp(-((r - 5.0) ** 2) / 0.01)
        + 0.4 * np.exp(-((r - 7.0) ** 2) / 0.01)
    )
    config = (((1, 0), 2), ((2, 0), 2), ((3, 0), 2))
    with pytest.raises(ValueError) as excinfo:
        _shell_table(r, v, v, config)
    message = str(excinfo.value)
    assert "GSZ" in message
    assert "4 maxima" in message
    assert "7" in message

@pytest.mark.parametrize("z", [2, 6, 10, 11, 14, 18])
def test_the_innermost_shell_has_no_measured_separation_under_either_model(z):
    c = compare_total_densities(z, z)
    k = c.shells[0]
    assert k.gsz_depth is None
    assert k.hf_depth is None

@pytest.mark.parametrize(
    "z,expected",
    [(2, 0.0003), (10, 0.0232), (11, 0.1218), (18, 0.0600)],
)
def test_the_measured_disagreement(z, expected):
    c = compare_total_densities(z, z)
    assert c.displaced_charge.value == pytest.approx(
        expected, abs=max(c.displaced_charge.provenance.error_estimate, 2e-3)
    )

@pytest.mark.parametrize("z", [2, 6, 10, 11, 18])
def test_the_window_costs_less_than_the_bar_it_is_folded_into(z):
    c = compare_total_densities(z, z)
    loss = _window_loss(c.gsz, c.grid) + _window_loss(c.hf, c.grid)
    assert loss < c.displaced_charge.provenance.error_estimate
    assert loss < 5e-3

def test_the_bar_is_four_measured_terms_and_no_one_of_them_carries_it():
    c = compare_total_densities(18, 18)
    cfg = aufbau_configuration(18)
    hf = hf_total_radial_density(18, 18, config=cfg, points=800)
    gsz = screened_total_radial_density(18, 18, config=cfg, points=800)
    terms = (
        hf.provenance.error_estimate,
        gsz.provenance.error_estimate,
        _window_loss(gsz, c.grid),
        _window_loss(hf, c.grid),
    )
    bar = c.displaced_charge.provenance.error_estimate
    assert sum(terms) == pytest.approx(bar, rel=1e-12)
    assert max(terms) / bar < 0.5
    assert bar / c.displaced_charge.value == pytest.approx(0.040, abs=0.01)

BOTH_MODELS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18)

def test_the_models_agree_far_better_than_their_energies_do():
    for z in BOTH_MODELS:
        c = compare_total_densities(z, z)
        assert c.displaced_charge.value / z < 0.015, f"Z={z}"

def test_lithium_is_the_atom_that_sets_the_bound():
    c = compare_total_densities(3, 3)
    assert c.displaced_charge.value / 3 == pytest.approx(0.0145, abs=5e-4)

def test_the_comparison_is_an_approximation_and_says_neither_is_truth():
    c = compare_total_densities(10, 10)
    assert c.provenance.fidelity is Fidelity.APPROXIMATION
    assert c.displaced_charge.unit == "electrons"
    assert any("not truth" in a for a in c.provenance.assumptions)

def test_a_thrown_switch_makes_the_comparison_counterfactual():
    c = compare_total_densities(10, 10, exchange=False)
    assert c.provenance.fidelity is Fidelity.COUNTERFACTUAL
    assert c.displaced_charge.provenance.fidelity is Fidelity.COUNTERFACTUAL

def test_both_models_take_the_same_configuration_with_pauli_off():
    c = compare_total_densities(10, 10, exchange=False, pauli=False)
    assert c.provenance.fidelity is Fidelity.COUNTERFACTUAL
    assert [s.label for s in c.shells] == ["K"]

def test_an_ion_is_refused_because_the_parameters_are_fitted_to_neutrals():
    with pytest.raises(ValueError, match="neutral"):
        compare_total_densities(10, 9)
