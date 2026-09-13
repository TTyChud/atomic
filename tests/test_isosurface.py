
import numpy as np
import pytest

from atomic.isosurface import (
    GRID_SIZES,
    default_half_width,
    fraction_above,
    hf_isosurface,
    isosurface,
    screened_isosurface,
    solve_level,
)
from atomic.numerics.marching_tets import edge_use_counts
from atomic.provenance import Fidelity
from atomic.sampling import sample_density


def hydrogen_1s_enclosed(radius: float) -> float:
    return 1.0 - np.exp(-2 * radius) * (1 + 2 * radius + 2 * radius**2)

def radius_enclosing(fraction: float) -> float:
    lo, hi = 0.0, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if hydrogen_1s_enclosed(mid) < fraction:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def test_solve_level_on_a_uniform_box_splits_it_by_count():
    grid = np.full((4, 4, 4), 0.5)
    cell = 1.0 / (0.5 * 64)
    assert fraction_above(grid, cell, solve_level(grid, cell, 0.5)) == pytest.approx(1.0)

def test_solve_level_refuses_a_target_the_box_cannot_hold():
    grid = np.full((4, 4, 4), 0.1)
    with pytest.raises(ValueError, match="cannot enclose"):
        solve_level(grid, 0.001, 0.9)

@pytest.mark.parametrize("target", [-0.1, 0.0, 1.0, 2.0])
def test_solve_level_rejects_targets_outside_the_open_unit_interval(target):
    with pytest.raises(ValueError, match="must be in"):
        solve_level(np.ones((3, 3, 3)), 1.0, target)

def test_hydrogen_1s_at_ninety_percent_is_the_textbook_sphere():
    expected_r = radius_enclosing(0.9)
    assert expected_r == pytest.approx(2.6612, abs=1e-3)

    surface = isosurface(1, 0, 0, target_fraction=0.9, resolution=96)
    radii = np.linalg.norm(surface.vertices, axis=1)

    assert radii.mean() == pytest.approx(expected_r, rel=2e-3)
    assert radii.std() < 0.02
    assert surface.mesh_volume.value == pytest.approx(4 / 3 * np.pi * expected_r**3, rel=0.01)
    assert surface.area.value == pytest.approx(4 * np.pi * expected_r**2, rel=0.01)

@pytest.mark.parametrize("fraction", [0.5, 0.95])
def test_the_radius_tracks_the_requested_fraction(fraction):
    surface = isosurface(1, 0, 0, target_fraction=fraction, resolution=64)
    radii = np.linalg.norm(surface.vertices, axis=1)
    assert radii.mean() == pytest.approx(radius_enclosing(fraction), rel=1e-2)

def test_a_tighter_contour_sits_inside_a_looser_one():
    inner = isosurface(1, 0, 0, target_fraction=0.5, resolution=48)
    outer = isosurface(1, 0, 0, target_fraction=0.95, resolution=48)
    assert inner.level.value > outer.level.value
    assert inner.mesh_volume.value < outer.mesh_volume.value

@pytest.mark.parametrize("n,l,m", [(1, 0, 0), (2, 1, 0), (3, 2, 1)])
def test_the_sampler_lands_inside_the_surface_at_the_stated_rate(n, l, m):
    from atomic.analytic.wavefunction import evaluate_state

    surface = isosurface(n, l, m, target_fraction=0.9, resolution=96)
    cloud = sample_density(n, l, m, 40_000, seed=7)
    density = np.abs(evaluate_state(n, l, m, cloud.positions).values) ** 2
    inside = float((density >= surface.level.value).mean())
    assert inside == pytest.approx(0.9, abs=0.01)

def test_the_box_grows_until_it_holds_essentially_everything():
    for n, l in ((1, 0), (4, 3)):
        surface = isosurface(n, l, 0, target_fraction=0.9, resolution=48)
        assert surface.escaped_fraction.value < 2e-3
        assert surface.half_width > default_half_width(n) * 0.5

def test_a_box_too_small_for_the_target_is_refused_rather_than_renormalized():
    with pytest.raises(ValueError, match="widen the box"):
        isosurface(1, 0, 0, target_fraction=0.9, resolution=48, half_width=1.0)

def test_the_escaped_mass_is_reported_even_when_it_is_tiny():
    surface = isosurface(1, 0, 0, target_fraction=0.9, resolution=64)
    assert surface.escaped_fraction.value >= 0.0
    assert any("outside the" in a and "box" in a for a in surface.provenance.assumptions)

def test_the_mesh_volume_agrees_with_the_cells_it_was_cut_from():
    surface = isosurface(2, 1, 0, target_fraction=0.9, resolution=96)
    assert surface.mesh_volume.value == pytest.approx(surface.voxel_volume.value, rel=0.05)
    assert surface.mesh_volume.value != surface.voxel_volume.value

def test_refinement_moves_the_answer_toward_the_closed_form():
    exact = 4 / 3 * np.pi * radius_enclosing(0.9) ** 3
    errors = [
        abs(isosurface(1, 0, 0, target_fraction=0.9, resolution=n).mesh_volume.value - exact)
        for n in (48, 96)
    ]
    assert errors[1] < errors[0]

def test_the_surface_is_watertight_for_a_lobed_orbital():
    surface = isosurface(3, 2, 0, target_fraction=0.9, resolution=64)
    counts = edge_use_counts(
        type("M", (), {"vertices": surface.vertices, "triangles": surface.triangles})()
    )
    assert set(np.unique(counts).tolist()) == {2}

def test_the_s_orbital_is_one_piece_and_the_p_orbital_is_two():
    assert isosurface(1, 0, 0, target_fraction=0.9, resolution=64).components == 1

    split = isosurface(2, 1, 0, target_fraction=0.5, basis="real", resolution=64)
    gap = 2 * np.abs(split.vertices[:, 2]).min()
    assert gap > 2 * split.half_width / 63
    assert split.components == 2

    fused = isosurface(2, 1, 0, target_fraction=0.9, basis="real", resolution=64)
    assert 2 * np.abs(fused.vertices[:, 2]).min() < 2 * fused.half_width / 63
    assert fused.components == 1
    assert any("narrower than" in a for a in fused.provenance.assumptions)

def test_the_two_lobes_carry_opposite_signs_of_psi():
    p_z = isosurface(2, 1, 0, target_fraction=0.9, basis="real", resolution=64)
    upper = p_z.vertex_phase[p_z.vertices[:, 2] > 0]
    lower = p_z.vertex_phase[p_z.vertices[:, 2] < 0]
    assert np.allclose(upper, 0.0)
    assert np.allclose(np.abs(lower), np.pi)

@pytest.mark.parametrize("m", [1, -1])
def test_a_complex_orbital_carries_a_real_phase_rather_than_a_sign(m):
    surface = isosurface(2, 1, m, target_fraction=0.9, basis="complex", resolution=64)
    phase = surface.vertex_phase
    assert phase.max() - phase.min() > 5.0
    azimuth = np.arctan2(surface.vertices[:, 1], surface.vertices[:, 0])
    residual = np.exp(1j * (phase - m * azimuth))
    assert np.abs(residual - residual[0]).max() < 1e-9

def test_an_exact_wavefunction_still_gives_a_numerical_surface():
    surface = isosurface(1, 0, 0, target_fraction=0.9, resolution=48)
    assert surface.provenance.fidelity is Fidelity.NUMERICAL
    assert surface.level.provenance.fidelity is Fidelity.NUMERICAL

def test_the_disclosure_states_the_complement_not_just_the_fraction():
    surface = isosurface(1, 0, 0, target_fraction=0.9, resolution=48)
    text = " ".join(surface.provenance.assumptions)
    assert "outside the surface" in text
    assert "no boundary" in text
    assert surface.outside_fraction == pytest.approx(1 - surface.enclosed_fraction.value)

def test_the_error_bar_is_a_grid_halving_on_the_fraction():
    surface = isosurface(1, 0, 0, target_fraction=0.9, resolution=64)
    assert surface.provenance.error_estimate is not None
    assert 0.0 <= surface.provenance.error_estimate < 0.02
    assert surface.enclosed_fraction.provenance.error_estimate is not None

def test_the_volume_gets_its_own_error_bar_because_the_fraction_one_is_blind():
    surface = isosurface(1, 0, 0, target_fraction=0.9, resolution=96)
    fraction_error = surface.provenance.error_estimate
    volume_error = surface.mesh_volume.provenance.error_estimate

    assert fraction_error is not None and volume_error is not None
    assert fraction_error < 1e-3
    assert volume_error / surface.mesh_volume.value > 1e-3
    text = " ".join(surface.provenance.assumptions)
    assert "enclosed volume by" in text
    assert "converges long before the surface does" in text

def test_the_volume_error_bar_shrinks_as_the_grid_refines():
    errors = [
        isosurface(2, 1, 0, target_fraction=0.9, resolution=n, basis="real")
        for n in (48, 96)
    ]
    relative = [s.mesh_volume.provenance.error_estimate / s.mesh_volume.value for s in errors]
    assert relative[1] < relative[0]

def test_the_achieved_fraction_is_what_is_reported_not_the_request():
    surface = isosurface(2, 1, 0, target_fraction=0.9, resolution=64)
    assert surface.enclosed_fraction.value == pytest.approx(0.9, abs=5e-3)
    assert surface.target_fraction == 0.9

@pytest.mark.parametrize("resolution", [7, 1000])
def test_an_unsupported_grid_size_is_refused(resolution):
    with pytest.raises(ValueError, match="resolution must be one of"):
        isosurface(1, 0, 0, resolution=resolution)

def test_the_offered_grid_sizes_are_the_ones_that_work():
    for size in GRID_SIZES:
        assert isosurface(1, 0, 0, target_fraction=0.5, resolution=size).components == 1

def test_a_screened_orbital_inherits_the_weaker_tier():
    surface = screened_isosurface(11, 11, 3, 0, 0, target_fraction=0.9, resolution=48)
    assert surface.provenance.fidelity is Fidelity.APPROXIMATION
    assert surface.enclosed_fraction.value == pytest.approx(0.9, abs=0.01)

def test_a_hartree_fock_orbital_gets_a_surface_too():
    surface = hf_isosurface(4, 4, 2, 0, 0, target_fraction=0.9, resolution=48)
    assert surface.provenance.fidelity is Fidelity.APPROXIMATION
    assert surface.enclosed_fraction.value == pytest.approx(0.9, abs=0.01)
    assert surface.components >= 1

def test_the_hartree_fock_valence_orbital_is_bigger_than_the_core_one():
    core = hf_isosurface(4, 4, 1, 0, 0, target_fraction=0.9, resolution=48)
    valence = hf_isosurface(4, 4, 2, 0, 0, target_fraction=0.9, resolution=48)
    assert valence.mesh_volume.value > core.mesh_volume.value
