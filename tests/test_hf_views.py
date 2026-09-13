import numpy as np
import pytest

from atomic.atoms import aufbau_configuration, parse_config
from atomic.hf_atom import evaluate_hf_state, hf_radial
from atomic.isosurface import hf_isosurface
from atomic.provenance import Fidelity


def test_explicit_configuration_reaches_the_orbital():
    ground = aufbau_configuration(11)
    excited = parse_config("1s2 2s2 2p6 3p1")

    r_ground, _ = hf_radial(11, 11, 2, 1, points=200, config=ground)
    r_excited, _ = hf_radial(11, 11, 2, 1, points=200, config=excited)

    assert not np.allclose(r_ground.values, r_excited.values, atol=1e-9)

def test_exchange_off_reaches_the_orbital_and_the_badge():
    config = aufbau_configuration(10)
    r_hf, _ = hf_radial(10, 10, 2, 1, points=200, config=config)
    r_hartree, _ = hf_radial(
        10, 10, 2, 1, points=200, config=config, exchange=False
    )

    assert not np.allclose(r_hf.values, r_hartree.values, atol=1e-9)
    assert r_hf.provenance.fidelity is Fidelity.APPROXIMATION
    assert r_hartree.provenance.fidelity is Fidelity.COUNTERFACTUAL

def test_evaluate_hf_state_inherits_the_counterfactual_tier():
    pos = np.array([[0.0, 0.0, 1.0], [0.5, 0.0, 0.5]])
    real = evaluate_hf_state(10, 10, 2, 1, 0, pos)
    hartree = evaluate_hf_state(10, 10, 2, 1, 0, pos, exchange=False)

    assert real.provenance.fidelity is Fidelity.APPROXIMATION
    assert hartree.provenance.fidelity is Fidelity.COUNTERFACTUAL

def test_pauli_off_refuses_every_subshell_but_the_one_that_exists():
    collapsed = aufbau_configuration(10, pauli=False)
    with pytest.raises(ValueError, match="occupancy cap"):
        hf_radial(
            10, 10, 2, 1, points=200,
            config=collapsed, exchange=False, pauli=False,
        )
    r, _ = hf_radial(
        10, 10, 1, 0, points=200, config=collapsed, exchange=False, pauli=False
    )
    assert r.provenance.fidelity is Fidelity.COUNTERFACTUAL

def test_orbital_carries_the_not_an_observable_claim():
    r, p = hf_radial(10, 10, 2, 1, points=200)
    joined = " ".join(r.provenance.assumptions)
    assert "not an observable" in joined
    assert "spherical" in joined
    assert joined == " ".join(p.provenance.assumptions)

def test_hf_sampling_reduces_to_hydrogen():
    from scipy import stats

    from atomic.sampling import sample_hf_density

    cloud = sample_hf_density(1, 1, 1, 0, 0, 20_000, seed=7)
    r = np.linalg.norm(cloud.positions.astype(np.float64), axis=1)

    def cdf(x):
        return 1.0 - np.exp(-2.0 * x) * (1.0 + 2.0 * x + 2.0 * x * x)

    assert stats.kstest(r, cdf).pvalue > 0.01

def test_hf_cloud_carries_the_solve_and_the_claim():
    from atomic.sampling import sample_hf_density

    cloud = sample_hf_density(10, 10, 2, 1, 0, 2_000, seed=1)
    joined = " ".join(cloud.provenance.assumptions)
    assert cloud.provenance.fidelity is Fidelity.APPROXIMATION
    assert "not an observable" in joined
    assert "correlation" in joined

def test_hf_cloud_goes_counterfactual_with_exchange_off():
    from atomic.sampling import sample_hf_density

    cloud = sample_hf_density(10, 10, 2, 1, 0, 2_000, seed=1, exchange=False)
    assert cloud.provenance.fidelity is Fidelity.COUNTERFACTUAL

def test_hf_plane_agrees_with_the_evaluator_it_is_built_on():
    from atomic.plane import hf_plane_grid

    pg = hf_plane_grid(10, 10, 2, 1, 0, quantity="psi", resolution=33)
    axis = pg.axis
    for i in (3, 16, 29):
        for j in (5, 16, 27):
            direct = evaluate_hf_state(
                10, 10, 2, 1, 0,
                np.array([[axis[j], 0.0, axis[i]]]),
            )
            assert pg.values[i, j] == pytest.approx(
                float(np.real(direct.values[0])), rel=1e-9, abs=1e-12
            )

def test_hf_psi_is_real_on_the_y_zero_plane():
    from atomic.plane import hf_plane_grid

    pg = hf_plane_grid(10, 10, 2, 1, 1, quantity="psi", resolution=33)
    pos = np.array([[0.7, 0.0, 0.9], [-1.3, 0.0, 0.4]])
    psi = evaluate_hf_state(10, 10, 2, 1, 1, pos).values
    assert np.max(np.abs(np.imag(psi))) < 1e-12
    assert "psi is real on y=0" in " ".join(pg.provenance.assumptions)

def test_hf_plane_inherits_the_counterfactual_tier():
    from atomic.plane import hf_plane_grid

    real = hf_plane_grid(10, 10, 2, 1, 0, resolution=17)
    hartree = hf_plane_grid(10, 10, 2, 1, 0, resolution=17, exchange=False)
    assert real.provenance.fidelity is Fidelity.APPROXIMATION
    assert hartree.provenance.fidelity is Fidelity.COUNTERFACTUAL
    assert not np.allclose(real.values, hartree.values)

def test_hf_isosurface_reduces_to_the_closed_form_hydrogen_radius():
    surf = hf_isosurface(1, 1, 1, 0, 0, target_fraction=0.9, resolution=96)
    radii = np.linalg.norm(surf.vertices, axis=1)
    assert radii.mean() == pytest.approx(2.6612, rel=5e-3)

def test_helium_hartree_and_hartree_fock_surfaces_are_bit_identical():
    with_x = hf_isosurface(2, 2, 1, 0, 0, resolution=64)
    without = hf_isosurface(2, 2, 1, 0, 0, resolution=64, exchange=False)
    assert np.array_equal(with_x.vertices, without.vertices)
    assert with_x.provenance.fidelity is Fidelity.APPROXIMATION
    assert without.provenance.fidelity is Fidelity.COUNTERFACTUAL

def test_multi_shell_atom_surfaces_differ_with_exchange_off():
    with_x = hf_isosurface(10, 10, 2, 1, 0, resolution=64)
    without = hf_isosurface(10, 10, 2, 1, 0, resolution=64, exchange=False)
    assert not np.array_equal(with_x.vertices, without.vertices)

def _orbital_mean_radius(z, n_electrons, n, l, config, exchange, pauli):
    _, p = hf_radial(
        z, n_electrons, n, l, points=4000,
        config=config, exchange=exchange, pauli=pauli,
    )
    return float(
        np.trapezoid(p.grid * p.values, p.grid) / np.trapezoid(p.values, p.grid)
    )

def test_pauli_collapse_orders_two_independent_measures_the_same_way():
    real_cfg = aufbau_configuration(4)
    collapsed_cfg = aufbau_configuration(4, pauli=False)

    real_surf = hf_isosurface(4, 4, 1, 0, 0, resolution=64, config=real_cfg)
    collapsed_surf = hf_isosurface(
        4, 4, 1, 0, 0, resolution=64,
        config=collapsed_cfg, exchange=False, pauli=False,
    )
    surface_sign = np.sign(
        np.linalg.norm(collapsed_surf.vertices, axis=1).mean()
        - np.linalg.norm(real_surf.vertices, axis=1).mean()
    )

    quadrature_sign = np.sign(
        _orbital_mean_radius(4, 4, 1, 0, collapsed_cfg, False, False)
        - _orbital_mean_radius(4, 4, 1, 0, real_cfg, True, True)
    )

    assert surface_sign != 0
    assert surface_sign == quadrature_sign

def test_the_collapsed_atom_shrinks_while_its_1s_swells():
    from atomic.hf_atom import hf_mean_radius, solve_hartree_fock

    real_cfg = aufbau_configuration(4)
    collapsed_cfg = aufbau_configuration(4, pauli=False)

    atom_real = hf_mean_radius(solve_hartree_fock(4, 4, real_cfg)).value
    atom_collapsed = hf_mean_radius(
        solve_hartree_fock(4, 4, collapsed_cfg, False, False)
    ).value
    assert atom_collapsed < atom_real

    orbital_real = _orbital_mean_radius(4, 4, 1, 0, real_cfg, True, True)
    orbital_collapsed = _orbital_mean_radius(
        4, 4, 1, 0, collapsed_cfg, False, False
    )
    assert orbital_collapsed > orbital_real

    assert atom_collapsed == pytest.approx(orbital_collapsed, rel=1e-3)
