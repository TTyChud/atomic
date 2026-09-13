import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid
from scipy.special import lpmv
from scipy.stats import kstest

from atomic.provenance import Fidelity
from atomic.sampling import SampleCloud, sample_density, sample_screened_density
from atomic.screened_atom import screened_radial

COUNT = 100_000

def _radii(cloud: SampleCloud) -> np.ndarray:
    return np.linalg.norm(cloud.positions.astype(float), axis=1)

def test_positions_shape_dtype_and_metadata():
    cloud = sample_density(2, 1, 0, count=5_000, seed=1)
    assert cloud.positions.shape == (5_000, 3)
    assert cloud.positions.dtype == np.float32
    assert (cloud.n, cloud.l, cloud.m) == (2, 1, 0)
    assert np.isfinite(cloud.positions).all()

def test_provenance_is_numerical_and_states_seed_and_count():
    cloud = sample_density(1, 0, 0, count=2_000, seed=7)
    assert cloud.provenance.fidelity is Fidelity.NUMERICAL
    joined = " ".join(cloud.provenance.assumptions)
    assert "seed=7" in joined
    assert "2000" in joined.replace(",", "").replace("_", "")

def test_1s_radial_distribution_ks_against_analytic_cdf():
    cloud = sample_density(1, 0, 0, count=COUNT, seed=42)
    r = _radii(cloud)
    ks = kstest(r, lambda x: 1.0 - np.exp(-2.0 * x) * (1.0 + 2.0 * x + 2.0 * x**2))
    assert ks.statistic < 0.01, ks

def test_1s_mean_radius():
    r = _radii(sample_density(1, 0, 0, count=COUNT, seed=42))
    assert r.mean() == pytest.approx(1.5, abs=0.02)

def test_2p_mean_radius():
    r = _radii(sample_density(2, 1, 0, count=COUNT, seed=3))
    assert r.mean() == pytest.approx(5.0, abs=0.05)

def test_1s_angular_isotropy():
    cloud = sample_density(1, 0, 0, count=COUNT, seed=11)
    r = _radii(cloud)
    cos_theta = cloud.positions[:, 2].astype(float) / r
    assert cos_theta.mean() == pytest.approx(0.0, abs=0.01)
    assert (cos_theta**2).mean() == pytest.approx(1.0 / 3.0, abs=0.01)

def test_2p_m0_angular_distribution():
    cloud = sample_density(2, 1, 0, count=COUNT, seed=5)
    r = _radii(cloud)
    cos_theta = cloud.positions[:, 2].astype(float) / r
    assert (cos_theta**2).mean() == pytest.approx(0.6, abs=0.01)

def test_seed_reproducibility():
    a = sample_density(3, 2, 1, count=1_000, seed=99)
    b = sample_density(3, 2, 1, count=1_000, seed=99)
    assert np.array_equal(a.positions, b.positions)

def test_progress_callback_monotonic_and_complete():
    calls: list[float] = []
    sample_density(1, 0, 0, count=10_000, seed=0, progress=calls.append, n_chunks=10)
    assert len(calls) == 10
    assert calls[-1] == pytest.approx(1.0)
    assert all(b >= a for a, b in zip(calls, calls[1:]))

def test_rejects_invalid_quantum_numbers():
    with pytest.raises(ValueError):
        sample_density(1, 1, 0, count=100)
    with pytest.raises(ValueError):
        sample_density(2, 1, 2, count=100)
    with pytest.raises(ValueError):
        sample_density(1, 0, 0, count=0)

def _phis(cloud: SampleCloud) -> np.ndarray:
    return np.mod(
        np.arctan2(cloud.positions[:, 1].astype(float), cloud.positions[:, 0].astype(float)),
        2.0 * np.pi,
    )

def test_real_basis_px_phi_marginal_ks():
    cloud = sample_density(2, 1, 1, count=COUNT, seed=21, basis="real")
    ks = kstest(
        _phis(cloud),
        lambda p: (p / 2.0 + np.sin(2.0 * p) / 4.0) / np.pi,
    )
    assert ks.statistic < 0.01, ks

def test_real_basis_px_angular_moment():
    cloud = sample_density(2, 1, 1, count=COUNT, seed=22, basis="real")
    r = _radii(cloud)
    assert ((cloud.positions[:, 0].astype(float) / r) ** 2).mean() == pytest.approx(
        0.6, abs=0.01
    )

def test_real_basis_dxy_sin_type():
    cloud = sample_density(3, 2, -2, count=COUNT, seed=23, basis="real")
    assert (np.sin(2.0 * _phis(cloud)) ** 2).mean() == pytest.approx(0.75, abs=0.01)

def test_real_m0_matches_complex_m0_statistically():
    a = sample_density(2, 1, 0, count=COUNT, seed=24, basis="real")
    b = sample_density(2, 1, 0, count=COUNT, seed=25, basis="complex")
    za = (a.positions[:, 2].astype(float) / _radii(a)) ** 2
    zb = (b.positions[:, 2].astype(float) / _radii(b)) ** 2
    assert za.mean() == pytest.approx(zb.mean(), abs=0.01)

def test_basis_recorded_in_cloud_and_provenance():
    cloud = sample_density(2, 1, 1, count=2_000, seed=1, basis="real")
    assert cloud.basis == "real"
    assert "real" in cloud.provenance.method
    default = sample_density(2, 1, 1, count=2_000, seed=1)
    assert default.basis == "complex"

def test_rejects_unknown_basis():
    with pytest.raises(ValueError):
        sample_density(1, 0, 0, count=1_000, basis="cartoon")

def test_screened_radial_marginal_matches_numerical_cdf():
    cloud = sample_screened_density(11, 11, 3, 0, 0, 20000, seed=1)
    r = np.linalg.norm(cloud.positions, axis=1)
    r_field, _ = screened_radial(11, 11, 3, 0, points=8192)
    grid, R = r_field.grid, r_field.values
    p = grid**2 * R**2
    cdf = cumulative_trapezoid(p, grid, initial=0.0)
    cdf /= cdf[-1]
    _, pval = kstest(r, lambda x: np.interp(x, grid, cdf))
    assert pval > 0.01

def test_screened_angular_marginal_matches_legendre():
    cloud = sample_screened_density(11, 11, 3, 1, 0, 20000, seed=2)
    cos_t = cloud.positions[:, 2] / np.linalg.norm(cloud.positions, axis=1)
    x = np.linspace(-1.0, 1.0, 4096)
    p = lpmv(0, 1, x) ** 2
    cdf = cumulative_trapezoid(p, x, initial=0.0)
    cdf /= cdf[-1]
    _, pval = kstest(cos_t, lambda v: np.interp(v, x, cdf))
    assert pval > 0.01

def test_screened_cloud_is_approximation_and_sane():
    cloud = sample_screened_density(11, 11, 3, 0, 0, 5000, seed=3)
    assert cloud.positions.shape == (5000, 3)
    assert cloud.provenance.fidelity is Fidelity.APPROXIMATION
    assert "GSZ" in cloud.provenance.method or "screen" in cloud.provenance.method.lower()
    r = np.linalg.norm(cloud.positions, axis=1)
    assert np.all(np.isfinite(r))
    assert 1.0 < float(r.mean()) < 20.0
