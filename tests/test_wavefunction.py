import numpy as np
import pytest

from atomic.analytic.wavefunction import WavefunctionValues, evaluate_state
from atomic.provenance import Fidelity


def _spherical_grid(n, r_points=800, ang_points=48):
    r_max = 30.0 * n * n
    r = np.linspace(1e-8, r_max, r_points)
    x, w = np.polynomial.legendre.leggauss(ang_points)
    theta = np.arccos(x)
    phi = np.linspace(0.0, 2.0 * np.pi, ang_points, endpoint=False)
    R, T, P = np.meshgrid(r, theta, phi, indexing="ij")
    pos = np.stack(
        [
            (R * np.sin(T) * np.cos(P)).ravel(),
            (R * np.sin(T) * np.sin(P)).ravel(),
            (R * np.cos(T)).ravel(),
        ],
        axis=1,
    )

    dr = np.gradient(r)
    wgt = (
        (dr * r * r)[:, None, None]
        * w[None, :, None]
        * np.full((1, 1, ang_points), 2.0 * np.pi / ang_points)
    )
    return pos, wgt.ravel()

@pytest.mark.parametrize(
    "n,l,m,basis", [(1, 0, 0, "complex"), (2, 1, 1, "complex"), (3, 2, -2, "real")]
)
def test_norm_is_one(n, l, m, basis):
    pos, wgt = _spherical_grid(n)
    psi = evaluate_state(n, l, m, pos, basis=basis).values
    norm = float(np.sum(np.abs(psi) ** 2 * wgt))
    assert norm == pytest.approx(1.0, abs=2e-3)

def test_complex_phase_is_exp_i_m_phi():
    phis = np.linspace(0.0, 2.0 * np.pi, 9, endpoint=False)
    pos = np.stack([2.0 * np.cos(phis), 2.0 * np.sin(phis), np.full_like(phis, 1.3)], axis=1)
    psi = evaluate_state(3, 2, 2, pos).values
    unwound = np.unwrap(np.angle(psi))
    assert np.diff(unwound) == pytest.approx(2.0 * np.diff(phis))

def test_real_basis_returns_real_dtype():
    pos = np.array([[1.0, 0.5, -0.3], [0.2, -1.0, 2.0]])
    wf = evaluate_state(2, 1, 1, pos, basis="real")
    assert wf.values.dtype == np.float64

def test_origin_is_finite():
    pos = np.zeros((1, 3))
    psi_s = evaluate_state(1, 0, 0, pos).values
    psi_p = evaluate_state(2, 1, 0, pos).values
    assert np.isfinite(psi_s).all() and psi_s[0] != 0.0
    assert psi_p[0] == pytest.approx(0.0)

def test_container_and_provenance():
    pos = np.array([[1.0, 0.0, 0.0]])
    wf = evaluate_state(2, 1, 0, pos, Z=2, mu_ratio=0.5, basis="real")
    assert isinstance(wf, WavefunctionValues)
    assert wf.provenance.fidelity is Fidelity.EXACT
    assert (wf.n, wf.l, wf.m, wf.Z, wf.mu_ratio, wf.basis) == (2, 1, 0, 2, 0.5, "real")

def test_rejects_bad_input():
    with pytest.raises(ValueError):
        evaluate_state(1, 1, 0, np.zeros((1, 3)))
    with pytest.raises(ValueError):
        evaluate_state(1, 0, 0, np.zeros((3,)))
