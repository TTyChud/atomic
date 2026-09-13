import numpy as np
import pytest

from atomic.analytic.angular import (
    AngularValues,
    real_orbital_label,
    spherical_harmonic,
    validate_angular,
)
from atomic.provenance import Fidelity

_X, _W = np.polynomial.legendre.leggauss(96)
_THETA = np.arccos(_X)
_PHI = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
_TT, _PP = np.meshgrid(_THETA, _PHI, indexing="ij")

def _inner(l1, m1, b1, l2, m2, b2):
    y1 = spherical_harmonic(l1, m1, _TT, _PP, basis=b1).values
    y2 = spherical_harmonic(l2, m2, _TT, _PP, basis=b2).values
    integrand = np.conj(y1) * y2
    phi_mean = integrand.mean(axis=1) * 2.0 * np.pi
    return float(np.real(np.sum(_W * phi_mean)))

@pytest.mark.parametrize("basis", ["complex", "real"])
def test_orthonormality_up_to_l3(basis):
    states = [(l, m) for l in range(4) for m in range(-l, l + 1)]
    for i, (l1, m1) in enumerate(states):
        for l2, m2 in states[i:]:
            expected = 1.0 if (l1, m1) == (l2, m2) else 0.0
            assert _inner(l1, m1, basis, l2, m2, basis) == pytest.approx(
                expected, abs=1e-10
            ), (l1, m1, l2, m2, basis)

def test_known_closed_forms():
    th, ph = np.array([0.7]), np.array([1.1])
    y00 = spherical_harmonic(0, 0, th, ph).values[0]
    assert y00 == pytest.approx(1.0 / np.sqrt(4.0 * np.pi))
    y10 = spherical_harmonic(1, 0, th, ph).values[0]
    assert np.real(y10) == pytest.approx(np.sqrt(3.0 / (4.0 * np.pi)) * np.cos(0.7))

def test_condon_shortley_phase():

    y11 = spherical_harmonic(1, 1, np.array([np.pi / 2]), np.array([0.0])).values[0]
    assert np.real(y11) == pytest.approx(-np.sqrt(3.0 / (8.0 * np.pi)))
    assert np.imag(y11) == pytest.approx(0.0, abs=1e-15)

def test_real_basis_is_real_and_correctly_oriented():

    px = spherical_harmonic(1, 1, np.array([np.pi / 2]), np.array([0.0]), basis="real")
    py = spherical_harmonic(1, -1, np.array([np.pi / 2]), np.array([np.pi / 2]), basis="real")
    peak = np.sqrt(3.0 / (4.0 * np.pi))
    assert px.values.dtype.kind == "f"
    assert px.values[0] == pytest.approx(peak)
    assert py.values[0] == pytest.approx(peak)

    dxy = spherical_harmonic(2, -2, np.array([np.pi / 2]), np.array([np.pi / 4]), basis="real")
    assert dxy.values[0] > 0.0

def test_complex_phase_winds_with_m():
    phis = np.linspace(0.0, 2.0 * np.pi, 7, endpoint=False)
    y = spherical_harmonic(2, 2, np.full_like(phis, 1.0), phis).values
    unwound = np.unwrap(np.angle(y))
    assert np.diff(unwound) == pytest.approx(2.0 * np.diff(phis))

def test_provenance_and_metadata():
    av = spherical_harmonic(2, 1, np.array([0.3]), np.array([0.4]), basis="real")
    assert isinstance(av, AngularValues)
    assert av.provenance.fidelity is Fidelity.EXACT
    assert av.basis == "real"
    assert (av.l, av.m) == (2, 1)

def test_labels():
    assert real_orbital_label(0, 0) == "s"
    assert real_orbital_label(1, 0) == "p_z"
    assert real_orbital_label(1, 1) == "p_x"
    assert real_orbital_label(1, -1) == "p_y"
    assert real_orbital_label(2, 0) == "d_z2"
    assert real_orbital_label(2, 2) == "d_x2-y2"
    assert real_orbital_label(2, -2) == "d_xy"
    assert real_orbital_label(3, 0) == "f_z3"
    assert real_orbital_label(4, 3) == "g(m=+3, cos)"

def test_validation():
    with pytest.raises(ValueError):
        validate_angular(-1, 0)
    with pytest.raises(ValueError):
        validate_angular(1, 2)
    with pytest.raises(ValueError):
        spherical_harmonic(1, 0, np.array([0.1]), np.array([0.1]), basis="chebyshev")
