import numpy as np
import pytest
from scipy.linalg import eigh_tridiagonal

from atomic.numerics.mesh import (
    RadialMesh,
    display_window,
    exponential_mesh,
    mesh_for_atom,
    mesh_for_atom_at_step,
    uniform_mesh,
)


def ground_state_energy(mesh: RadialMesh, z: float = 1.0) -> float:
    diag, offdiag = mesh.hamiltonian_bands(-z / mesh.r, 0)
    return float(
        eigh_tridiagonal(diag, offdiag, select="i", select_range=(0, 0), eigvals_only=True)[0]
    )

def blunt_wall(mesh: RadialMesh) -> RadialMesh:
    return RadialMesh(
        r=mesh.r, jacobian=mesh.jacobian, step=mesh.step,
        kinetic_diag=mesh.kinetic_diag, kinetic_offdiag=mesh.kinetic_offdiag,
        inner_wall_coupling=0.0, inner_ghost_ratio=0.0,
    )

class TestUniformMesh:
    def test_wall_sits_on_the_origin(self):
        mesh = uniform_mesh(30.0, 2000)
        assert mesh.r[0] == pytest.approx(30.0 / 2001)
        assert mesh.inner_ghost_ratio == 0.0

    def test_rejects_bad_args(self):
        with pytest.raises(ValueError, match="positive"):
            uniform_mesh(0.0, 100)
        with pytest.raises(ValueError, match="at least 3 points"):
            uniform_mesh(10.0, 2)

    def test_integrate_is_the_trapezoid_rule(self):
        mesh = uniform_mesh(20.0, 500)
        f = np.exp(-mesh.r) * mesh.r**2
        assert mesh.integrate(f) == pytest.approx(np.trapezoid(f, mesh.r), rel=1e-15)

    def test_wall_correction_vanishes_on_the_origin(self):
        mesh = uniform_mesh(30.0, 500)
        v = -1.0 / mesh.r
        diag, _ = mesh.hamiltonian_bands(v, 0)
        assert diag == pytest.approx(mesh.kinetic_diag + v, rel=1e-15)

class TestExponentialMesh:
    def test_endpoints_and_jacobian(self):
        mesh = exponential_mesh(1e-3, 60.0, 1000)
        assert mesh.r[0] == pytest.approx(1e-3)
        assert mesh.r[-1] == pytest.approx(60.0)
        assert mesh.jacobian == pytest.approx(mesh.r)
        assert np.all(np.diff(mesh.r) > 0.0)

    def test_rejects_bad_args(self):
        with pytest.raises(ValueError, match="strictly above zero"):
            exponential_mesh(0.0, 40.0, 100)
        with pytest.raises(ValueError, match="must exceed inner radius"):
            exponential_mesh(1.0, 1.0, 100)
        with pytest.raises(ValueError, match="at least 3 points"):
            exponential_mesh(1e-3, 10.0, 2)

    def test_mesh_for_atom_places_the_optimal_inner_wall(self):
        mesh = mesh_for_atom(18, 40.0, 1200)
        assert mesh.r[0] == pytest.approx(1e-3 / 18)
        assert mesh.r[-1] == pytest.approx(40.0)
        with pytest.raises(ValueError, match="Z must be"):
            mesh_for_atom(0, 40.0, 100)

class TestBandsAndTransforms:
    def test_band_shapes_and_guards(self):
        mesh = uniform_mesh(10.0, 100)
        diag, offdiag = mesh.hamiltonian_bands(-1.0 / mesh.r, 2)
        assert diag.shape == (100,)
        assert offdiag.shape == (99,)
        with pytest.raises(ValueError, match=">= 0"):
            mesh.hamiltonian_bands(-1.0 / mesh.r, -1)
        with pytest.raises(ValueError, match="sampled on this mesh"):
            mesh.hamiltonian_bands(np.ones(50), 0)

    def test_ghost_correction_buys_two_orders(self):
        mesh = exponential_mesh(1e-2, 60.0, 800)
        want = 0.5
        assert abs(ground_state_energy(mesh) + want) / want < (
            abs(ground_state_energy(blunt_wall(mesh)) + want) / want / 50
        )

    def test_hydrogen_1s_matches_the_exact_value(self):
        got = ground_state_energy(mesh_for_atom(1, 60.0, 1200))
        assert abs(got + 0.5) / 0.5 < 5e-6

    def test_transforms_round_trip_and_normalize(self):
        mesh = mesh_for_atom(1, 60.0, 2000)
        p = 2.0 * mesh.r * np.exp(-mesh.r)
        assert mesh.to_p(mesh.to_s(p)) == pytest.approx(p, rel=1e-9)
        assert mesh.integrate(mesh.normalized(p) ** 2) == pytest.approx(1.0, rel=1e-12)
        s = mesh.to_s(mesh.normalized(p))
        assert float(s @ s) == pytest.approx(1.0, rel=1e-6)

    def test_cumulative_ends_at_the_total(self):
        mesh = mesh_for_atom(1, 40.0, 1500)
        f = mesh.r * np.exp(-mesh.r)
        assert mesh.cumulative(f)[-1] == pytest.approx(mesh.integrate(f), rel=1e-12)

class TestSizingByStep:
    def test_step_direction_and_shared_endpoints(self):
        coarse = mesh_for_atom_at_step(18, 60.0, 0.01)
        fine = mesh_for_atom_at_step(18, 60.0, 0.005)
        assert coarse.step >= 0.01
        assert fine.r[0] == coarse.r[0]
        assert fine.r[-1] == coarse.r[-1]
        assert fine.points > coarse.points
        by_count = mesh_for_atom(18, 60.0, coarse.points)
        assert np.array_equal(coarse.r, by_count.r)

    def test_rejects_bad_step(self):
        with pytest.raises(ValueError, match="step must be positive"):
            mesh_for_atom_at_step(2, 60.0, 0.0)

class TestRefusals:
    def test_decreasing_mesh_rejected(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            RadialMesh(
                r=np.array([3.0, 2.0, 1.0]), jacobian=np.ones(3), step=1.0,
                kinetic_diag=np.ones(3), kinetic_offdiag=np.ones(2),
                inner_wall_coupling=0.0, inner_ghost_ratio=0.0,
            )

    def test_kinetic_bands_must_match_node_count(self):
        with pytest.raises(ValueError, match="one entry per interior interval"):
            RadialMesh(
                r=np.array([1.0, 2.0, 3.0]), jacobian=np.ones(3), step=1.0,
                kinetic_diag=np.ones(3), kinetic_offdiag=np.ones(3),
                inner_wall_coupling=0.0, inner_ghost_ratio=0.0,
            )

    def test_ghost_node_above_first_point_rejected(self):
        with pytest.raises(ValueError, match=r"below r\[0\]"):
            RadialMesh(
                r=np.array([1.0, 2.0, 3.0]), jacobian=np.ones(3), step=1.0,
                kinetic_diag=np.ones(3), kinetic_offdiag=np.ones(2),
                inner_wall_coupling=0.0, inner_ghost_ratio=1.5,
            )

    def test_off_mesh_integrand_rejected(self):
        mesh = uniform_mesh(10.0, 100)
        with pytest.raises(ValueError, match="sampled on this mesh"):
            mesh.integrate(np.ones(50))

class TestDisplayWindow:
    def test_empty_grid_returns_zero(self):
        assert display_window(np.array([]), np.array([])) == 0.0

    def test_concentrated_density_windows_inside_the_box(self):
        mesh = mesh_for_atom(1, 60.0, 2000)
        density = (2.0 * mesh.r * np.exp(-mesh.r)) ** 2
        window = display_window(mesh.r, density)
        assert window < mesh.r[-1]
        assert window >= 2.0 * mesh.r[int(np.argmax(density))]

    def test_flat_density_keeps_the_box(self):
        r = np.linspace(0.1, 10.0, 100)
        assert display_window(r, np.ones_like(r)) == pytest.approx(10.0)
