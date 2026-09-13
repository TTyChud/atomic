
import numpy as np
import pytest

from atomic.numerics.marching_tets import (
    connected_components,
    edge_use_counts,
    enclosed_volume,
    marching_tets,
    surface_area,
)


def _sphere_field(n: int, half_width: float = 2.0):
    axis = np.linspace(-half_width, half_width, n)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    return -(x**2 + y**2 + z**2), axis[1] - axis[0]

def _sphere_mesh(n: int, radius: float, half_width: float = 2.0):
    field, step = _sphere_field(n, half_width)
    return marching_tets(field, -(radius**2), origin=(-half_width,) * 3, spacing=step)

def test_every_tetrahedron_case_emits_a_closed_ring_of_cut_edges():
    from atomic.numerics.marching_tets import _TET_CASES, _TET_EDGES

    for code in range(16):
        inside = {c for c in range(4) if code & (1 << c)}
        expected = {
            e for e, (a, b) in enumerate(map(tuple, _TET_EDGES)) if (a in inside) != (b in inside)
        }
        used = {int(e) for tri in _TET_CASES[code] for e in tri if e >= 0}
        assert used == expected, f"case {code} cuts the wrong edges"

def test_the_two_and_two_cases_emit_two_triangles_and_the_rest_emit_one():
    from atomic.numerics.marching_tets import _TET_CASES

    for code in range(16):
        n_tris = int((_TET_CASES[code][:, 0] >= 0).sum())
        n_inside = bin(code).count("1")
        expected = 0 if n_inside in (0, 4) else (2 if n_inside == 2 else 1)
        assert n_tris == expected

def test_the_six_tetrahedra_tile_the_cell_exactly():
    from atomic.numerics.marching_tets import _CORNER_OFFSETS, _TETS

    total = 0.0
    for tet in _TETS:
        p = _CORNER_OFFSETS[tet].astype(float)
        total += abs(np.dot(p[1] - p[0], np.cross(p[2] - p[0], p[3] - p[0]))) / 6
    assert total == pytest.approx(1.0, abs=1e-12)

@pytest.mark.parametrize("n", [24, 48, 72])
def test_sphere_volume_and_area_converge_on_the_formulas(n):
    radius = 1.0
    mesh = _sphere_mesh(n, radius)
    tolerance = {24: 0.03, 48: 8e-3, 72: 4e-3}[n]
    assert enclosed_volume(mesh) == pytest.approx(4 / 3 * np.pi * radius**3, rel=tolerance)
    assert surface_area(mesh) == pytest.approx(4 * np.pi * radius**2, rel=tolerance)

def test_refining_the_grid_actually_reduces_the_error():
    exact = 4 / 3 * np.pi
    errors = [abs(enclosed_volume(_sphere_mesh(n, 1.0)) - exact) for n in (24, 48, 96)]
    assert errors[1] < errors[0] / 2
    assert errors[2] < errors[1] / 2

def test_vertices_land_on_the_sphere_to_first_order():
    n, half_width = 48, 2.0
    step = 2 * half_width / (n - 1)
    radii = np.linalg.norm(_sphere_mesh(n, 1.0, half_width).vertices, axis=1)
    assert np.abs(radii - 1.0).max() < 0.1 * step

def test_a_plane_is_reproduced_exactly():
    axis = np.linspace(-1.0, 1.0, 21)
    x, _, _ = np.meshgrid(axis, axis, axis, indexing="ij")
    mesh = marching_tets(-x, -0.31, origin=(-1.0,) * 3, spacing=axis[1] - axis[0])
    assert np.abs(mesh.vertices[:, 0] - 0.31).max() < 1e-12

def test_the_surface_is_watertight():
    counts = edge_use_counts(_sphere_mesh(48, 1.0))
    assert set(np.unique(counts).tolist()) == {2}

def test_orientation_is_outward_everywhere():
    mesh = _sphere_mesh(36, 1.0)
    v = mesh.vertices[mesh.triangles]
    signed = np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2]))
    assert (signed > 0).all()

def test_vertices_are_welded_rather_than_duplicated_per_triangle():
    mesh = _sphere_mesh(36, 1.0)
    assert connected_components(mesh) == 1
    assert mesh.vertices.shape[0] < mesh.triangles.shape[0]

def test_two_separated_balls_come_out_as_two_components():
    axis = np.linspace(-3.0, 3.0, 61)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    left = (x + 1.5) ** 2 + y**2 + z**2
    right = (x - 1.5) ** 2 + y**2 + z**2
    mesh = marching_tets(
        -np.minimum(left, right), -0.36, origin=(-3.0,) * 3, spacing=axis[1] - axis[0]
    )
    assert connected_components(mesh) == 2

def test_a_level_no_cell_crosses_gives_an_empty_mesh_not_a_crash():
    mesh = _sphere_mesh(16, 0.0001)
    assert mesh.is_empty
    assert enclosed_volume(mesh) == 0.0

def test_a_level_the_whole_box_is_above_gives_an_empty_mesh():
    field = np.ones((8, 8, 8))
    assert marching_tets(field, 0.5).is_empty

def test_a_field_that_is_not_three_dimensional_is_refused():
    with pytest.raises(ValueError, match="3-D"):
        marching_tets(np.zeros((4, 4)), 0.5)

def test_chunking_does_not_change_the_answer():
    field, step = _sphere_field(40)
    whole = marching_tets(field, -1.0, origin=(-2.0,) * 3, spacing=step, chunk=1000)
    slabbed = marching_tets(field, -1.0, origin=(-2.0,) * 3, spacing=step, chunk=3)
    assert whole.vertices.shape == slabbed.vertices.shape
    assert enclosed_volume(whole) == pytest.approx(enclosed_volume(slabbed), rel=1e-12)
    assert set(np.unique(edge_use_counts(slabbed)).tolist()) == {2}
