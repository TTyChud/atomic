import pytest

from atomic.classical import classical_ghost
from atomic.provenance import Fidelity


def test_hydrogen_ground_collapse_time_matches_literature():
    g = classical_ghost(n=1, system="h")
    assert g.collapse_time_s.value == pytest.approx(1.556e-11, rel=0.02)
    assert g.collapse_time_s.unit == "s"
    assert g.collapse_time_s.provenance.fidelity is Fidelity.COUNTERFACTUAL

def test_hydrogen_ground_orbit_count_matches_literature():
    g = classical_ghost(n=1, system="h")
    assert g.orbit_count.value == pytest.approx(2.05e5, rel=0.03)
    assert g.orbit_count.provenance.fidelity is Fidelity.COUNTERFACTUAL

def test_bohr_radius_scales_as_n_squared_over_z():
    g = classical_ghost(n=3, system="h")
    assert tuple(o.n for o in g.orbits) == (1, 2, 3)
    r1 = g.orbits[0].radius_bohr.value
    r3 = g.orbits[2].radius_bohr.value
    assert r3 / r1 == pytest.approx(9.0, rel=1e-6)
    assert g.orbits[0].radius_bohr.provenance.fidelity is Fidelity.APPROXIMATION

def test_r0_is_current_n_radius():
    g = classical_ghost(n=2, system="h")
    assert g.r0_bohr.value == pytest.approx(g.orbits[-1].radius_bohr.value, rel=1e-12)

def test_higher_z_collapses_faster():
    h = classical_ghost(n=1, system="h")
    he = classical_ghost(n=1, system="he+")
    assert he.collapse_time_s.value < h.collapse_time_s.value

def test_muonic_hydrogen_smaller_and_faster():
    h = classical_ghost(n=1, system="h")
    mu = classical_ghost(n=1, system="mu-h")
    assert mu.r0_bohr.value < h.r0_bohr.value
    assert mu.collapse_time_s.value < h.collapse_time_s.value
