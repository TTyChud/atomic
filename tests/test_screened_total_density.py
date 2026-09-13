
import numpy as np
import pytest
from fastapi.testclient import TestClient

from atomic.atoms import aufbau_configuration
from atomic.provenance import Fidelity
from atomic.screened_atom import screened_total_radial_density
from atomic.server.app import create_app


def _peaks(field, floor=0.01):
    v = field.values
    big = v > floor * v.max()
    return [
        field.grid[i]
        for i in range(1, len(v) - 1)
        if big[i] and v[i] > v[i - 1] and v[i] > v[i + 1]
    ]

@pytest.mark.parametrize("z", [2, 3, 6, 10, 18])
def test_the_curve_integrates_to_the_electron_count(z):
    d = screened_total_radial_density(z, z)
    assert np.trapezoid(d.values, d.grid) == pytest.approx(z, abs=5e-3)

def test_the_residual_is_reported_as_the_error_bar():
    d = screened_total_radial_density(18, 18)
    assert d.unit == "electrons/bohr"
    assert d.provenance.error_estimate is not None
    assert d.provenance.error_estimate < 5e-3
    assert d.provenance.error_estimate == pytest.approx(
        abs(np.trapezoid(d.values, d.grid) - 18.0), rel=1e-9
    )

def test_refining_the_display_grid_does_not_make_it_worse():
    errs = [
        abs(np.trapezoid(d.values, d.grid) - 18.0)
        for d in (screened_total_radial_density(18, 18, points=p)
                  for p in (400, 800, 1600))
    ]
    assert max(errs) < 5e-3
    assert errs[2] <= errs[0] * 1.5

@pytest.mark.parametrize("z,shells", [(2, 1), (10, 2), (18, 3)])
def test_it_has_one_peak_per_shell(z, shells):
    assert len(_peaks(screened_total_radial_density(z, z))) == shells

def test_the_k_shell_integrates_to_more_than_two_electrons():
    from atomic.hf_atom import hf_total_radial_density

    for d in (screened_total_radial_density(18, 18, points=4000),
              hf_total_radial_density(18, 18, points=4000)):
        v, g = d.values, d.grid
        first_min = next(
            i for i in range(1, len(v) - 1)
            if v[i] < v[i - 1] and v[i] < v[i + 1] and v[i] > 1e-3 * v.max()
        )
        k_shell = np.trapezoid(v[: first_min + 1], g[: first_min + 1])
        assert k_shell == pytest.approx(2.2, abs=0.05)

def test_the_shells_land_where_the_shells_are():
    k, l_, m = _peaks(screened_total_radial_density(18, 18))
    assert 0.03 < k < 0.09
    assert 0.2 < l_ < 0.45
    assert 0.9 < m < 1.8

def test_it_is_labelled_as_the_screened_model_and_not_as_hartree_fock():
    d = screened_total_radial_density(18, 18)
    assert d.provenance.fidelity is Fidelity.APPROXIMATION
    assert "N = 18" in d.label
    joined = " ".join(d.provenance.assumptions).lower()
    assert "fitted" in joined or "not fitted" in joined

def test_a_non_ground_configuration_is_honoured():
    assert aufbau_configuration(10) == (((1, 0), 2), ((2, 0), 2), ((2, 1), 6))
    excited = (((1, 0), 2), ((2, 0), 2), ((2, 1), 5), ((3, 0), 1))
    d = screened_total_radial_density(10, 10, config=excited)
    assert np.trapezoid(d.values, d.grid) == pytest.approx(10.0, abs=5e-3)

def test_sulfur_is_refused_here_too():
    with pytest.raises(ValueError, match="no sourced GSZ parameters"):
        screened_total_radial_density(16, 16)

@pytest.fixture()
def client():
    with TestClient(create_app()) as c:
        yield c

def test_the_radial_endpoint_ships_it_under_the_screened_model(client):
    d = client.get("/api/radial/3/1?system=ar").json()["total_density"]
    assert np.trapezoid(d["values"], d["grid"]) == pytest.approx(18.0, abs=5e-3)
    assert d["unit"] == "electrons/bohr"

def test_it_follows_the_configuration_the_request_names(client):
    excited = "1s2 2s2 2p6 3s2 3p5 4s1"
    a = client.get("/api/radial/3/1?system=ar").json()["total_density"]
    b = client.get(f"/api/radial/3/1?system=ar&config={excited}").json()["total_density"]
    assert np.trapezoid(b["values"], b["grid"]) == pytest.approx(18.0, abs=5e-3)
    assert sum(b["values"][-40:]) > sum(a["values"][-40:])

def test_a_one_electron_system_still_gets_none(client):
    assert client.get("/api/radial/3/1?system=h").json()["total_density"] is None
