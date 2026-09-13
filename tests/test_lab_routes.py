import pytest
from fastapi.testclient import TestClient

from atomic.server.app import create_app


@pytest.fixture()
def client():
    with TestClient(create_app()) as c:
        yield c

def test_constants_real_universe(client):
    body = client.get("/api/constants").json()
    assert body["altered"] is False
    assert body["alpha"]["changed"] is False
    assert body["alpha"]["quantity"]["provenance"]["fidelity"] == "exact"

def test_constants_degeneracy_pair(client):
    body = client.get("/api/constants", params={"e": 2.0, "eps0": 4.0}).json()
    assert body["altered"] is True
    assert body["alpha"]["changed"] is False
    assert body["bohr_radius_pm"]["changed"] is False
    assert body["hartree_ev"]["changed"] is False
    assert body["alpha"]["quantity"]["provenance"]["fidelity"] == "counterfactual"

def test_constants_rejects_out_of_range(client):
    assert client.get("/api/constants", params={"e": 10.0}).status_code == 422
    assert client.get("/api/constants", params={"c": 0.01}).status_code == 422

def test_classical_hydrogen_collapse(client):
    body = client.get("/api/classical", params={"system": "h", "n": 1}).json()
    assert body["z"] == 1
    assert body["collapse_time_s"]["value"] == pytest.approx(1.556e-11, rel=0.02)
    assert body["collapse_time_s"]["provenance"]["fidelity"] == "counterfactual"
    assert [o["n"] for o in body["orbits"]] == [1]

def test_classical_rejects_bad_n(client):
    assert client.get("/api/classical", params={"n": 0}).status_code == 422
    assert client.get("/api/classical", params={"system": "uranium"}).status_code == 404
