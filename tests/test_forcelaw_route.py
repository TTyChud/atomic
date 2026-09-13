import pytest
from fastapi.testclient import TestClient

from atomic.analytic.hydrogen import energy
from atomic.server.app import create_app
from atomic.systems import get_system

H_MU = get_system("h").mu_ratio.value

@pytest.fixture()
def client():
    with TestClient(create_app()) as c:
        yield c

def test_powerlaw_default_is_hydrogen(client):
    body = client.get("/api/forcelaw").json()
    assert body["preset"] == "powerlaw"
    assert body["params"] == {"p": 1.0}
    assert body["bound_count"] == 4
    energies = [c["energy"]["value"] for c in body["counterfactual"]]
    assert energies[0] == pytest.approx(energy(1, mu_ratio=H_MU).value, rel=2e-4)
    assert body["reference"]["kind"] == "levels"
    assert body["counterfactual"][0]["energy"]["provenance"]["fidelity"] == "numerical"
    assert len(body["potential_curve"]["r"]) == 256

def test_custom_expr_recovers_hydrogen(client):
    body = client.get("/api/forcelaw", params={"preset": "custom", "expr": "-1/r"}).json()
    assert body["preset"] == "custom"
    assert body["expression"] == "-1/r"
    assert all(c["trusted"] for c in body["counterfactual"][:3])

def test_forcelaw_validates(client):
    assert client.get("/api/forcelaw", params={"preset": "nope"}).status_code == 422
    assert client.get("/api/forcelaw", params={"p": 1.9}).status_code == 422
    assert client.get("/api/forcelaw", params={"l": -1}).status_code == 422
    assert client.get("/api/forcelaw", params={"n_states": 0}).status_code == 422
    assert client.get("/api/forcelaw", params={"preset": "custom"}).status_code == 422
    assert client.get(
        "/api/forcelaw", params={"preset": "custom", "expr": "foo(r)"}
    ).status_code == 422
