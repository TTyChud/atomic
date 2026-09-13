import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from atomic.analytic.hydrogen import energy, mean_radius
from atomic.constants import HARTREE_EV
from atomic.server.app import create_app
from atomic.systems import get_system

H_MU = get_system("h").mu_ratio.value

@pytest.fixture()
def client():
    with TestClient(create_app()) as c:
        yield c

def test_health(client):
    data = client.get("/api/health").json()
    assert data == {"status": "ok", "version": "0.1.0"}

def test_cors_allows_configured_split_deploy_origins(monkeypatch):
    """A UI hosted away from the engine (Vercel in front of Fly) names itself
    in ATOMIC_ALLOWED_ORIGINS; unknown origins get no CORS grant."""
    monkeypatch.setenv("ATOMIC_ALLOWED_ORIGINS", "https://atomic.vercel.app")
    with TestClient(create_app()) as c:
        allowed = c.get("/api/health", headers={"Origin": "https://atomic.vercel.app"})
        assert allowed.headers["access-control-allow-origin"] == "https://atomic.vercel.app"
        stranger = c.get("/api/health", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in stranger.headers

def test_cors_defaults_to_dev_origins_only(monkeypatch):
    monkeypatch.delenv("ATOMIC_ALLOWED_ORIGINS", raising=False)
    with TestClient(create_app()) as c:
        vite = c.get("/api/health", headers={"Origin": "http://localhost:5173"})
        assert vite.headers["access-control-allow-origin"] == "http://localhost:5173"
        other = c.get("/api/health", headers={"Origin": "https://atomic.vercel.app"})
        assert "access-control-allow-origin" not in other.headers

def test_systems_lists_hydrogenic_presets(client):
    systems = client.get("/api/systems").json()["systems"]
    keys = [s["key"] for s in systems]
    assert keys[:6] == ["h", "d", "t", "mu-h", "ps", "he+"]
    assert systems[0]["mu_ratio"]["unit"] == "m_e"
    assert systems[0]["nuclear_radius"]["unit"] == "bohr"
    assert systems[4]["nuclear_radius"] is None

def test_systems_lists_s_cl_as_hartree_fock_only(client):
    systems = {s["key"]: s for s in client.get("/api/systems").json()["systems"]}
    assert systems["ne"]["kind"] == "screened"
    assert systems["ne"]["has_gsz"] is True
    assert systems["s"]["has_gsz"] is False
    assert systems["cl"]["has_gsz"] is False
    assert "Hartree-Fock" in systems["s"]["description"]

def test_state_ground_state_values(client):
    r = client.get("/api/state/1/0/0")
    assert r.status_code == 200
    body = r.json()
    assert body["energy"]["value"] == pytest.approx(energy(1, mu_ratio=H_MU).value)
    assert body["energy_ev"]["value"] == pytest.approx(
        energy(1, mu_ratio=H_MU).value * HARTREE_EV, abs=0.1
    )
    assert body["mean_radius"]["value"] == pytest.approx(
        mean_radius(1, 0, mu_ratio=H_MU).value
    )
    assert body["angular_momentum"]["value"] == pytest.approx(0.0)
    assert (body["radial_nodes"], body["angular_nodes"]) == (0, 0)
    assert body["system"]["key"] == "h"
    assert body["energy"]["provenance"]["fidelity"] == "exact"

def test_state_generic_ion(client):
    body = client.get("/api/state/1/0/0", params={"system": "z3"}).json()
    assert body["energy"]["value"] == pytest.approx(-4.5)
    assert body["system"]["z"] == 3

def test_state_rejects_bad_quantum_numbers(client):
    assert client.get("/api/state/0/0/0").status_code == 422
    assert client.get("/api/state/1/0/1").status_code == 422

def test_state_unknown_system_is_404(client):
    r = client.get("/api/state/1/0/0", params={"system": "uranium"})
    assert r.status_code == 404

def test_levels_ladder(client):
    body = client.get("/api/levels").json()
    assert [lv["n"] for lv in body["gross"]] == [1, 2, 3, 4, 5, 6]
    assert body["gross"][0]["energy"]["value"] == pytest.approx(
        energy(1, mu_ratio=H_MU).value
    )
    assert body["gross"][1]["degeneracy"] == 8
    assert body["fine"] is None
    assert len(client.get("/api/levels", params={"n_max": 2}).json()["gross"]) == 2
    assert client.get("/api/levels", params={"n_max": 0}).status_code == 422
    assert client.get("/api/levels", params={"n_max": 21}).status_code == 422

def test_radial_shape_and_normalization(client):
    body = client.get("/api/radial/1/0", params={"points": 2000}).json()
    assert len(body["r_wavefunction"]["values"]) == 2000
    assert len(body["r_wavefunction"]["grid"]) == 2000
    assert body["r_wavefunction"]["unit"] == "bohr^-3/2"
    grid = np.array(body["radial_probability"]["grid"])
    prob = np.array(body["radial_probability"]["values"])
    assert np.trapezoid(prob, grid) == pytest.approx(1.0, rel=1e-3)
    assert client.get("/api/radial/1/0", params={"points": 10}).status_code == 422
    assert client.get("/api/radial/2/2").status_code == 422

def _wait_done(client, job_id, timeout=30.0):
    start = time.time()
    while time.time() - start < timeout:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.05)
    raise TimeoutError(f"job {job_id} not done after {timeout}s")

def test_sample_job_full_lifecycle(client):
    posted = client.post(
        "/api/jobs/sample",
        json={"n": 1, "l": 0, "m": 0, "count": 1000, "seed": 7},
    )
    assert posted.status_code == 200
    job_id = posted.json()["id"]
    job = _wait_done(client, job_id)
    assert job["status"] == "done"
    assert job["progress"] == pytest.approx(1.0)

    meta = client.get(f"/api/jobs/{job_id}/meta").json()
    assert meta["kind"] == "sample"
    assert meta["count"] == 1000
    assert [c["name"] for c in meta["channels"]] == ["positions", "density", "phase"]
    assert meta["provenance"]["fidelity"] == "numerical"

    positions = client.get(f"/api/jobs/{job_id}/data").content
    assert len(positions) == 1000 * 3 * 4
    density = client.get(f"/api/jobs/{job_id}/data", params={"channel": "density"}).content
    assert len(density) == 1000 * 4
    phase = client.get(f"/api/jobs/{job_id}/data", params={"channel": "phase"}).content
    assert len(phase) == 1000 * 4
    bad = client.get(f"/api/jobs/{job_id}/data", params={"channel": "bogus"})
    assert bad.status_code == 422

def test_sample_job_real_basis_has_no_phase_channel(client):
    job_id = client.post(
        "/api/jobs/sample",
        json={"n": 2, "l": 1, "m": 1, "count": 1000, "basis": "real"},
    ).json()["id"]
    _wait_done(client, job_id)
    meta = client.get(f"/api/jobs/{job_id}/meta").json()
    assert [c["name"] for c in meta["channels"]] == ["positions", "density"]
    r = client.get(f"/api/jobs/{job_id}/data", params={"channel": "phase"})
    assert r.status_code == 422

def test_sample_job_validates_before_creating(client):
    r = client.post("/api/jobs/sample", json={"n": 1, "l": 0, "m": 1, "count": 1000})
    assert r.status_code == 422
    r = client.post("/api/jobs/sample", json={"n": 1, "l": 0, "m": 0, "count": 10})
    assert r.status_code == 422

def test_plane_job_lifecycle(client):
    job_id = client.post(
        "/api/jobs/plane",
        json={"n": 3, "l": 1, "m": 0, "quantity": "density", "resolution": 32},
    ).json()["id"]
    job = _wait_done(client, job_id)
    assert job["status"] == "done"
    meta = client.get(f"/api/jobs/{job_id}/meta").json()
    assert meta["kind"] == "plane"
    assert meta["resolution"] == 32
    assert meta["half_extent"] > 0
    assert meta["unit"] == "bohr^-3"
    data = client.get(f"/api/jobs/{job_id}/data").content
    assert len(data) == 32 * 32 * 4
    assert client.get(
        f"/api/jobs/{job_id}/data", params={"channel": "density"}
    ).status_code == 422

def test_plane_job_psi_quantity(client):
    job_id = client.post(
        "/api/jobs/plane",
        json={"n": 2, "l": 1, "m": 0, "quantity": "psi", "resolution": 16},
    ).json()["id"]
    _wait_done(client, job_id)
    meta = client.get(f"/api/jobs/{job_id}/meta").json()
    assert meta["quantity"] == "psi"
    assert meta["unit"] == "bohr^-3/2"

def test_plane_job_validates(client):
    assert client.post(
        "/api/jobs/plane", json={"n": 1, "l": 0, "m": 0, "resolution": 8}
    ).status_code == 422
    assert client.post(
        "/api/jobs/plane", json={"n": 1, "l": 0, "m": 0, "quantity": "bogus"}
    ).status_code == 422
    assert client.post(
        "/api/jobs/plane", json={"n": 1, "l": 1, "m": 0}
    ).status_code == 422

def test_screened_sample_job_lifecycle(client):
    job_id = client.post(
        "/api/jobs/sample",
        json={"n": 1, "l": 0, "m": 0, "count": 1000, "system": "he", "seed": 3},
    ).json()["id"]
    job = _wait_done(client, job_id)
    assert job["status"] == "done"
    meta = client.get(f"/api/jobs/{job_id}/meta").json()
    assert meta["kind"] == "sample"
    assert meta["model"] == "screened"
    assert meta["system"] == "he"
    assert meta["provenance"]["fidelity"] == "approximation"
    assert [c["name"] for c in meta["channels"]] == ["positions", "density", "phase"]

def test_screened_plane_job(client):
    job_id = client.post(
        "/api/jobs/plane",
        json={"n": 1, "l": 0, "m": 0, "quantity": "density", "resolution": 32, "system": "he"},
    ).json()["id"]
    _wait_done(client, job_id)
    meta = client.get(f"/api/jobs/{job_id}/meta").json()
    assert meta["kind"] == "plane"
    assert meta["model"] == "screened"
    assert len(client.get(f"/api/jobs/{job_id}/data").content) == 32 * 32 * 4

def test_s_cl_refused_with_named_reason(client):
    r = client.post(
        "/api/jobs/sample",
        json={"n": 3, "l": 1, "m": 0, "count": 1000, "system": "s"},
    )
    assert r.status_code == 400
    assert "GSZ" in r.json()["detail"]
    assert client.get("/api/levels", params={"system": "cl"}).status_code == 400
    assert client.get("/api/radial/3/1", params={"system": "s"}).status_code == 400

def test_screened_levels_shape(client):
    body = client.get("/api/levels", params={"system": "ne"}).json()
    assert body["config"] == "1s2 2s2 2p6"
    assert body["is_ground"] is True
    assert body["system"]["kind"] == "screened"
    assert len(body["orbitals"]) > 0
    assert body["total_energy"]["unit"] == "hartree"
    assert body["orbitals"][0]["energy"]["provenance"]["fidelity"] == "approximation"

def test_levels_screened_bad_config_422(client):
    assert client.get("/api/levels", params={"system": "li", "config": "1s5"}).status_code == 422

def test_levels_screened_wrong_electron_count_422(client):
    assert client.get("/api/levels", params={"system": "li", "config": "1s2"}).status_code == 422

def test_levels_screened_excited_config_not_ground(client):
    body = client.get(
        "/api/levels", params={"system": "na", "config": "1s2 2s2 2p6 3p1"}
    ).json()
    assert body["is_ground"] is False
    assert body["config"] == "1s2 2s2 2p6 3p1"

def test_screened_radial_shape(client):
    body = client.get("/api/radial/1/0", params={"system": "he", "points": 100}).json()
    assert len(body["r_wavefunction"]["values"]) == 100
    assert body["system"]["kind"] == "screened"
    assert body["r_wavefunction"]["provenance"]["fidelity"] == "approximation"

def test_unknown_job_is_404_and_unfinished_meta_is_409(client):
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get("/api/jobs/nope/meta").status_code == 404
    job_id = client.post(
        "/api/jobs/sample",
        json={"n": 1, "l": 0, "m": 0, "count": 1000000},
    ).json()["id"]
    assert client.get(f"/api/jobs/{job_id}/meta").status_code == 409

def test_levels_endpoint_gross(client):
    body = client.get("/api/levels?n_max=3").json()
    assert body["n_max"] == 3 and body["fine"] is None
    assert [g["n"] for g in body["gross"]] == [1, 2, 3]
    assert [g["degeneracy"] for g in body["gross"]] == [2, 8, 18]
    e1 = body["gross"][0]["energy"]
    assert e1["unit"] == "hartree"
    assert e1["value"] == pytest.approx(-0.4997278, rel=1e-5)
    assert body["gross"][0]["energy_ev"]["unit"] == "eV"

def test_levels_endpoint_fine_structure(client):
    body = client.get("/api/levels?n_max=2&fine_structure=true").json()
    fine = body["fine"]
    assert [(f["n"], f["l"], f["j"]) for f in fine] == [
        (1, 0, 0.5), (2, 0, 0.5), (2, 1, 0.5), (2, 1, 1.5),
    ]
    for f in fine:
        assert f["shift"]["provenance"]["fidelity"] == "approximation"
        assert f["shift_ev"]["unit"] == "eV"
    assert fine[2]["energy"]["value"] < fine[3]["energy"]["value"]

def test_levels_fine_absent_without_flag(client):
    body = client.get("/api/levels?n_max=2").json()
    assert body["fine_structure"] is False
    assert body["fine"] is None

def test_levels_dirac_is_exact_and_degenerate(client):
    r = client.get("/api/levels", params={"system": "h", "n_max": 3, "dirac": "true"})
    assert r.status_code == 200
    body = r.json()
    assert body["dirac"] is True
    fine = body["fine"]
    assert fine[0]["energy"]["provenance"]["fidelity"] == "exact"
    n2 = [f for f in fine if f["n"] == 2 and f["j"] == 0.5]
    assert len(n2) == 2
    assert n2[0]["energy"]["value"] == pytest.approx(n2[1]["energy"]["value"], abs=1e-14)

def test_levels_perturbative_still_default(client):
    r = client.get("/api/levels", params={"system": "h", "n_max": 2, "fine_structure": "true"})
    body = r.json()
    assert body["dirac"] is False
    assert body["fine"][0]["energy"]["provenance"]["fidelity"] == "approximation"

def test_levels_dirac_supercritical_rejected(client):
    r = client.get("/api/levels", params={"system": "z200", "n_max": 1, "dirac": "true"})
    assert r.status_code == 422

def test_levels_default_alpha_is_real_and_approximation(client):
    body = client.get("/api/levels?fine_structure=true").json()
    assert body["alpha"] == pytest.approx(1 / 137.035999084, rel=1e-6)
    assert body["fine"][0]["shift"]["provenance"]["fidelity"] == "approximation"

def test_levels_altered_alpha_is_counterfactual(client):
    real = client.get("/api/levels?fine_structure=true").json()
    alt = client.get("/api/levels?fine_structure=true&alpha=0.05").json()
    assert alt["alpha"] == pytest.approx(0.05)
    assert alt["fine"][0]["shift"]["provenance"]["fidelity"] == "counterfactual"
    assert abs(alt["fine"][0]["shift"]["value"]) > abs(real["fine"][0]["shift"]["value"])

def test_levels_rejects_bad_alpha_and_z(client):
    assert client.get("/api/levels?alpha=0").status_code == 422
    assert client.get("/api/levels?alpha=0.6").status_code == 422
    assert client.get("/api/levels?system=z0").status_code == 422
    assert client.get("/api/levels?system=z99").status_code == 200

def test_levels_zeeman_splits_fine_levels(client):
    r = client.get("/api/levels?system=h&n_max=3&fine_structure=true&b_field=2")
    assert r.status_code == 200
    body = r.json()
    assert body["b_field"] == 2.0
    p = next(f for f in body["fine"] if f["n"] == 2 and f["l"] == 1 and f["j"] == 1.5)
    assert p["sublevels"] is not None and len(p["sublevels"]) == 4
    s0 = p["sublevels"][0]
    assert s0["energy"]["provenance"]["fidelity"] == "approximation"
    assert "m_l" in s0["high_field_label"]

def test_levels_zeeman_absent_without_field(client):
    body = client.get("/api/levels?system=h&n_max=2&fine_structure=true").json()
    assert body["b_field"] == 0.0
    assert all(f.get("sublevels") is None for f in body["fine"])

def test_levels_zeeman_negative_field_rejected(client):
    r = client.get("/api/levels?system=h&n_max=2&fine_structure=true&b_field=-1")
    assert r.status_code == 422

def test_levels_zeeman_ignored_for_screened(client):
    r = client.get("/api/levels?system=he&fine_structure=true&b_field=5")
    assert r.status_code == 200
    assert "orbitals" in r.json()

def test_levels_stark_splits_gross_levels(client):
    r = client.get("/api/levels?system=h&n_max=3&e_field=50")
    assert r.status_code == 200
    body = r.json()
    assert body["e_field"] == 50.0
    g2 = next(g for g in body["gross"] if g["n"] == 2)
    assert g2["sublevels"] is not None and len(g2["sublevels"]) == 4
    s0 = g2["sublevels"][0]
    assert s0["energy"]["provenance"]["fidelity"] == "approximation"
    assert "k" in s0 and "n1" in s0

def test_levels_stark_absent_without_field(client):
    body = client.get("/api/levels?system=h&n_max=2").json()
    assert body["e_field"] == 0.0
    assert all(g.get("sublevels") is None for g in body["gross"])

def test_levels_stark_independent_of_fine_structure(client):
    r = client.get("/api/levels?system=h&n_max=2&fine_structure=false&e_field=30")
    assert r.status_code == 200
    g2 = next(g for g in r.json()["gross"] if g["n"] == 2)
    assert g2["sublevels"] is not None and len(g2["sublevels"]) == 4

def test_levels_stark_negative_field_rejected(client):
    r = client.get("/api/levels?system=h&n_max=2&e_field=-1")
    assert r.status_code == 422

def test_levels_stark_ignored_for_screened(client):
    r = client.get("/api/levels?system=he&e_field=50")
    assert r.status_code == 200
    assert "orbitals" in r.json()

def test_levels_hyperfine_splits_hydrogen_ground_state(client):
    r = client.get("/api/levels?system=h&n_max=2&hyperfine=true")
    assert r.status_code == 200
    body = r.json()
    assert body["hyperfine"] is True
    shells = body["hyperfine_shells"]
    assert shells is not None
    s1 = next(s for s in shells if s["n"] == 1)
    assert s1["available"] is True
    assert s1["nucleus"] == "proton" and s1["I"] == 0.5
    assert sorted(lv["F"] for lv in s1["levels"]) == [0.0, 1.0]
    assert s1["A"]["provenance"]["fidelity"] == "approximation"
    split_ev = (max(lv["energy_ev"]["value"] for lv in s1["levels"])
                - min(lv["energy_ev"]["value"] for lv in s1["levels"]))
    assert split_ev == pytest.approx(5.874e-6, rel=2e-2)

def test_levels_hyperfine_absent_without_flag(client):
    body = client.get("/api/levels?system=h&n_max=2").json()
    assert body["hyperfine"] is False
    assert body["hyperfine_shells"] is None

def test_levels_hyperfine_spin_zero_nucleus_does_not_split(client):
    body = client.get("/api/levels?system=he%2B&n_max=1&hyperfine=true").json()
    s1 = body["hyperfine_shells"][0]
    assert s1["available"] is True and s1["I"] == 0.0
    assert len(s1["levels"]) == 1
    assert "spin" in (s1["note"] or "").lower()

def test_levels_hyperfine_unavailable_for_positronium(client):
    body = client.get("/api/levels?system=ps&n_max=2&hyperfine=true").json()
    shells = body["hyperfine_shells"]
    assert shells is not None and shells[0]["available"] is False
    assert shells[0]["reason"]

def test_levels_hyperfine_ignored_for_screened(client):
    r = client.get("/api/levels?system=he&hyperfine=true")
    assert r.status_code == 200
    assert "orbitals" in r.json()

def test_spectrum_endpoint_with_comparison(client):
    body = client.get("/api/spectrum?system=h&n_max=6").json()
    assert body["reference_citation"] and "NIST" in body["reference_citation"]
    assert len(body["lines"]) > 10
    assert body["comparison"] is not None
    assert all(c["within_tolerance"] for c in body["comparison"])

def test_spectrum_without_reference_data(client):
    body = client.get("/api/spectrum?system=ps&n_max=3").json()
    assert body["comparison"] is None
    assert body["reference_citation"] is None
    assert len(body["lines"]) > 0

def test_spectrum_rejects_bad_n_max(client):
    assert client.get("/api/spectrum?n_max=1").status_code == 422
    assert client.get("/api/spectrum?n_max=11").status_code == 422
