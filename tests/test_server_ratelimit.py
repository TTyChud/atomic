
import logging

import pytest
from fastapi.testclient import TestClient

from atomic.server.app import create_app

SAMPLE = {"n": 1, "l": 0, "m": 0, "count": 1000}

@pytest.fixture
def limited(monkeypatch):
    monkeypatch.setenv("ATOMIC_RATE_LIMIT", "on")
    monkeypatch.setenv("ATOMIC_RATE_LIMIT_BURST", "2")
    monkeypatch.setenv("ATOMIC_RATE_LIMIT_PERIOD", "600")
    with TestClient(create_app()) as client:
        yield client

def test_the_burst_is_allowed_and_the_next_job_is_refused(limited):
    assert limited.post("/api/jobs/sample", json=SAMPLE).status_code == 200
    assert limited.post("/api/jobs/sample", json=SAMPLE).status_code == 200

    refused = limited.post("/api/jobs/sample", json=SAMPLE)
    assert refused.status_code == 429
    assert "retry" in refused.json()["detail"].lower()

def test_a_refusal_says_when_to_come_back(limited):
    for _ in range(3):
        response = limited.post("/api/jobs/sample", json=SAMPLE)
    assert response.status_code == 429
    retry_after = int(response.headers["Retry-After"])
    assert retry_after == 300

def test_reads_are_never_charged(limited):
    for _ in range(4):
        limited.post("/api/jobs/sample", json=SAMPLE)

    assert limited.get("/api/systems").status_code == 200
    assert limited.get("/api/state/1/0/0").status_code == 200
    assert limited.get("/api/levels?system=h&n_max=3").status_code == 200

def test_the_limiter_is_off_when_the_environment_says_so(monkeypatch):
    monkeypatch.setenv("ATOMIC_RATE_LIMIT", "off")
    monkeypatch.setenv("ATOMIC_RATE_LIMIT_BURST", "1")
    with TestClient(create_app()) as client:
        codes = [client.post("/api/jobs/sample", json=SAMPLE).status_code for _ in range(5)]
    assert codes == [200] * 5

def test_a_named_proxy_header_separates_clients_behind_one_address(monkeypatch):
    monkeypatch.setenv("ATOMIC_RATE_LIMIT", "on")
    monkeypatch.setenv("ATOMIC_RATE_LIMIT_BURST", "1")
    monkeypatch.setenv("ATOMIC_RATE_LIMIT_PERIOD", "600")
    monkeypatch.setenv("ATOMIC_CLIENT_IP_HEADER", "Fly-Client-IP")
    with TestClient(create_app()) as client:
        first = client.post("/api/jobs/sample", json=SAMPLE, headers={"Fly-Client-IP": "1.1.1.1"})
        same = client.post("/api/jobs/sample", json=SAMPLE, headers={"Fly-Client-IP": "1.1.1.1"})
        other = client.post("/api/jobs/sample", json=SAMPLE, headers={"Fly-Client-IP": "2.2.2.2"})
    assert first.status_code == 200
    assert same.status_code == 429
    assert other.status_code == 200

def test_a_spoofed_prefix_does_not_buy_a_fresh_bucket(monkeypatch):
    monkeypatch.setenv("ATOMIC_RATE_LIMIT", "on")
    monkeypatch.setenv("ATOMIC_RATE_LIMIT_BURST", "1")
    monkeypatch.setenv("ATOMIC_RATE_LIMIT_PERIOD", "600")
    monkeypatch.setenv("ATOMIC_CLIENT_IP_HEADER", "X-Forwarded-For")
    with TestClient(create_app()) as client:
        first = client.post(
            "/api/jobs/sample", json=SAMPLE, headers={"X-Forwarded-For": "spoof-a, 3.3.3.3"}
        )
        rotated = client.post(
            "/api/jobs/sample", json=SAMPLE, headers={"X-Forwarded-For": "spoof-b, 3.3.3.3"}
        )
    assert first.status_code == 200
    assert rotated.status_code == 429

def test_a_refusal_names_who_was_charged(monkeypatch, caplog):
    monkeypatch.setenv("ATOMIC_RATE_LIMIT", "on")
    monkeypatch.setenv("ATOMIC_RATE_LIMIT_BURST", "1")
    monkeypatch.setenv("ATOMIC_RATE_LIMIT_PERIOD", "600")
    monkeypatch.setenv("ATOMIC_CLIENT_IP_HEADER", "Fly-Client-IP")
    with caplog.at_level(logging.WARNING, logger="atomic.server.app"):
        with TestClient(create_app()) as client:
            client.post("/api/jobs/sample", json=SAMPLE, headers={"Fly-Client-IP": "5.5.5.5"})
            refused = client.post(
                "/api/jobs/sample", json=SAMPLE, headers={"Fly-Client-IP": "5.5.5.5"}
            )
    assert refused.status_code == 429
    assert "5.5.5.5" in caplog.text

def test_an_unnamed_proxy_header_is_ignored(monkeypatch):
    monkeypatch.setenv("ATOMIC_RATE_LIMIT", "on")
    monkeypatch.setenv("ATOMIC_RATE_LIMIT_BURST", "1")
    monkeypatch.setenv("ATOMIC_RATE_LIMIT_PERIOD", "600")
    monkeypatch.delenv("ATOMIC_CLIENT_IP_HEADER", raising=False)
    with TestClient(create_app()) as client:
        first = client.post("/api/jobs/sample", json=SAMPLE, headers={"Fly-Client-IP": "1.1.1.1"})
        spoofed = client.post("/api/jobs/sample", json=SAMPLE, headers={"Fly-Client-IP": "9.9.9.9"})
    assert first.status_code == 200
    assert spoofed.status_code == 429
