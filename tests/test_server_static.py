
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from atomic.server.app import _configure_logging, _web_dist, create_app


@pytest.fixture()
def built(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text('<div id="root"></div>', encoding="utf-8")
    return dist

def test_the_default_is_the_source_checkout(monkeypatch):
    monkeypatch.delenv("ATOMIC_WEB_DIST", raising=False)
    expected = Path(__file__).resolve().parents[1] / "web" / "dist"
    assert _web_dist() == expected

def test_the_override_names_where_the_build_is(monkeypatch, built):
    monkeypatch.setenv("ATOMIC_WEB_DIST", str(built))
    assert _web_dist() == built

    with TestClient(create_app()) as client:
        served = client.get("/")
    assert served.status_code == 200
    assert 'id="root"' in served.text

def test_a_missing_build_is_said_out_loud(monkeypatch, tmp_path, caplog):
    absent = tmp_path / "never-built"
    monkeypatch.setenv("ATOMIC_WEB_DIST", str(absent))

    with caplog.at_level(logging.WARNING, logger="atomic.server.app"):
        with TestClient(create_app()) as client:
            served = client.get("/")

    assert served.status_code == 404
    assert str(absent) in caplog.text

def test_a_mounted_build_is_said_out_loud(monkeypatch, built, caplog):
    monkeypatch.setenv("ATOMIC_WEB_DIST", str(built))

    with caplog.at_level(logging.INFO, logger="atomic.server.app"):
        create_app()

    assert str(built) in caplog.text
    assert "Mounted the UI from" in caplog.text

def test_an_unconfigured_root_is_given_a_handler():
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    try:
        root.handlers.clear()
        _configure_logging()
        assert root.handlers
        assert root.level <= logging.INFO
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)

def test_existing_logging_configuration_is_left_alone():
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    try:
        sentinel = logging.NullHandler()
        root.handlers[:] = [sentinel]
        _configure_logging()
        assert root.handlers == [sentinel]
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)

def test_the_api_still_answers_without_a_build(monkeypatch, tmp_path):
    monkeypatch.setenv("ATOMIC_WEB_DIST", str(tmp_path / "never-built"))
    with TestClient(create_app()) as client:
        assert client.get("/api/health").json()["status"] == "ok"
