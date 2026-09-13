import subprocess
import sys

import pytest
import uvicorn

import atomic.cli as cli


def test_parser_defaults():
    args = cli.build_parser().parse_args(["serve"])
    assert args.command == "serve"
    assert args.port is None
    assert args.no_browser is False

def test_serve_invokes_uvicorn_on_loopback(monkeypatch):
    captured = {}

    def fake_run(app, host, port):
        captured["host"] = host
        captured["port"] = port

    opened = []
    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(cli, "_open_browser_soon", lambda url: opened.append(url))

    cli.main(["serve", "--port", "8123"])
    assert captured == {"host": "127.0.0.1", "port": 8123}
    assert opened == ["http://127.0.0.1:8123"]

def test_env_overrides_apply_when_flags_absent(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        uvicorn, "run", lambda app, host, port: captured.update(host=host, port=port)
    )
    monkeypatch.setattr(cli, "_open_browser_soon", lambda url: None)
    monkeypatch.setenv("ATOMIC_HOST", "0.0.0.0")
    monkeypatch.setenv("ATOMIC_PORT", "9000")

    cli.main(["serve"])
    assert captured == {"host": "0.0.0.0", "port": 9000}

def test_flags_take_precedence_over_env(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        uvicorn, "run", lambda app, host, port: captured.update(host=host, port=port)
    )
    monkeypatch.setattr(cli, "_open_browser_soon", lambda url: None)
    monkeypatch.setenv("ATOMIC_HOST", "0.0.0.0")
    monkeypatch.setenv("ATOMIC_PORT", "9000")

    cli.main(["serve", "--host", "127.0.0.1", "--port", "8001"])
    assert captured == {"host": "127.0.0.1", "port": 8001}

def test_remote_bind_suppresses_browser(monkeypatch):
    """Serving on a non-loopback interface (a separate server) must not try to
    open a desktop browser — there is no desktop."""
    opened = []
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: None)
    monkeypatch.setattr(cli, "_open_browser_soon", lambda url: opened.append(url))

    cli.main(["serve", "--host", "0.0.0.0"])
    assert opened == []

def test_invalid_port_env_falls_back_to_default(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        uvicorn, "run", lambda app, host, port: captured.update(port=port)
    )
    monkeypatch.setattr(cli, "_open_browser_soon", lambda url: None)
    monkeypatch.setenv("ATOMIC_PORT", "not-a-port")

    cli.main(["serve"])
    assert captured["port"] == 8000

def test_platform_port_env_is_honored(monkeypatch):
    """PaaS convention (Render, Heroku, ...): the port arrives as PORT."""
    captured = {}
    monkeypatch.setattr(
        uvicorn, "run", lambda app, host, port: captured.update(port=port)
    )
    monkeypatch.setattr(cli, "_open_browser_soon", lambda url: None)
    monkeypatch.delenv("ATOMIC_PORT", raising=False)
    monkeypatch.setenv("PORT", "10000")

    cli.main(["serve"])
    assert captured["port"] == 10000

def test_atomic_port_beats_platform_port(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        uvicorn, "run", lambda app, host, port: captured.update(port=port)
    )
    monkeypatch.setattr(cli, "_open_browser_soon", lambda url: None)
    monkeypatch.setenv("ATOMIC_PORT", "8080")
    monkeypatch.setenv("PORT", "10000")

    cli.main(["serve"])
    assert captured["port"] == 8080

def test_no_browser_flag(monkeypatch):
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: None)
    opened = []
    monkeypatch.setattr(cli, "_open_browser_soon", lambda url: opened.append(url))
    cli.main(["serve", "--no-browser"])
    assert opened == []

def test_importing_the_cli_does_not_drag_in_the_server_stack():
    probe = (
        "import sys, atomic.cli; "
        "print(','.join(m for m in ('fastapi', 'uvicorn', 'matplotlib') "
        "if m in sys.modules))"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "", (
        f"atomic.cli eagerly imported: {done.stdout.strip()}"
    )

@pytest.mark.parametrize("module", ["atomic", "atomic.cli"])
def test_module_entry_point_runs_the_parser(module):
    done = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    assert "usage: atomic" in done.stdout
    assert "serve" in done.stdout

@pytest.mark.parametrize("module", ["atomic", "atomic.cli"])
def test_module_entry_point_requires_a_subcommand(module):
    done = subprocess.run(
        [sys.executable, "-m", module],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 2
    assert "required" in done.stderr
