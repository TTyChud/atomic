import argparse
import os
import threading
import webbrowser

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atomic")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="launch the server and open the app")
    serve.add_argument(
        "--host",
        default=None,
        help=(
            "interface to bind (default: ATOMIC_HOST or 127.0.0.1; "
            "containers need 0.0.0.0)"
        ),
    )
    serve.add_argument(
        "--port", type=int, default=None, help="port (default: ATOMIC_PORT or 8000)"
    )
    serve.add_argument("--no-browser", action="store_true")
    return parser


def _resolve_serve_target(args: argparse.Namespace) -> tuple[str, int]:
    """Precedence: explicit flag > ATOMIC_HOST/ATOMIC_PORT env > PORT (PaaS
    convention: Render, Heroku, ...) > loopback default."""
    host = args.host
    if host is None:
        host = os.environ.get("ATOMIC_HOST", DEFAULT_HOST)
    port = args.port
    if port is None:
        raw = os.environ.get("ATOMIC_PORT") or os.environ.get("PORT")
        try:
            port = int(raw) if raw is not None else DEFAULT_PORT
        except ValueError:
            port = DEFAULT_PORT
    return host, port


def _open_browser_soon(url: str) -> None:
    threading.Timer(1.5, webbrowser.open, args=(url,)).start()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        import uvicorn

        from atomic.server.app import create_app

        host, port = _resolve_serve_target(args)
        if host == DEFAULT_HOST and not args.no_browser:
            _open_browser_soon(f"http://{host}:{port}")
        uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
