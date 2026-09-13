"""In-browser engine support (Pyodide/WebAssembly).

The engine worker imports the real FastAPI app and drives it over ASGI.
Two small shims make that work where there is no thread pool and no real
socket:

- `InlineExecutor` replaces the app's ThreadPoolExecutor. WebAssembly keeps
  a single thread, so job creation runs the job to completion inline; the
  client's meta poll then finds the job DONE on its first ask (it already
  tolerates the 409-and-retry path for the slower cases).

- `install_inline_threadpool()` re-points anyio's `to_thread.run_sync` at an
  inline call. FastAPI executes sync endpoints through that pool, and the
  real pool starts one OS thread per call — impossible under WebAssembly
  ("can't start new thread"). The device engine has exactly one thread and
  it is already the right place to run the work.

- `dispatch()` runs one ASGI round-trip and returns a plain result record
  `{status, content_type, body_b64}` — one shape for JSON routes and binary
  job channels alike, so true status codes survive the worker boundary.

This module ships inside the wheel, so the browser runs exactly the code
the deployed server runs and the tests pin.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from urllib.parse import urlsplit


class InlineExecutor:
    """Drop-in stand-in for a ThreadPoolExecutor that just runs inline."""

    def submit(self, fn: Callable[..., object], /, *args: object, **kwargs: object) -> None:
        fn(*args, **kwargs)

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        pass


def install_inline_threadpool() -> None:
    """Make anyio's ``to_thread.run_sync`` run its work inline.

    Must be called before the first request. Idempotent: the replacement is
    a plain function, so re-installing just overwrites it again.
    """
    from anyio._backends import _asyncio as asyncio_backend

    async def run_sync_inline(
        func: Callable[..., object],
        args: tuple[object, ...],
        abandon_on_cancel: bool = False,  # noqa: ARG001 - signature compat
        limiter: object | None = None,  # noqa: ARG001 - inline uses no capacity
    ) -> object:
        return func(*args)

    asyncio_backend.AsyncIOBackend.run_sync_in_worker_thread = staticmethod(run_sync_inline)  # type: ignore[method-assign]


async def dispatch(
    app: Callable[..., object],
    method: str,
    url: str,
    body_b64: str | None,
) -> str:
    """Run one ASGI request against `app`; return the result as JSON.

    `url` is path + query as the client addresses the engine. A JSON request
    body crosses as base64 (the worker posts bodies as text). The reply is a
    JSON string: {"status", "content_type", "body_b64"}.
    """
    parts = urlsplit(url)
    body = base64.b64decode(body_b64) if body_b64 else b""
    headers = [(b"content-type", b"application/json")] if body else []
    scope: dict[str, object] = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": parts.path or "/",
        "raw_path": (parts.path or "/").encode(),
        "query_string": parts.query.encode(),
        "root_path": "",
        "headers": headers,
        "client": ("device", 0),
        "server": ("engine", 0),
    }
    holder: dict[str, object] = {"status": 500, "headers": {}}
    chunks: list[bytes] = []

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(msg: dict[str, object]) -> None:
        if msg["type"] == "http.response.start":
            holder["status"] = msg["status"]
            holder["headers"] = {
                k.decode().lower(): v.decode() for k, v in msg["headers"]  # type: ignore[union-attr]
            }
        elif msg["type"] == "http.response.body":
            chunks.append(msg.get("body", b""))  # type: ignore[arg-type]

    await app(scope, receive, send)  # type: ignore[arg-type]
    headers_out = holder["headers"]
    assert isinstance(headers_out, dict)
    return json.dumps(
        {
            "status": holder["status"],
            "content_type": headers_out.get("content-type", "application/json"),
            "body_b64": base64.b64encode(b"".join(chunks)).decode(),
        }
    )
