from __future__ import annotations

import base64
import json
from collections.abc import Callable
from urllib.parse import urlsplit


class InlineExecutor:
    def submit(self, fn: Callable[..., object], /, *args: object, **kwargs: object) -> None:
        fn(*args, **kwargs)

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        pass

def install_inline_threadpool() -> None:
    from anyio._backends import _asyncio as asyncio_backend

    async def run_sync_inline(
        func: Callable[..., object],
        args: tuple[object, ...],
        abandon_on_cancel: bool = False,  # noqa: ARG001
        limiter: object | None = None,  # noqa: ARG001
    ) -> object:
        return func(*args)
    asyncio_backend.AsyncIOBackend.run_sync_in_worker_thread = staticmethod(run_sync_inline)  # type: ignore[method-assign]

async def dispatch(
    app: Callable[..., object],
    method: str,
    url: str,
    body_b64: str | None,
) -> str:
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
