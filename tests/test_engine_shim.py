"""Evidence for the in-browser engine shim.

The device engine runs the real app through these shims, so the tests pin
the same behaviors the browser relies on: JSON routes, binary channels, true
status codes, and jobs that complete inside their create call.
"""

import base64
import json

import numpy as np
from anyio._backends import _asyncio as _asyncio_backend

from atomic.server.app import create_app
from atomic.server.engine_shim import (
    InlineExecutor,
    dispatch,
    install_inline_threadpool,
)

_real_run_sync_in_worker_thread = (
    _asyncio_backend.AsyncIOBackend.run_sync_in_worker_thread
)

def test_install_inline_threadpool_runs_sync_endpoints_without_a_thread():
    """The WASM case: anyio must not spawn a thread for a sync endpoint."""
    import asyncio

    async def scenario():
        from anyio._backends import _asyncio as asyncio_backend

        install_inline_threadpool()
        try:
            return await asyncio_backend.AsyncIOBackend.run_sync_in_worker_thread(
                lambda: "inline", ()
            )
        finally:

            asyncio_backend.AsyncIOBackend.run_sync_in_worker_thread = (
                _real_run_sync_in_worker_thread
            )

    assert asyncio.run(scenario()) == "inline"

def test_inline_executor_runs_the_call_now():
    seen = []

    def work(report):
        seen.append("ran")
        report(1.0)

    InlineExecutor().submit(work, lambda f: None)
    assert seen == ["ran"]

def test_dispatch_answers_json_with_true_status():
    import asyncio

    app = create_app()
    result = json.loads(asyncio.run(dispatch(app, "GET", "/api/systems", None)))
    assert result["status"] == 200
    assert result["content_type"].startswith("application/json")
    body = json.loads(base64.b64decode(result["body_b64"]))
    assert body["systems"][0]["key"] == "h"

def test_dispatch_answers_unknown_route_as_404():
    import asyncio

    app = create_app()
    result = json.loads(asyncio.run(dispatch(app, "GET", "/api/nope", None)))
    assert result["status"] == 404

def test_dispatch_carries_a_post_body_and_runs_the_job_inline():
    import asyncio
    import json as json_mod

    app = create_app()
    app.state.executor = InlineExecutor()
    payload = json_mod.dumps(
        {"n": 2, "l": 1, "m": 0, "count": 1000, "seed": 0, "basis": "complex", "system": "h"}
    ).encode()
    body_b64 = base64.b64encode(payload).decode()

    created = json.loads(
        asyncio.run(dispatch(app, "POST", "/api/jobs/sample", body_b64))
    )
    assert created["status"] == 200
    job_id = json_mod.loads(base64.b64decode(created["body_b64"]))["id"]

    meta = json.loads(asyncio.run(dispatch(app, "GET", f"/api/jobs/{job_id}/meta", None)))
    assert meta["status"] == 200
    meta_body = json_mod.loads(base64.b64decode(meta["body_b64"]))

    assert meta_body["kind"] == "sample"

    data = json.loads(
        asyncio.run(dispatch(app, "GET", f"/api/jobs/{job_id}/data?channel=density", None))
    )
    assert data["status"] == 200
    assert data["content_type"] == "application/octet-stream"
    floats = np.frombuffer(base64.b64decode(data["body_b64"]), dtype=np.float32)
    assert floats.size == 1000 and np.all(floats > 0)

def test_dispatch_passes_the_query_string_through():
    import asyncio
    import json as json_mod

    app = create_app()
    result = json.loads(
        asyncio.run(dispatch(app, "GET", "/api/state/2/1/0?system=he%2B", None))
    )
    assert result["status"] == 200
    body = json_mod.loads(base64.b64decode(result["body_b64"]))
    assert body["system"]["key"] == "he+"
