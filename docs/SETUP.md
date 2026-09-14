# Setting up atomic locally

Everything runs on one machine: a Python server that computes the physics and
serves the compiled web UI. There is no database, no external service, no
network calls — the NIST reference data is vendored in `src/atomic/data/`.

## Prerequisites

- Python ≥ 3.11
- Node ≥ 22 (for the web UI)

## Option A: venv

```bash
make setup          # venv + pip install -e ".[dev]" + npm ci
make serve          # builds web/dist, then atomic serve on 127.0.0.1:8000
```

The browser opens on its own. `make stop` kills a running instance.

Manual equivalent:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cd web && npm ci && npm run build && cd ..
.venv/bin/atomic serve
```

## Option B: conda

```bash
conda env create -f environment.yml
conda activate atomic
cd web && npm ci && npm run build && cd ..
atomic serve
```

## The on-device engine

Locally (localhost), the physics runs **in your browser**: a Web Worker boots
the real Python engine — the same `atomic` package, same FastAPI app, same
schemas the server runs — via Pyodide/WebAssembly. No server participates in
a computation; there is nothing to store and nothing to share between
visitors.

The runtime is not committed (it is ~42 MB). Stage it once per checkout:

```bash
make engine-stage      # builds the atomic wheel + vendors the Pyodide runtime
```

That writes `web/public/atomic-engine/` (gitignored). Then use the app as
usual — `make serve` or `make web-dev` — and the top bar shows
`device engine` with a tooltip saying where the physics is running.

- Without staging, localhost transparently falls back to the network API, so
  a fresh clone works immediately against `atomic serve`.
- `?engine=server` forces the network transport; `?engine=device` forces the
  in-browser runtime.
- First boot downloads ~42 MB (cached by the browser afterwards); the
  production UI chunk itself is unchanged and still ~124 kB gzipped.

## Deploying the UI to Vercel

`vercel.json` at the repo root builds the UI and the staged engine runtime on
Vercel's side, so a deployed visitor gets the same in-browser engine without
any server of ours. Import the repo at vercel.com (framework preset: Other),
and the config does the rest:

- `installCommand`: `npm ci` inside `web/`
- `buildCommand`: builds the `atomic` wheel, stages
  `web/public/atomic-engine/` (`stage_engine_wheel.py` fetches the Pyodide
  runtime + package wheels from the jsDelivr CDN at build time), then `vite
  build` → `web/dist`
- `/atomic-engine/*` and `/assets/*` serve immutable cache headers; the SPA
  rewrite keeps deep links working

The deployed site runs physics entirely on the visitor's device. If you also
run `atomic serve` somewhere public (Fly, a VPS, …), point a split deploy at
it with `?engine=server&api=<url>` and set `ATOMIC_ALLOWED_ORIGINS` on the
server to your Vercel domain.

## Useful flags and environment

| Flag | Env var | Default | Meaning |
|---|---|---|---|
| `--host` | `ATOMIC_HOST` | `127.0.0.1` | Bind interface. `0.0.0.0` exposes the server to your network — containers and remote deploys need it. |
| `--port` | `ATOMIC_PORT` | `8000` | Listen port. |
| `--no-browser` | — | off | Don't auto-open a browser. Implied whenever `--host` is not loopback. |

Other server knobs (see `src/atomic/server/`): `ATOMIC_WEB_DIST` (where the
compiled UI lives), `ATOMIC_JOB_WORKERS` (background job threads),
`ATOMIC_RATE_LIMIT*` (token bucket for job creation),
`ATOMIC_CLIENT_IP_HEADER` (client-IP header when behind a proxy).

## Verify

```bash
curl http://127.0.0.1:8000/api/health   # {"status":"ok",...}
make test                               # pytest + ruff + vitest
make build                              # the full gate CI runs
```
