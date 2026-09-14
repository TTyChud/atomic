# Setup

Python server that computes the physics and serves the web UI. No database, no external services.

Prerequisites: Python ≥ 3.11, Node ≥ 22.

## Run

```bash
make setup    # venv + pip install + npm ci
make serve    # builds the UI, serves on 127.0.0.1:8000
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate atomic
cd web && npm ci && npm run build && cd ..
atomic serve
```

## The on-device engine

On localhost, the physics runs in the browser: a Web Worker boots the real Python engine (same `atomic` package, same FastAPI app) via Pyodide. The runtime is not committed (~42 MB). Stage it once:

```bash
make engine-stage
```

Writes `web/public/atomic-engine/` (gitignored). Without it, localhost falls back to the network API. `?engine=server` / `?engine=device` force a transport.

## Server config

| Flag | Env var | Default | Meaning |
|---|---|---|---|
| `--host` | `ATOMIC_HOST` | `127.0.0.1` | Bind interface |
| `--port` | `ATOMIC_PORT` | `8000` | Listen port |
| `--no-browser` | — | off | Don't open a browser |

Other env vars: `ATOMIC_WEB_DIST`, `ATOMIC_JOB_WORKERS`, `ATOMIC_RATE_LIMIT*`, `ATOMIC_CLIENT_IP_HEADER`, `ATOMIC_ALLOWED_ORIGINS` (CORS for split deploys).

## Deploy (Vercel)

`vercel.json` at the repo root handles it: import the repo on Vercel, nothing else to configure. The build stages the engine runtime and serves the UI statically; visitors run the physics on their own device.

To use a remote engine instead, run `atomic serve` somewhere public, then `?engine=server&api=<url>` and set `ATOMIC_ALLOWED_ORIGINS` on the server to your Vercel domain.

## Verify

```bash
make test    # pytest + ruff + vitest
make build   # full CI gate
```
