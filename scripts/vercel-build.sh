#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv /tmp/atomic-venv
/tmp/atomic-venv/bin/pip install --quiet build
/tmp/atomic-venv/bin/pip install --quiet -e .
/tmp/atomic-venv/bin/python scripts/stage_engine_wheel.py
cd web && npm run build
