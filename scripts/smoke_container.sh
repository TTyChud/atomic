#!/usr/bin/env bash
# Build the container image, boot it, and verify the served instrument end to
# end: /api/health warm, an API route answering, and the compiled UI mounted.
# Usage: bash scripts/smoke_container.sh [image-tag]   (default: atomic:ci)
set -euo pipefail

IMAGE="${1:-atomic:ci}"
PORT="${ATOMIC_SMOKE_PORT:-8123}"
NAME="atomic-smoke"
BASE="http://127.0.0.1:${PORT}"
TIMEOUT=60

say() { printf 'smoke: %s\n' "$*"; }
die() { printf 'smoke: FAIL: %s\n' "$*" >&2; exit 1; }
cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

command -v docker >/dev/null 2>&1 || die "docker is not installed"

say "building ${IMAGE}"
docker build -t "$IMAGE" .

cleanup
say "booting container on port ${PORT}"
docker run -d --name "$NAME" -p "127.0.0.1:${PORT}:8080" "$IMAGE" >/dev/null

say "waiting for /api/health (up to ${TIMEOUT}s)"
healthy=""
for _ in $(seq 1 "$TIMEOUT"); do
  if curl -fsS "${BASE}/api/health" >/dev/null 2>&1; then
    healthy=1
    break
  fi
  sleep 1
done
[ -n "$healthy" ] || { docker logs "$NAME" >&2 || true; die "/api/health never went warm"; }

HEALTH=$(curl -fsS "${BASE}/api/health")
say "health: ${HEALTH}"
printf '%s' "$HEALTH" | grep -q '"status"' || die "health payload missing status"

say "checking an API route (GET /api/systems)"
curl -fsS "${BASE}/api/systems" | grep -q '"h"' || die "/api/systems did not answer"

say "checking the compiled UI is mounted"
curl -fsS "${BASE}/" | grep -qi '<div id="root">' || die "UI not mounted at /"

say "OK — container serves API + UI"
