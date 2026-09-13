# atomic — two-stage build
#   stage 1 (web):   Node toolchain compiles web/ into static assets
#   stage 2 (serve): slim Python runtime ships only the package + assets
# The UI is baked in at ATOMIC_WEB_DIST, so the container serves the whole
# instrument from one process — no separate static host needed.

# ---- stage 1: web build -----------------------------------------------------
FROM node:22-bookworm-slim AS web

WORKDIR /build/web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY web/ ./
RUN npm run build


# ---- stage 2: python runtime ------------------------------------------------
FROM python:3.12-slim-bookworm AS serve

# Run as a non-root user; uvicorn and the job threads get no more than they need.
RUN useradd --create-home --shell /usr/sbin/nologin atomic

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

# Drop the compiled UI into place and point ATOMIC_WEB_DIST at it.
COPY --from=web /build/web/dist /app/web-dist
ENV ATOMIC_WEB_DIST=/app/web-dist

# Serving behind Fly's proxy: trust its client-IP header for rate limiting.
ENV ATOMIC_HOST=0.0.0.0 \
    ATOMIC_PORT=8080 \
    ATOMIC_CLIENT_IP_HEADER=fly-client-ip

USER atomic
EXPOSE 8080

# ASGI consensus port + healthcheck against the vendored evidence suite.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.environ['ATOMIC_PORT'], timeout=2)" || exit 1

CMD ["atomic", "serve", "--no-browser"]
