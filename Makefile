.PHONY: help setup serve stop web-build web-dev test test-py test-web lint build engine-stage clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Create venv and install backend + web deps
	python3 -m venv .venv
	.venv/bin/pip install -e ".[dev]"
	cd web && npm ci

serve: web-build stop ## Build the web UI, kill any old instance, start the server
	.venv/bin/atomic serve

stop: ## Kill any running atomic serve instance
	pkill -f '[a]tomic serve' || true

web-build: ## Production build of the web UI (required before serve)
	cd web && npm run build

engine-stage: ## Stage the Pyodide runtime + atomic wheel for the in-browser engine
	python3 scripts/stage_engine_wheel.py

web-dev: ## Vite dev server with hot reload
	cd web && npm run dev

test: test-py test-web ## Run all tests
test-py: ## Backend: pytest + ruff
	.venv/bin/pytest
	.venv/bin/ruff check .
test-web: ## Frontend: vitest
	cd web && npm test

lint: ## Ruff only
	.venv/bin/ruff check .

build: test-py test-web web-build ## Full gate: backend tests, web tests, web build

clean: ## Remove venv, web build, caches
	rm -rf .venv web/dist
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache \) -exec rm -rf {} +
