# Developer commands for paperfacts. `make help` lists all targets.
# The CI workflow (.github/workflows/ci.yml) runs the same gates as `make check`.

LINT_PATHS := src tests runners eval

.PHONY: help test test-cov lint format format-check check e2e deploy

help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "} {printf "  %-14s %s\n", $$1, $$2}'

test: ## Run the pytest suite (no models, no network)
	uv run --locked pytest

test-cov: ## Run the pytest suite with coverage
	uv run --locked pytest --cov=paperfacts

lint: ## Ruff lint check
	uv run --locked ruff check $(LINT_PATHS)

format: ## Rewrite files with ruff format
	uv run --locked ruff format $(LINT_PATHS)

format-check: ## Verify formatting without rewriting (CI gate)
	uv run --locked ruff format --check $(LINT_PATHS)

check: lint format-check test ## Run all CI gates locally

e2e: ## Browser e2e suite (headless Chromium; installs the browser on first run)
	uv run --with playwright python -m playwright install chromium
	PYTHONPATH=src uv run --with playwright pytest -m e2e

deploy: ## Deploy the workstation service; extra flags: make deploy ARGS="--check"
	scripts/deploy.sh $(ARGS)
