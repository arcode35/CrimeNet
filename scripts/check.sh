#!/usr/bin/env bash

set -euo pipefail

uv lock --check
uv export --locked --no-dev --no-emit-project --no-hashes --format requirements-txt | diff - requirements-prod.txt
uv sync --locked --all-groups
uv run ruff check .
uv run mypy src
uv run pytest --cov=crimenet --cov-report=term-missing --cov-fail-under=80
uv build
databricks bundle validate --target dev
