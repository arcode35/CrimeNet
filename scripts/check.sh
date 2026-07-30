#!/usr/bin/env bash

set -euo pipefail

TARGET="${1:-dev}"

uv run ruff check src tests
uv run pytest tests/unit tests/integration -m "not databricks"
databricks bundle validate --target "${TARGET}"
