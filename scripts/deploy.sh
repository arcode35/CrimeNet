#!/usr/bin/env bash

set -euo pipefail

TARGET="${1:-dev}"

uv run ruff check src tests
uv run pytest
databricks bundle validate --target "${TARGET}"
databricks bundle deploy --target "${TARGET}"
