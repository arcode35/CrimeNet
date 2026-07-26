#!/usr/bin/env bash

set -euo pipefail

uv run ruff check src tests
uv run pytest
databricks bundle validate --target dev
