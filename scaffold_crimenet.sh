#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${1:-crimenet}"
PACKAGE_NAME="crimenet"

if [[ -e "${PROJECT_DIR}" ]]; then
    echo "Error: '${PROJECT_DIR}' already exists." >&2
    echo "Choose another directory or remove the existing one." >&2
    exit 1
fi

echo "Creating CrimeNet project in: ${PROJECT_DIR}"

mkdir -p \
    "${PROJECT_DIR}/targets" \
    "${PROJECT_DIR}/resources/jobs" \
    "${PROJECT_DIR}/src/${PACKAGE_NAME}/jobs" \
    "${PROJECT_DIR}/src/${PACKAGE_NAME}/ingestion" \
    "${PROJECT_DIR}/src/${PACKAGE_NAME}/transforms" \
    "${PROJECT_DIR}/src/${PACKAGE_NAME}/contracts" \
    "${PROJECT_DIR}/src/${PACKAGE_NAME}/quality" \
    "${PROJECT_DIR}/src/${PACKAGE_NAME}/config" \
    "${PROJECT_DIR}/src/${PACKAGE_NAME}/utils" \
    "${PROJECT_DIR}/tests/unit" \
    "${PROJECT_DIR}/tests/integration" \
    "${PROJECT_DIR}/tests/fixtures/dallas" \
    "${PROJECT_DIR}/tests/fixtures/houston" \
    "${PROJECT_DIR}/tests/fixtures/fort_worth" \
    "${PROJECT_DIR}/notebooks/exploration" \
    "${PROJECT_DIR}/docs" \
    "${PROJECT_DIR}/scripts"

# ---------------------------------------------------------------------------
# Root bundle configuration
# ---------------------------------------------------------------------------

cat > "${PROJECT_DIR}/databricks.yml" <<'YAML'
bundle:
  name: crimenet
  databricks_cli_version: ">=0.218.0"

include:
  - resources/jobs/*.yml
  - targets/*.yml

variables:
  catalog:
    description: Unity Catalog catalog for the deployment target

  bronze_schema:
    description: Bronze table schema
    default: bronze

  silver_schema:
    description: Silver table schema
    default: silver

  gold_schema:
    description: Gold table schema
    default: gold

  ops_schema:
    description: Operational metadata schema
    default: ops

  data_quality_schema:
    description: Data-quality and quarantine schema
    default: data_quality

  raw_files_schema:
    description: Schema containing raw-file volumes
    default: raw_files

  landing_volume:
    description: Volume containing downloaded source files
    default: landing

artifacts:
  default:
    type: whl
    path: .
    build: uv build

sync:
  exclude:
    - .venv/**
    - tests/**
    - docs/**
    - notebooks/**
    - dist/**
    - build/**
YAML

# ---------------------------------------------------------------------------
# Deployment targets
# ---------------------------------------------------------------------------

cat > "${PROJECT_DIR}/targets/dev.yml" <<'YAML'
targets:
  dev:
    default: true
    mode: development

    workspace:
      root_path: /Workspace/Users/${workspace.current_user.userName}/.bundle/${bundle.name}/${bundle.target}

    variables:
      catalog: crimenet_dev
YAML

cat > "${PROJECT_DIR}/targets/prod.yml" <<'YAML'
targets:
  prod:
    mode: production

    git:
      branch: main

    workspace:
      root_path: /Workspace/Shared/.bundle/${bundle.name}/${bundle.target}

    variables:
      catalog: crimenet_prod

    # Replace this with a service principal before production deployment.
    run_as:
      user_name: ${workspace.current_user.userName}

    permissions:
      - user_name: ${workspace.current_user.userName}
        level: CAN_MANAGE
YAML

# ---------------------------------------------------------------------------
# Lakeflow Job resource
# ---------------------------------------------------------------------------

cat > "${PROJECT_DIR}/resources/jobs/crime_pipeline.job.yml" <<'YAML'
resources:
  jobs:
    crime_pipeline:
      name: CrimeNet crime-data pipeline
      description: >
        Ingests Dallas, Houston, and Fort Worth crime data into Bronze,
        standardizes the sources into Silver, validates quality, and refreshes
        Gold tables.

      max_concurrent_runs: 1

      tasks:
        - task_key: bronze_dallas
          environment_key: default

          python_wheel_task:
            package_name: crimenet
            entry_point: bronze_ingestion
            parameters:
              - --catalog
              - ${var.catalog}
              - --bronze-schema
              - ${var.bronze_schema}
              - --city
              - dallas
              - --input-path
              - /Volumes/${var.catalog}/${var.raw_files_schema}/${var.landing_volume}/dallas

          max_retries: 2
          min_retry_interval_millis: 60000

        - task_key: bronze_houston
          environment_key: default

          python_wheel_task:
            package_name: crimenet
            entry_point: bronze_ingestion
            parameters:
              - --catalog
              - ${var.catalog}
              - --bronze-schema
              - ${var.bronze_schema}
              - --city
              - houston
              - --input-path
              - /Volumes/${var.catalog}/${var.raw_files_schema}/${var.landing_volume}/houston

          max_retries: 2
          min_retry_interval_millis: 60000

        - task_key: bronze_fort_worth
          environment_key: default

          python_wheel_task:
            package_name: crimenet
            entry_point: bronze_ingestion
            parameters:
              - --catalog
              - ${var.catalog}
              - --bronze-schema
              - ${var.bronze_schema}
              - --city
              - fort_worth
              - --input-path
              - /Volumes/${var.catalog}/${var.raw_files_schema}/${var.landing_volume}/fort_worth

          max_retries: 2
          min_retry_interval_millis: 60000

        - task_key: silver_transform
          environment_key: default

          depends_on:
            - task_key: bronze_dallas
            - task_key: bronze_houston
            - task_key: bronze_fort_worth

          python_wheel_task:
            package_name: crimenet
            entry_point: silver_transform
            parameters:
              - --catalog
              - ${var.catalog}
              - --bronze-schema
              - ${var.bronze_schema}
              - --silver-schema
              - ${var.silver_schema}
              - --data-quality-schema
              - ${var.data_quality_schema}

          max_retries: 1
          min_retry_interval_millis: 60000

        - task_key: quality_checks
          environment_key: default

          depends_on:
            - task_key: silver_transform

          python_wheel_task:
            package_name: crimenet
            entry_point: quality_checks
            parameters:
              - --catalog
              - ${var.catalog}
              - --silver-schema
              - ${var.silver_schema}
              - --data-quality-schema
              - ${var.data_quality_schema}

        - task_key: gold_refresh
          environment_key: default

          depends_on:
            - task_key: quality_checks

          python_wheel_task:
            package_name: crimenet
            entry_point: gold_refresh
            parameters:
              - --catalog
              - ${var.catalog}
              - --silver-schema
              - ${var.silver_schema}
              - --gold-schema
              - ${var.gold_schema}

      environments:
        - environment_key: default
          spec:
            environment_version: "4"
            dependencies:
              - ../../dist/*.whl
YAML

# ---------------------------------------------------------------------------
# Python package configuration
# ---------------------------------------------------------------------------

cat > "${PROJECT_DIR}/pyproject.toml" <<'TOML'
[project]
name = "crimenet"
version = "0.1.0"
description = "Multi-city crime data lakehouse built on Databricks and Delta Lake"
requires-python = ">=3.11"
dependencies = []

[project.scripts]
bronze_ingestion = "crimenet.jobs.bronze_ingestion:main"
silver_transform = "crimenet.jobs.silver_transform:main"
quality_checks = "crimenet.jobs.quality_checks:main"
gold_refresh = "crimenet.jobs.gold_refresh:main"

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.11",
    "mypy>=1.15",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/crimenet"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"

[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = [
    "E",
    "F",
    "I",
    "B",
    "UP",
    "SIM",
]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
TOML

cat > "${PROJECT_DIR}/.python-version" <<'EOF'
3.11
EOF

cat > "${PROJECT_DIR}/.gitignore" <<'EOF'
# Databricks
.databricks/

# Python
.venv/
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/

# Testing and linting
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/

# Editors
.idea/
.vscode/

# Operating systems
.DS_Store

# Local environment files
.env
.env.*
!.env.example
EOF

# ---------------------------------------------------------------------------
# Python package modules
# ---------------------------------------------------------------------------

find "${PROJECT_DIR}/src/${PACKAGE_NAME}" \
    -type d \
    -exec touch "{}/__init__.py" \;

cat > "${PROJECT_DIR}/src/${PACKAGE_NAME}/jobs/bronze_ingestion.py" <<'PYTHON'
"""Databricks entry point for source-specific Bronze ingestion."""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--bronze-schema", required=True)
    parser.add_argument(
        "--city",
        required=True,
        choices=("dallas", "houston", "fort_worth"),
    )
    parser.add_argument("--input-path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raise NotImplementedError(
        "Implement Bronze ingestion for "
        f"{args.city!r} from {args.input_path!r}."
    )


if __name__ == "__main__":
    main()
PYTHON

cat > "${PROJECT_DIR}/src/${PACKAGE_NAME}/jobs/silver_transform.py" <<'PYTHON'
"""Databricks entry point for the unified Silver transformation."""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--bronze-schema", required=True)
    parser.add_argument("--silver-schema", required=True)
    parser.add_argument("--data-quality-schema", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raise NotImplementedError(
        "Implement the Dallas, Houston, and Fort Worth canonical "
        f"transformations for catalog {args.catalog!r}."
    )


if __name__ == "__main__":
    main()
PYTHON

cat > "${PROJECT_DIR}/src/${PACKAGE_NAME}/jobs/quality_checks.py" <<'PYTHON'
"""Databricks entry point for Silver data-quality validation."""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--silver-schema", required=True)
    parser.add_argument("--data-quality-schema", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raise NotImplementedError(
        f"Implement Silver quality checks for catalog {args.catalog!r}."
    )


if __name__ == "__main__":
    main()
PYTHON

cat > "${PROJECT_DIR}/src/${PACKAGE_NAME}/jobs/gold_refresh.py" <<'PYTHON'
"""Databricks entry point for Gold-table refreshes."""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--silver-schema", required=True)
    parser.add_argument("--gold-schema", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raise NotImplementedError(
        f"Implement Gold refreshes for catalog {args.catalog!r}."
    )


if __name__ == "__main__":
    main()
PYTHON

# Create implementation modules.
touch \
    "${PROJECT_DIR}/src/${PACKAGE_NAME}/ingestion/readers.py" \
    "${PROJECT_DIR}/src/${PACKAGE_NAME}/ingestion/metadata.py" \
    "${PROJECT_DIR}/src/${PACKAGE_NAME}/ingestion/column_names.py" \
    "${PROJECT_DIR}/src/${PACKAGE_NAME}/transforms/common.py" \
    "${PROJECT_DIR}/src/${PACKAGE_NAME}/transforms/dallas.py" \
    "${PROJECT_DIR}/src/${PACKAGE_NAME}/transforms/houston.py" \
    "${PROJECT_DIR}/src/${PACKAGE_NAME}/transforms/fort_worth.py" \
    "${PROJECT_DIR}/src/${PACKAGE_NAME}/transforms/canonical.py" \
    "${PROJECT_DIR}/src/${PACKAGE_NAME}/contracts/bronze.py" \
    "${PROJECT_DIR}/src/${PACKAGE_NAME}/contracts/silver.py" \
    "${PROJECT_DIR}/src/${PACKAGE_NAME}/contracts/gold.py" \
    "${PROJECT_DIR}/src/${PACKAGE_NAME}/quality/rules.py" \
    "${PROJECT_DIR}/src/${PACKAGE_NAME}/quality/quarantine.py" \
    "${PROJECT_DIR}/src/${PACKAGE_NAME}/config/resources.py" \
    "${PROJECT_DIR}/src/${PACKAGE_NAME}/config/validation.py" \
    "${PROJECT_DIR}/src/${PACKAGE_NAME}/utils/logging.py" \
    "${PROJECT_DIR}/src/${PACKAGE_NAME}/utils/spark.py"

# ---------------------------------------------------------------------------
# Tests and fixtures
# ---------------------------------------------------------------------------

cat > "${PROJECT_DIR}/tests/unit/test_project_import.py" <<'PYTHON'
def test_package_import() -> None:
    import crimenet

    assert crimenet is not None
PYTHON

touch \
    "${PROJECT_DIR}/tests/integration/.gitkeep" \
    "${PROJECT_DIR}/tests/fixtures/dallas/.gitkeep" \
    "${PROJECT_DIR}/tests/fixtures/houston/.gitkeep" \
    "${PROJECT_DIR}/tests/fixtures/fort_worth/.gitkeep" \
    "${PROJECT_DIR}/notebooks/exploration/.gitkeep"

# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------

cat > "${PROJECT_DIR}/docs/architecture.md" <<'EOF'
# Architecture

Document the CrimeNet ingestion, Bronze, Silver, Gold, orchestration, storage,
and data-serving architecture.
EOF

cat > "${PROJECT_DIR}/docs/data_contracts.md" <<'EOF'
# Data Contracts

Document the Dallas, Houston, Fort Worth, and canonical Silver contracts,
including row grain, identifiers, timestamps, coordinates, and nullability.
EOF

cat > "${PROJECT_DIR}/docs/data_quality.md" <<'EOF'
# Data Quality

Document validation rules, quarantine behavior, severity levels, thresholds,
and operational ownership.
EOF

cat > "${PROJECT_DIR}/docs/operations_runbook.md" <<'EOF'
# Operations Runbook

Document deployments, backfills, retries, incident handling, recovery,
monitoring, and rollback procedures.
EOF

# ---------------------------------------------------------------------------
# Developer scripts
# ---------------------------------------------------------------------------

cat > "${PROJECT_DIR}/scripts/check.sh" <<'BASH'
#!/usr/bin/env bash

set -euo pipefail

uv run ruff check src tests
uv run pytest
databricks bundle validate --target dev
BASH

cat > "${PROJECT_DIR}/scripts/deploy.sh" <<'BASH'
#!/usr/bin/env bash

set -euo pipefail

TARGET="${1:-dev}"

uv run ruff check src tests
uv run pytest
databricks bundle validate --target "${TARGET}"
databricks bundle deploy --target "${TARGET}"
BASH

chmod +x \
    "${PROJECT_DIR}/scripts/check.sh" \
    "${PROJECT_DIR}/scripts/deploy.sh"

# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------

cat > "${PROJECT_DIR}/README.md" <<'EOF'
# CrimeNet

CrimeNet is a multi-city crime-data lakehouse implemented with Databricks,
Delta Lake, Unity Catalog, PySpark, and Declarative Automation Bundles.

## Sources

- Dallas
- Houston
- Fort Worth

## Setup

```bash
uv sync
databricks auth login
databricks bundle validate --target dev
databricks bundle deploy --target dev
databricks bundle run --target dev crime_pipeline