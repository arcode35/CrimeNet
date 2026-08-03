#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(pwd)"

if [[ "$(basename "$PROJECT_ROOT")" != "crimenet_ml" ]]; then
  echo "Run this script from the crimenet_ml project directory."
  exit 1
fi

write_if_missing() {
  local path="$1"

  if [[ -e "$path" ]]; then
    echo "Keeping existing file: $path"
    cat >/dev/null
    return
  fi

  mkdir -p "$(dirname "$path")"
  cat >"$path"
  echo "Created: $path"
}

echo "Scaffolding project in: $PROJECT_ROOT"

mkdir -p \
  configs/experiments \
  src/crimenet_ml/data \
  src/crimenet_ml/features \
  src/crimenet_ml/training \
  src/crimenet_ml/evaluation \
  src/crimenet_ml/experiments \
  tests \
  notebooks \
  scripts \
  artifacts \
  models \
  reports/figures

touch \
  src/crimenet_ml/__init__.py \
  src/crimenet_ml/data/__init__.py \
  src/crimenet_ml/features/__init__.py \
  src/crimenet_ml/training/__init__.py \
  src/crimenet_ml/evaluation/__init__.py \
  src/crimenet_ml/experiments/__init__.py \
  artifacts/.gitkeep \
  models/.gitkeep \
  reports/figures/.gitkeep

write_if_missing ".gitignore" <<'EOF'
# Python
__pycache__/
*.py[cod]
*.so
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/

# Environments
venv/
.venv/

# IDE and operating system
.vscode/
.idea/
.DS_Store
Thumbs.db

# Secrets
.env
.env.*
!.env.example

# Generated artifacts
artifacts/*
!artifacts/.gitkeep

models/*
!models/.gitkeep

reports/figures/*
!reports/figures/.gitkeep

mlruns/
wandb/

# Local datasets
crimenet_datasets/
data/

# Notebooks
.ipynb_checkpoints/
EOF

write_if_missing ".env.example" <<'EOF'
CRIMENET_DATA_DIR=crimenet_datasets
CRIMENET_ARTIFACT_DIR=artifacts
CRIMENET_MODEL_DIR=models
EOF

write_if_missing "pyproject.toml" <<'EOF'
[build-system]
requires = ["setuptools>=75", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "crimenet-ml"
version = "0.1.0"
description = "CrimeNet machine-learning training and experimentation"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "numpy>=2.0",
    "polars>=1.30",
    "pyarrow>=18",
    "pandas>=2.2",
    "scikit-learn>=1.6",
    "xgboost>=3.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.9",
    "mypy>=1.14",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]

[tool.mypy]
python_version = "3.11"
strict = true
packages = ["crimenet_ml"]
EOF

write_if_missing "README.md" <<'EOF'
# CrimeNet ML

Local model-training and experimentation repository for CrimeNet.

## Project layout

```text
crimenet_ml/
├── crimenet_datasets/
├── configs/
│   └── experiments/
├── src/
│   └── crimenet_ml/
│       ├── data/
│       ├── evaluation/
│       ├── experiments/
│       ├── features/
│       └── training/
├── tests/
├── notebooks/
├── scripts/
├── artifacts/
├── models/
└── reports/