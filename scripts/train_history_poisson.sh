#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.." &&
  pwd
)"

cd "$PROJECT_ROOT"

python -m crimenet_ml.training.train_xgboost_poisson \
  --config "$PROJECT_ROOT/configs/xgb_history_poisson_v1.yaml" \
  "$@"