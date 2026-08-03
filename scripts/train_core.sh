#!/usr/bin/env bash
set -euo pipefail

python -m crimenet_ml.training.train_xgboost   --config configs/xgb_core_v1.yaml   "$@"
