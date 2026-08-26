#!/usr/bin/env bash
set -euo pipefail

# Aggressive defaults aimed at ~8 high-end NVIDIA GPUs, ~256-384 CPU cores,
# hundreds of GB of RAM, and >= ~2.5-3 TiB fast local NVMe.
#
# Prerequisite: configure an rclone remote named `b2` for the CrimeNet B2 bucket.
# The script never deletes the REMOTE bronze imagery.

WORK_DIR="${WORK_DIR:-/workspace/crimenet-sentinel-vast}"
SCRIPT="${SCRIPT:-./crimenet_sentinel_vast.py}"

ulimit -n 131072 2>/dev/null || true

python "$SCRIPT" \
  --phase all \
  --work-dir "$WORK_DIR" \
  --b2-source-remote b2:crimenet-data/bronze/imagery/sentinel2/national \
  --existing-gold-remote b2:crimenet-data/gold/imagery/embeddings/foundation_v1/sentinel2 \
  --publish-remote b2:crimenet-data/gold/imagery/embeddings/foundation_v1_b2_staging/sentinel2 \
  --rclone-transfers 96 \
  --rclone-checkers 192 \
  --rclone-multi-thread-streams 8 \
  --candidate-workers 160 \
  --context-download-workers 96 \
  --frame-buckets 64 \
  --frame-workers 192 \
  --frame-scene-cache-per-thread 4 \
  --frames-per-shard 384 \
  --gpus 0 \
  --precision bf16 \
  --gpu-batch-size 128 \
  --gpu-prep-threads 12 \
  --writer-threads-per-gpu 2 \
  --rows-per-output-shard 25000 \
  --strict \
  --strict-context \
  --resume \
  --verify-publish
