#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_crimenet_sentinel_vast.sh preprocess   # CPU/RAM/NVMe host -> B2 prepared cache
#   ./run_crimenet_sentinel_vast.sh gpu          # GPU host <- B2 prepared cache -> Gold staging
#   ./run_crimenet_sentinel_vast.sh all          # legacy single-host end-to-end path
#
# Prerequisite: an rclone remote named `b2` that can read CrimeNet Bronze/Gold and
# write the temporary prepared-cache + Gold staging prefixes.

MODE="${1:-all}"
WORK_DIR="${WORK_DIR:-/workspace/crimenet-sentinel-vast}"
SCRIPT="${SCRIPT:-./crimenet_sentinel_vast.py}"
PREPROCESSED_REMOTE="${PREPROCESSED_REMOTE:-b2:crimenet-data/tmp/sentinel_prepared_v1}"
PUBLISH_REMOTE="${PUBLISH_REMOTE:-b2:crimenet-data/gold/imagery/embeddings/foundation_v1_b2_staging/sentinel2}"

ulimit -n 131072 2>/dev/null || true

COMMON=(
  --work-dir "$WORK_DIR"
  --b2-source-remote b2:crimenet-data/bronze/imagery/sentinel2/national
  --existing-gold-remote b2:crimenet-data/gold/imagery/embeddings/foundation_v1/sentinel2
  --preprocessed-remote "$PREPROCESSED_REMOTE"
  --publish-remote "$PUBLISH_REMOTE"
  --rclone-multi-thread-streams 8
  --frame-buckets 64
  --strict
  --strict-context
  --resume
)

case "$MODE" in
  preprocess)
    exec python "$SCRIPT" \
      --phase preprocess \
      "${COMMON[@]}" \
      --rclone-transfers 96 \
      --rclone-checkers 192 \
      --candidate-workers "${CANDIDATE_WORKERS:-160}" \
      --context-download-workers "${CONTEXT_DOWNLOAD_WORKERS:-96}" \
      --frame-workers "${FRAME_WORKERS:-192}" \
      --frame-scene-cache-per-thread 4 \
      --frames-per-shard 384 \
      --cache-transfers "${CACHE_TRANSFERS:-64}" \
      --cache-checkers "${CACHE_CHECKERS:-128}" \
      --verify-cache-transfer \
      "${EXTRA_ARGS[@]:-}"
    ;;
  gpu)
    exec python "$SCRIPT" \
      --phase gpu \
      "${COMMON[@]}" \
      --rclone-transfers 32 \
      --rclone-checkers 64 \
      --cache-transfers "${CACHE_TRANSFERS:-64}" \
      --cache-checkers "${CACHE_CHECKERS:-128}" \
      --verify-cache-transfer \
      --gpus "${GPUS:-0}" \
      --precision "${PRECISION:-bf16}" \
      --gpu-batch-size "${GPU_BATCH_SIZE:-128}" \
      --gpu-prep-threads "${GPU_PREP_THREADS:-12}" \
      --writer-threads-per-gpu "${WRITER_THREADS_PER_GPU:-2}" \
      --rows-per-output-shard "${ROWS_PER_OUTPUT_SHARD:-25000}" \
      --publish-transfers "${PUBLISH_TRANSFERS:-32}" \
      --publish-checkers "${PUBLISH_CHECKERS:-64}" \
      --verify-publish \
      "${EXTRA_ARGS[@]:-}"
    ;;
  all)
    exec python "$SCRIPT" \
      --phase all \
      "${COMMON[@]}" \
      --rclone-transfers 96 \
      --rclone-checkers 192 \
      --candidate-workers "${CANDIDATE_WORKERS:-160}" \
      --context-download-workers "${CONTEXT_DOWNLOAD_WORKERS:-96}" \
      --frame-workers "${FRAME_WORKERS:-192}" \
      --frame-scene-cache-per-thread 4 \
      --frames-per-shard 384 \
      --gpus "${GPUS:-0}" \
      --precision "${PRECISION:-bf16}" \
      --gpu-batch-size "${GPU_BATCH_SIZE:-128}" \
      --gpu-prep-threads "${GPU_PREP_THREADS:-12}" \
      --writer-threads-per-gpu "${WRITER_THREADS_PER_GPU:-2}" \
      --rows-per-output-shard "${ROWS_PER_OUTPUT_SHARD:-25000}" \
      --publish-transfers "${PUBLISH_TRANSFERS:-32}" \
      --publish-checkers "${PUBLISH_CHECKERS:-64}" \
      --verify-publish \
      "${EXTRA_ARGS[@]:-}"
    ;;
  *)
    echo "usage: $0 {preprocess|gpu|all}" >&2
    exit 2
    ;;
esac
