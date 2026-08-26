#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/run_crimenet_sentinel_vast.sh" preprocess
