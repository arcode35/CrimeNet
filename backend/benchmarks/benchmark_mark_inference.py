from __future__ import annotations

import json
import sys
from pathlib import Path

# Keep direct script execution working with package imports.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.inference.mark_runtime import MarkRuntime


if __name__ == "__main__":
    # Standalone diagnostics explicitly opt into benchmarking. Normal API
    # startup defaults to deterministic CPU serving and never reaches this path.
    runtime = MarkRuntime(inference_mode="auto")
    print(json.dumps(runtime.inference_benchmark, indent=2, sort_keys=True))
