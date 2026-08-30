from __future__ import annotations

import json

from mark_runtime import MarkRuntime


if __name__ == "__main__":
    runtime = MarkRuntime()
    print(json.dumps(runtime.inference_benchmark, indent=2, sort_keys=True))
