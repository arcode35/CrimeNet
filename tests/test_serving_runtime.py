from __future__ import annotations

import sys
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest


BACKEND_ROOT = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from mark_runtime import MarkRuntime  # noqa: E402
from viewport_response import build_viewport_rows  # noqa: E402


def test_vectorized_viewport_rows_match_reference_values() -> None:
    cells = ["a", "b", "c", "d"]
    positions = np.asarray([0, 2, 3], dtype=np.int64)
    intensity = np.asarray([0.125, 0.000_123, 1.5], dtype=np.float32)
    children = np.asarray([1, 17, 0], dtype=np.uint32)

    expected = []
    for index, position in enumerate(positions):
        hourly = float(intensity[index]) * 3600.0
        child_count = int(children[index])
        expected.append(
            {
                "h3": cells[int(position)],
                "events_per_hour": hourly,
                "mean_r9_events_per_hour": hourly / child_count if child_count else 0.0,
                "modeled_r9_cells": child_count,
            }
        )

    assert build_viewport_rows(cells, positions, intensity, children) == expected


def _runtime_without_artifacts() -> MarkRuntime:
    runtime = MarkRuntime.__new__(MarkRuntime)
    runtime.num_classes = 3
    runtime.class_labels = ["a", "b", "c"]
    runtime.labels_available = True
    runtime.inference_device = "cpu"
    runtime.snapshot_id = "snapshot-a"
    runtime.state_lock = threading.RLock()
    runtime.cache_lock = threading.Lock()
    runtime.cache = OrderedDict()
    runtime.inflight_lock = threading.Lock()
    runtime.inflight = {}
    return runtime


def test_mark_duplicate_misses_are_coalesced_and_repeat_hits_cache(monkeypatch) -> None:
    runtime = _runtime_without_artifacts()
    inference_started = threading.Event()
    release_inference = threading.Event()
    initial_cache_lookups = threading.Barrier(2)
    calls = 0

    original_cache_get = runtime._cache_get

    def synchronized_cache_get(key):
        if not runtime.inflight and original_cache_get(key) is None:
            initial_cache_lookups.wait(timeout=1)
        return original_cache_get(key)

    def build_feature_row_for_snapshot(**_kwargs):
        return 0, np.zeros((1, 38), dtype=np.float32)

    def predict_margins(_features):
        nonlocal calls
        calls += 1
        inference_started.set()
        release_inference.wait(timeout=1)
        return np.asarray([1.0, 2.0, 3.0], dtype=np.float32)

    monkeypatch.setattr(runtime, "build_feature_row_for_snapshot", build_feature_row_for_snapshot)
    monkeypatch.setattr(runtime, "_predict_margins", predict_margins)
    monkeypatch.setattr(runtime, "_cache_get", synchronized_cache_get)
    arguments = {
        "cell": "cell-a",
        "snapshot_id": "snapshot-a",
        "intensity_snapshot_path": Path("unused"),
        "top_k": 3,
    }

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_timings: dict[str, float] = {}
        second_timings: dict[str, float] = {}
        first = pool.submit(runtime.predict, **arguments, timings=first_timings)
        second = pool.submit(runtime.predict, **arguments, timings=second_timings)
        assert inference_started.wait(timeout=1)
        release_inference.set()
        first_result = first.result(timeout=1)
        second_result = second.result(timeout=1)

    assert calls == 1
    assert first_result["distribution"] == second_result["distribution"]
    assert sum(item["probability"] for item in first_result["distribution"]) == pytest.approx(1.0)
    assert sorted([first_timings["coalesced_wait_ms"], second_timings["coalesced_wait_ms"]])[0] == 0
    assert sorted([first_timings["coalesced_wait_ms"], second_timings["coalesced_wait_ms"]])[1] >= 0

    runtime.predict(**arguments)
    assert calls == 1


def test_stale_snapshot_result_cannot_repopulate_active_cache() -> None:
    runtime = _runtime_without_artifacts()
    runtime.snapshot_id = "snapshot-new"
    runtime._cache_put(("snapshot-old", "cell"), np.asarray([1.0], dtype=np.float32))
    assert runtime._cache_get(("snapshot-old", "cell")) is None
