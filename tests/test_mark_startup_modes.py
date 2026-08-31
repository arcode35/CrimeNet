from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import numpy as np


BACKEND_ROOT = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from mark_runtime import MarkRuntime, resolve_mark_inference_mode  # noqa: E402


class _FakeBooster:
    feature_names = ["static_feature", "lighting_condition"] + [
        f"dynamic_{index}" for index in range(36)
    ]
    feature_types = ["float"] * 38

    @staticmethod
    def save_config() -> str:
        return json.dumps(
            {
                "learner": {
                    "learner_model_param": {
                        "num_class": "87",
                    }
                }
            }
        )


def _runtime_shell() -> MarkRuntime:
    runtime = MarkRuntime.__new__(MarkRuntime)
    runtime.cpu_bst = object()
    runtime.gpu_bst = object()
    runtime.bst = runtime.cpu_bst
    runtime.cp = None
    runtime.gpu_queue = None
    runtime.inference_device = "unconfigured"
    runtime.inference_benchmark = None
    return runtime


def _forbid_diagnostic_paths(monkeypatch: pytest.MonkeyPatch, runtime: MarkRuntime) -> None:
    monkeypatch.setattr(
        runtime,
        "_select_inference_device",
        Mock(side_effect=AssertionError("benchmark must not run")),
    )
    monkeypatch.setattr(
        runtime,
        "_initialize_gpu_runtime",
        Mock(side_effect=AssertionError("CUDA must not initialize")),
    )
    monkeypatch.setattr(
        runtime,
        "_start_gpu_batch_worker",
        Mock(side_effect=AssertionError("GPU worker must not start")),
    )


def test_unset_environment_defaults_to_cpu_without_benchmark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CRIMENET_MARK_INFERENCE", raising=False)
    runtime = _runtime_shell()
    _forbid_diagnostic_paths(monkeypatch, runtime)

    mode = resolve_mark_inference_mode()
    runtime._activate_inference_mode(mode)

    assert mode == "cpu"
    assert runtime.inference_device == "cpu"
    assert runtime.inference_benchmark is None
    assert runtime.gpu_queue is None


def test_real_unset_startup_runs_zero_benchmarks_and_never_initializes_cuda(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("CRIMENET_MARK_INFERENCE", raising=False)
    model_path = tmp_path / "model.ubj"
    model_path.write_bytes(b"model-placeholder")
    monkeypatch.setattr("mark_runtime.MARK_MODEL_PATH", model_path)

    def fake_np_load(path, mmap_mode=None):
        path = Path(path)
        if path.name == "features.npy":
            return np.zeros((1, 1), dtype=np.float32)
        return np.zeros(1, dtype=np.uint64)

    def fake_load_json(path):
        if Path(path).name == "r6_timezones.json":
            return {"timezones": {"0": "UTC"}}
        return {"features": ["static_feature"]}

    benchmark = Mock(side_effect=AssertionError("benchmark must not run"))
    initialize_gpu = Mock(side_effect=AssertionError("CUDA must not initialize"))
    start_worker = Mock(side_effect=AssertionError("GPU worker must not start"))
    monkeypatch.setattr("mark_runtime.np.load", fake_np_load)
    monkeypatch.setattr("mark_runtime.load_json", fake_load_json)
    monkeypatch.setattr(MarkRuntime, "_load_booster", lambda _self, _device: _FakeBooster())
    monkeypatch.setattr(
        MarkRuntime,
        "_load_labels",
        lambda _self: ([f"class_{index}" for index in range(87)], True),
    )
    monkeypatch.setattr(MarkRuntime, "_select_inference_device", benchmark)
    monkeypatch.setattr(MarkRuntime, "_initialize_gpu_runtime", initialize_gpu)
    monkeypatch.setattr(MarkRuntime, "_start_gpu_batch_worker", start_worker)

    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        runtime = MarkRuntime()

    benchmark.assert_not_called()
    initialize_gpu.assert_not_called()
    start_worker.assert_not_called()
    assert runtime.configured_inference_mode == "cpu"
    assert runtime.inference_benchmark is None
    assert runtime.cp is None
    assert runtime.gpu_queue is None
    mode_logs = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("mark_inference mode=")
    ]
    assert mode_logs == ["mark_inference mode=cpu benchmark=false"]


def test_explicit_cpu_does_not_benchmark_or_start_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRIMENET_MARK_INFERENCE", "cpu")
    runtime = _runtime_shell()
    _forbid_diagnostic_paths(monkeypatch, runtime)

    runtime._activate_inference_mode(resolve_mark_inference_mode())

    assert runtime.inference_device == "cpu"
    assert runtime.inference_benchmark is None
    assert runtime.gpu_queue is None


def test_gpu_batch_initializes_gpu_and_worker_without_benchmark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRIMENET_MARK_INFERENCE", "gpu_batch")
    runtime = _runtime_shell()
    benchmark = Mock(side_effect=AssertionError("benchmark must not run"))
    initialize_gpu = Mock()
    start_worker = Mock(side_effect=lambda: setattr(runtime, "gpu_queue", object()))
    monkeypatch.setattr(runtime, "_select_inference_device", benchmark)
    monkeypatch.setattr(runtime, "_initialize_gpu_runtime", initialize_gpu)
    monkeypatch.setattr(runtime, "_start_gpu_batch_worker", start_worker)

    runtime._activate_inference_mode(resolve_mark_inference_mode())

    benchmark.assert_not_called()
    initialize_gpu.assert_called_once_with()
    start_worker.assert_called_once_with()
    assert runtime.inference_device == "gpu_batch"
    assert runtime.inference_benchmark is None
    assert runtime.gpu_queue is not None


def test_auto_runs_benchmark_only_when_explicitly_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRIMENET_MARK_INFERENCE", "auto")
    runtime = _runtime_shell()
    report = {"recommended_serving_mode": "cpu"}
    benchmark = Mock(return_value=report)
    start_worker = Mock(side_effect=AssertionError("mock benchmark selected CPU"))
    monkeypatch.setattr(runtime, "_select_inference_device", benchmark)
    monkeypatch.setattr(runtime, "_start_gpu_batch_worker", start_worker)

    runtime._activate_inference_mode(resolve_mark_inference_mode())

    benchmark.assert_called_once_with()
    start_worker.assert_not_called()
    assert runtime.inference_benchmark is report


def test_invalid_mode_fails_before_loading_serving_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRIMENET_MARK_INFERENCE", "foo")
    artifact_load = Mock(side_effect=AssertionError("artifacts must not load"))
    monkeypatch.setattr("mark_runtime.np.load", artifact_load)

    with pytest.raises(
        RuntimeError,
        match=(
            "Invalid CRIMENET_MARK_INFERENCE='foo'. "
            "Expected one of: cpu, gpu_batch, auto."
        ),
    ):
        MarkRuntime()

    artifact_load.assert_not_called()


def test_cpu_model_initialization_loads_no_gpu_booster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime_shell()
    loaded_devices: list[str] = []
    monkeypatch.setattr(
        runtime,
        "_load_booster",
        lambda device: loaded_devices.append(device) or object(),
    )

    runtime.cpu_bst = None
    runtime.gpu_bst = None
    runtime._initialize_model_for_mode("cpu")

    assert loaded_devices == ["cpu"]
    assert runtime.cpu_bst is runtime.bst
    assert runtime.gpu_bst is None


def test_gpu_model_initialization_loads_no_cpu_booster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime_shell()
    loaded_devices: list[str] = []
    monkeypatch.setattr(
        runtime,
        "_load_booster",
        lambda device: loaded_devices.append(device) or object(),
    )

    runtime.cpu_bst = None
    runtime.gpu_bst = None
    runtime._initialize_model_for_mode("gpu_batch")

    assert loaded_devices == ["cuda:0"]
    assert runtime.gpu_bst is runtime.bst
    assert runtime.cpu_bst is None
