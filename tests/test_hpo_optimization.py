from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from machine_learning.data.snapshot_stage import (
    HPO_STAGE_MANIFEST,
    assert_test_split_not_staged,
    plan_hpo_snapshot_stage,
    stage_hpo_snapshot,
)
from machine_learning.models.xgboost.model import (
    _CudaMacroCityMetric,
    _CudaPointProcessObjective,
    _point_process_eval_values,
    _point_process_grad_hess_cpu,
    _prepare_xy,
)


class _FakeS3Client:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.listed_prefixes: list[str] = []
        self.downloads: list[str] = []
        self._lock = threading.Lock()

    def list_objects_v2(self, *, Bucket, Prefix, **_kwargs):
        self.listed_prefixes.append(Prefix)
        contents = [
            {"Key": key, "Size": len(value)}
            for key, value in sorted(self.objects.items())
            if key.startswith(Prefix)
        ]
        return {"Contents": contents, "IsTruncated": False}

    def head_object(self, *, Bucket, Key):
        return {"ContentLength": len(self.objects[Key])}

    def download_file(self, bucket, key, filename):
        with self._lock:
            self.downloads.append(key)
        Path(filename).write_bytes(self.objects[key])


class _FakeLake:
    def __init__(self, client: _FakeS3Client) -> None:
        self._client = client

    def s3_client(self):
        return self._client

    def storage_options_for(self, _uri: str) -> dict[str, object]:
        return {}


def _snapshot_objects(tmp_path: Path) -> tuple[str, str, dict[str, bytes]]:
    snapshot_id = "immutable-1"
    snapshot_uri = f"s3://bucket/final/snapshot_id={snapshot_id}"
    prefix = f"final/snapshot_id={snapshot_id}"
    columns = [
        "model_row_id",
        "row_type",
        "event_indicator",
        "is_observed_event",
        "event_count",
        "integration_weight_cell_seconds",
        "source_city",
        "split",
        "feature",
    ]
    objects: dict[str, bytes] = {
        f"{prefix}/manifest.json": json.dumps(
            {
                "snapshot_id": snapshot_id,
                "snapshot_uri": snapshot_uri,
                "schema_version": "v1",
                "columns": columns,
            }
        ).encode(),
        f"{prefix}/_SUCCESS": b"",
        f"{prefix}/split=test/source_city=sealed/must-not-touch": b"sealed",
    }
    for split in ("train", "validation"):
        path = tmp_path / f"{split}.parquet"
        pl.DataFrame(
            {
                "model_row_id": [f"{split}-1"],
                "row_type": ["event"],
                "event_indicator": [1],
                "is_observed_event": [True],
                "event_count": [1],
                "integration_weight_cell_seconds": [None],
                "feature": [1.0],
            },
            schema_overrides={"integration_weight_cell_seconds": pl.Float64},
        ).write_parquet(path)
        objects[f"{prefix}/split={split}/source_city=alpha/part.parquet"] = (
            path.read_bytes()
        )
    return snapshot_id, snapshot_uri, objects


def test_snapshot_staging_is_test_sealed_lineage_preserving_and_resumable(
    tmp_path: Path,
) -> None:
    snapshot_id, snapshot_uri, objects = _snapshot_objects(tmp_path)
    client = _FakeS3Client(objects)
    lake = _FakeLake(client)
    plan = plan_hpo_snapshot_stage(
        snapshot_uri=snapshot_uri,
        snapshot_id=snapshot_id,
        lake=lake,  # type: ignore[arg-type]
    )
    assert all("split=test" not in prefix for prefix in client.listed_prefixes)
    assert all("split=test" not in item.relative_path for item in plan.objects)

    local_root = tmp_path / "stage" / f"snapshot_id={snapshot_id}"
    train_item = next(
        item for item in plan.objects if item.relative_path.startswith("split=train/")
    )
    partial = local_root / f"{train_item.relative_path}.part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(objects[train_item.key])

    staged = stage_hpo_snapshot(
        plan=plan,
        stage_dir=tmp_path / "stage",
        lake=lake,  # type: ignore[arg-type]
        workers=4,
    )
    assert staged == local_root
    assert not (staged / "split=test").exists()
    assert train_item.key not in client.downloads
    manifest = json.loads((staged / "manifest.json").read_text())
    assert manifest["snapshot_id"] == snapshot_id
    assert manifest["snapshot_uri"] == snapshot_uri
    stage_manifest = json.loads((staged / HPO_STAGE_MANIFEST).read_text())
    assert stage_manifest["complete"] is True
    assert stage_manifest["test_split_staged"] is False

    downloads_after_first_run = list(client.downloads)
    stage_hpo_snapshot(
        plan=plan,
        stage_dir=tmp_path / "stage",
        lake=lake,  # type: ignore[arg-type]
        workers=4,
    )
    assert client.downloads == downloads_after_first_run


def test_staging_rejects_preexisting_test_split(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    (root / "split=test").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="split=test"):
        assert_test_split_not_staged(root)


def test_polars_feature_preparation_matches_pandas_category_semantics() -> None:
    train = pl.DataFrame(
        {"numeric": [1, None, 3], "category": ["beta", "alpha", None]}
    )
    validation = pl.DataFrame(
        {"numeric": [4, 5], "category": ["alpha", "unknown"]}
    )
    X_train, _, _, levels = _prepare_xy(
        train.with_columns(
            pl.Series("event_count", [1, 0, 0]),
            pl.Series("integration_weight_cell_seconds", [None, 1.0, 1.0]),
            pl.Series("event_indicator", [1, 0, 0]),
            pl.Series("is_observed_event", [True, False, False]),
            pl.Series("row_type", ["event", "integration", "integration"]),
        ),
        feature_columns=["numeric", "category"],
        categorical_columns=["category"],
    )
    X_validation, _, _, _ = _prepare_xy(
        validation.with_columns(
            pl.Series("event_count", [1, 0]),
            pl.Series("integration_weight_cell_seconds", [None, 1.0]),
            pl.Series("event_indicator", [1, 0]),
            pl.Series("is_observed_event", [True, False]),
            pl.Series("row_type", ["event", "integration"]),
        ),
        feature_columns=["numeric", "category"],
        categorical_columns=["category"],
        category_levels=levels,
    )
    pandas_codes = pd.Categorical(
        validation["category"].to_list(), categories=levels["category"]
    ).codes
    assert X_train.schema["numeric"] == pl.Float32
    assert levels == {"category": ["alpha", "beta"]}
    assert X_validation["category"].is_null().to_list() == [False, True]
    assert pandas_codes.tolist() == [0, -1]


def _cuda_available() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.skipif(not _cuda_available(), reason="CUDA/CuPy unavailable")
def test_cuda_point_process_objective_matches_cpu() -> None:
    import cupy as cp

    rng = np.random.default_rng(42)
    pred = np.concatenate(
        [
            np.array([-100.0, -5.0, 0.0, 100.0, -20.0], dtype=np.float32),
            rng.normal(size=1_000).astype(np.float32),
        ]
    )
    y = np.concatenate(
        [
            np.array([1.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
            rng.integers(0, 2, size=1_000).astype(np.float32),
        ]
    )
    exposure = np.concatenate(
        [
            np.array([0.0, 1.0, 0.0, 0.0, 1e-30], dtype=np.float64),
            rng.lognormal(size=1_000),
        ]
    )
    expected_grad, expected_hess = _point_process_grad_hess_cpu(
        pred=pred,
        y=y,
        exposure=exposure,
        min_log_intensity=-30.0,
        max_log_intensity=15.0,
        hessian_floor=1e-6,
    )
    objective = _CudaPointProcessObjective(
        y=y,
        exposure=exposure,
        min_log_intensity=-30.0,
        max_log_intensity=15.0,
        hessian_floor=1e-6,
    )
    grad, hess = objective(pred, None)
    cp.testing.assert_allclose(grad, expected_grad, rtol=1e-6, atol=1e-7)
    cp.testing.assert_allclose(hess, expected_hess, rtol=1e-6, atol=1e-7)
    objective.release()


@pytest.mark.skipif(not _cuda_available(), reason="CUDA/CuPy unavailable")
def test_cuda_macro_city_metric_matches_cpu() -> None:
    y = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    exposure = np.array([0.0, 2.0, 0.0, 3.0], dtype=np.float64)
    margins = np.array([-2.0, -2.0, -1.0, -1.0], dtype=np.float32)
    city_codes = np.array([0, 0, 1, 1], dtype=np.int64)
    _, expected = _point_process_eval_values(
        y=y,
        exposure=exposure,
        margin=margins,
        city_codes=city_codes,
        city_count=2,
        min_log_intensity=-30.0,
        max_log_intensity=15.0,
    )
    metric = _CudaMacroCityMetric(
        y=y,
        exposure=exposure,
        city_codes=city_codes,
        city_count=2,
        min_log_intensity=-30.0,
        max_log_intensity=15.0,
    )
    name, actual = metric(margins, None)
    assert name == "macro_city_pp_nll_per_event"
    assert actual == pytest.approx(expected, rel=1e-7, abs=1e-8)
    metric.release()
