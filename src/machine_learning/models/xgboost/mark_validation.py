"""Full split-authoritative geographic validation for mark models."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import polars as pl
import xgboost as xgb

from machine_learning.data.metrics import geographic_mark_metrics
from machine_learning.data.model_table import resolve_model_table
from machine_learning.models.xgboost.mark_model import (
    MACHINE_LEARNING_ROOT,
    _prepare_xy,
)


def validate(config: dict[str, Any], *, run_id: str) -> dict[str, Any]:
    model_config = config["model"]
    data_config = config["data"]
    validation_config = config.get("validation", {})
    artifact_dir = (
        MACHINE_LEARNING_ROOT
        / config["artifacts"]["output_root"]
        / model_config["name"]
        / run_id
    )
    model_path = artifact_dir / "model.json"
    metadata_path = artifact_dir / "metadata.json"
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Missing model artifacts under {artifact_dir}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("run_id") not in (None, run_id):
        raise RuntimeError("Requested run ID does not match model metadata")

    table_ref = resolve_model_table(
        snapshot_override_uri=str(metadata["final_model_snapshot_uri"]),
        local_root=data_config.get("local_snapshot_root"),
    )
    if table_ref.snapshot_id != str(metadata["final_model_snapshot_id"]):
        raise RuntimeError("Validation snapshot differs from training snapshot")
    validation_split = str(data_config.get("validation_split", "validation"))
    table = table_ref.scan_split(validation_split)
    holdouts = sorted(set(validation_config.get("geographic_holdout_cities", [])))
    if holdouts:
        table = table.filter(pl.col("source_city").is_in(holdouts))

    features = list(metadata["features"])
    target = str(metadata["target_column"])
    frame = (
        table.filter(pl.col("is_observed_event"))
        .filter(pl.col("event_count") == 1)
        .filter(pl.col(target).is_not_null())
        .select(features + ["source_city", target])
        .collect()
    )
    if frame.is_empty():
        raise RuntimeError("Geographic mark-validation frame is empty")
    class_to_index = {
        str(key): int(value) for key, value in metadata["class_to_index"].items()
    }
    X, y, _ = _prepare_xy(
        frame,
        feature_columns=features,
        categorical_columns=list(metadata["categorical_columns"]),
        target_column=target,
        class_to_index=class_to_index,
        category_levels={
            str(key): [str(value) for value in values]
            for key, values in metadata["category_levels"].items()
        },
    )
    dmatrix = xgb.DMatrix(X, label=y, enable_categorical=True, nthread=-1)
    booster = xgb.Booster()
    booster.load_model(model_path)
    probabilities = booster.predict(dmatrix)
    classes = list(metadata["classes"])
    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(-1, len(classes))
    priors = np.asarray(
        [metadata["training_class_priors"][class_name] for class_name in classes],
        dtype=np.float64,
    )
    report = geographic_mark_metrics(
        source_cities=frame["source_city"].cast(pl.String).to_list(),
        labels=y,
        probabilities=probabilities,
        training_priors=priors,
    )
    city_path = artifact_dir / "full_validation_by_city.csv"
    summary_path = artifact_dir / "full_validation_summary.json"
    pl.DataFrame(report["per_city"]).write_csv(city_path)
    summary_path.write_text(json.dumps(report, indent=2))
    metrics = {
        "pooled_log_loss": float(report["global"]["log_loss"]),
        "pooled_bits_gain": float(report["global"]["bits_gain"]),
        "macro_city_log_loss": float(report["macro_city"]["mean_log_loss"]),
        "macro_city_bits_gain": float(report["macro_city"]["mean_bits_gain"]),
        "macro_city_accuracy": float(report["macro_city"]["mean_accuracy"]),
    }
    return {
        "metrics": metrics,
        "city_metrics": {
            str(row["source_city"]): {
                key: float(value)
                for key, value in row.items()
                if key != "source_city"
            }
            for row in report["per_city"]
        },
        "artifacts": [str(city_path), str(summary_path)],
        "summary": {
            "validation_split": validation_split,
            "geographic_holdout_cities": holdouts,
            "final_model_snapshot_id": table_ref.snapshot_id,
            "unseen_validation_label_policy": "fail",
            "test_split_used": False,
        },
    }
