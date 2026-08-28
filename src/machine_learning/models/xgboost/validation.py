"""Full split-authoritative geographic validation for intensity models."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import polars as pl
import xgboost as xgb

from machine_learning.data.metrics import geographic_point_process_metrics
from machine_learning.data.model_table import resolve_model_table
from machine_learning.data.geographic_cv import validate_exact_modeling_cities
from machine_learning.models.xgboost.model import (
    MACHINE_LEARNING_ROOT,
    _prepare_xy,
    _validate_point_process_rows,
)


def validate(config: dict[str, Any], *, run_id: str) -> dict[str, Any]:
    model_config = config["model"]
    data_config = config["data"]
    numerics = config["numerics"]
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
    categorical = list(metadata["categorical_columns"])
    frame = table.select(
        features
        + [
            "source_city",
            "row_type",
            "event_indicator",
            "is_observed_event",
            "event_count",
            "integration_weight_cell_seconds",
        ]
    ).collect()
    if frame.is_empty():
        raise RuntimeError("Geographic validation frame is empty")
    if metadata.get("training_strategy") == "final_all_city_train":
        validate_exact_modeling_cities(
            frame["source_city"].cast(pl.String).unique().to_list(),
            label="final in-domain temporal validation",
        )
    X, y, exposure, _ = _prepare_xy(
        frame,
        feature_columns=features,
        categorical_columns=categorical,
        category_levels={
            str(key): [str(value) for value in values]
            for key, values in metadata["category_levels"].items()
        },
    )
    _validate_point_process_rows(
        name="validation",
        y=y,
        exposure=exposure,
        event_exposure_tolerance=float(
            numerics.get("event_exposure_tolerance", 1e-12)
        ),
    )
    dmatrix = xgb.DMatrix(
        X,
        label=y,
        base_margin=np.full(y.shape, float(metadata["initial_log_lambda"])),
        enable_categorical=True,
        nthread=-1,
    )
    booster = xgb.Booster()
    booster.load_model(model_path)
    margin = booster.predict(dmatrix, output_margin=True)
    report = geographic_point_process_metrics(
        frame,
        log_intensity=margin,
        constant_log_intensity=float(metadata["initial_log_lambda"]),
        min_log_intensity=float(numerics["min_log_intensity"]),
        max_log_intensity=float(numerics["max_log_intensity"]),
    )
    validation_label = (
        "final_in_domain_temporal_validation"
        if metadata.get("training_strategy") == "final_all_city_train"
        else "geographic_validation"
    )
    city_path = artifact_dir / f"{validation_label}_by_city.csv"
    summary_path = artifact_dir / f"{validation_label}_summary.json"
    pl.DataFrame(report["per_city"]).write_csv(city_path)
    summary_path.write_text(json.dumps(report, indent=2))
    global_metrics = report["global"]
    macro = report["macro_city"]
    metrics = {
        "pooled_nll_per_event": float(global_metrics["nll_per_event"]),
        "pooled_bits_per_event": float(global_metrics["bits_per_event"]),
        "pooled_expected_observed": float(global_metrics["expected_observed_ratio"]),
        "macro_city_nll_per_event": float(macro["mean_nll_per_event"]),
        "macro_city_bits_per_event": float(macro["mean_bits_per_event"]),
        "worst_city_bits_per_event": float(macro["worst_city_bits_per_event"]),
        "mean_absolute_city_calibration_error_pct": float(
            macro["mean_absolute_calibration_error_pct"]
        ),
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
            "validation_label": validation_label,
            "test_split_used": False,
        },
    }
