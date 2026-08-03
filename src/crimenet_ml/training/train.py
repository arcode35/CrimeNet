"""Thin command-line orchestration for CrimeNet Poisson training."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import mlflow.xgboost
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

from crimenet_ml.config import AppConfig, load_config
from crimenet_ml.data.databricks import DatabricksTableLoader
from crimenet_ml.data.local import LocalParquetLoader
from crimenet_ml.data.splits import build_dataset_bundle
from crimenet_ml.evaluation.metrics import point_process_metrics
from crimenet_ml.features.feature_set import get_feature_set
from crimenet_ml.inference.pyfunc_model import CalibratedXGBPoissonPyfunc
from crimenet_ml.models.xgb_poisson import XGBPoissonModel
from crimenet_ml.tracking.configuration import configure_mlflow
from crimenet_ml.tracking.logging import (
    feature_importance_frame,
    flatten_metrics,
    flatten_parameters,
    git_state,
    log_dataset_inputs,
    log_evaluation_history,
    log_json_artifact,
    log_yaml_artifact,
)


@dataclass
class TrainingOutcome:
    run_id: str | None
    model_uri: str | None
    raw_model_uri: str | None
    artifact_uri: str | None
    tracking_uri: str | None
    calibration_factor: float
    metrics: dict[str, dict[str, float | None]]
    in_process_example_prediction: list[float]


def _model_parameters(config: AppConfig) -> dict[str, Any]:
    values = config.model.model_dump()
    values.pop("family")
    return values


def _loader(config: AppConfig):
    if config.data.backend == "local":
        return LocalParquetLoader(
            config.data.path, config.data.split_column, config.feature_set, config.random_seed
        )
    return DatabricksTableLoader(
        config.data.table or "",
        config.data.split_column,
        config.feature_set,
        config.data.collection_row_limit,
        config.data.allow_large_collection,
        config.random_seed,
    )


def train(
    config: AppConfig,
    *,
    limit_per_split: int | None = None,
    device: str | None = None,
    run_name: str | None = None,
    data_table: str | None = None,
    experiment_name: str | None = None,
) -> TrainingOutcome:
    if device:
        config.model.device = device  # CLI is an intentional resolved-config override.
    if data_table:
        config.data.table = data_table
    if experiment_name:
        config.tracking.experiment_name = experiment_name
    feature_set = get_feature_set(config.feature_set)
    required_columns = list(
        dict.fromkeys(
            list(feature_set.features)
            + [config.data.event_count_column, config.data.integration_weight_column]
            + config.data.identifier_columns
            + config.data.metadata_columns
        )
    )
    bundle = build_dataset_bundle(
        _loader(config),
        required_columns,
        list(feature_set.features),
        config.data.event_count_column,
        config.data.integration_weight_column,
        config.data.identifier_columns,
        config.data.allow_missing_features,
        limit_per_split,
    )
    model = XGBPoissonModel(
        list(feature_set.features),
        config.data.categorical_columns,
        _model_parameters(config),
        config.random_seed,
    )
    model.fit_category_vocabularies(bundle.train.frame)
    train_matrix = model.make_matrix(
        bundle.train.frame, config.data.event_count_column, config.data.integration_weight_column
    )
    validation_matrix = model.make_matrix(
        bundle.validation.frame,
        config.data.event_count_column,
        config.data.integration_weight_column,
    )
    test_matrix = model.make_matrix(
        bundle.test.frame, config.data.event_count_column, config.data.integration_weight_column
    )
    result = model.fit(train_matrix, validation_matrix)
    validation_raw = model.predict_raw(bundle.validation.frame)
    validation_calibrated = model.predict_calibrated(bundle.validation.frame)
    test_raw = model.predict_raw(bundle.test.frame)
    test_calibrated = model.predict_calibrated(bundle.test.frame)
    metrics = {
        "validation_raw": point_process_metrics(
            validation_matrix.event_count,
            validation_matrix.sample_weight,
            validation_raw,
        ),
        "validation_calibrated": point_process_metrics(
            validation_matrix.event_count,
            validation_matrix.sample_weight,
            validation_calibrated,
        ),
        "test_raw": point_process_metrics(
            test_matrix.event_count,
            test_matrix.sample_weight,
            test_raw,
        ),
        "test_calibrated": point_process_metrics(
            test_matrix.event_count,
            test_matrix.sample_weight,
            test_calibrated,
        ),
    }
    output_dir = config.output.artifact_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    example = bundle.validation.frame.loc[:, feature_set.features].head(5).copy()
    example_prediction = model.predict_calibrated(example)
    if config.output.write_test_predictions:
        prediction_columns = list(
            dict.fromkeys(
                config.data.identifier_columns
                + config.data.metadata_columns
                + [config.data.event_count_column, config.data.integration_weight_column]
            )
        )
        predictions = bundle.test.frame.loc[:, prediction_columns].copy()
        predictions["raw_predicted_intensity"] = test_raw
        predictions["predicted_intensity"] = test_calibrated
        predictions["raw_predicted_event_mass"] = (
            predictions[config.data.integration_weight_column] * test_raw
        )
        predictions["predicted_event_mass"] = (
            predictions[config.data.integration_weight_column] * test_calibrated
        )
        predictions.to_parquet(output_dir / "test_predictions.parquet", index=False)

    if not config.tracking.enabled:
        return TrainingOutcome(
            None,
            None,
            None,
            None,
            None,
            result.calibration_factor,
            metrics,
            example_prediction.tolist(),
        )

    configure_mlflow(config.tracking)
    repo_root = Path(__file__).resolve().parents[3]
    commit, dirty = git_state(repo_root)
    tags = {
        "environment": config.environment,
        "data_backend": config.data.backend,
        "model_family": "xgb_poisson",
        "objective": config.model.objective,
        "feature_set": config.feature_set,
        "feature_definition_version": config.feature_definition_version,
        "split_definition_version": config.split_definition_version,
        "device": config.model.device,
        "git_commit": commit,
        "git_dirty": str(dirty).lower(),
    }
    with mlflow.start_run(run_name=run_name, tags=tags) as active_run:
        split_frames = {
            "train": bundle.train.frame,
            "validation": bundle.validation.frame,
            "test": bundle.test.frame,
        }
        statistics = {
            name: {
                "row_count": len(frame),
                "observed_event_count": float(frame[config.data.event_count_column].sum()),
                "total_integration_weight": float(
                    frame[config.data.integration_weight_column].sum()
                ),
            }
            for name, frame in split_frames.items()
        }
        parameters = {
            "xgboost": _model_parameters(config),
            "random_seed": config.random_seed,
            "feature_count": len(feature_set.features),
            "categorical_columns": config.data.categorical_columns,
            "split_names": ["train", "validation", "test"],
            "target_column": config.data.event_count_column,
            "integration_weight_column": config.data.integration_weight_column,
            "calibration_strategy": "validation_observed_over_raw_predicted_event_mass",
            "data_source": bundle.identity.table_or_path,
            "limit_per_split": limit_per_split,
            "splits": statistics,
        }
        mlflow.log_params(flatten_parameters(parameters))
        flattened_metrics = flatten_metrics(metrics)
        flattened_metrics["intensity_calibration_factor"] = result.calibration_factor
        mlflow.log_metrics(flattened_metrics)
        log_evaluation_history(result.evals_result)
        log_dataset_inputs(
            split_frames,
            bundle.identity.table_or_path,
            str(bundle.identity.fingerprint_or_delta_version or "") or None,
        )
        resolved = config.model_dump(mode="json")
        log_yaml_artifact(resolved, "resolved_config.yaml", output_dir)
        log_json_artifact(metrics, "metrics.json", output_dir)
        log_json_artifact(result.evals_result, "evals_result.json", output_dir)
        log_json_artifact(asdict(bundle.identity), "dataset_identity.json", output_dir)
        metadata = {
            "feature_names": list(result.feature_names),
            "categorical_columns": list(result.categorical_columns),
            "category_vocabularies": result.category_vocabularies,
            "baseline_intensity": result.baseline_intensity,
            "calibration_factor": result.calibration_factor,
        }
        log_json_artifact(metadata, "metadata.json", output_dir)
        importance = feature_importance_frame(result.model, result.feature_names)
        importance_path = output_dir / "feature_importance.csv"
        importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(str(importance_path))
        if config.output.write_test_predictions:
            mlflow.log_artifact(str(output_dir / "test_predictions.parquet"))
        raw_info = mlflow.xgboost.log_model(
            result.model,
            artifact_path=config.tracking.raw_model_artifact_name,
        )
        wrapper = CalibratedXGBPoissonPyfunc(
            result.model,
            result.feature_names,
            result.categorical_columns,
            result.category_vocabularies,
            result.calibration_factor,
        )
        output_example = pd.DataFrame({"predicted_intensity": example_prediction})
        signature = infer_signature(example, output_example)
        model_info = mlflow.pyfunc.log_model(
            artifact_path=config.tracking.calibrated_model_artifact_name,
            python_model=wrapper,
            input_example=example,
            signature=signature,
            pip_requirements=["mlflow", "pandas", "numpy", "xgboost"],
        )
        loaded = mlflow.pyfunc.load_model(model_info.model_uri)
        round_trip = loaded.predict(example)["predicted_intensity"].to_numpy()
        np.testing.assert_allclose(round_trip, example_prediction, rtol=1e-6, atol=1e-8)
        artifact_uri = mlflow.get_artifact_uri()
        run_id = active_run.info.run_id
    return TrainingOutcome(
        run_id,
        model_info.model_uri,
        raw_info.model_uri,
        artifact_uri,
        mlflow.get_tracking_uri(),
        result.calibration_factor,
        metrics,
        example_prediction.tolist(),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--run-name")
    parser.add_argument("--data-table", help="Override the configured Unity Catalog table")
    parser.add_argument("--experiment-name", help="Override the configured MLflow experiment")
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    outcome = train(
        load_config(arguments.config),
        limit_per_split=arguments.limit_per_split,
        device=arguments.device,
        run_name=arguments.run_name,
        data_table=arguments.data_table,
        experiment_name=arguments.experiment_name,
    )
    print(f"run_id={outcome.run_id or 'tracking-disabled'}")
    print(f"tracking_location={outcome.tracking_uri or 'disabled'}")
    print(f"artifact_uri={outcome.artifact_uri or 'disabled'}")
    print(f"model_uri={outcome.model_uri or 'disabled'}")


if __name__ == "__main__":
    main()
