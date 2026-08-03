from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import polars as pl
import sklearn
import xgboost
import yaml
from xgboost import XGBClassifier

from crimenet_ml.config import load_yaml_config
from crimenet_ml.data.dataloader import (
    ModelMatrix,
    build_category_vocabularies,
    load_split,
    normalize_sample_weights,
    to_model_matrix,
)
from crimenet_ml.evaluation import evaluate_binary_predictions
from crimenet_ml.features import get_feature_set
from crimenet_ml.paths import PROJECT_ROOT


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _print_split_summary(
    name: str,
    rows: int,
    positives: int,
    positive_rate: float,
) -> None:
    print(
        f"{name:<10} rows={rows:>12,} "
        f"positives={positives:>10,} "
        f"positive_rate={positive_rate:.6f}"
    )


def _create_run_directory(
    project_root: Path,
    output_config: Mapping[str, Any],
    experiment_name: str,
    run_name: str | None,
) -> Path:
    artifact_dir = Path(str(output_config.get("artifact_dir", "artifacts/xgboost")))
    if not artifact_dir.is_absolute():
        artifact_dir = project_root / artifact_dir

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    resolved_run_name = run_name or f"{experiment_name}_{timestamp}"
    run_directory = artifact_dir.resolve() / resolved_run_name
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def _normalize_model_parameters(
    model_config: Mapping[str, Any],
    random_seed: int,
    device_override: str | None,
) -> dict[str, Any]:
    parameters = dict(model_config)

    eval_metric = parameters.get("eval_metric")
    if isinstance(eval_metric, tuple):
        parameters["eval_metric"] = list(eval_metric)

    if device_override is not None:
        parameters["device"] = device_override

    parameters["random_state"] = random_seed
    parameters["enable_categorical"] = True
    parameters["verbosity"] = 1

    return parameters


def _save_feature_importance(
    model: XGBClassifier,
    output_path: Path,
) -> None:
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    cover = booster.get_score(importance_type="cover")

    feature_names = booster.feature_names or []
    rows = [
        {
            "feature": feature,
            "gain": float(gain.get(feature, 0.0)),
            "weight": float(weight.get(feature, 0.0)),
            "cover": float(cover.get(feature, 0.0)),
        }
        for feature in feature_names
    ]

    (
        pl.DataFrame(rows)
        .sort("gain", descending=True)
        .write_csv(output_path)
    )


def _save_predictions(
    matrix: ModelMatrix,
    probability: np.ndarray,
    output_path: Path,
) -> None:
    prediction_frame = matrix.metadata.with_columns(
        pl.Series("label", matrix.y),
        pl.Series("prediction", probability),
    )

    if matrix.sample_weight is not None:
        prediction_frame = prediction_frame.with_columns(
            pl.Series("sample_weight", matrix.sample_weight)
        )

    prediction_frame.write_parquet(output_path)


def _dependency_versions() -> dict[str, str]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "polars": pl.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
    }


def train_from_config(
    config_path: str | Path,
    *,
    limit_per_split: int | None = None,
    device_override: str | None = None,
    run_name: str | None = None,
    explain: bool = False,
) -> Path:
    config_path = Path(config_path).expanduser().resolve()
    config = load_yaml_config(config_path)

    experiment_config = config["experiment"]
    data_config = config["data"]
    split_config = config["split"]
    model_config = config["model"]
    output_config = config.get("output", {})
    training_config = config.get("training", {})

    experiment_name = str(experiment_config["name"])
    feature_set_name = str(experiment_config["feature_set"])
    random_seed = int(experiment_config.get("random_seed", 42))
    feature_names = get_feature_set(feature_set_name)

    np.random.seed(random_seed)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Experiment:   {experiment_name}")
    print(f"Feature set:  {feature_set_name} ({len(feature_names)} features)")
    print(f"Split mode:   {split_config.get('mode', 'temporal')}")

    loaded_train = load_split(
        project_root=PROJECT_ROOT,
        data_config=data_config,
        split_config=split_config,
        split_name="train",
        feature_names=feature_names,
        limit=limit_per_split,
        explain=explain,
    )
    loaded_validation = load_split(
        project_root=PROJECT_ROOT,
        data_config=data_config,
        split_config=split_config,
        split_name="validation",
        feature_names=feature_names,
        limit=limit_per_split,
        explain=explain,
    )
    loaded_test = load_split(
        project_root=PROJECT_ROOT,
        data_config=data_config,
        split_config=split_config,
        split_name="test",
        feature_names=feature_names,
        limit=limit_per_split,
        explain=explain,
    )

    print("\nLoaded splits")
    _print_split_summary(
        "train",
        loaded_train.row_count,
        loaded_train.positive_count,
        loaded_train.positive_rate,
    )
    _print_split_summary(
        "validation",
        loaded_validation.row_count,
        loaded_validation.positive_count,
        loaded_validation.positive_rate,
    )
    _print_split_summary(
        "test",
        loaded_test.row_count,
        loaded_test.positive_count,
        loaded_test.positive_rate,
    )

    categorical_columns = tuple(data_config.get("categorical_columns", ()))
    category_vocabularies = build_category_vocabularies(
        loaded_train,
        categorical_columns,
    )

    train_matrix = to_model_matrix(
        loaded_train,
        category_vocabularies,
    )
    validation_matrix = to_model_matrix(
        loaded_validation,
        category_vocabularies,
    )
    test_matrix = to_model_matrix(
        loaded_test,
        category_vocabularies,
    )

    weight_normalization_divisor = 1.0
    if bool(data_config.get("normalize_sample_weight", True)):
        (
            train_matrix,
            validation_matrix,
            test_matrix,
            weight_normalization_divisor,
        ) = normalize_sample_weights(
            train_matrix,
            validation_matrix,
            test_matrix,
        )

    parameters = _normalize_model_parameters(
        model_config=model_config,
        random_seed=random_seed,
        device_override=device_override,
    )

    model = XGBClassifier(**parameters)

    print("\nTraining XGBoost")
    print(
        f"device={parameters.get('device', 'cpu')} "
        f"tree_method={parameters.get('tree_method', 'auto')} "
        f"n_estimators={parameters.get('n_estimators')}"
    )

    model.fit(
        train_matrix.X,
        train_matrix.y,
        sample_weight=train_matrix.sample_weight,
        eval_set=[
            (train_matrix.X, train_matrix.y),
            (validation_matrix.X, validation_matrix.y),
        ],
        sample_weight_eval_set=[
            train_matrix.sample_weight,
            validation_matrix.sample_weight,
        ]
        if train_matrix.sample_weight is not None
        else None,
        verbose=int(training_config.get("verbose_eval", 25)),
    )

    validation_probability = model.predict_proba(validation_matrix.X)[:, 1]
    test_probability = model.predict_proba(test_matrix.X)[:, 1]

    metrics = {
        "validation": evaluate_binary_predictions(
            validation_matrix.y,
            validation_probability,
            validation_matrix.sample_weight,
        ),
        "test": evaluate_binary_predictions(
            test_matrix.y,
            test_probability,
            test_matrix.sample_weight,
        ),
    }

    run_directory = _create_run_directory(
        project_root=PROJECT_ROOT,
        output_config=output_config,
        experiment_name=experiment_name,
        run_name=run_name,
    )

    # UBJSON preserves categorical split information.
    model.save_model(run_directory / "model.ubj")

    with (run_directory / "config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    with (run_directory / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, default=_json_default)

    with (run_directory / "evals_result.json").open("w", encoding="utf-8") as file:
        json.dump(model.evals_result(), file, indent=2, default=_json_default)

    model_metadata = {
        "experiment_name": experiment_name,
        "feature_set_name": feature_set_name,
        "feature_names": list(feature_names),
        "categorical_columns": list(categorical_columns),
        "category_vocabularies": category_vocabularies,
        "label_column": data_config["label_column"],
        "weight_column": data_config.get("weight_column"),
        "sample_weight_normalization_divisor": weight_normalization_divisor,
        "best_iteration": getattr(model, "best_iteration", None),
        "best_score": getattr(model, "best_score", None),
        "dependencies": _dependency_versions(),
    }
    with (run_directory / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(model_metadata, file, indent=2, default=_json_default)

    _save_feature_importance(
        model,
        run_directory / "feature_importance.csv",
    )

    if bool(output_config.get("save_test_predictions", True)):
        _save_predictions(
            test_matrix,
            test_probability,
            run_directory / "test_predictions.parquet",
        )

    print("\nValidation metrics")
    print(json.dumps(metrics["validation"], indent=2))

    print("\nTest metrics")
    print(json.dumps(metrics["test"], indent=2))

    print(f"\nBest iteration: {getattr(model, 'best_iteration', None)}")
    print(f"Artifacts:      {run_directory}")

    return run_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an XGBoost CrimeNet baseline"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to an experiment YAML file",
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="Load only this many rows per split for a smoke test",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
        help="Override the device configured in YAML",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional fixed artifact-directory name",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Print optimized Polars query plans",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_from_config(
        args.config,
        limit_per_split=args.limit_per_split,
        device_override=args.device,
        run_name=args.run_name,
        explain=args.explain,
    )


if __name__ == "__main__":
    main()
