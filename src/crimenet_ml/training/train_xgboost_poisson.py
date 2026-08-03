from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import polars as pl
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBRegressor

from crimenet_ml.config import load_yaml_config
from crimenet_ml.features import get_feature_set
from crimenet_ml.paths import PROJECT_ROOT


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return value.item()

    raise TypeError(
        f"Cannot serialize object of type {type(value).__name__}"
    )


def resolve_dataset_glob(
    data_config: Mapping[str, Any],
) -> str:
    dataset_directory = Path(
        str(data_config["dataset_dir"])
    ).expanduser()

    if not dataset_directory.is_absolute():
        dataset_directory = PROJECT_ROOT / dataset_directory

    dataset_directory = dataset_directory.resolve()

    if not dataset_directory.is_dir():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {dataset_directory}"
        )

    parquet_glob = str(
        data_config.get("parquet_glob", "**/*.parquet")
    )

    data_glob = dataset_directory / parquet_glob

    if next(dataset_directory.rglob("*.parquet"), None) is None:
        raise FileNotFoundError(
            f"No Parquet files found under {dataset_directory}"
        )

    return str(data_glob)


def validate_required_columns(
    schema: pl.Schema,
    required_columns: Sequence[str],
) -> None:
    missing = sorted(
        set(required_columns) - set(schema.names())
    )

    if missing:
        formatted = "\n".join(
            f"  - {column}" for column in missing
        )

        raise ValueError(
            f"Dataset is missing required columns:\n{formatted}"
        )


def load_split(
    *,
    data_glob: str,
    split_name: str,
    feature_names: Sequence[str],
    data_config: Mapping[str, Any],
    limit: int | None,
    explain: bool,
) -> pl.DataFrame:
    split_column = str(
        data_config.get("split_column", "dataset_split")
    )
    event_count_column = str(
        data_config["event_count_column"]
    )
    weight_column = str(
        data_config["integration_weight_column"]
    )
    metadata_columns = tuple(
        data_config.get("metadata_columns", ())
    )

    query = pl.scan_parquet(
        data_glob,
        glob=True,
        hive_partitioning=True,
        use_statistics=True,
    )

    schema = query.collect_schema()

    required_columns = list(
        dict.fromkeys(
            [
                *feature_names,
                event_count_column,
                weight_column,
                split_column,
                *metadata_columns,
            ]
        )
    )

    validate_required_columns(
        schema,
        required_columns,
    )

    output_columns = list(
        dict.fromkeys(
            [
                *feature_names,
                event_count_column,
                weight_column,
                *metadata_columns,
            ]
        )
    )

    query = (
        query
        .filter(pl.col(split_column) == split_name)
        .select(output_columns)
    )

    if limit is not None:
        if limit <= 0:
            raise ValueError(
                "--limit-per-split must be greater than zero"
            )

        if "source_city" not in output_columns:
            raise ValueError(
                "source_city must be included in metadata_columns "
                "for stratified smoke-test sampling"
            )

        cities = (
            query
            .select("source_city")
            .unique()
            .collect()
            .get_column("source_city")
            .drop_nulls()
            .sort()
            .to_list()
        )

        if not cities:
            raise ValueError(
                f"No cities found in split {split_name!r}"
            )

        rows_per_city = limit // len(cities)
        remainder = limit % len(cities)

        city_queries: list[pl.LazyFrame] = []

        for city_index, city in enumerate(cities):
            city_limit = rows_per_city + (
                1 if city_index < remainder else 0
            )

            if city_limit == 0:
                continue

            city_queries.append(
                query
                .filter(pl.col("source_city") == city)
                .head(city_limit)
            )

        query = pl.concat(
            city_queries,
            how="vertical",
        )

    if explain:
        print(f"\nOptimized plan for {split_name}:")
        print(query.explain(optimized=True))

    frame = query.collect()

    if frame.is_empty():
        raise ValueError(
            f"The {split_name!r} split contains zero rows"
        )

    validate_split(
        frame=frame,
        split_name=split_name,
        event_count_column=event_count_column,
        weight_column=weight_column,
    )

    return frame


def validate_split(
    *,
    frame: pl.DataFrame,
    split_name: str,
    event_count_column: str,
    weight_column: str,
) -> None:
    invalid_event_counts = frame.select(
        (
            pl.col(event_count_column).is_null()
            | ~pl.col(event_count_column).is_finite()
            | (pl.col(event_count_column) < 0)
        ).sum()
    ).item()

    if invalid_event_counts:
        raise ValueError(
            f"{split_name!r} contains "
            f"{invalid_event_counts:,} invalid event counts"
        )

    invalid_weights = frame.select(
        (
            pl.col(weight_column).is_null()
            | ~pl.col(weight_column).is_finite()
            | (pl.col(weight_column) <= 0)
        ).sum()
    ).item()

    if invalid_weights:
        raise ValueError(
            f"{split_name!r} contains "
            f"{invalid_weights:,} invalid integration weights"
        )

    total_events = float(
        frame.get_column(event_count_column).sum()
    )

    if total_events <= 0:
        raise ValueError(
            f"{split_name!r} contains no observed events"
        )


def build_category_vocabularies(
    training_frame: pl.DataFrame,
    categorical_columns: Sequence[str],
) -> dict[str, list[Any]]:
    vocabularies: dict[str, list[Any]] = {}

    for column in categorical_columns:
        categories = (
            training_frame
            .get_column(column)
            .drop_nulls()
            .unique()
            .sort()
            .to_list()
        )

        if not categories:
            raise ValueError(
                f"Categorical feature {column!r} "
                "contains no training categories"
            )

        vocabularies[column] = categories

    return vocabularies


def build_model_matrix(
    *,
    frame: pl.DataFrame,
    feature_names: Sequence[str],
    event_count_column: str,
    weight_column: str,
    category_vocabularies: Mapping[str, Sequence[Any]],
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    features = frame.select(feature_names).to_pandas()

    for column, categories in category_vocabularies.items():
        if column not in features.columns:
            continue

        category_dtype = pd.CategoricalDtype(
            categories=list(categories)
        )

        features[column] = features[column].astype(
            category_dtype
        )

    event_count = (
        frame
        .get_column(event_count_column)
        .cast(pl.Float64)
        .to_numpy()
    )

    integration_weight = (
        frame
        .get_column(weight_column)
        .cast(pl.Float64)
        .to_numpy()
    )

    # Berman–Turner quadrature representation:
    #
    #     target_i = event_count_i / integration_weight_i
    #     sample_weight_i = integration_weight_i
    #
    # This causes the weighted Poisson loss to represent:
    #
    #     weight_i * intensity_i
    #     - event_count_i * log(intensity_i)
    poisson_target = event_count / integration_weight

    return (
        features,
        poisson_target,
        integration_weight,
        event_count,
    )


def point_process_metrics(
    *,
    event_count: np.ndarray,
    integration_weight: np.ndarray,
    predicted_intensity: np.ndarray,
) -> dict[str, float | int | None]:
    intensity = np.clip(
        np.asarray(
            predicted_intensity,
            dtype=np.float64,
        ),
        1e-15,
        None,
    )

    event_count = np.asarray(
        event_count,
        dtype=np.float64,
    )

    integration_weight = np.asarray(
        integration_weight,
        dtype=np.float64,
    )

    observed_events = float(event_count.sum())

    predicted_events = float(
        np.sum(integration_weight * intensity)
    )

    point_process_nll = float(
        np.sum(
            integration_weight * intensity
            - event_count * np.log(intensity)
        )
    )

    baseline_intensity = float(
        observed_events / integration_weight.sum()
    )

    baseline_nll = float(
        np.sum(
            integration_weight * baseline_intensity
            - event_count * np.log(baseline_intensity)
        )
    )

    event_indicator = event_count > 0

    average_precision: float | None = None
    roc_auc: float | None = None

    if np.unique(event_indicator).size == 2:
        average_precision = float(
            average_precision_score(
                event_indicator,
                intensity,
            )
        )

        roc_auc = float(
            roc_auc_score(
                event_indicator,
                intensity,
            )
        )

    return {
        "rows": int(event_count.size),
        "observed_events": observed_events,
        "predicted_events": predicted_events,
        "predicted_to_observed_ratio": (
            predicted_events / observed_events
        ),
        "mean_predicted_intensity": float(
            intensity.mean()
        ),
        "baseline_intensity": baseline_intensity,
        "point_process_nll": point_process_nll,
        "baseline_point_process_nll": baseline_nll,
        "nll_improvement_over_baseline": (
            baseline_nll - point_process_nll
        ),
        "nll_per_event": (
            point_process_nll / observed_events
        ),
        "average_precision": average_precision,
        "roc_auc": roc_auc,
    }


def create_run_directory(
    *,
    experiment_name: str,
    output_config: Mapping[str, Any],
    run_name: str | None,
) -> Path:
    artifact_directory = Path(
        str(
            output_config.get(
                "artifact_dir",
                "artifacts/xgboost",
            )
        )
    )

    if not artifact_directory.is_absolute():
        artifact_directory = (
            PROJECT_ROOT / artifact_directory
        )

    timestamp = datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    resolved_run_name = (
        run_name
        or f"{experiment_name}_{timestamp}"
    )

    run_directory = (
        artifact_directory.resolve()
        / resolved_run_name
    )

    run_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    return run_directory


def save_feature_importance(
    *,
    model: XGBRegressor,
    output_path: Path,
) -> None:
    booster = model.get_booster()
    feature_names = booster.feature_names or []

    gain = booster.get_score(
        importance_type="gain"
    )
    split_count = booster.get_score(
        importance_type="weight"
    )
    cover = booster.get_score(
        importance_type="cover"
    )

    rows = [
        {
            "feature": feature,
            "gain": float(gain.get(feature, 0.0)),
            "split_count": float(
                split_count.get(feature, 0.0)
            ),
            "cover": float(
                cover.get(feature, 0.0)
            ),
        }
        for feature in feature_names
    ]

    if rows:
        (
            pl.DataFrame(rows)
            .sort("gain", descending=True)
            .write_csv(output_path)
        )
    else:
        pl.DataFrame(
            schema={
                "feature": pl.String,
                "gain": pl.Float64,
                "split_count": pl.Float64,
                "cover": pl.Float64,
            }
        ).write_csv(output_path)

def save_predictions(
    *,
    frame: pl.DataFrame,
    metadata_columns: Sequence[str],
    event_count_column: str,
    weight_column: str,
    raw_intensity: np.ndarray,
    calibrated_intensity: np.ndarray,
    output_path: Path,
) -> None:
    available_metadata = [
        column
        for column in metadata_columns
        if column in frame.columns
    ]

    prediction_frame = (
        frame
        .select(
            [
                *available_metadata,
                event_count_column,
                weight_column,
            ]
        )
        .with_columns(
            pl.Series(
                "raw_predicted_intensity",
                raw_intensity,
            ),
            pl.Series(
                "predicted_intensity",
                calibrated_intensity,
            ),
        )
        .with_columns(
            (
                pl.col(weight_column)
                * pl.col("raw_predicted_intensity")
            ).alias("raw_predicted_event_mass"),
            (
                pl.col(weight_column)
                * pl.col("predicted_intensity")
            ).alias("predicted_event_mass"),
        )
    )

    prediction_frame.write_parquet(output_path)

def train_from_config(
    *,
    config_path: Path,
    limit_per_split: int | None,
    device_override: str | None,
    run_name: str | None,
    explain: bool,
) -> Path:
    config = load_yaml_config(config_path)

    experiment_config = config["experiment"]
    data_config = config["data"]
    model_config = dict(config["model"])
    training_config = config.get("training", {})
    output_config = config.get("output", {})

    experiment_name = str(
        experiment_config["name"]
    )

    feature_set_name = str(
        experiment_config["feature_set"]
    )

    random_seed = int(
        experiment_config.get(
            "random_seed",
            42,
        )
    )

    feature_names = get_feature_set(
        feature_set_name
    )

    event_count_column = str(
        data_config["event_count_column"]
    )

    weight_column = str(
        data_config["integration_weight_column"]
    )

    categorical_columns = tuple(
        data_config.get(
            "categorical_columns",
            (),
        )
    )

    metadata_columns = tuple(
        data_config.get(
            "metadata_columns",
            (),
        )
    )

    data_glob = resolve_dataset_glob(
        data_config
    )

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Experiment:   {experiment_name}")
    print(
        f"Feature set:  {feature_set_name} "
        f"({len(feature_names)} features)"
    )
    print("Objective:    weighted Poisson")
    print(f"Data glob:    {data_glob}")

    train_frame = load_split(
        data_glob=data_glob,
        split_name="train",
        feature_names=feature_names,
        data_config=data_config,
        limit=limit_per_split,
        explain=explain,
    )

    validation_frame = load_split(
        data_glob=data_glob,
        split_name="validation",
        feature_names=feature_names,
        data_config=data_config,
        limit=limit_per_split,
        explain=explain,
    )

    test_frame = load_split(
        data_glob=data_glob,
        split_name="test",
        feature_names=feature_names,
        data_config=data_config,
        limit=limit_per_split,
        explain=explain,
    )

    for split_name, frame in (
        ("train", train_frame),
        ("validation", validation_frame),
        ("test", test_frame),
    ):
        event_count = float(
            frame.get_column(
                event_count_column
            ).sum()
        )

        total_weight = float(
            frame.get_column(
                weight_column
            ).sum()
        )

        print(
            f"{split_name:<10} "
            f"rows={frame.height:>12,} "
            f"events={event_count:>12,.0f} "
            f"exposure={total_weight:.6e} "
            f"rate={event_count / total_weight:.6e}"
        )

    category_vocabularies = (
        build_category_vocabularies(
            train_frame,
            categorical_columns,
        )
    )

    (
        train_X,
        train_y,
        train_weight,
        train_event_count,
    ) = build_model_matrix(
        frame=train_frame,
        feature_names=feature_names,
        event_count_column=event_count_column,
        weight_column=weight_column,
        category_vocabularies=category_vocabularies,
    )

    (
        validation_X,
        validation_y,
        validation_weight,
        validation_event_count,
    ) = build_model_matrix(
        frame=validation_frame,
        feature_names=feature_names,
        event_count_column=event_count_column,
        weight_column=weight_column,
        category_vocabularies=category_vocabularies,
    )

    (
        test_X,
        test_y,
        test_weight,
        test_event_count,
    ) = build_model_matrix(
        frame=test_frame,
        feature_names=feature_names,
        event_count_column=event_count_column,
        weight_column=weight_column,
        category_vocabularies=category_vocabularies,
    )

    baseline_intensity = float(
        train_event_count.sum()
        / train_weight.sum()
    )

    if device_override is not None:
        model_config["device"] = device_override

    model_config["random_state"] = random_seed
    model_config["enable_categorical"] = True
    model_config["base_score"] = max(
        baseline_intensity,
        1e-15,
    )

    print("\nTraining XGBoost")
    print(
        f"device={model_config.get('device', 'cpu')} "
        f"n_estimators="
        f"{model_config.get('n_estimators')} "
        f"base_score={baseline_intensity:.6e}"
    )

    model = XGBRegressor(
        **model_config
    )

    model.fit(
        train_X,
        train_y,
        sample_weight=train_weight,
        eval_set=[
            (train_X, train_y),
            (validation_X, validation_y),
        ],
        sample_weight_eval_set=[
            train_weight,
            validation_weight,
        ],
        verbose=int(
            training_config.get(
                "verbose_eval",
                25,
            )
        ),
    )

    validation_intensity = model.predict(
        validation_X
    )

    test_intensity = model.predict(
        test_X
    )

    validation_predicted_events = float(
        np.sum(
            validation_weight
            * validation_intensity
        )
    )

    validation_observed_events = float(
        validation_event_count.sum()
    )

    if validation_predicted_events <= 0:
        raise ValueError(
            "Validation predicted event mass must be positive"
        )

    calibration_factor = (
        validation_observed_events
        / validation_predicted_events
    )

    calibrated_validation_intensity = (
        validation_intensity
        * calibration_factor
    )

    calibrated_test_intensity = (
        test_intensity
        * calibration_factor
    )

    print(
        "\nIntensity calibration"
    )
    print(
        f"Validation-derived factor: "
        f"{calibration_factor:.9f}"
    )
    metrics = {
        "validation_raw": point_process_metrics(
            event_count=validation_event_count,
            integration_weight=validation_weight,
            predicted_intensity=validation_intensity,
        ),
        "validation_calibrated": point_process_metrics(
            event_count=validation_event_count,
            integration_weight=validation_weight,
            predicted_intensity=(
                calibrated_validation_intensity
            ),
        ),
        "test_raw": point_process_metrics(
            event_count=test_event_count,
            integration_weight=test_weight,
            predicted_intensity=test_intensity,
        ),
        "test_calibrated": point_process_metrics(
            event_count=test_event_count,
            integration_weight=test_weight,
            predicted_intensity=(
                calibrated_test_intensity
            ),
        ),
        "intensity_calibration_factor": calibration_factor,
    }

    run_directory = create_run_directory(
        experiment_name=experiment_name,
        output_config=output_config,
        run_name=run_name,
    )

    model.save_model(
        run_directory / "model.ubj"
    )

    with (
        run_directory / "config.yaml"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
        )

    with (
        run_directory / "metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
            default=json_default,
        )

    with (
        run_directory / "evals_result.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            model.evals_result(),
            file,
            indent=2,
            default=json_default,
        )

    metadata = {
        "experiment_name": experiment_name,
        "feature_set_name": feature_set_name,
        "feature_names": list(feature_names),
        "categorical_columns": list(
            categorical_columns
        ),
        "category_vocabularies": (
            category_vocabularies
        ),
        "event_count_column": (
            event_count_column
        ),
        "integration_weight_column": (
            weight_column
        ),
        "training_baseline_intensity": (
            baseline_intensity
        ),
        "intensity_calibration_factor": (
            calibration_factor
        ),
        "best_iteration": getattr(
            model,
            "best_iteration",
            None,
        ),
        "best_score": getattr(
            model,
            "best_score",
            None,
        ),
    }

    with (
        run_directory / "metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            default=json_default,
        )

    save_feature_importance(
        model=model,
        output_path=(
            run_directory
            / "feature_importance.csv"
        ),
    )

    if bool(
        output_config.get(
            "save_test_predictions",
            True,
        )
    ):
        save_predictions(
            frame=test_frame,
            metadata_columns=metadata_columns,
            event_count_column=event_count_column,
            weight_column=weight_column,
            raw_intensity=test_intensity,
            calibrated_intensity=calibrated_test_intensity,
            output_path=(
                run_directory
                / "test_predictions.parquet"
            ),
        )
    print("\nValidation metrics — raw")
    print(
        json.dumps(
            metrics["validation_raw"],
            indent=2,
        )
    )

    print("\nValidation metrics — calibrated")
    print(
        json.dumps(
            metrics["validation_calibrated"],
            indent=2,
        )
    )

    print("\nTest metrics — raw")
    print(
        json.dumps(
            metrics["test_raw"],
            indent=2,
        )
    )

    print("\nTest metrics — calibrated")
    print(
        json.dumps(
            metrics["test_calibrated"],
            indent=2,
        )
    )

    print(
        "\nBest iteration: "
        f"{getattr(model, 'best_iteration', None)}"
    )

    print(
        f"Artifacts:      {run_directory}"
    )

    return run_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the CrimeNet weighted "
            "Poisson XGBoost model"
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
    )

    parser.add_argument(
        "--run-name",
        default=None,
    )

    parser.add_argument(
        "--explain",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_from_config(
        config_path=args.config,
        limit_per_split=args.limit_per_split,
        device_override=args.device,
        run_name=args.run_name,
        explain=args.explain,
    )


if __name__ == "__main__":
    main()