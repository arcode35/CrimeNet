from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

from crimenet_data.assets.model_table.transformations import (
    CONTEXT_FEATURE_COLUMNS,
    HISTORY_FEATURE_COLUMNS,
    LIGHTING_FEATURE_COLUMNS,
)


MACHINE_LEARNING_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


CALENDAR_FEATURE_COLUMNS = [
    "local_hour",
    "local_day_of_week",
    "local_hour_sin",
    "local_hour_cos",
    "local_day_of_week_sin",
    "local_day_of_week_cos",
]

AUXILIARY_COLUMNS = [
    "event_count",
    "integration_weight_cell_seconds",
]


def _safe_margin(
    x: np.ndarray,
    *,
    min_log_intensity: float,
    max_log_intensity: float,
) -> np.ndarray:
    return np.clip(
        np.asarray(
            x,
            dtype=np.float64,
        ),
        min_log_intensity,
        max_log_intensity,
    )


def _resolve_feature_columns(
    config: dict[str, Any],
) -> tuple[list[str], list[str]]:
    feature_config = config["features"]

    feature_columns: list[str] = []

    if feature_config.get(
        "include_source_city",
        True,
    ):
        feature_columns.append(
            "source_city"
        )

    if feature_config.get(
        "include_calendar",
        True,
    ):
        feature_columns.extend(
            CALENDAR_FEATURE_COLUMNS
        )

    if feature_config.get(
        "include_context",
        True,
    ):
        feature_columns.extend(
            CONTEXT_FEATURE_COLUMNS
        )

    if feature_config.get(
        "include_lighting",
        True,
    ):
        feature_columns.extend(
            LIGHTING_FEATURE_COLUMNS
        )

    if feature_config.get(
        "include_history",
        True,
    ):
        feature_columns.extend(
            HISTORY_FEATURE_COLUMNS
        )

    # Preserve order while removing accidental duplicates.
    feature_columns = list(
        dict.fromkeys(
            feature_columns
        )
    )

    categorical_columns = [
        column
        for column in (
            "source_city",
            "lighting_condition",
        )
        if column in feature_columns
    ]

    return (
        feature_columns,
        categorical_columns,
    )


def _deterministic_split_sample(
    *,
    table: pl.LazyFrame,
    split: str,
    fraction: float,
    seed: int,
    feature_columns: list[str],
) -> pl.DataFrame:
    if not (
        0.0
        < fraction
        <= 1.0
    ):
        raise ValueError(
            f"Sampling fraction must be in (0, 1], "
            f"got {fraction}."
        )

    buckets = 1_000_000

    threshold = int(
        fraction
        * buckets
    )

    return (
        table

        .filter(
            pl.col("split")
            == split
        )

        .filter(
            (
                pl.col("model_row_id")
                .hash(seed=seed)
                % buckets
            )
            < threshold
        )

        .select(
            [
                *feature_columns,
                *AUXILIARY_COLUMNS,
            ]
        )

        .collect()
    )


def _sample_summary(
    name: str,
    frame: pl.DataFrame,
) -> dict[str, float]:
    rows = int(
        frame.height
    )

    observed_events = float(
        frame
        .get_column(
            "event_count"
        )
        .sum()
    )

    integration_rows = int(
        (
            frame
            .get_column(
                "event_count"
            )
            == 0
        )
        .sum()
    )

    integration_weight = float(
        frame
        .get_column(
            "integration_weight_cell_seconds"
        )
        .sum()
    )

    print(
        f"\n{name.upper()}"
    )

    print(
        f"Rows:               "
        f"{rows:,}"
    )

    print(
        f"Observed events:    "
        f"{observed_events:,.0f}"
    )

    print(
        f"Integration rows:   "
        f"{integration_rows:,}"
    )

    print(
        f"Integration weight: "
        f"{integration_weight:.6e}"
    )

    return {
        "rows":
            float(rows),

        "observed_events":
            observed_events,

        "integration_rows":
            float(
                integration_rows
            ),

        "integration_weight":
            integration_weight,
    }


def _prepare_xy(
    frame: pl.DataFrame,
    *,
    feature_columns: list[str],
    categorical_columns: list[str],
    category_levels:
        dict[str, list[str]]
        | None = None,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    dict[str, list[str]],
]:
    y = (
        frame
        .get_column(
            "event_count"
        )
        .to_numpy()
        .astype(
            np.float32,
            copy=False,
        )
    )

    exposure = (
        frame
        .get_column(
            "integration_weight_cell_seconds"
        )
        .to_numpy()
        .astype(
            np.float64,
            copy=False,
        )
    )

    X = (
        frame
        .select(
            feature_columns
        )
        .to_pandas()
    )

    learned_categories: dict[
        str,
        list[str],
    ] = {}

    for column in categorical_columns:
        if category_levels is None:
            categories = sorted(
                X[column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        else:
            categories = (
                category_levels[
                    column
                ]
            )

        learned_categories[
            column
        ] = categories

        X[column] = pd.Categorical(
            X[column],
            categories=categories,
        )

    for column in feature_columns:
        if column in categorical_columns:
            continue

        X[column] = (
            pd.to_numeric(
                X[column],
                errors="coerce",
            )
            .astype(
                np.float32
            )
        )

    return (
        X,
        y,
        exposure,
        learned_categories,
    )


def _validate_point_process_rows(
    *,
    name: str,
    y: np.ndarray,
    exposure: np.ndarray,
    event_exposure_tolerance: float,
) -> None:
    if not np.isfinite(
        exposure
    ).all():
        raise ValueError(
            f"{name}: exposure contains "
            f"NaN or infinity."
        )

    if not (
        exposure >= 0
    ).all():
        raise ValueError(
            f"{name}: exposure contains "
            f"negative values."
        )

    if not np.isfinite(
        y
    ).all():
        raise ValueError(
            f"{name}: event_count contains "
            f"NaN or infinity."
        )

    if not set(
        np.unique(y)
    ).issubset(
        {0.0, 1.0}
    ):
        raise ValueError(
            f"{name}: event_count is not binary."
        )

    event_mask = (
        y == 1
    )

    integration_mask = (
        y == 0
    )

    if (
        event_mask.any()
        and float(
            exposure[
                event_mask
            ].max()
        )
        > event_exposure_tolerance
    ):
        raise ValueError(
            f"{name}: observed-event rows "
            f"contain non-zero exposure."
        )

    if (
        integration_mask.any()
        and not (
            exposure[
                integration_mask
            ]
            > 0
        ).all()
    ):
        raise ValueError(
            f"{name}: integration rows "
            f"contain zero or negative exposure."
        )


def _constant_nll_per_event(
    *,
    y: np.ndarray,
    exposure: np.ndarray,
    log_lambda: float,
) -> float:
    nll = np.sum(
        exposure
        * np.exp(
            log_lambda
        )
        -
        y
        * log_lambda
    )

    return float(
        nll
        /
        max(
            float(
                y.sum()
            ),
            1.0,
        )
    )


def _evaluate_split(
    *,
    name: str,
    booster: xgb.Booster,
    dmatrix: xgb.DMatrix,
    exposure: np.ndarray,
    min_log_intensity: float,
    max_log_intensity: float,
) -> dict[str, float]:
    y = (
        dmatrix
        .get_label()
        .astype(
            np.float64,
            copy=False,
        )
    )

    margin = booster.predict(
        dmatrix,
        output_margin=True,
    )

    margin = _safe_margin(
        margin,
        min_log_intensity=
            min_log_intensity,
        max_log_intensity=
            max_log_intensity,
    )

    intensity = np.exp(
        margin
    )

    expected_events = float(
        np.sum(
            exposure
            * intensity
        )
    )

    observed_events = float(
        y.sum()
    )

    nll = float(
        np.sum(
            exposure
            * intensity
            -
            y
            * margin
        )
    )

    nll_per_event = (
        nll
        /
        max(
            observed_events,
            1.0,
        )
    )

    expected_observed_ratio = (
        expected_events
        /
        max(
            observed_events,
            1.0,
        )
    )

    calibration_error_pct = (
        (
            expected_observed_ratio
            - 1.0
        )
        * 100.0
    )

    mean_log_lambda = float(
        margin.mean()
    )

    print(
        f"\n{name.upper()}"
    )

    print(
        f"Observed events:  "
        f"{observed_events:,.0f}"
    )

    print(
        f"Expected events:  "
        f"{expected_events:,.2f}"
    )

    print(
        f"Expected/actual:  "
        f"{expected_observed_ratio:.4f}"
    )

    print(
        f"NLL/event:        "
        f"{nll_per_event:.6f}"
    )

    print(
        f"Mean log lambda:  "
        f"{mean_log_lambda:.6f}"
    )

    return {
        "observed_events":
            observed_events,

        "expected_events":
            expected_events,

        "expected_observed_ratio":
            expected_observed_ratio,

        "calibration_error_pct":
            calibration_error_pct,

        "nll":
            nll,

        "nll_per_event":
            nll_per_event,

        "mean_log_lambda":
            mean_log_lambda,
    }


def train(
    config: dict[str, Any],
    *,
    run_id: str,
    config_hash: str,
) -> dict[str, Any]:
    """Train one XGBoost CrimeNet experiment.

    Model-specific code lives here. This function does not create or
    manipulate MLflow runs. It returns plain metrics, history, artifact
    paths, and a summary for the generic experiment orchestrator.
    """

    model_config = config["model"]
    data_config = config["data"]
    architecture_config = config[
        "architecture"
    ]
    optimization_config = config[
        "optimization"
    ]
    training_config = config["training"]
    numerics_config = config["numerics"]
    artifact_config = config["artifacts"]

    seed = int(
        data_config["seed"]
    )

    train_fraction = float(
        data_config[
            "train_fraction"
        ]
    )

    validation_fraction = float(
        data_config[
            "validation_fraction"
        ]
    )

    max_bin = int(
        architecture_config[
            "max_bin"
        ]
    )

    min_log_intensity = float(
        numerics_config[
            "min_log_intensity"
        ]
    )

    max_log_intensity = float(
        numerics_config[
            "max_log_intensity"
        ]
    )

    hessian_floor = float(
        numerics_config[
            "hessian_floor"
        ]
    )

    event_exposure_tolerance = float(
        numerics_config.get(
            "event_exposure_tolerance",
            1e-12,
        )
    )

    (
        feature_columns,
        categorical_columns,
    ) = _resolve_feature_columns(
        config
    )

    if not feature_columns:
        raise ValueError(
            "Resolved feature set is empty."
        )

    artifact_dir = (
        MACHINE_LEARNING_ROOT
        / artifact_config[
            "output_root"
        ]
        / model_config["name"]
        / run_id
    )

    artifact_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    model_path = (
        artifact_dir
        / "model.json"
    )

    metadata_path = (
        artifact_dir
        / "metadata.json"
    )

    importance_path = (
        artifact_dir
        / "feature_importance.json"
    )

    training_history_path = (
        artifact_dir
        / "training_history.json"
    )

    print(
        f"Experiment artifacts: "
        f"{artifact_dir}"
    )

    credentials = (
        pl.CredentialProviderGCP()
    )

    table = pl.scan_delta(
        data_config[
            "model_table_root"
        ],
        credential_provider=
            credentials,
    )

    print(
        "Loading deterministic "
        "training sample..."
    )

    train_frame = (
        _deterministic_split_sample(
            table=table,
            split=data_config[
                "train_split"
            ],
            fraction=train_fraction,
            seed=seed,
            feature_columns=
                feature_columns,
        )
    )

    print(
        "Loading deterministic "
        "validation sample..."
    )

    validation_frame = (
        _deterministic_split_sample(
            table=table,
            split=data_config[
                "validation_split"
            ],
            fraction=
                validation_fraction,
            seed=seed,
            feature_columns=
                feature_columns,
        )
    )

    train_summary = _sample_summary(
        "train",
        train_frame,
    )

    validation_summary = (
        _sample_summary(
            "validation",
            validation_frame,
        )
    )

    (
        X_train,
        y_train,
        exposure_train,
        categories,
    ) = _prepare_xy(
        train_frame,
        feature_columns=
            feature_columns,
        categorical_columns=
            categorical_columns,
    )

    (
        X_validation,
        y_validation,
        exposure_validation,
        _,
    ) = _prepare_xy(
        validation_frame,
        feature_columns=
            feature_columns,
        categorical_columns=
            categorical_columns,
        category_levels=
            categories,
    )

    del train_frame
    del validation_frame

    gc.collect()

    _validate_point_process_rows(
        name="train",
        y=y_train,
        exposure=exposure_train,
        event_exposure_tolerance=
            event_exposure_tolerance,
    )

    _validate_point_process_rows(
        name="validation",
        y=y_validation,
        exposure=
            exposure_validation,
        event_exposure_tolerance=
            event_exposure_tolerance,
    )

    train_events = float(
        y_train.sum()
    )

    train_exposure = float(
        exposure_train.sum()
    )

    if (
        train_events <= 0
        or train_exposure <= 0
    ):
        raise ValueError(
            "Training sample must contain "
            "positive events and exposure."
        )

    initial_lambda = (
        train_events
        /
        train_exposure
    )

    initial_log_lambda = float(
        np.log(
            initial_lambda
        )
    )

    print(
        "\nInitial constant intensity"
    )

    print(
        f"events:       "
        f"{train_events:,.0f}"
    )

    print(
        f"exposure:     "
        f"{train_exposure:,.3e}"
    )

    print(
        f"lambda0:      "
        f"{initial_lambda:.8e}"
    )

    print(
        f"log(lambda0): "
        f"{initial_log_lambda:.6f}"
    )

    train_margin = np.full(
        y_train.shape,
        initial_log_lambda,
        dtype=np.float32,
    )

    validation_margin = np.full(
        y_validation.shape,
        initial_log_lambda,
        dtype=np.float32,
    )

    dtrain = xgb.QuantileDMatrix(
        X_train,
        label=y_train,
        base_margin=train_margin,
        enable_categorical=True,
        max_bin=max_bin,
        nthread=-1,
    )

    dvalidation = (
        xgb.QuantileDMatrix(
            X_validation,
            label=y_validation,
            base_margin=
                validation_margin,
            enable_categorical=True,
            max_bin=max_bin,
            ref=dtrain,
            nthread=-1,
        )
    )

    del X_train
    del X_validation

    gc.collect()

    def point_process_poisson_objective(
        predt: np.ndarray,
        dtrain_: xgb.DMatrix,
    ):
        y = (
            dtrain_
            .get_label()
            .astype(
                np.float64,
                copy=False,
            )
        )

        f_safe = _safe_margin(
            predt,
            min_log_intensity=
                min_log_intensity,
            max_log_intensity=
                max_log_intensity,
        )

        integrated_intensity = (
            exposure_train
            * np.exp(
                f_safe
            )
        )

        grad = (
            integrated_intensity
            - y
        )

        hess = np.maximum(
            integrated_intensity,
            hessian_floor,
        )

        return (
            grad.astype(
                np.float32
            ),
            hess.astype(
                np.float32
            ),
        )

    exposure_lookup = {
        id(dtrain):
            exposure_train,

        id(dvalidation):
            exposure_validation,
    }

    def point_process_nll(
        predt: np.ndarray,
        dmatrix: xgb.DMatrix,
    ):
        y = (
            dmatrix
            .get_label()
            .astype(
                np.float64,
                copy=False,
            )
        )

        exposure = (
            exposure_lookup[
                id(dmatrix)
            ]
        )

        f_safe = _safe_margin(
            predt,
            min_log_intensity=
                min_log_intensity,
            max_log_intensity=
                max_log_intensity,
        )

        nll = np.sum(
            exposure
            * np.exp(
                f_safe
            )
            -
            y
            * f_safe
        )

        return (
            "pp_nll_per_event",
            float(
                nll
                /
                max(
                    float(
                        y.sum()
                    ),
                    1.0,
                )
            ),
        )

    params = {
        "tree_method":
            architecture_config[
                "tree_method"
            ],

        "device":
            architecture_config[
                "device"
            ],

        "max_bin":
            max_bin,

        "max_depth":
            int(
                architecture_config[
                    "max_depth"
                ]
            ),

        "eta":
            float(
                optimization_config[
                    "learning_rate"
                ]
            ),

        "subsample":
            float(
                optimization_config[
                    "subsample"
                ]
            ),

        "colsample_bytree":
            float(
                optimization_config[
                    "colsample_bytree"
                ]
            ),

        "min_child_weight":
            float(
                optimization_config[
                    "min_child_weight"
                ]
            ),

        "max_delta_step":
            float(
                optimization_config[
                    "max_delta_step"
                ]
            ),

        "reg_lambda":
            float(
                optimization_config[
                    "reg_lambda"
                ]
            ),

        "reg_alpha":
            float(
                optimization_config[
                    "reg_alpha"
                ]
            ),

        "max_cat_to_onehot":
            int(
                architecture_config[
                    "max_cat_to_onehot"
                ]
            ),

        "seed":
            seed,

        "nthread":
            -1,

        "disable_default_eval_metric":
            True,
    }

    constant_train_nll = (
        _constant_nll_per_event(
            y=y_train,
            exposure=exposure_train,
            log_lambda=
                initial_log_lambda,
        )
    )

    constant_validation_nll = (
        _constant_nll_per_event(
            y=y_validation,
            exposure=
                exposure_validation,
            log_lambda=
                initial_log_lambda,
        )
    )

    print(
        "Constant train NLL/event:",
        constant_train_nll,
    )

    print(
        "Constant validation NLL/event:",
        constant_validation_nll,
    )

    print(
        "\nTraining XGBoost "
        "Poisson point-process baseline...\n"
    )

    evals_result: dict[
        str,
        dict[
            str,
            list[float],
        ],
    ] = {}

    booster = xgb.train(
        params=params,

        dtrain=dtrain,

        num_boost_round=int(
            training_config[
                "num_boost_round"
            ]
        ),

        evals=[
            (
                dtrain,
                "train",
            ),
            (
                dvalidation,
                "validation",
            ),
        ],

        obj=
            point_process_poisson_objective,

        custom_metric=
            point_process_nll,

        early_stopping_rounds=int(
            training_config[
                "early_stopping_rounds"
            ]
        ),

        evals_result=
            evals_result,

        verbose_eval=
            training_config[
                "verbose_eval"
            ],
    )

    best_iteration = getattr(
        booster,
        "best_iteration",
        None,
    )

    if best_iteration is not None:
        booster = booster[
            : best_iteration + 1
        ]

    train_metrics = _evaluate_split(
        name="train",
        booster=booster,
        dmatrix=dtrain,
        exposure=exposure_train,
        min_log_intensity=
            min_log_intensity,
        max_log_intensity=
            max_log_intensity,
    )

    validation_metrics = (
        _evaluate_split(
            name="validation",
            booster=booster,
            dmatrix=dvalidation,
            exposure=
                exposure_validation,
            min_log_intensity=
                min_log_intensity,
            max_log_intensity=
                max_log_intensity,
        )
    )

    train_nll_gain = (
        constant_train_nll
        -
        train_metrics[
            "nll_per_event"
        ]
    )

    validation_nll_gain = (
        constant_validation_nll
        -
        validation_metrics[
            "nll_per_event"
        ]
    )

    train_bits_per_event = (
        train_nll_gain
        /
        np.log(2.0)
    )

    validation_bits_per_event = (
        validation_nll_gain
        /
        np.log(2.0)
    )

    importance = {
        "gain":
            booster.get_score(
                importance_type="gain"
            ),

        "weight":
            booster.get_score(
                importance_type="weight"
            ),

        "cover":
            booster.get_score(
                importance_type="cover"
            ),
    }

    print(
        "\nTOP 30 FEATURES BY GAIN"
    )

    sorted_gain = sorted(
        importance[
            "gain"
        ].items(),
        key=lambda item:
            item[1],
        reverse=True,
    )

    for (
        feature,
        gain,
    ) in sorted_gain[:30]:
        print(
            f"{feature:<50} "
            f"{gain:,.6f}"
        )

    booster.save_model(
        model_path
    )

    metadata = {
        "model_name":
            model_config[
                "name"
            ],

        "run_id":
            run_id,

        "config_hash":
            config_hash,

        "model_table_root":
            data_config[
                "model_table_root"
            ],

        "train_fraction":
            train_fraction,

        "validation_fraction":
            validation_fraction,

        "seed":
            seed,

        "features":
            feature_columns,

        "feature_set":
            config[
                "features"
            ].get(
                "feature_set"
            ),

        "categorical_columns":
            categorical_columns,

        "category_levels":
            categories,

        "initial_lambda":
            initial_lambda,

        "initial_log_lambda":
            initial_log_lambda,

        "best_iteration":
            best_iteration,

        "test_split_used":
            False,
    }

    with metadata_path.open(
        "w"
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    with importance_path.open(
        "w"
    ) as file:
        json.dump(
            importance,
            file,
            indent=2,
        )

    with training_history_path.open(
        "w"
    ) as file:
        json.dump(
            evals_result,
            file,
            indent=2,
        )

    print(
        f"\nSaved model: "
        f"{model_path}"
    )

    print(
        "TEST SPLIT HAS NOT BEEN ACCESSED."
    )

    metrics = {
        "sample_train_rows":
            train_summary[
                "rows"
            ],

        "sample_validation_rows":
            validation_summary[
                "rows"
            ],

        "sample_train_events":
            train_summary[
                "observed_events"
            ],

        "sample_validation_events":
            validation_summary[
                "observed_events"
            ],

        "sample_train_exposure":
            train_summary[
                "integration_weight"
            ],

        "sample_validation_exposure":
            validation_summary[
                "integration_weight"
            ],

        "initial_lambda":
            initial_lambda,

        "initial_log_lambda":
            initial_log_lambda,

        "constant_train_nll_per_event":
            constant_train_nll,

        "constant_validation_nll_per_event":
            constant_validation_nll,

        "sample_train_nll_per_event":
            train_metrics[
                "nll_per_event"
            ],

        "sample_validation_nll_per_event":
            validation_metrics[
                "nll_per_event"
            ],

        "sample_train_expected_observed":
            train_metrics[
                "expected_observed_ratio"
            ],

        "sample_validation_expected_observed":
            validation_metrics[
                "expected_observed_ratio"
            ],

        "sample_train_calibration_error_pct":
            train_metrics[
                "calibration_error_pct"
            ],

        "sample_validation_calibration_error_pct":
            validation_metrics[
                "calibration_error_pct"
            ],

        "sample_train_nll_gain_per_event":
            train_nll_gain,

        "sample_validation_nll_gain_per_event":
            validation_nll_gain,

        "sample_train_bits_per_event":
            train_bits_per_event,

        "sample_validation_bits_per_event":
            validation_bits_per_event,

        "best_iteration":
            float(
                best_iteration
                if best_iteration
                is not None
                else -1
            ),
    }

    return {
        "metrics":
            metrics,

        "history":
            evals_result,

        "artifacts": [
            str(
                model_path
            ),
            str(
                metadata_path
            ),
            str(
                importance_path
            ),
            str(
                training_history_path
            ),
        ],

        "summary": {
            "best_iteration":
                best_iteration,

            "feature_count":
                len(
                    feature_columns
                ),

            "artifact_dir":
                str(
                    artifact_dir
                ),

            "test_split_used":
                False,
        },
    }
