from __future__ import annotations

import gc
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

from machine_learning.data.features import resolve_feature_contract
from machine_learning.data.geography import (
    deterministic_sample,
    geographic_frames,
    validate_holdout_membership,
)
from machine_learning.data.geographic_cv import (
    CANONICAL_GEOCV_VERSION,
    CANONICAL_MODELING_CITIES,
    resolve_geographic_folds,
    validate_exact_modeling_cities,
)
from machine_learning.data.metrics import geographic_point_process_metrics
from machine_learning.data.model_table import resolve_model_table_from_config
from machine_learning.data.point_process import prepare_target_exposure
from machine_learning.experiments.experiment_logging import git_commit, git_dirty


MACHINE_LEARNING_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


AUXILIARY_COLUMNS = [
    "model_row_id",
    "source_city",
    "row_type",
    "event_indicator",
    "is_observed_event",
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


def _point_process_eval_values(
    *,
    y: np.ndarray,
    exposure: np.ndarray,
    margin: np.ndarray,
    city_codes: np.ndarray,
    city_count: int,
    min_log_intensity: float,
    max_log_intensity: float,
) -> tuple[float, float]:
    """Return pooled and equal-city point-process NLL/event."""

    safe_margin = _safe_margin(
        margin,
        min_log_intensity=min_log_intensity,
        max_log_intensity=max_log_intensity,
    )
    row_nll = exposure * np.exp(safe_margin) - y * safe_margin
    observed_by_city = np.bincount(
        city_codes, weights=y, minlength=city_count
    ).astype(np.float64, copy=False)
    if city_count <= 0 or (observed_by_city <= 0).any():
        raise ValueError("Macro-city early stopping requires observed events in every city")
    nll_by_city = np.bincount(
        city_codes, weights=row_nll, minlength=city_count
    ).astype(np.float64, copy=False)
    total_observed = float(observed_by_city.sum())
    pooled = float(row_nll.sum() / total_observed)
    macro = float(np.mean(nll_by_city / observed_by_city))
    if not np.isfinite(pooled) or not np.isfinite(macro):
        raise ValueError("Point-process evaluation metric is non-finite")
    return pooled, macro


def _city_metric_index(frame: pl.DataFrame) -> tuple[np.ndarray, int]:
    cities = frame["source_city"].cast(pl.String).to_numpy()
    _, city_codes = np.unique(cities, return_inverse=True)
    return city_codes.astype(np.int64, copy=False), int(city_codes.max()) + 1


def _resolve_feature_columns(
    config: dict[str, Any],
    *,
    available_columns: list[str],
) -> tuple[list[str], list[str]]:
    contract = resolve_feature_contract(
        config["features"], available_columns=available_columns
    )
    config["features"]["resolved_numeric"] = list(contract.numeric)
    config["features"]["resolved_categorical"] = list(contract.categorical)
    config["features"]["feature_contract_hash"] = contract.contract_hash
    return list(contract.all_features), list(contract.categorical)


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

    return (
        deterministic_sample(table, fraction=fraction, seed=seed)

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
    y, exposure = prepare_target_exposure(frame)
    y = y.astype(np.float32, copy=False)

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

    table_ref = resolve_model_table_from_config(data_config)
    data_config.update(table_ref.lineage)
    data_config["test_split_used"] = False
    train_table = table_ref.scan_split(str(data_config.get("train_split", "train")))
    validation_table = table_ref.scan_split(
        str(data_config.get("validation_split", "validation"))
    )
    validation_config = config.get("validation", {})
    geocv_enabled = bool(config.get("geographic_cv", {}).get("enabled", False))
    resolved_geocv_folds = resolve_geographic_folds(config) if geocv_enabled else {}
    holdout_cities = list(validation_config.get("geographic_holdout_cities", []))
    final_training_config = config.get("final_training", {})
    is_final_production = bool(final_training_config.get("use_all_cities", False))
    if is_final_production:
        if holdout_cities:
            raise ValueError("Final production training cannot exclude geographic cities")
        if train_fraction != 1.0 or validation_fraction != 1.0:
            raise ValueError("Final production training requires full train/validation fractions")
        validate_exact_modeling_cities(
            train_table.select("source_city").unique().collect()["source_city"].to_list(),
            label="final production train split",
        )
        validate_exact_modeling_cities(
            validation_table.select("source_city").unique().collect()["source_city"].to_list(),
            label="final in-domain validation split",
        )
    train_table, validation_table, in_domain_table = geographic_frames(
        train=train_table,
        validation=validation_table,
        holdout_cities=holdout_cities,
        report_in_domain=bool(
            validation_config.get("report_in_domain_validation", True)
        ),
    )

    (
        feature_columns,
        categorical_columns,
    ) = _resolve_feature_columns(
        config,
        available_columns=train_table.collect_schema().names(),
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

    print(
        "Loading deterministic "
        "training sample..."
    )

    train_frame = (
        _deterministic_split_sample(
            table=train_table,
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
            table=validation_table,
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
    validate_holdout_membership(
        training=train_frame,
        validation=validation_frame,
        holdout_cities=holdout_cities,
        expected_modeling_cities=(CANONICAL_MODELING_CITIES if geocv_enabled else None),
    )

    in_domain_summary = None
    if in_domain_table is not None:
        in_domain_frame = _deterministic_split_sample(
            table=in_domain_table,
            split=str(data_config.get("validation_split", "validation")),
            fraction=validation_fraction,
            seed=seed,
            feature_columns=feature_columns,
        )
        if in_domain_frame.height:
            in_domain_summary = _sample_summary("in-domain validation", in_domain_frame)

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

    X_in_domain = y_in_domain = exposure_in_domain = None
    if in_domain_table is not None and "in_domain_frame" in locals() and in_domain_frame.height:
        X_in_domain, y_in_domain, exposure_in_domain, _ = _prepare_xy(
            in_domain_frame,
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
            category_levels=categories,
        )

    train_city_codes, train_city_count = _city_metric_index(train_frame)
    validation_city_codes, validation_city_count = _city_metric_index(validation_frame)
    in_domain_city_metric_index = (
        _city_metric_index(in_domain_frame)
        if in_domain_table is not None
        and "in_domain_frame" in locals()
        and in_domain_frame.height
        else None
    )

    del train_frame
    gc.collect()

    _validate_point_process_rows(
        name="train",
        y=y_train,
        exposure=exposure_train,
        event_exposure_tolerance=
            event_exposure_tolerance,
    )
    if y_in_domain is not None and exposure_in_domain is not None:
        _validate_point_process_rows(
            name="in-domain validation",
            y=y_in_domain,
            exposure=exposure_in_domain,
            event_exposure_tolerance=event_exposure_tolerance,
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

    din_domain = None
    if X_in_domain is not None and y_in_domain is not None:
        din_domain = xgb.QuantileDMatrix(
            X_in_domain,
            label=y_in_domain,
            base_margin=np.full(y_in_domain.shape, initial_log_lambda, dtype=np.float32),
            enable_categorical=True,
            max_bin=max_bin,
            ref=dtrain,
            nthread=-1,
        )

    del X_train
    del X_validation
    if X_in_domain is not None:
        del X_in_domain

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
    if din_domain is not None and exposure_in_domain is not None:
        exposure_lookup[id(din_domain)] = exposure_in_domain

    city_metric_lookup = {
        id(dtrain): (train_city_codes, train_city_count),
        id(dvalidation): (validation_city_codes, validation_city_count),
    }
    if din_domain is not None and in_domain_city_metric_index is not None:
        city_metric_lookup[id(din_domain)] = in_domain_city_metric_index

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

        city_codes, city_count = city_metric_lookup[id(dmatrix)]
        pooled_nll, macro_city_nll = _point_process_eval_values(
            y=y,
            exposure=exposure,
            margin=predt,
            city_codes=city_codes,
            city_count=city_count,
            min_log_intensity=min_log_intensity,
            max_log_intensity=max_log_intensity,
        )

        # XGBoost early stopping uses the final metric on the final eval set.
        # Preserve pooled NLL for diagnostics and place equal-city macro NLL last.
        return [
            ("pp_nll_per_event", pooled_nll),
            ("macro_city_pp_nll_per_event", macro_city_nll),
        ]

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

        "gamma": float(optimization_config.get("gamma", 0.0)),

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

    fixed_rounds = bool(training_config.get("fixed_num_boost_round", False))
    early_stopping_callbacks = (
        []
        if fixed_rounds
        else [
            xgb.callback.EarlyStopping(
                rounds=int(training_config["early_stopping_rounds"]),
                metric_name="macro_city_pp_nll_per_event",
                data_name="validation",
                maximize=False,
                save_best=False,
            )
        ]
    )
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

        early_stopping_rounds=None,

        callbacks=early_stopping_callbacks,

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

    if best_iteration is not None and not fixed_rounds:
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

    validation_margin_for_geo = booster.predict(
        dvalidation, output_margin=True
    )
    geographic_metrics = geographic_point_process_metrics(
        validation_frame,
        log_intensity=validation_margin_for_geo,
        constant_log_intensity=initial_log_lambda,
        min_log_intensity=min_log_intensity,
        max_log_intensity=max_log_intensity,
    )
    per_city_path = artifact_dir / (
        "final_in_domain_temporal_validation_by_city.csv"
        if is_final_production
        else "geographic_validation_by_city.csv"
    )
    pl.DataFrame(geographic_metrics["per_city"]).write_csv(per_city_path)
    in_domain_metrics = None
    in_domain_city_path = None
    if (
        din_domain is not None
        and exposure_in_domain is not None
        and "in_domain_frame" in locals()
    ):
        in_domain_margin = booster.predict(din_domain, output_margin=True)
        in_domain_metrics = geographic_point_process_metrics(
            in_domain_frame,
            log_intensity=in_domain_margin,
            constant_log_intensity=initial_log_lambda,
            min_log_intensity=min_log_intensity,
            max_log_intensity=max_log_intensity,
        )
        in_domain_city_path = artifact_dir / "validation_in_domain_by_city.csv"
        pl.DataFrame(in_domain_metrics["per_city"]).write_csv(in_domain_city_path)
    del validation_frame
    if in_domain_table is not None and "in_domain_frame" in locals():
        del in_domain_frame

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

        "git_commit": git_commit(),

        "git_dirty": git_dirty(),

        "runtime_versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "polars": pl.__version__,
            "xgboost": xgb.__version__,
        },

        **table_ref.lineage,

        "feature_contract_hash": config["features"]["feature_contract_hash"],

        "geographic_holdout_cities": sorted(set(holdout_cities)),

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

        "geographic_validation": (
            None if is_final_production else geographic_metrics
        ),

        "geographic_oof_validation": config.get("hpo", {}).get(
            "geographic_oof_validation"
        ),

        "final_in_domain_temporal_validation": (
            geographic_metrics if is_final_production else None
        ),

        "in_domain_summary": in_domain_summary,

        "in_domain_validation": in_domain_metrics,

        "training_strategy": (
            "final_all_city_train" if is_final_production else "geographic_holdout_fit"
        ),

        "geographic_cv": {
            "enabled": geocv_enabled,
            "fold_version": (
                CANONICAL_GEOCV_VERSION if geocv_enabled else None
            ),
            "fold_count": len(resolved_geocv_folds),
            "folds": (
                {
                    name: list(cities)
                    for name, cities in resolved_geocv_folds.items()
                }
                if geocv_enabled
                else {}
            ),
            "primary_metric": "geocv_macro_nll_per_event",
        },

        "final_training": {
            "train_split": str(data_config.get("train_split", "train")),
            "city_count": len(CANONICAL_MODELING_CITIES) if is_final_production else None,
            "excluded_cities": [] if is_final_production else sorted(set(holdout_cities)),
            "train_fraction": train_fraction,
        },

        "validation_strategy": {
            "selection_metric_source": "geographic_oof_cv",
            "final_diagnostic_source": "full_validation_split_in_domain",
        },

        "validation": {
            "selection_metric_source": "geographic_oof_cv",
            "final_diagnostic_source": "full_validation_split_in_domain",
        },

        "hyperparameters": {
            "architecture": architecture_config,
            "optimization": optimization_config,
            "num_boost_round": int(training_config["num_boost_round"]),
            "fixed_num_boost_round": fixed_rounds,
            "early_stopping_metric": (
                None
                if fixed_rounds
                else "validation-macro_city_pp_nll_per_event"
            ),
        },

        "hpo": config.get("hpo"),
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

        "geographic_macro_nll_per_event": float(
            geographic_metrics["macro_city"]["mean_nll_per_event"]
        ),

        "geographic_macro_bits_per_event": float(
            geographic_metrics["macro_city"]["mean_bits_per_event"]
        ),

        "geographic_worst_city_bits_per_event": float(
            geographic_metrics["macro_city"]["worst_city_bits_per_event"]
        ),

        "geographic_mean_abs_calibration_error_pct": float(
            geographic_metrics["macro_city"]["mean_absolute_calibration_error_pct"]
        ),

        "best_iteration":
            float(
                best_iteration
                if best_iteration
                is not None
                else -1
            ),
    }

    if in_domain_metrics is not None:
        metrics["in_domain_macro_nll_per_event"] = float(
            in_domain_metrics["macro_city"]["mean_nll_per_event"]
        )

    return {
        "metrics":
            metrics,

        "geographic_validation": (
            None if is_final_production else geographic_metrics
        ),

        "final_in_domain_temporal_validation": (
            geographic_metrics if is_final_production else None
        ),

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
            str(per_city_path),
        ] + ([str(in_domain_city_path)] if in_domain_city_path is not None else []),

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
