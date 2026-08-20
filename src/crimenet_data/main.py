from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

from crimenet_data.assets.model_table.transformations import (
    CONTEXT_FEATURE_COLUMNS,
    HISTORY_FEATURE_COLUMNS,
    LIGHTING_FEATURE_COLUMNS,
)


# ============================================================
# Configuration
# ============================================================

MODEL_TABLE_ROOT = (
    "gs://crimenet/gold_staging_/model_table_nyc_timestamp_fix"
)

OUTPUT_DIR = Path("artifacts/xgb_poisson_local")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SEED = 42

# Same sampling probability for observed + integration
# rows within each split. This preserves the point-process
# objective up to a constant factor.
TRAIN_FRACTION = 0.05
VALIDATION_FRACTION = 0.10

MAX_BIN = 256

# ============================================================
# MLflow
# ============================================================

MLFLOW_EXPERIMENT_NAME = "crimenet-xgb-point-process"
MLFLOW_RUN_NAME = "xgb_pp_baseline_v1"

# Override this with the MLFLOW_TRACKING_URI environment
# variable when you move tracking to a remote server.
LOCAL_MLFLOW_DIR = Path("mlruns").resolve()
LOCAL_MLFLOW_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    LOCAL_MLFLOW_DIR.as_uri(),
)

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

mlflow.set_experiment(
    MLFLOW_EXPERIMENT_NAME
)

# Let MLflow capture native xgb.train parameters and the
# per-boosting-round metrics, but DO NOT let autologging save
# the model or feature importance. xgb.train returns the
# post-early-stopping booster, while below we explicitly slice
# to best_iteration before logging the canonical model.
mlflow.xgboost.autolog(
    log_models=False,
    log_datasets=False,
    importance_types=[],
    exclusive=False,
)


# ============================================================
# Features
# ============================================================

CALENDAR_FEATURE_COLUMNS = [
    "local_hour",
    "local_day_of_week",
    "local_hour_sin",
    "local_hour_cos",
    "local_day_of_week_sin",
    "local_day_of_week_cos",
]


# Add source_city to the 62 feature columns in the model table.
#
# This gives 63 model inputs.
#
# Deliberately excluded:
# - row_type
# - is_observed_event
# - event_count
# - integration_weight_cell_seconds
# - timestamps
# - H3 IDs
# - offense marks
#
# Those would either reveal the target/control role of a row,
# create leakage, or encourage high-cardinality memorization.
FEATURE_COLUMNS = [
    "source_city",
    *CALENDAR_FEATURE_COLUMNS,
    *CONTEXT_FEATURE_COLUMNS,
    *LIGHTING_FEATURE_COLUMNS,
    *HISTORY_FEATURE_COLUMNS,
]


CATEGORICAL_COLUMNS = [
    "source_city",
    "lighting_condition",
]


AUXILIARY_COLUMNS = [
    "event_count",
    "integration_weight_cell_seconds",
]

MIN_LOG_INTENSITY = -30.0
MAX_LOG_INTENSITY = 15.0

def safe_margin(x):
    return np.clip(
        np.asarray(x, dtype=np.float64),
        MIN_LOG_INTENSITY,
        MAX_LOG_INTENSITY,
    )

# ============================================================
# Read model table
# ============================================================

credentials = (
    pl.CredentialProviderGCP()
)

table = pl.scan_delta(
    MODEL_TABLE_ROOT,
    credential_provider=credentials,
)


# ============================================================
# Deterministic sampling
# ============================================================

def deterministic_split_sample(
    split: str,
    fraction: float,
) -> pl.DataFrame:

    buckets = 1_000_000

    threshold = int(
        fraction * buckets
    )

    result = (
        table

        .filter(
            pl.col("split")
            == split
        )

        .filter(
            (
                pl.col("model_row_id")
                .hash(seed=SEED)
                % buckets
            )
            < threshold
        )

        .select(
            [
                *FEATURE_COLUMNS,
                *AUXILIARY_COLUMNS,
            ]
        )

        .collect()
    )

    return result


print(
    "Loading deterministic "
    "training sample..."
)

train = deterministic_split_sample(
    "train",
    TRAIN_FRACTION,
)

print(
    "Loading deterministic "
    "validation sample..."
)

validation = deterministic_split_sample(
    "validation",
    VALIDATION_FRACTION,
)


# ============================================================
# Basic sample diagnostics
# ============================================================

def sample_summary(
    name: str,
    frame: pl.DataFrame,
) -> None:

    summary = (
        frame
        .select(
            pl.len()
            .alias("rows"),

            pl.col("event_count")
            .sum()
            .alias("observed_events"),

            (
                pl.col("event_count")
                == 0
            )
            .sum()
            .alias(
                "integration_rows"
            ),

            pl.col(
                "integration_weight_cell_seconds"
            )
            .sum()
            .alias(
                "integration_weight"
            ),
        )
    )

    print(
        f"\n{name.upper()}"
    )

    print(summary)


sample_summary(
    "train",
    train,
)

sample_summary(
    "validation",
    validation,
)


# ============================================================
# Convert Polars -> pandas for XGBoost
# ============================================================

def prepare_xy(
    frame: pl.DataFrame,
    *,
    category_levels:
        dict[str, list[str]]
        | None = None,
):
    y = (
        frame
        .get_column("event_count")
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
            FEATURE_COLUMNS
        )
        .to_pandas()
    )

    # --------------------------------------------
    # Categorical encoding.
    # --------------------------------------------

    learned_categories = {}

    for column in CATEGORICAL_COLUMNS:

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
                category_levels[column]
            )

        learned_categories[
            column
        ] = categories

        X[column] = pd.Categorical(
            X[column],
            categories=categories,
        )

    # --------------------------------------------
    # Use float32 for all numerical features.
    #
    # XGBoost handles NaN as missing values, so
    # DO NOT globally impute from train+validation.
    # --------------------------------------------

    for column in FEATURE_COLUMNS:

        if (
            column
            in CATEGORICAL_COLUMNS
        ):
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


(
    X_train,
    y_train,
    exposure_train,
    categories,
) = prepare_xy(
    train
)

(
    X_validation,
    y_validation,
    exposure_validation,
    _,
) = prepare_xy(
    validation,
    category_levels=categories,
)


# Free the Polars materializations.
del train
del validation

gc.collect()


# ============================================================
# Sanity checks
# ============================================================

assert (
    np.isfinite(
        exposure_train
    )
    .all()
)

assert (
    np.isfinite(
        exposure_validation
    )
    .all()
)

assert (
    exposure_train
    >= 0
).all()

assert (
    exposure_validation
    >= 0
).all()

assert set(
    np.unique(y_train)
).issubset(
    {0.0, 1.0}
)

assert set(
    np.unique(y_validation)
).issubset(
    {0.0, 1.0}
)


# ============================================================
# Initial constant log intensity
# ============================================================

#
# If the model predicted one constant intensity:
#
#     lambda = N_events / total_exposure
#
# This is the MLE.
#

train_events = float(
    y_train.sum()
)

train_exposure = float(
    exposure_train.sum()
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


# ============================================================
# Quantile DMatrix
# ============================================================

#
# base_margin is the raw score.
#
# Here it is log(lambda0).
#

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
    max_bin=MAX_BIN,
    nthread=-1,
)

dvalidation = xgb.QuantileDMatrix(
    X_validation,
    label=y_validation,
    base_margin=validation_margin,
    enable_categorical=True,
    max_bin=MAX_BIN,
    ref=dtrain,
    nthread=-1,
)


# QuantileDMatrix is designed for hist training and reduces
# memory use by constructing quantized data directly.
# We no longer need the pandas frames.
del X_train
del X_validation

gc.collect()


# ============================================================
# Point-process Poisson objective
# ============================================================

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

    f = (
        predt
        .astype(
            np.float64,
            copy=False,
        )
    )

    f_safe = safe_margin(f)

    intensity = np.exp(
        f_safe
    )

    integrated_intensity = (
        exposure_train
        * intensity
    )

    # Exact gradient of:
    #
    #     w * exp(f) - y * f
    #
    grad = (
        integrated_intensity
        - y
    )

    # Exact Hessian:
    #
    #     w * exp(f)
    #
    hess = np.maximum(
        integrated_intensity,
        1e-6,
    )

    return (
        grad.astype(np.float32),
        hess.astype(np.float32),
    )

# ============================================================
# Point-process NLL metric
# ============================================================

EXPOSURE_LOOKUP = {
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

    exposure = EXPOSURE_LOOKUP[id(dmatrix)]
    
    f = np.asarray(
        predt,
        dtype=np.float64,
    )

    f_safe = safe_margin(f)

    intensity = np.exp(
        f_safe
    )

    nll = np.sum(
        exposure * intensity
        -
        y * f_safe
    )

    return (
        "pp_nll_per_event",
        float(
            nll
            /
            max(
                y.sum(),
                1.0,
            )
        ),
    )

# ============================================================
# Train
# ============================================================
params = {
    "tree_method": "hist",
    "device": "cpu",

    "max_bin": 256,

    "max_depth": 6,
    "eta": 0.03,

    "subsample": 0.90,
    "colsample_bytree": 0.90,

    # Minimum summed point-process curvature
    # required in each child.
    "min_child_weight": 50.0,

    # Very important for this objective.
    "max_delta_step": 1.0,

    "reg_lambda": 10.0,
    "reg_alpha": 0.0,

    "max_cat_to_onehot": 4,

    "seed": SEED,
    "nthread": -1,

    "disable_default_eval_metric": True,
}


evals_result = {}


def constant_nll_per_event(
    y: np.ndarray,
    exposure: np.ndarray,
    log_lambda: float,
) -> float:
    intensity = np.exp(
        log_lambda
    )

    nll = np.sum(
        exposure * intensity
        -
        y * log_lambda
    )

    return float(
        nll
        /
        max(
            y.sum(),
            1.0,
        )
    )


constant_train_nll_per_event = (
    constant_nll_per_event(
        y_train,
        exposure_train,
        initial_log_lambda,
    )
)

constant_validation_nll_per_event = (
    constant_nll_per_event(
        y_validation,
        exposure_validation,
        initial_log_lambda,
    )
)


print(
    "Constant train NLL/event:",
    constant_train_nll_per_event,
)

print(
    "Constant validation NLL/event:",
    constant_validation_nll_per_event,
)


# ============================================================
# Start canonical MLflow experiment run
# ============================================================

with mlflow.start_run(
    run_name=MLFLOW_RUN_NAME
) as mlflow_run:

    mlflow_run_id = (
        mlflow_run.info.run_id
    )

    print(
        f"\nMLflow run ID: "
        f"{mlflow_run_id}"
    )

    print(
        f"MLflow tracking URI: "
        f"{mlflow.get_tracking_uri()}"
    )

    mlflow.set_tags(
        {
            "project":
                "CrimeNet",

            "model_family":
                "xgboost",

            "task":
                "poisson_point_process",

            "experiment_stage":
                "baseline",

            "feature_set":
                "full_v1",

            "test_split_used":
                "false",
        }
    )

    # CrimeNet-specific experiment semantics. Native XGBoost
    # parameters are logged automatically by MLflow autologging.
    mlflow.log_params(
        {
            "model_table_root":
                MODEL_TABLE_ROOT,

            "train_fraction":
                TRAIN_FRACTION,

            "validation_fraction":
                VALIDATION_FRACTION,

            "sampling_strategy":
                "deterministic_model_row_id_hash",

            "sample_seed":
                SEED,

            "feature_count":
                len(FEATURE_COLUMNS),

            "categorical_feature_count":
                len(CATEGORICAL_COLUMNS),

            "objective_name":
                "point_process_poisson",

            "hessian":
                "exact",

            "base_margin":
                "train_global_log_intensity",

            "min_log_intensity":
                MIN_LOG_INTENSITY,

            "max_log_intensity":
                MAX_LOG_INTENSITY,
        }
    )

    mlflow.log_metrics(
        {
            "sample_train_rows":
                float(
                    dtrain.num_row()
                ),

            "sample_validation_rows":
                float(
                    dvalidation.num_row()
                ),

            "sample_train_events":
                train_events,

            "sample_validation_events":
                float(
                    y_validation.sum()
                ),

            "sample_train_exposure":
                train_exposure,

            "sample_validation_exposure":
                float(
                    exposure_validation.sum()
                ),

            "initial_lambda":
                initial_lambda,

            "initial_log_lambda":
                initial_log_lambda,

            "constant_train_nll_per_event":
                constant_train_nll_per_event,

            "constant_validation_nll_per_event":
                constant_validation_nll_per_event,
        }
    )


    print(
        "\nTraining XGBoost "
        "Poisson point-process baseline...\n"
    )


    booster = xgb.train(
        params=params,

        dtrain=dtrain,

        num_boost_round=1000,

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

        early_stopping_rounds=
            50,

        evals_result=
            evals_result,

        verbose_eval=
            10,
    )


# ============================================================
    # Final sampled validation diagnostics
    # ============================================================

    best_iteration = getattr(
        booster,
        "best_iteration",
        None,
    )

    if best_iteration is not None:
        booster = booster[
            : best_iteration + 1
        ]


    def evaluate_split(
        name: str,
        dmatrix: xgb.DMatrix,
        exposure: np.ndarray,
    ):
        y = (
            dmatrix
            .get_label()
            .astype(
                np.float64
            )
        )

        margin = booster.predict(
            dmatrix,
            output_margin=True,
        )

        margin = safe_margin(margin)

        intensity = np.exp(
            margin
        )

        predicted_events = float(
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

        calibration_ratio = (
            predicted_events
            /
            max(
                observed_events,
                1.0,
            )
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
            f"{predicted_events:,.2f}"
        )

        print(
            f"Expected/actual:  "
            f"{calibration_ratio:.4f}"
        )

        print(
            f"NLL/event:        "
            f"{nll_per_event:.6f}"
        )

        print(
            f"Mean log lambda:  "
            f"{margin.mean():.6f}"
        )

        return {
            "observed_events":
                observed_events,

            "expected_events":
                predicted_events,

            "expected_observed_ratio":
                calibration_ratio,

            "nll_per_event":
                nll_per_event,

            "mean_log_lambda":
                float(
                    margin.mean()
                ),
        }


    train_metrics = evaluate_split(
        "train",
        dtrain,
        exposure_train,
    )

    validation_metrics = evaluate_split(
        "validation",
        dvalidation,
        exposure_validation,
    )


    sample_train_nll_gain = (
        constant_train_nll_per_event
        -
        train_metrics[
            "nll_per_event"
        ]
    )

    sample_validation_nll_gain = (
        constant_validation_nll_per_event
        -
        validation_metrics[
            "nll_per_event"
        ]
    )

    sample_train_bits_per_event = (
        sample_train_nll_gain
        /
        np.log(2.0)
    )

    sample_validation_bits_per_event = (
        sample_validation_nll_gain
        /
        np.log(2.0)
    )


    mlflow.log_metrics(
        {
            "final_best_iteration":
                float(
                    best_iteration
                    if best_iteration is not None
                    else -1
                ),

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

            "sample_train_nll_gain_per_event":
                sample_train_nll_gain,

            "sample_validation_nll_gain_per_event":
                sample_validation_nll_gain,

            "sample_train_bits_per_event":
                sample_train_bits_per_event,

            "sample_validation_bits_per_event":
                sample_validation_bits_per_event,
        }
    )


    # ============================================================
    # Feature importance
    # ============================================================

    importance_gain = (
        booster
        .get_score(
            importance_type="gain"
        )
    )

    importance_weight = (
        booster
        .get_score(
            importance_type="weight"
        )
    )

    importance_cover = (
        booster
        .get_score(
            importance_type="cover"
        )
    )

    importance = sorted(
        importance_gain.items(),
        key=lambda item:
            item[1],
        reverse=True,
    )


    print(
        "\nTOP 30 FEATURES BY GAIN"
    )

    for (
        feature,
        gain,
    ) in importance[:30]:

        print(
            f"{feature:<50} "
            f"{gain:,.6f}"
        )


    mlflow.log_dict(
        importance_gain,
        "feature_importance/gain.json",
    )

    mlflow.log_dict(
        importance_weight,
        "feature_importance/weight.json",
    )

    mlflow.log_dict(
        importance_cover,
        "feature_importance/cover.json",
    )

    mlflow.log_dict(
        evals_result,
        "training/evals_result.json",
    )


    # ============================================================
    # Save
    # ============================================================

    model_path = (
        OUTPUT_DIR
        / "xgb_poisson_point_process.json"
    )

    booster.save_model(
        model_path
    )


    metadata = {
        "train_fraction":
            TRAIN_FRACTION,

        "validation_fraction":
            VALIDATION_FRACTION,

        "seed":
            SEED,

        "features":
            FEATURE_COLUMNS,

        "categorical_columns":
            CATEGORICAL_COLUMNS,

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

        "mlflow_run_id":
            mlflow_run_id,

        "mlflow_tracking_uri":
            mlflow.get_tracking_uri(),

        "mlflow_experiment_name":
            MLFLOW_EXPERIMENT_NAME,

        "mlflow_run_name":
            MLFLOW_RUN_NAME,
    }


    with (
        OUTPUT_DIR
        / "metadata.json"
    ).open(
        "w"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )


    metadata_path = (
        OUTPUT_DIR
        / "metadata.json"
    )


    # Log the exact sliced booster as the canonical MLflow model.
    # Downstream inference must still restore the training
    # categories and base margin from metadata.
    mlflow.xgboost.log_model(
        xgb_model=booster,
        name="model",
        model_format="json",
        metadata={
            "best_iteration":
                best_iteration,

            "initial_lambda":
                initial_lambda,

            "initial_log_lambda":
                initial_log_lambda,

            "requires_base_margin":
                True,

            "categorical_columns":
                CATEGORICAL_COLUMNS,
        },
    )


    # Preserve the exact native files used by validation.py.
    mlflow.log_artifact(
        str(model_path),
        artifact_path="native_artifacts",
    )

    mlflow.log_artifact(
        str(metadata_path),
        artifact_path="native_artifacts",
    )


    print(
        f"\nSaved model: "
        f"{model_path}"
    )

    print(
        f"MLflow run ID: "
        f"{mlflow_run_id}"
    )

    print(
        f"MLflow experiment: "
        f"{MLFLOW_EXPERIMENT_NAME}"
    )

    print(
        f"MLflow tracking URI: "
        f"{mlflow.get_tracking_uri()}"
    )

    print(
        "TEST SPLIT HAS NOT "
        "BEEN ACCESSED."
    )
