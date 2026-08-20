from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb


# ============================================================
# Configuration
# ============================================================

MODEL_TABLE_ROOT = (
    "gs://crimenet/gold_staging_/model_table_nyc_timestamp_fix"
)

ARTIFACT_DIR = Path(
    "artifacts/xgb_poisson_local"
)

MODEL_PATH = (
    ARTIFACT_DIR
    / "xgb_poisson_point_process.json"
)

METADATA_PATH = (
    ARTIFACT_DIR
    / "metadata.json"
)

OUTPUT_PATH = (
    ARTIFACT_DIR
    / "full_validation_2024_by_city.csv"
)

MIN_LOG_INTENSITY = -30.0
MAX_LOG_INTENSITY = 15.0

EVENT_EXPOSURE_TOL = 1e-12


# ============================================================
# Numerical helper
# ============================================================

def safe_margin(
    x: np.ndarray,
) -> np.ndarray:
    return np.clip(
        np.asarray(
            x,
            dtype=np.float64,
        ),
        MIN_LOG_INTENSITY,
        MAX_LOG_INTENSITY,
    )


# ============================================================
# Load model metadata
# ============================================================

with METADATA_PATH.open() as file:
    metadata = json.load(file)


FEATURE_COLUMNS = (
    metadata["features"]
)

CATEGORICAL_COLUMNS = (
    metadata["categorical_columns"]
)

CATEGORY_LEVELS = (
    metadata["category_levels"]
)

INITIAL_LOG_LAMBDA = float(
    metadata["initial_log_lambda"]
)

INITIAL_LAMBDA = float(
    metadata["initial_lambda"]
)

BEST_ITERATION = (
    metadata.get("best_iteration")
)


print(
    f"Initial lambda:     "
    f"{INITIAL_LAMBDA:.8e}"
)

print(
    f"Initial log lambda: "
    f"{INITIAL_LOG_LAMBDA:.6f}"
)

print(
    f"Features:           "
    f"{len(FEATURE_COLUMNS)}"
)

print(
    f"Best iteration:     "
    f"{BEST_ITERATION}"
)


# ============================================================
# Load saved booster
#
# IMPORTANT:
# The training script already slices the booster to
# best_iteration before saving it.
#
# Therefore DO NOT specify iteration_range here.
# ============================================================

booster = xgb.Booster()

booster.load_model(
    MODEL_PATH
)


# ============================================================
# Open Delta model table lazily
# ============================================================

credentials = (
    pl.CredentialProviderGCP()
)

table = pl.scan_delta(
    MODEL_TABLE_ROOT,
    credential_provider=credentials,
)


# ============================================================
# Discover all 2024 validation cities
# ============================================================

cities = (
    table

    .filter(
        pl.col("split")
        == "validation"
    )

    .filter(
        pl.col("row_year")
        == 2024
    )

    .select(
        "source_city"
    )

    .unique()

    .sort(
        "source_city"
    )

    .collect()

    .get_column(
        "source_city"
    )

    .to_list()
)


print(
    f"\nValidation cities: "
    f"{cities}"
)


if not cities:
    raise RuntimeError(
        "No 2024 validation cities found."
    )


# ============================================================
# Validate categorical compatibility
# ============================================================

def validate_categories(
    city: str,
    frame: pl.DataFrame,
) -> None:

    for column in CATEGORICAL_COLUMNS:

        known_categories = {
            str(value)
            for value
            in CATEGORY_LEVELS[column]
        }

        raw_categories = (
            frame

            .get_column(
                column
            )

            .drop_nulls()

            .cast(
                pl.String
            )

            .unique()

            .to_list()
        )

        unknown_categories = sorted(
            value
            for value
            in raw_categories
            if value not in known_categories
        )

        if unknown_categories:

            print(
                f"\nWARNING: "
                f"{city} / {column} "
                f"contains categories unseen "
                f"during training:"
            )

            print(
                unknown_categories
            )


# ============================================================
# Convert one city to XGBoost representation
# ============================================================

def prepare_city(
    city: str,
    frame: pl.DataFrame,
):
    validate_categories(
        city,
        frame,
    )

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


    # --------------------------------------------------------
    # Basic target/exposure sanity checks
    # --------------------------------------------------------

    if not np.isfinite(
        exposure
    ).all():
        raise ValueError(
            f"{city}: exposure contains "
            f"NaN or inf values."
        )

    if not (
        exposure >= 0
    ).all():
        raise ValueError(
            f"{city}: exposure contains "
            f"negative values."
        )

    if not np.isfinite(
        y
    ).all():
        raise ValueError(
            f"{city}: target contains "
            f"NaN or inf values."
        )

    if not set(
        np.unique(y)
    ).issubset(
        {0.0, 1.0}
    ):
        raise ValueError(
            f"{city}: event_count is "
            f"not binary."
        )


    # --------------------------------------------------------
    # Verify point-process row semantics.
    #
    # Event rows should have zero exposure.
    # Integration rows should have positive exposure.
    # --------------------------------------------------------

    event_mask = (
        y == 1
    )

    integration_mask = (
        y == 0
    )

    if event_mask.any():

        max_event_exposure = float(
            exposure[
                event_mask
            ].max()
        )

        if (
            max_event_exposure
            > EVENT_EXPOSURE_TOL
        ):
            raise ValueError(
                f"{city}: observed event rows "
                f"have non-zero exposure. "
                f"Max={max_event_exposure}"
            )


    if integration_mask.any():

        if not (
            exposure[
                integration_mask
            ]
            > 0
        ).all():
            raise ValueError(
                f"{city}: integration rows "
                f"contain zero/non-positive "
                f"exposure."
            )


    # --------------------------------------------------------
    # Feature matrix
    # --------------------------------------------------------

    X = (
        frame

        .select(
            FEATURE_COLUMNS
        )

        .to_pandas()
    )


    # --------------------------------------------------------
    # Restore EXACT categorical levels from training.
    # --------------------------------------------------------

    for column in (
        CATEGORICAL_COLUMNS
    ):
        X[column] = pd.Categorical(
            X[column],
            categories=
                CATEGORY_LEVELS[
                    column
                ],
        )


    # --------------------------------------------------------
    # Numerical values -> float32.
    #
    # Preserve NaN. XGBoost handles missing values natively.
    # --------------------------------------------------------

    for column in (
        FEATURE_COLUMNS
    ):

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
    )


# ============================================================
# Evaluate one city
# ============================================================

def evaluate_city(
    city: str,
) -> dict[str, object]:

    print(
        f"\n========================================"
    )

    print(
        f"Loading {city}..."
    )

    print(
        f"========================================"
    )


    frame = (
        table

        .filter(
            pl.col("split")
            == "validation"
        )

        .filter(
            pl.col("row_year")
            == 2024
        )

        .filter(
            pl.col("source_city")
            == city
        )

        .select(
            [
                *FEATURE_COLUMNS,
                "event_count",
                "integration_weight_cell_seconds",
            ]
        )

        .collect()
    )


    rows = frame.height


    if rows == 0:
        raise RuntimeError(
            f"{city}: no validation rows."
        )


    (
        X,
        y,
        exposure,
    ) = prepare_city(
        city,
        frame,
    )


    observed_events = float(
        y.sum()
    )

    integration_rows = int(
        np.sum(
            y == 0
        )
    )

    total_exposure = float(
        exposure.sum()
    )


    if observed_events <= 0:
        raise RuntimeError(
            f"{city}: no observed events."
        )

    if total_exposure <= 0:
        raise RuntimeError(
            f"{city}: no positive exposure."
        )


    # --------------------------------------------------------
    # Training used a constant raw base margin equal to
    # initial_log_lambda.
    #
    # Prediction must use the same base margin.
    # --------------------------------------------------------

    base_margin = np.full(
        y.shape,
        INITIAL_LOG_LAMBDA,
        dtype=np.float32,
    )


    dmatrix = xgb.DMatrix(
        X,
        label=y,
        base_margin=base_margin,
        enable_categorical=True,
        nthread=-1,
    )


    # --------------------------------------------------------
    # Predict raw log intensity.
    #
    # Booster is already sliced to the best iteration.
    # --------------------------------------------------------

    margin = booster.predict(
        dmatrix,
        output_margin=True,
    )

    margin = safe_margin(
        margin
    )

    intensity = np.exp(
        margin
    )


    # --------------------------------------------------------
    # Model point-process likelihood
    #
    # NLL =
    #
    #   Σ_j w_j λ_j
    #   -
    #   Σ_i log λ_i
    #
    # Under the unified row representation:
    #
    #   Σ exposure * exp(f)
    #   -
    #   Σ y * f
    # --------------------------------------------------------

    expected_events = float(
        np.sum(
            exposure
            * intensity
        )
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
        observed_events
    )


    expected_observed_ratio = (
        expected_events
        /
        observed_events
    )


    calibration_error_pct = (
        (
            expected_observed_ratio
            - 1.0
        )
        * 100.0
    )


    # --------------------------------------------------------
    # Frozen GLOBAL training constant-intensity baseline.
    #
    # IMPORTANT:
    # This uses lambda fitted on TRAIN, not 2024 validation.
    # --------------------------------------------------------

    constant_nll = float(
        np.sum(
            exposure
            * INITIAL_LAMBDA

            -

            y
            * INITIAL_LOG_LAMBDA
        )
    )


    constant_nll_per_event = (
        constant_nll
        /
        observed_events
    )


    # --------------------------------------------------------
    # Information gain over homogeneous Poisson baseline
    # --------------------------------------------------------

    nll_gain_per_event = (
        constant_nll_per_event
        -
        nll_per_event
    )


    bits_per_event = (
        nll_gain_per_event
        /
        np.log(2.0)
    )


    # --------------------------------------------------------
    # Additional intensity diagnostics
    # --------------------------------------------------------

    mean_log_intensity = float(
        margin.mean()
    )

    median_log_intensity = float(
        np.median(
            margin
        )
    )

    min_log_intensity = float(
        margin.min()
    )

    max_log_intensity = float(
        margin.max()
    )


    result = {
        "source_city":
            city,

        "rows":
            rows,

        "observed_events":
            int(
                observed_events
            ),

        "integration_rows":
            integration_rows,

        "integration_weight":
            total_exposure,

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

        "constant_nll":
            constant_nll,

        "constant_nll_per_event":
            constant_nll_per_event,

        "nll_gain_per_event":
            nll_gain_per_event,

        "bits_per_event":
            bits_per_event,

        "mean_log_intensity":
            mean_log_intensity,

        "median_log_intensity":
            median_log_intensity,

        "min_log_intensity":
            min_log_intensity,

        "max_log_intensity":
            max_log_intensity,
    }


    # --------------------------------------------------------
    # Print city diagnostics
    # --------------------------------------------------------

    print(
        f"\n{city}"
    )

    print(
        f"  Rows:              "
        f"{rows:,}"
    )

    print(
        f"  Observed events:   "
        f"{observed_events:,.0f}"
    )

    print(
        f"  Integration rows:  "
        f"{integration_rows:,}"
    )

    print(
        f"  Exposure:          "
        f"{total_exposure:.6e}"
    )

    print(
        f"  Expected events:   "
        f"{expected_events:,.2f}"
    )

    print(
        f"  Expected/actual:   "
        f"{expected_observed_ratio:.4f}"
    )

    print(
        f"  Calibration error: "
        f"{calibration_error_pct:+.2f}%"
    )

    print(
        f"  Model NLL/event:   "
        f"{nll_per_event:.6f}"
    )

    print(
        f"  Constant NLL/event:"
        f" {constant_nll_per_event:.6f}"
    )

    print(
        f"  NLL gain/event:    "
        f"{nll_gain_per_event:.6f}"
    )

    print(
        f"  Bits/event:        "
        f"{bits_per_event:.6f}"
    )


    # --------------------------------------------------------
    # Release city memory before loading the next one.
    # --------------------------------------------------------

    del frame
    del X
    del y
    del exposure
    del base_margin
    del dmatrix
    del margin
    del intensity

    gc.collect()


    return result


# ============================================================
# Evaluate ALL 2024 validation cities
# ============================================================

results = []


for city in cities:

    result = evaluate_city(
        city
    )

    results.append(
        result
    )


results_df = pl.DataFrame(
    results
)


# ============================================================
# City-by-city table
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "FULL 2024 VALIDATION — BY CITY"
)

print(
    "============================================================"
)


city_table = (
    results_df

    .select(
        [
            "source_city",
            "rows",
            "observed_events",
            "integration_rows",
            "expected_events",
            "expected_observed_ratio",
            "calibration_error_pct",
            "nll_per_event",
            "constant_nll_per_event",
            "nll_gain_per_event",
            "bits_per_event",
        ]
    )

    .sort(
        "source_city"
    )
)


print(
    city_table
)


# ============================================================
# Global metrics
#
# IMPORTANT:
# Aggregate RAW NLL / exposure / event counts first.
#
# Do NOT average per-city NLL values.
# ============================================================

total_rows = int(
    results_df[
        "rows"
    ].sum()
)

total_observed = int(
    results_df[
        "observed_events"
    ].sum()
)

total_integration = int(
    results_df[
        "integration_rows"
    ].sum()
)

total_exposure = float(
    results_df[
        "integration_weight"
    ].sum()
)

total_expected = float(
    results_df[
        "expected_events"
    ].sum()
)

total_nll = float(
    results_df[
        "nll"
    ].sum()
)

total_constant_nll = float(
    results_df[
        "constant_nll"
    ].sum()
)


if total_observed <= 0:
    raise RuntimeError(
        "Global validation has zero observed events."
    )


global_nll_per_event = (
    total_nll
    /
    total_observed
)


global_constant_nll_per_event = (
    total_constant_nll
    /
    total_observed
)


global_nll_gain_per_event = (
    global_constant_nll_per_event
    -
    global_nll_per_event
)


global_bits_per_event = (
    global_nll_gain_per_event
    /
    np.log(2.0)
)


global_calibration = (
    total_expected
    /
    total_observed
)


global_calibration_error_pct = (
    (
        global_calibration
        - 1.0
    )
    * 100.0
)


# ============================================================
# Global report
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "FULL 2024 VALIDATION — OVERALL"
)

print(
    "============================================================"
)


print(
    f"Rows:                     "
    f"{total_rows:,}"
)

print(
    f"Observed events:          "
    f"{total_observed:,}"
)

print(
    f"Integration rows:         "
    f"{total_integration:,}"
)

print(
    f"Integration exposure:     "
    f"{total_exposure:.6e}"
)

print(
    f"Expected events:          "
    f"{total_expected:,.2f}"
)

print(
    f"Expected / observed:      "
    f"{global_calibration:.6f}"
)

print(
    f"Calibration error:        "
    f"{global_calibration_error_pct:+.3f}%"
)

print(
    f"Model NLL / event:        "
    f"{global_nll_per_event:.6f}"
)

print(
    f"Constant NLL / event:     "
    f"{global_constant_nll_per_event:.6f}"
)

print(
    f"NLL gain / event:         "
    f"{global_nll_gain_per_event:.6f}"
)

print(
    f"Bits / event:             "
    f"{global_bits_per_event:.6f}"
)


# ============================================================
# Save detailed city diagnostics
# ============================================================

results_df.write_csv(
    OUTPUT_PATH
)


print(
    f"\nSaved diagnostics: "
    f"{OUTPUT_PATH}"
)

print(
    "\nTEST SPLIT HAS NOT BEEN ACCESSED."
)