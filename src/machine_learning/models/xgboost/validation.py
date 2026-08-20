from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb


MACHINE_LEARNING_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


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


def _validate_categories(
    *,
    city: str,
    frame: pl.DataFrame,
    categorical_columns: list[str],
    category_levels: dict[str, list[str]],
) -> None:
    for column in categorical_columns:
        known_categories = {
            str(value)
            for value
            in category_levels[column]
        }

        raw_categories = (
            frame
            .get_column(column)
            .drop_nulls()
            .cast(pl.String)
            .unique()
            .to_list()
        )

        unknown_categories = sorted(
            value
            for value in raw_categories
            if value not in known_categories
        )

        if unknown_categories:
            print(
                f"\nWARNING: {city} / {column} contains "
                "categories unseen during training:"
            )
            print(
                unknown_categories
            )


def _prepare_city(
    *,
    city: str,
    frame: pl.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    category_levels: dict[str, list[str]],
    event_exposure_tolerance: float,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
]:
    _validate_categories(
        city=city,
        frame=frame,
        categorical_columns=categorical_columns,
        category_levels=category_levels,
    )

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

    if not np.isfinite(
        exposure
    ).all():
        raise ValueError(
            f"{city}: exposure contains NaN or inf values."
        )

    if not (
        exposure >= 0
    ).all():
        raise ValueError(
            f"{city}: exposure contains negative values."
        )

    if not np.isfinite(
        y
    ).all():
        raise ValueError(
            f"{city}: target contains NaN or inf values."
        )

    if not set(
        np.unique(y)
    ).issubset(
        {0.0, 1.0}
    ):
        raise ValueError(
            f"{city}: event_count is not binary."
        )

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
            > event_exposure_tolerance
        ):
            raise ValueError(
                f"{city}: observed event rows have "
                "non-zero exposure. "
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
                f"{city}: integration rows contain "
                "zero/non-positive exposure."
            )

    X = (
        frame
        .select(
            feature_columns
        )
        .to_pandas()
    )

    for column in categorical_columns:
        X[column] = pd.Categorical(
            X[column],
            categories=
                category_levels[
                    column
                ],
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
    )


def validate(
    config: dict[str, Any],
    *,
    run_id: str,
) -> dict[str, Any]:
    model_config = config[
        "model"
    ]

    data_config = config[
        "data"
    ]

    numerics_config = config[
        "numerics"
    ]

    validation_config = config.get(
        "validation",
        {},
    )

    validation_year = int(
        validation_config.get(
            "year",
            2024,
        )
    )

    artifact_root = (
        MACHINE_LEARNING_ROOT
        / config[
            "artifacts"
        ][
            "output_root"
        ]
    )

    artifact_dir = (
        artifact_root
        / model_config[
            "name"
        ]
        / run_id
    )

    model_path = (
        artifact_dir
        / "model.json"
    )

    metadata_path = (
        artifact_dir
        / "metadata.json"
    )

    output_path = (
        artifact_dir
        / (
            f"full_validation_"
            f"{validation_year}_by_city.csv"
        )
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model artifact: {model_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing model metadata: {metadata_path}"
        )

    with metadata_path.open() as file:
        metadata = json.load(
            file
        )

    feature_columns = (
        metadata[
            "features"
        ]
    )

    categorical_columns = (
        metadata[
            "categorical_columns"
        ]
    )

    category_levels = (
        metadata[
            "category_levels"
        ]
    )

    initial_log_lambda = float(
        metadata[
            "initial_log_lambda"
        ]
    )

    initial_lambda = float(
        metadata[
            "initial_lambda"
        ]
    )

    best_iteration = (
        metadata.get(
            "best_iteration"
        )
    )

    metadata_run_id = (
        metadata.get(
            "run_id"
        )
    )

    if (
        metadata_run_id is not None
        and metadata_run_id != run_id
    ):
        raise RuntimeError(
            "Requested run ID does not match model metadata. "
            f"Requested={run_id}, metadata={metadata_run_id}"
        )

    metadata_table_root = (
        metadata.get(
            "model_table_root"
        )
    )

    configured_table_root = (
        data_config[
            "model_table_root"
        ]
    )

    if (
        metadata_table_root is not None
        and metadata_table_root
        != configured_table_root
    ):
        raise RuntimeError(
            "Model-table path in training metadata differs "
            "from the supplied experiment config."
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

    event_exposure_tolerance = float(
        numerics_config.get(
            "event_exposure_tolerance",
            1e-12,
        )
    )

    print(
        f"Run ID:             {run_id}"
    )

    print(
        f"Validation year:    {validation_year}"
    )

    print(
        f"Initial lambda:     {initial_lambda:.8e}"
    )

    print(
        f"Initial log lambda: {initial_log_lambda:.6f}"
    )

    print(
        f"Features:           {len(feature_columns)}"
    )

    print(
        f"Best iteration:     {best_iteration}"
    )

    booster = xgb.Booster()

    booster.load_model(
        model_path
    )

    credentials = (
        pl.CredentialProviderGCP()
    )

    table = pl.scan_delta(
        configured_table_root,
        credential_provider=
            credentials,
    )

    validation_split = (
        data_config.get(
            "validation_split",
            "validation",
        )
    )

    cities = (
        table

        .filter(
            pl.col("split")
            == validation_split
        )

        .filter(
            pl.col("row_year")
            == validation_year
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
        f"\nValidation cities: {cities}"
    )

    if not cities:
        raise RuntimeError(
            f"No {validation_year} validation cities found."
        )

    def evaluate_city(
        city: str,
    ) -> dict[str, object]:
        print(
            "\n========================================"
        )

        print(
            f"Loading {city}..."
        )

        print(
            "========================================"
        )

        frame = (
            table

            .filter(
                pl.col("split")
                == validation_split
            )

            .filter(
                pl.col("row_year")
                == validation_year
            )

            .filter(
                pl.col("source_city")
                == city
            )

            .select(
                [
                    *feature_columns,
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
        ) = _prepare_city(
            city=city,
            frame=frame,
            feature_columns=
                feature_columns,
            categorical_columns=
                categorical_columns,
            category_levels=
                category_levels,
            event_exposure_tolerance=
                event_exposure_tolerance,
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

        base_margin = np.full(
            y.shape,
            initial_log_lambda,
            dtype=np.float32,
        )

        dmatrix = xgb.DMatrix(
            X,
            label=y,
            base_margin=base_margin,
            enable_categorical=True,
            nthread=-1,
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

        constant_nll = float(
            np.sum(
                exposure
                * initial_lambda

                -

                y
                * initial_log_lambda
            )
        )

        constant_nll_per_event = (
            constant_nll
            /
            observed_events
        )

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
                float(
                    margin.mean()
                ),

            "median_log_intensity":
                float(
                    np.median(
                        margin
                    )
                ),

            "min_log_intensity":
                float(
                    margin.min()
                ),

            "max_log_intensity":
                float(
                    margin.max()
                ),
        }

        print(
            f"\n{city}"
        )

        print(
            f"  Rows:              {rows:,}"
        )

        print(
            f"  Observed events:   {observed_events:,.0f}"
        )

        print(
            f"  Integration rows:  {integration_rows:,}"
        )

        print(
            f"  Exposure:          {total_exposure:.6e}"
        )

        print(
            f"  Expected events:   {expected_events:,.2f}"
        )

        print(
            f"  Expected/actual:   {expected_observed_ratio:.4f}"
        )

        print(
            f"  Calibration error: {calibration_error_pct:+.2f}%"
        )

        print(
            f"  Model NLL/event:   {nll_per_event:.6f}"
        )

        print(
            "  Constant NLL/event:"
            f" {constant_nll_per_event:.6f}"
        )

        print(
            f"  NLL gain/event:    {nll_gain_per_event:.6f}"
        )

        print(
            f"  Bits/event:        {bits_per_event:.6f}"
        )

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

    results = [
        evaluate_city(
            city
        )
        for city in cities
    ]

    results_df = pl.DataFrame(
        results
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        f"FULL {validation_year} VALIDATION — BY CITY"
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

    print(
        "\n"
        "============================================================"
    )

    print(
        f"FULL {validation_year} VALIDATION — OVERALL"
    )

    print(
        "============================================================"
    )

    print(
        f"Rows:                     {total_rows:,}"
    )

    print(
        f"Observed events:          {total_observed:,}"
    )

    print(
        f"Integration rows:         {total_integration:,}"
    )

    print(
        f"Integration exposure:     {total_exposure:.6e}"
    )

    print(
        f"Expected events:          {total_expected:,.2f}"
    )

    print(
        f"Expected / observed:      {global_calibration:.6f}"
    )

    print(
        f"Calibration error:        "
        f"{global_calibration_error_pct:+.3f}%"
    )

    print(
        f"Model NLL / event:        {global_nll_per_event:.6f}"
    )

    print(
        "Constant NLL / event:     "
        f"{global_constant_nll_per_event:.6f}"
    )

    print(
        f"NLL gain / event:         {global_nll_gain_per_event:.6f}"
    )

    print(
        f"Bits / event:             {global_bits_per_event:.6f}"
    )

    results_df.write_csv(
        output_path
    )

    print(
        f"\nSaved diagnostics: {output_path}"
    )

    print(
        "\nTEST SPLIT HAS NOT BEEN ACCESSED."
    )

    overall_metrics = {
        "rows":
            float(
                total_rows
            ),

        "observed_events":
            float(
                total_observed
            ),

        "integration_rows":
            float(
                total_integration
            ),

        "integration_exposure":
            total_exposure,

        "expected_events":
            total_expected,

        "expected_observed":
            global_calibration,

        "calibration_error_pct":
            global_calibration_error_pct,

        "nll_per_event":
            global_nll_per_event,

        "constant_nll_per_event":
            global_constant_nll_per_event,

        "nll_gain_per_event":
            global_nll_gain_per_event,

        "bits_per_event":
            global_bits_per_event,
    }

    city_metrics = {
        str(row["source_city"]): {
            key: float(value)
            for key, value
            in row.items()
            if (
                key
                != "source_city"
                and isinstance(
                    value,
                    (
                        int,
                        float,
                    ),
                )
            )
        }
        for row
        in results
    }

    return {
        "metrics":
            overall_metrics,

        "city_metrics":
            city_metrics,

        "artifacts": [
            str(
                output_path
            ),
        ],

        "summary": {
            "validation_year":
                validation_year,

            "city_count":
                len(
                    cities
                ),

            "test_split_used":
                False,
        },
    }
