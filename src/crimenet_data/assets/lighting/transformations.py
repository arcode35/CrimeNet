from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import polars_h3 as plh3
import pvlib


LIGHTING_DEFINITION_VERSION = (
    "pvlib_nrel_numpy_v1"
)

LIGHTING_KEY = [
    "weather_query_cell_id",
    "solar_timestamp_hour",
]


LIGHTING_BATCH_SCHEMA = pl.Schema(
    {
        "weather_query_cell_id":
            pl.Int64,

        "solar_timestamp_hour":
            pl.Datetime(
                "us",
                time_zone="UTC",
            ),

        "query_latitude":
            pl.Float64,

        "query_longitude":
            pl.Float64,

        "solar_year":
            pl.Int32,

        "solar_elevation_deg":
            pl.Float64,

        "apparent_solar_elevation_deg":
            pl.Float64,

        "solar_zenith_deg":
            pl.Float64,

        "solar_azimuth_deg":
            pl.Float64,

        "lighting_condition":
            pl.String,

        "is_daylight":
            pl.Boolean,

        "pvlib_version":
            pl.String,

        "lighting_definition_version":
            pl.String,
    }
)


# =============================================================================
# Required lighting universe
# =============================================================================

def build_required_lighting_keys(
    *,
    events: pl.LazyFrame,
    integration: pl.LazyFrame,
) -> pl.LazyFrame:
    observed_keys = (
        events
        .select(
            pl.col(
                "weather_query_cell_id"
            ).cast(pl.Int64),

            pl.col(
                "weather_timestamp"
            )
            .cast(
                pl.Datetime(
                    "us",
                    time_zone="UTC",
                )
            )
            .alias(
                "solar_timestamp_hour"
            ),
        )
    )

    integration_keys = (
        integration
        .select(
            pl.col(
                "weather_query_cell_id"
            ).cast(pl.Int64),

            pl.col(
                "sample_timestamp_utc"
            )
            .dt.truncate("1h")
            .cast(
                pl.Datetime(
                    "us",
                    time_zone="UTC",
                )
            )
            .alias(
                "solar_timestamp_hour"
            ),
        )
    )

    return (
        pl.concat(
            [
                observed_keys,
                integration_keys,
            ],
            how="vertical_relaxed",
        )
        .drop_nulls(
            [
                "weather_query_cell_id",
                "solar_timestamp_hour",
            ]
        )
        .unique(
            subset=[
                "weather_query_cell_id",
                "solar_timestamp_hour",
            ]
        )
        .with_columns(
            plh3.cell_to_lat(
                "weather_query_cell_id"
            )
            .cast(pl.Float64)
            .alias("query_latitude"),

            plh3.cell_to_lng(
                "weather_query_cell_id"
            )
            .cast(pl.Float64)
            .alias("query_longitude"),
        )
        .filter(
            pl.col("query_latitude")
            .is_finite()
            &
            pl.col("query_longitude")
            .is_finite()
            &
            pl.col("query_latitude")
            .is_between(-90.0, 90.0)
            &
            pl.col("query_longitude")
            .is_between(-180.0, 180.0)
        )
        .with_columns(
            pl.col(
                "solar_timestamp_hour"
            )
            .dt.year()
            .cast(pl.Int32)
            .alias("solar_year")
        )
    )


# =============================================================================
# Lighting classification
# =============================================================================


def classify_lighting_condition(
    elevation: np.ndarray,
) -> np.ndarray:
    """
    Classify lighting from geometric solar elevation.

    >=   0° : daylight
    >=  -6° : civil twilight
    >= -12° : nautical twilight
    >= -18° : astronomical twilight
    <  -18° : night
    """

    return np.select(
        [
            elevation >= 0.0,
            elevation >= -6.0,
            elevation >= -12.0,
            elevation >= -18.0,
        ],
        [
            "daylight",
            "civil_twilight",
            "nautical_twilight",
            "astronomical_twilight",
        ],
        default="night",
    )


# =============================================================================
# pvlib calculation
# =============================================================================


def calculate_solar_batch(
    batch: pl.DataFrame,
) -> pl.DataFrame:
    """
    Calculate pvlib solar position for a Polars batch.

    pvlib receives one scalar latitude/longitude
    per H3 cell and a vector of UTC timestamps.
    """

    if batch.is_empty():
        return pl.DataFrame(
            schema=LIGHTING_BATCH_SCHEMA
        )

    outputs: list[pl.DataFrame] = []

    groups = batch.partition_by(
        "weather_query_cell_id",
        maintain_order=False,
    )

    for group in groups:
        latitude = float(
            group[
                "query_latitude"
            ][0]
        )

        longitude = float(
            group[
                "query_longitude"
            ][0]
        )

        timestamps = pd.DatetimeIndex(
            pd.to_datetime(
                group[
                    "solar_timestamp_hour"
                ].to_list(),
                utc=True,
            )
        )

        solar_position = (
            pvlib.solarposition
            .get_solarposition(
                time=timestamps,
                latitude=latitude,
                longitude=longitude,
                method="nrel_numpy",
            )
        )

        elevation = (
            solar_position[
                "elevation"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        apparent_elevation = (
            solar_position[
                "apparent_elevation"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        zenith = (
            solar_position[
                "zenith"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        azimuth = (
            solar_position[
                "azimuth"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        lighting_condition = (
            classify_lighting_condition(
                elevation
            )
        )

        result = (
            group
            .with_columns(
                pl.Series(
                    "solar_elevation_deg",
                    elevation,
                ),

                pl.Series(
                    "apparent_solar_elevation_deg",
                    apparent_elevation,
                ),

                pl.Series(
                    "solar_zenith_deg",
                    zenith,
                ),

                pl.Series(
                    "solar_azimuth_deg",
                    azimuth,
                ),

                pl.Series(
                    "lighting_condition",
                    lighting_condition,
                ),

                pl.Series(
                    "is_daylight",
                    elevation >= 0.0,
                ),

                pl.lit(
                    pvlib.__version__
                )
                .alias(
                    "pvlib_version"
                ),

                pl.lit(
                    LIGHTING_DEFINITION_VERSION
                )
                .alias(
                    "lighting_definition_version"
                ),
            )

            .select(
                LIGHTING_BATCH_SCHEMA.names()
            )
        )

        outputs.append(
            result
        )

    return pl.concat(
        outputs,
        how="vertical",
    )


def compute_lighting_conditions(
    required_keys: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Apply pvlib computation to the complete
    required cell-hour universe.
    """

    return (
        required_keys
        .map_batches(
            calculate_solar_batch,
            schema=LIGHTING_BATCH_SCHEMA,
            validate_output_schema=True,
            streamable=True,
        )
    )


# =============================================================================
# Validation
# =============================================================================


def expected_lighting_condition() -> pl.Expr:
    elevation = pl.col(
        "solar_elevation_deg"
    )

    return (
        pl.when(
            elevation >= 0.0
        )
        .then(
            pl.lit("daylight")
        )

        .when(
            elevation >= -6.0
        )
        .then(
            pl.lit(
                "civil_twilight"
            )
        )

        .when(
            elevation >= -12.0
        )
        .then(
            pl.lit(
                "nautical_twilight"
            )
        )

        .when(
            elevation >= -18.0
        )
        .then(
            pl.lit(
                "astronomical_twilight"
            )
        )

        .otherwise(
            pl.lit("night")
        )
    )


def validate_required_lighting_keys(
    keys: pl.LazyFrame,
) -> dict[str, object]:
    metrics = (
        keys
        .select(
            pl.len()
            .cast(pl.Int64)
            .alias("rows"),

            pl.struct(
                LIGHTING_KEY
            )
            .n_unique()
            .cast(pl.Int64)
            .alias(
                "unique_keys"
            ),

            pl.col(
                "weather_query_cell_id"
            )
            .n_unique()
            .cast(pl.Int64)
            .alias(
                "cells"
            ),

            pl.col(
                "solar_timestamp_hour"
            )
            .min()
            .alias(
                "min_timestamp"
            ),

            pl.col(
                "solar_timestamp_hour"
            )
            .max()
            .alias(
                "max_timestamp"
            ),

            pl.col(
                "query_latitude"
            )
            .is_null()
            .sum()
            .cast(pl.Int64)
            .alias(
                "null_latitudes"
            ),

            pl.col(
                "query_longitude"
            )
            .is_null()
            .sum()
            .cast(pl.Int64)
            .alias(
                "null_longitudes"
            ),
        )
        .collect()
        .row(
            0,
            named=True,
        )
    )

    if metrics["rows"] == 0:
        raise ValueError(
            "Required lighting universe is empty."
        )

    if (
        metrics["unique_keys"]
        != metrics["rows"]
    ):
        raise ValueError(
            "Required lighting key uniqueness "
            "violated: "
            f"rows={metrics['rows']:,}, "
            f"unique={metrics['unique_keys']:,}"
        )

    if (
        metrics["null_latitudes"] != 0
        or metrics["null_longitudes"] != 0
    ):
        raise ValueError(
            "Required lighting keys contain "
            "missing coordinates."
        )

    return metrics


def validate_lighting_results(
    lighting: pl.LazyFrame,
    *,
    expected_rows: int,
) -> dict[str, int]:
    metrics = (
        lighting
        .select(
            pl.len()
            .cast(pl.Int64)
            .alias("rows"),

            pl.struct(
                LIGHTING_KEY
            )
            .n_unique()
            .cast(pl.Int64)
            .alias(
                "unique_keys"
            ),

            pl.col(
                "solar_elevation_deg"
            )
            .is_null()
            .sum()
            .cast(pl.Int64)
            .alias(
                "null_elevation"
            ),

            pl.col(
                "apparent_solar_elevation_deg"
            )
            .is_null()
            .sum()
            .cast(pl.Int64)
            .alias(
                "null_apparent_elevation"
            ),

            pl.col(
                "solar_zenith_deg"
            )
            .is_null()
            .sum()
            .cast(pl.Int64)
            .alias(
                "null_zenith"
            ),

            pl.col(
                "solar_azimuth_deg"
            )
            .is_null()
            .sum()
            .cast(pl.Int64)
            .alias(
                "null_azimuth"
            ),

            (
                pl.col(
                    "lighting_condition"
                )
                != expected_lighting_condition()
            )
            .sum()
            .cast(pl.Int64)
            .alias(
                "classification_mismatches"
            ),

            (
                pl.col(
                    "is_daylight"
                )
                != (
                    pl.col(
                        "solar_elevation_deg"
                    )
                    >= 0.0
                )
            )
            .sum()
            .cast(pl.Int64)
            .alias(
                "daylight_mismatches"
            ),

            (
                ~pl.col(
                    "solar_elevation_deg"
                )
                .is_between(
                    -90.0,
                    90.0,
                )
            )
            .sum()
            .cast(pl.Int64)
            .alias(
                "invalid_elevation"
            ),

            (
                ~pl.col(
                    "solar_zenith_deg"
                )
                .is_between(
                    0.0,
                    180.0,
                )
            )
            .sum()
            .cast(pl.Int64)
            .alias(
                "invalid_zenith"
            ),

            (
                ~pl.col(
                    "solar_azimuth_deg"
                )
                .is_between(
                    0.0,
                    360.0,
                )
            )
            .sum()
            .cast(pl.Int64)
            .alias(
                "invalid_azimuth"
            ),
        )
        .collect()
        .row(
            0,
            named=True,
        )
    )

    if (
        metrics["rows"]
        != expected_rows
    ):
        raise ValueError(
            "Lighting row-count mismatch: "
            f"expected={expected_rows:,}, "
            f"actual={metrics['rows']:,}"
        )

    if (
        metrics["unique_keys"]
        != metrics["rows"]
    ):
        raise ValueError(
            "Lighting key uniqueness violated: "
            f"rows={metrics['rows']:,}, "
            f"unique={metrics['unique_keys']:,}"
        )

    failure_fields = [
        "null_elevation",
        "null_apparent_elevation",
        "null_zenith",
        "null_azimuth",
        "classification_mismatches",
        "daylight_mismatches",
        "invalid_elevation",
        "invalid_zenith",
        "invalid_azimuth",
    ]

    failures = {
        field: int(
            metrics[field]
        )
        for field in failure_fields
        if metrics[field] != 0
    }

    if failures:
        raise ValueError(
            "Lighting validation failed: "
            f"{failures}"
        )

    return {
        key: int(value)
        for key, value in metrics.items()
    }