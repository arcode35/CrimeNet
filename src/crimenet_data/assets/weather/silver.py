from datetime import UTC, datetime

import dagster as dg
import polars as pl

from crimenet_data.assets.weather.transformations import (
    COASTAL_HOURLY_FIELDS,
    EXPECTED_COASTAL_UNITS,
    EXPECTED_LAND_UNITS,
    LAND_HOURLY_FIELDS,
    build_coastal_weather_hourly,
    build_land_weather_hourly,
    count_duplicate_weather_keys,
    count_hourly_length_violations,
    count_unit_violations,
)
from crimenet_data.observability.logger import get_logger
from crimenet_data.resources.crime_lake import CrimeLakeResources


log = get_logger(__name__)

GCP = pl.CredentialProviderGCP()


def _bronze_weather_uri(
    crime_lake: CrimeLakeResources,
    mode: str,
) -> str:
    return (
        f"{crime_lake.bronze_root.rstrip('/')}"
        f"/weather/open_meteo/era5_{mode}"
    )


def _silver_weather_uri(
    crime_lake: CrimeLakeResources,
    table: str,
) -> str:
    return (
        f"{crime_lake.silver_root.rstrip('/')}"
        f"/weather/{table}"
    )


# =====================================================================
# Land
# =====================================================================


@dg.asset(
    name="silver_weather_land_hourly",
    group_name="silver_weather",
    deps=["bronze_weather_land"],
)
def silver_weather_land_hourly(
    context: dg.AssetExecutionContext,
    crime_lake: CrimeLakeResources,
) -> dg.MaterializeResult:

    processed_at = datetime.now(UTC)

    source_uri = _bronze_weather_uri(
        crime_lake,
        "land",
    )

    target_uri = _silver_weather_uri(
        crime_lake,
        "land_hourly",
    )

    bronze_lf = pl.scan_delta(
        source_uri,
        credential_provider=GCP,
    )

    length_violations = count_hourly_length_violations(
        bronze_lf,
        LAND_HOURLY_FIELDS,
    )

    if length_violations:
        raise ValueError(
            "Land weather contains malformed parallel "
            f"hourly arrays: {length_violations:,}"
        )

    unit_violations = count_unit_violations(
        bronze_lf,
        EXPECTED_LAND_UNITS,
    )

    if unit_violations:
        raise ValueError(
            "Land weather contains unexpected units: "
            f"{unit_violations:,}"
        )

    weather_lf = build_land_weather_hourly(
        bronze_lf,
        processed_at=processed_at,
    )
    weather_lf = build_land_weather_hourly(
        bronze_lf,
        processed_at=processed_at,
    )
    weather_lf = (
        weather_lf
        .with_columns(
            pl.col("temperature_2m_c")
            .is_not_null()
            .cast(pl.Int8)
            .alias("_has_temperature")
        )
        .sort(
            [
                "weather_query_cell_id",
                "weather_timestamp",
                "_has_temperature",
                "_ingested_at_utc",
                "request_id",
            ]
        )
        .unique(
            subset=[
                "weather_query_cell_id",
                "weather_timestamp",
            ],
            keep="last",
        )
        .drop([
            "_has_temperature",
            "_ingested_at_utc",  # ← drop after it served its purpose
        ])
    )

    duplicate_keys = count_duplicate_weather_keys(
        weather_lf
    )

    if duplicate_keys:
        raise ValueError(
            "Land weather contains duplicate cell/hour "
            f"keys: {duplicate_keys:,}"
        )
    duplicate_keys = count_duplicate_weather_keys(
        weather_lf
    )

    if duplicate_keys:
        raise ValueError(
            "Land weather contains duplicate cell/hour "
            f"keys: {duplicate_keys:,}"
        )

    stats = (
        weather_lf
        .select(
            pl.len().alias("rows"),

            pl.col("weather_query_cell_id")
            .n_unique()
            .alias("weather_cells"),

            pl.col("temperature_2m_c")
            .is_null()
            .sum()
            .alias("null_temperature_rows"),

            pl.col("weather_timestamp")
            .min()
            .alias("minimum_timestamp"),

            pl.col("weather_timestamp")
            .max()
            .alias("maximum_timestamp"),
        )
        .collect()
        .row(0, named=True)
    )

    crime_lake.write_crimenet_table(
        weather_lf,
        target_uri,
        partitioning_columns=[
            "weather_year",
        ],
    )

    return dg.MaterializeResult(
        metadata={
            "rows": stats["rows"],
            "weather_cells": stats["weather_cells"],
            "null_temperature_rows":
                stats["null_temperature_rows"],
            "duplicate_keys": duplicate_keys,
            "minimum_timestamp":
                str(stats["minimum_timestamp"]),
            "maximum_timestamp":
                str(stats["maximum_timestamp"]),
            "target_uri": target_uri,
        }
    )


# =====================================================================
# Coastal fallback source
# =====================================================================


@dg.asset(
    name="silver_weather_coastal_hourly",
    group_name="silver_weather",
    deps=["bronze_weather_coastal"],
)
def silver_weather_coastal_hourly(
    context: dg.AssetExecutionContext,
    crime_lake: CrimeLakeResources,
) -> dg.MaterializeResult:

    processed_at = datetime.now(UTC)

    source_uri = _bronze_weather_uri(
        crime_lake,
        "coastal",
    )

    target_uri = _silver_weather_uri(
        crime_lake,
        "coastal_hourly",
    )

    bronze_lf = pl.scan_delta(
        source_uri,
        credential_provider=GCP,
    )

    length_violations = count_hourly_length_violations(
        bronze_lf,
        COASTAL_HOURLY_FIELDS,
    )

    if length_violations:
        raise ValueError(
            "Coastal weather contains malformed parallel "
            f"hourly arrays: {length_violations:,}"
        )

    unit_violations = count_unit_violations(
        bronze_lf,
        EXPECTED_COASTAL_UNITS,
    )

    if unit_violations:
        raise ValueError(
            "Coastal weather contains unexpected units: "
            f"{unit_violations:,}"
        )

    weather_lf = build_coastal_weather_hourly(
        bronze_lf,
        processed_at=processed_at,
    )

    weather_lf = (
        weather_lf
        .with_columns(
            pl.col("temperature_2m_c")
            .is_not_null()
            .cast(pl.Int8)
            .alias("_has_temperature")
        )
        .sort(
            [
                "weather_query_cell_id",
                "weather_timestamp",
                "_has_temperature",
                "_ingested_at_utc",
                "request_id",
            ]
        )
        .unique(
            subset=[
                "weather_query_cell_id",
                "weather_timestamp",
            ],
            keep="last",
        )
        .drop(
            [
                "_has_temperature",
                "_ingested_at_utc",
            ]
        )
    )
    duplicate_keys = count_duplicate_weather_keys(
        weather_lf
    )

    if duplicate_keys:
        raise ValueError(
            "Coastal weather contains duplicate cell/hour "
            f"keys: {duplicate_keys:,}"
        )

    stats = (
        weather_lf
        .select(
            pl.len().alias("rows"),

            pl.col("weather_query_cell_id")
            .n_unique()
            .alias("weather_cells"),

            pl.col("temperature_2m_c")
            .is_null()
            .sum()
            .alias("null_temperature_rows"),

            pl.col("weather_timestamp")
            .min()
            .alias("minimum_timestamp"),

            pl.col("weather_timestamp")
            .max()
            .alias("maximum_timestamp"),
        )
        .collect()
        .row(0, named=True)
    )

    crime_lake.write_crimenet_table(
        weather_lf,
        target_uri,
        partitioning_columns=[
            "weather_year",
        ],
    )

    return dg.MaterializeResult(
        metadata={
            "rows": stats["rows"],
            "weather_cells": stats["weather_cells"],
            "null_temperature_rows":
                stats["null_temperature_rows"],
            "duplicate_keys": duplicate_keys,
            "minimum_timestamp":
                str(stats["minimum_timestamp"]),
            "maximum_timestamp":
                str(stats["maximum_timestamp"]),
            "target_uri": target_uri,
        }
    )


weather_silver_assets = [
    silver_weather_land_hourly,
    silver_weather_coastal_hourly,
]