from datetime import datetime

import polars as pl


WEATHER_KEY = [
    "weather_query_cell_id",
    "weather_timestamp",
]


LAND_HOURLY_FIELDS = [
    "time",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "snowfall",
    "weather_code",
    "cloud_cover",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
]


COASTAL_HOURLY_FIELDS = [
    "time",
    "temperature_2m",
]


EXPECTED_LAND_UNITS = {
    "temperature_2m": "°C",
    "relative_humidity_2m": "%",
    "precipitation": "mm",
    "rain": "mm",
    "snowfall": "cm",
    "cloud_cover": "%",
    "surface_pressure": "hPa",
    "wind_speed_10m": "km/h",
    "wind_direction_10m": "°",
    "wind_gusts_10m": "km/h",
}


EXPECTED_COASTAL_UNITS = {
    "temperature_2m": "°C",
}


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------


def count_hourly_length_violations(
    lf: pl.LazyFrame,
    fields: list[str],
) -> int:
    """
    All parallel arrays inside `hourly` must have the same length.
    """

    length_columns = [
        pl.col("hourly")
        .struct.field(field)
        .list.len()
        .alias(f"__len_{field}")
        for field in fields
    ]

    invalid = (
        pl.col("__len_time").is_null()
        | (pl.col("__len_time") == 0)
    )

    for field in fields:
        if field == "time":
            continue

        invalid = invalid | (
            pl.col(f"__len_{field}")
            != pl.col("__len_time")
        )

    return (
        lf
        .with_columns(length_columns)
        .filter(invalid)
        .select(pl.len().alias("violations"))
        .collect()
        .item()
    )


def count_unit_violations(
    lf: pl.LazyFrame,
    expected_units: dict[str, str],
) -> int:
    """
    Fail rather than silently label values with incorrect canonical units.
    """

    invalid = pl.lit(False)

    for field, expected_unit in expected_units.items():
        unit = (
            pl.col("hourly_units")
            .struct.field(field)
        )

        invalid = invalid | (
            unit.is_null()
            | (unit != pl.lit(expected_unit))
        )

    return (
        lf
        .filter(invalid)
        .select(pl.len().alias("violations"))
        .collect()
        .item()
    )


def count_duplicate_weather_keys(
    lf: pl.LazyFrame,
) -> int:
    return (
        lf
        .group_by(WEATHER_KEY)
        .len()
        .filter(pl.col("len") > 1)
        .select(pl.len().alias("duplicate_keys"))
        .collect()
        .item()
    )


# ---------------------------------------------------------------------
# Land Bronze -> canonical hourly
# ---------------------------------------------------------------------


def build_land_weather_hourly(
    bronze_lf: pl.LazyFrame,
    *,
    processed_at: datetime,
) -> pl.LazyFrame:
    """
    Convert one-row-per-request Open-Meteo land responses into
    one-row-per-weather-cell-hour canonical observations.
    """

    list_columns = [
        f"__{field}"
        for field in LAND_HOURLY_FIELDS
    ]

    return (
        bronze_lf

        # Extract parallel hourly arrays from the struct.
        .with_columns(
            [
                pl.col("hourly")
                .struct.field(field)
                .alias(f"__{field}")
                for field in LAND_HOURLY_FIELDS
            ]
        )

        # One row per API hour.
        .explode(list_columns)

        # Canonical projection.
        .select(
            pl.col("provider").cast(pl.String),
            pl.col("model").cast(pl.String),
            pl.col("request_id").cast(pl.String),
            pl.col("_ingested_at_utc"),
            pl.col("weather_query_cell_id")
            .cast(pl.Int64),

            pl.col("h3_resolution")
            .cast(pl.Int8),

            pl.col("query_latitude")
            .cast(pl.Float64),

            pl.col("query_longitude")
            .cast(pl.Float64),

            pl.col("grid_latitude")
            .cast(pl.Float64),

            pl.col("grid_longitude")
            .cast(pl.Float64),

            pl.col("grid_elevation")
            .cast(pl.Float64),

            pl.col("__time")
            .str.to_datetime(
                time_zone="UTC",
                strict=True,
            )
            .alias("weather_timestamp"),

            pl.col("__temperature_2m")
            .cast(pl.Float64)
            .alias("temperature_2m_c"),

            pl.col("__relative_humidity_2m")
            .cast(pl.Float64)
            .alias("relative_humidity_2m_pct"),

            pl.col("__precipitation")
            .cast(pl.Float64)
            .alias("precipitation_mm"),

            pl.col("__rain")
            .cast(pl.Float64)
            .alias("rain_mm"),

            pl.col("__snowfall")
            .cast(pl.Float64)
            .alias("snowfall_cm"),

            pl.col("__cloud_cover")
            .cast(pl.Float64)
            .alias("cloud_cover_pct"),

            pl.col("__surface_pressure")
            .cast(pl.Float64)
            .alias("surface_pressure_hpa"),

            pl.col("__weather_code")
            .cast(pl.Int16)
            .alias("weather_code"),

            pl.col("__wind_speed_10m")
            .cast(pl.Float64)
            .alias("wind_speed_10m_kmh"),

            pl.col("__wind_direction_10m")
            .cast(pl.Float64)
            .alias("wind_direction_10m_deg"),

            pl.col("__wind_gusts_10m")
            .cast(pl.Float64)
            .alias("wind_gusts_10m_kmh"),

            pl.col("timezone").cast(pl.String),

            pl.col("utc_offset_seconds")
            .cast(pl.Int32),

            pl.col("cell_selection")
            .cast(pl.String),

            pl.col("hourly_units")
            .struct.field("temperature_2m")
            .cast(pl.String)
            .alias("temperature_unit"),
        )

        .with_columns(
            pl.col("weather_timestamp")
            .dt.date()
            .alias("weather_date"),

            pl.col("weather_timestamp")
            .dt.year()
            .cast(pl.Int16)
            .alias("weather_year"),

            pl.lit(False)
            .alias("temperature_was_patched"),

            pl.lit("land")
            .alias("temperature_source"),

            pl.lit(None, dtype=pl.String)
            .alias("temperature_patch_request_id"),

            pl.lit(processed_at)
            .alias("silver_processed_at_utc"),
        )
    )


# ---------------------------------------------------------------------
# Coastal Bronze -> canonical temperature patch
# ---------------------------------------------------------------------
def build_coastal_weather_hourly(
    bronze_lf: pl.LazyFrame,
    *,
    processed_at: datetime,
) -> pl.LazyFrame:

    return (
        bronze_lf
        .with_columns(
            pl.col("hourly")
            .struct.field("time")
            .alias("__time"),

            pl.col("hourly")
            .struct.field("temperature_2m")
            .alias("__temperature_2m"),
        )
        .explode(
            "__time",
            "__temperature_2m",
        )
        .select(
            pl.col("_weather_query_cell_id")
            .cast(pl.Int64)
            .alias("weather_query_cell_id"),

            pl.lit(6)
            .cast(pl.Int8)
            .alias("h3_resolution"),

            pl.col("__time")
            .str.to_datetime(
                time_zone="UTC",
                strict=True,
            )
            .alias("weather_timestamp"),

            pl.col("__temperature_2m")
            .cast(pl.Float64)
            .alias("temperature_2m_c"),

            pl.col("_requested_latitude")
            .cast(pl.Float64)
            .alias("query_latitude"),

            pl.col("_requested_longitude")
            .cast(pl.Float64)
            .alias("query_longitude"),

            pl.col("latitude")
            .cast(pl.Float64)
            .alias("grid_latitude"),

            pl.col("longitude")
            .cast(pl.Float64)
            .alias("grid_longitude"),

            pl.col("elevation")
            .cast(pl.Float64)
            .alias("grid_elevation"),

            pl.col("_provider")
            .cast(pl.String)
            .alias("provider"),

            pl.col("_model")
            .cast(pl.String)
            .alias("model"),

            pl.col("_request_id")
            .cast(pl.String)
            .alias("request_id"),
            pl.col("_ingested_at_utc"),
        )
        .with_columns(
            pl.col("weather_timestamp")
            .dt.year()
            .cast(pl.Int16)
            .alias("weather_year"),

            pl.lit(processed_at)
            .alias("silver_processed_at_utc"),
        )
    )
