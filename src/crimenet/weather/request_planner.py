from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from crimenet.spatial.h3 import (
    add_weather_query_cell,
    extract_h3_centers,
)
from crimenet.weather.open_meteo_client import (
    VALID_MODELS,
    normalize_hourly_variables,
)

PROVIDER = "open_meteo"
TIMEZONE = "GMT"
CELL_SELECTION = "nearest"

# ERA5 and ERA5-Land normally trail real time by
# approximately five days. Seven days gives a small
# availability buffer.
DEFAULT_AVAILABILITY_LAG_DAYS = 7

MODEL_START_DATES = {
    "era5": date(1940, 1, 1),
    "era5_land": date(1950, 1, 1),
}


def build_weather_request_manifest(
    crime_df: DataFrame,
    *,
    model: str,
    hourly_variables: Sequence[str],
    h3_resolution: int = 6,
    availability_cutoff: date | None = None,
) -> DataFrame:
    normalized_model = model.strip().lower()

    if normalized_model not in VALID_MODELS:
        supported = ", ".join(
            sorted(VALID_MODELS)
        )
        raise ValueError(
            f"Unsupported model {model!r}. "
            f"Supported models: {supported}"
        )

    model_start_date = MODEL_START_DATES[
        normalized_model
    ]

    if not 0 <= h3_resolution <= 15:
        raise ValueError(
            "h3_resolution must be between 0 and 15"
        )

    effective_cutoff = (
        availability_cutoff
        if availability_cutoff is not None
        else date.today()
        - timedelta(
            days=DEFAULT_AVAILABILITY_LAG_DAYS
        )
    )

    required_columns = {
        "latitude",
        "longitude",
        "occurred_at",
    }

    missing_columns = (
        required_columns - set(crime_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    normalized_variables = normalize_hourly_variables(
        hourly_variables
    )

    valid_crimes = (
        crime_df
        .select(
            F.col("latitude")
            .cast("double")
            .alias("latitude"),
            F.col("longitude")
            .cast("double")
            .alias("longitude"),
            F.col("occurred_at")
            .cast("timestamp")
            .alias("occurred_at"),
        )
        .filter(
            F.col("latitude").isNotNull()
            & F.col("longitude").isNotNull()
            & ~F.isnan("latitude")
            & ~F.isnan("longitude")
            & F.col("latitude").between(
                -90.0,
                90.0,
            )
            & F.col("longitude").between(
                -180.0,
                180.0,
            )
            & F.col("occurred_at").isNotNull()
            & (
                F.to_date("occurred_at")
                >= F.lit(model_start_date)
            )
            & (
                F.to_date("occurred_at")
                <= F.lit(effective_cutoff)
            )
        )
    )
    crimes_with_cells = add_weather_query_cell(
        valid_crimes,
        resolution=h3_resolution,
    )

    cell_years = (
        crimes_with_cells
        .withColumn(
            "weather_year",
            F.year("occurred_at"),
        )
        .groupBy(
            "weather_query_cell_id",
            "weather_year",
        )
        .agg(
            F.count(F.lit(1)).alias(
                "crime_record_count"
            )
        )
    )

    unique_cells = (
        cell_years
        .select("weather_query_cell_id")
        .distinct()
    )

    cell_centers = extract_h3_centers(
        unique_cells
    )

    manifest = (
        cell_years
        .join(
            cell_centers,
            on="weather_query_cell_id",
            how="inner",
        )
        .withColumn(
            "provider",
            F.lit(PROVIDER),
        )
        .withColumn(
            "model",
            F.lit(normalized_model),
        )
        .withColumn(
            "h3_resolution",
            F.lit(h3_resolution),
        )
        .withColumn(
            "start_date",
            F.make_date(
                F.col("weather_year"),
                F.lit(1),
                F.lit(1),
            ),
        )
        .withColumn(
            "end_date",
            F.least(
                F.make_date(
                    F.col("weather_year"),
                    F.lit(12),
                    F.lit(31),
                ),
                F.lit(effective_cutoff),
            ),
        )
        # This is where the date-range filter belongs.
        .filter(
            F.col("start_date")
            <= F.col("end_date")
        )
        .withColumn(
            "timezone",
            F.lit(TIMEZONE),
        )
        .withColumn(
            "cell_selection",
            F.lit(CELL_SELECTION),
        )
        .withColumn(
            "hourly_variables",
            F.array(
                *[
                    F.lit(variable)
                    for variable
                    in normalized_variables
                ]
            ),
        )
        .withColumn(
            "request_id",
            F.sha2(
                F.concat_ws(
                    "|",
                    F.col("provider"),
                    F.col("model"),
                    F.col(
                        "weather_query_cell_id"
                    ).cast("string"),
                    F.col("start_date").cast(
                        "string"
                    ),
                    F.col("end_date").cast(
                        "string"
                    ),
                    F.concat_ws(
                        ",",
                        F.col("hourly_variables"),
                    ),
                    F.col("timezone"),
                    F.col("cell_selection"),
                    F.col("h3_resolution").cast(
                        "string"
                    ),
                ),
                256,
            ),
        )
        .select(
            "request_id",
            "provider",
            "model",
            "weather_query_cell_id",
            "h3_resolution",
            "query_latitude",
            "query_longitude",
            "start_date",
            "end_date",
            "timezone",
            "cell_selection",
            "hourly_variables",
            "crime_record_count",
        )
    )

    return manifest
