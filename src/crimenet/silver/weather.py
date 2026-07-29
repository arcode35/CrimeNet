"""Transform raw Open-Meteo responses into hourly weather records."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

WEATHER_QUARANTINE_MESSAGES = {
    "RESCUED_SCHEMA_DATA": "Auto Loader rescued unexpected weather fields.",
    "MISSING_REQUEST_KEY": "A required weather request key is missing.",
    "UNSUPPORTED_PROVIDER_OR_MODEL": "The weather provider or model is unsupported.",
    "MISSING_HOURLY_DATA": "The response contains no hourly arrays.",
    "HOURLY_ARRAY_LENGTH_MISMATCH": (
        "Hourly timestamps and temperature arrays have different lengths."
    ),
    "EMPTY_HOURLY_DATA": "The response contains no hourly observations.",
    "INVALID_HOURLY_TIMESTAMP": "At least one hourly timestamp is invalid.",
    "INVALID_QUERY_COORDINATES": "Weather query coordinates are invalid.",
}
WEATHER_DEFINITION_VERSION = "open_meteo_hourly_v1"


def annotate_weather_validation(bronze_dataframe: DataFrame) -> DataFrame:
    """Attach auditable response-level validation reason codes."""
    hourly = F.col("hourly")
    timestamp_count = F.size(F.col("hourly.time"))
    temperature_count = F.size(F.col("hourly.temperature_2m"))
    invalid_timestamp = F.exists(
        F.col("hourly.time"),
        lambda value: F.try_to_timestamp(
            value,
            F.lit("yyyy-MM-dd'T'HH:mm"),
        ).isNull(),
    )
    reasons = F.array_compact(
        F.array(
            F.when(
                F.col("rescued_data").isNotNull(),
                F.lit("RESCUED_SCHEMA_DATA"),
            ),
            F.when(
                F.col("request_id").isNull()
                | F.col("weather_query_cell_id").isNull(),
                F.lit("MISSING_REQUEST_KEY"),
            ),
            F.when(
                F.col("provider").isNull()
                | F.col("model").isNull()
                | (F.col("provider") != "open_meteo")
                | ~F.col("model").isin("era5", "era5_land"),
                F.lit("UNSUPPORTED_PROVIDER_OR_MODEL"),
            ),
            F.when(
                hourly.isNull()
                | F.col("hourly.time").isNull()
                | F.col("hourly.temperature_2m").isNull(),
                F.lit("MISSING_HOURLY_DATA"),
            ),
            F.when(
                timestamp_count != temperature_count,
                F.lit("HOURLY_ARRAY_LENGTH_MISMATCH"),
            ),
            F.when(timestamp_count <= 0, F.lit("EMPTY_HOURLY_DATA")),
            F.when(
                F.coalesce(invalid_timestamp, F.lit(False)),
                F.lit("INVALID_HOURLY_TIMESTAMP"),
            ),
            F.when(
                F.col("query_latitude").isNull()
                | F.col("query_longitude").isNull()
                | F.isnan("query_latitude")
                | F.isnan("query_longitude")
                | ~F.col("query_latitude").between(-90.0, 90.0)
                | ~F.col("query_longitude").between(-180.0, 180.0),
                F.lit("INVALID_QUERY_COORDINATES"),
            ),
        )
    )
    return bronze_dataframe.withColumn("_quarantine_reason_codes", reasons)


def transform_open_meteo_weather(
    bronze_dataframe: DataFrame,
) -> DataFrame:
    """Parse and explode raw Open-Meteo responses to hourly grain."""

    parsed_dataframe = (
        annotate_weather_validation(bronze_dataframe)
        .withColumn("_hourly", F.col("hourly"))
        .withColumn("_hourly_units", F.col("hourly_units"))
        .withColumn(
            "_timestamp_count",
            F.size(
                F.col("_hourly.time")
            ),
        )
        .withColumn(
            "_temperature_count",
            F.size(
                F.col(
                    "_hourly.temperature_2m"
                )
            ),
        )
    )

    valid_dataframe = parsed_dataframe.filter(
        F.size("_quarantine_reason_codes") == 0
    )

    exploded_dataframe = (
        valid_dataframe
        .withColumn(
            "_observation",
            F.explode(
                F.arrays_zip(
                    F.col("_hourly.time"),
                    F.col(
                        "_hourly.temperature_2m"
                    ),
                )
            ),
        )
    )

    silver_dataframe = (
        exploded_dataframe
        .select(
            F.col("provider"),
            F.col("model"),
            F.col("request_id"),
            F.col("weather_query_cell_id")
            .cast("long")
            .alias("weather_query_cell_id"),
            F.col("h3_resolution")
            .cast("int")
            .alias("h3_resolution"),
            F.col("query_latitude")
            .cast("double")
            .alias("query_latitude"),
            F.col("query_longitude")
            .cast("double")
            .alias("query_longitude"),
            F.col("grid_latitude")
            .cast("double")
            .alias("grid_latitude"),
            F.col("grid_longitude")
            .cast("double")
            .alias("grid_longitude"),
            F.col("grid_elevation")
            .cast("double")
            .alias("grid_elevation"),
            F.to_timestamp(
                F.col("_observation.time"),
                "yyyy-MM-dd'T'HH:mm",
            ).alias("weather_timestamp"),
            F.col(
                "_observation.temperature_2m"
            )
            .cast("double")
            .alias("temperature_2m_c"),
            F.col(
                "_hourly_units.temperature_2m"
            ).alias("temperature_unit"),
            F.col("timezone"),
            F.col("utc_offset_seconds")
            .cast("int")
            .alias("utc_offset_seconds"),
            F.col("source_file"),
            F.col("source_row_hash"),
            F.col("source_contract_version"),
            F.col("ingested_at")
            .alias("bronze_ingested_at"),
            F.current_timestamp()
            .alias("silver_processed_at"),
        )
        .withColumn(
            "weather_date",
            F.to_date(
                F.col("weather_timestamp")
            ),
        )
        .withColumn(
            "weather_definition_version",
            F.lit(WEATHER_DEFINITION_VERSION),
        )
    )

    return silver_dataframe
