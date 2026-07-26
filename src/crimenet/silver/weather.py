"""Transform raw Open-Meteo responses into hourly weather records."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    StringType,
    StructField,
    StructType,
)


HOURLY_WEATHER_SCHEMA = StructType(
    [
        StructField(
            "time",
            ArrayType(
                StringType(),
                containsNull=True,
            ),
            nullable=True,
        ),
        StructField(
            "temperature_2m",
            ArrayType(
                DoubleType(),
                containsNull=True,
            ),
            nullable=True,
        ),
    ]
)


HOURLY_UNITS_SCHEMA = StructType(
    [
        StructField(
            "time",
            StringType(),
            nullable=True,
        ),
        StructField(
            "temperature_2m",
            StringType(),
            nullable=True,
        ),
    ]
)


def transform_open_meteo_weather(
    bronze_dataframe: DataFrame,
) -> DataFrame:
    """Parse and explode raw Open-Meteo responses to hourly grain."""

    parsed_dataframe = (
        bronze_dataframe
        .withColumn(
            "_hourly",
            F.from_json(
                F.col("hourly"),
                HOURLY_WEATHER_SCHEMA,
            ),
        )
        .withColumn(
            "_hourly_units",
            F.from_json(
                F.col("hourly_units"),
                HOURLY_UNITS_SCHEMA,
            ),
        )
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
        F.col("_hourly").isNotNull()
        & (
            F.col("_timestamp_count")
            == F.col("_temperature_count")
        )
        & (
            F.col("_timestamp_count") > 0
        )
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
        .filter(
            F.col("provider").isNotNull()
            & F.col("model").isNotNull()
            & F.col(
                "weather_query_cell_id"
            ).isNotNull()
            & F.col(
                "weather_timestamp"
            ).isNotNull()
        )
    )

    return silver_dataframe