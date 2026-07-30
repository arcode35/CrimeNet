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
from pyspark.sql.window import Window

WEATHER_MERGE_KEYS = (
    "provider",
    "model",
    "weather_query_cell_id",
    "weather_timestamp",
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


HOURLY_WEATHER_JSON_SCHEMA = StructType(
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
                StringType(),
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


def _require_column(
    dataframe: DataFrame,
    column_name: str,
) -> None:
    if column_name not in dataframe.columns:
        raise ValueError(
            "Weather input is missing required column "
            f"{column_name!r}."
        )


def _parsed_hourly_column(
    dataframe: DataFrame,
) -> F.Column:
    """Return one typed hourly struct from JSON text or a Spark struct."""
    _require_column(dataframe, "hourly")

    hourly_type = dataframe.schema["hourly"].dataType

    if isinstance(hourly_type, StringType):
        raw_hourly = F.from_json(
            F.col("hourly"),
            HOURLY_WEATHER_JSON_SCHEMA,
        )
    elif isinstance(hourly_type, StructType):
        raw_hourly = F.col("hourly")
    else:
        raise TypeError(
            "Weather column 'hourly' must be JSON text or a struct; "
            f"found {hourly_type.simpleString()}."
        )

    return F.struct(
        F.transform(
            raw_hourly["time"],
            lambda value: value.cast("string"),
        ).alias("time"),
        F.transform(
            raw_hourly["temperature_2m"],
            lambda value: value.try_cast("double"),
        ).alias("temperature_2m"),
    )


def _parsed_hourly_units_column(
    dataframe: DataFrame,
) -> F.Column:
    """Return hourly units from JSON text or a Spark struct."""
    _require_column(dataframe, "hourly_units")

    units_type = dataframe.schema[
        "hourly_units"
    ].dataType

    if isinstance(units_type, StringType):
        raw_units = F.from_json(
            F.col("hourly_units"),
            HOURLY_UNITS_SCHEMA,
        )
    elif isinstance(units_type, StructType):
        raw_units = F.col("hourly_units")
    else:
        raise TypeError(
            "Weather column 'hourly_units' must be JSON text or a "
            f"struct; found {units_type.simpleString()}."
        )

    return F.struct(
        raw_units["time"].cast("string").alias("time"),
        raw_units["temperature_2m"]
        .cast("string")
        .alias("temperature_2m"),
    )


def transform_open_meteo_weather(
    bronze_dataframe: DataFrame,
) -> DataFrame:
    """Parse and explode raw Open-Meteo responses to hourly grain."""

    parsed_dataframe = (
        bronze_dataframe
        .withColumn(
            "_hourly",
            _parsed_hourly_column(
                bronze_dataframe
            ),
        )
        .withColumn(
            "_hourly_units",
            _parsed_hourly_units_column(
                bronze_dataframe
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
            .try_cast("long")
            .alias("weather_query_cell_id"),
            F.col("h3_resolution")
            .try_cast("int")
            .alias("h3_resolution"),
            F.col("query_latitude")
            .try_cast("double")
            .alias("query_latitude"),
            F.col("query_longitude")
            .try_cast("double")
            .alias("query_longitude"),
            F.col("grid_latitude")
            .try_cast("double")
            .alias("grid_latitude"),
            F.col("grid_longitude")
            .try_cast("double")
            .alias("grid_longitude"),
            F.col("grid_elevation")
            .try_cast("double")
            .alias("grid_elevation"),
            F.try_to_timestamp(
                F.col("_observation.time"),
                F.lit("yyyy-MM-dd'T'HH:mm"),
            ).alias("weather_timestamp"),
            F.col(
                "_observation.temperature_2m"
            )
            .try_cast("double")
            .alias("temperature_2m_c"),
            F.col(
                "_hourly_units.temperature_2m"
            ).alias("temperature_unit"),
            F.col("timezone"),
            F.col("utc_offset_seconds")
            .try_cast("int")
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
            & (
                F.col("weather_timestamp")
                == F.date_trunc(
                    "hour",
                    F.col("weather_timestamp"),
                )
            )
        )
    )

    return silver_dataframe


def deduplicate_weather_records(
    dataframe: DataFrame,
    *,
    keys: tuple[str, ...] = WEATHER_MERGE_KEYS,
) -> DataFrame:
    """Select one deterministic weather row for each materialization key."""
    missing_columns = [
        column_name
        for column_name in keys
        if column_name not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Weather deduplication is missing key columns: "
            f"{missing_columns}"
        )

    stable_columns = sorted(
        column_name
        for column_name in dataframe.columns
        if column_name
        not in {
            "bronze_ingested_at",
            "silver_processed_at",
        }
    )

    fingerprint = F.sha2(
        F.to_json(
            F.struct(
                *[
                    F.col(column_name)
                    for column_name in stable_columns
                ]
            ),
            options={"ignoreNullFields": "false"},
        ),
        256,
    )

    latest_record_window = (
        Window
        .partitionBy(*keys)
        .orderBy(
            F.col("bronze_ingested_at")
            .desc_nulls_last(),
            fingerprint.desc(),
            F.col("source_row_hash")
            .desc_nulls_last(),
            F.col("source_file")
            .desc_nulls_last(),
        )
    )

    return (
        dataframe
        .withColumn(
            "_weather_deduplication_rank",
            F.row_number().over(
                latest_record_window
            ),
        )
        .filter(
            F.col(
                "_weather_deduplication_rank"
            )
            == 1
        )
        .drop(
            "_weather_deduplication_rank"
        )
    )
