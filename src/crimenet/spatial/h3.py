from __future__ import annotations

import os
from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StructField,
    StructType,
)

try:
    from pyspark.databricks.sql import functions as dbf
except ModuleNotFoundError:  # Ordinary local Spark and CI.
    dbf = None  # type: ignore[assignment]

CENTER_SCHEMA = """
    type STRING,
    coordinates ARRAY<DOUBLE>
"""

DEFAULT_WEATHER_H3_RESOLUTION = 6


def _local_h3_module() -> Any:
    if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        raise RuntimeError(
            "Databricks H3 functions are unavailable in this runtime; "
            "refusing to use the local Python compatibility path in a "
            "Databricks production process."
        )
    try:
        import h3
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Databricks H3 SQL functions are unavailable and the local "
            "'h3' compatibility package is not installed."
        ) from exc
    return h3


def add_weather_query_cell(
    df: DataFrame,
    *,
    resolution: int = DEFAULT_WEATHER_H3_RESOLUTION,
    latitude_column: str = "latitude",
    longitude_column: str = "longitude",
    output_column: str = "weather_query_cell_id",
) -> DataFrame:
    if not 0 <= resolution <= 15:
        raise ValueError(
            "H3 resolution must be between 0 and 15"
        )

    if dbf is not None:
        expression = dbf.h3_longlatash3(
            F.col(longitude_column),
            F.col(latitude_column),
            resolution,
        )
    else:
        h3 = _local_h3_module()

        @F.udf(returnType=LongType())
        def local_h3(latitude: float | None, longitude: float | None) -> int | None:
            if latitude is None or longitude is None:
                return None
            return int(h3.latlng_to_cell(latitude, longitude, resolution), 16)

        expression = local_h3(
            F.col(latitude_column),
            F.col(longitude_column),
        )

    return df.withColumn(output_column, expression)


def extract_h3_centers(
    weather_query_cell_df: DataFrame,
    *,
    cell_column: str = "weather_query_cell_id",
) -> DataFrame:
    if dbf is not None:
        with_center = weather_query_cell_df.withColumn(
            "_center",
            F.from_json(
                dbf.h3_centerasgeojson(F.col(cell_column)),
                CENTER_SCHEMA,
            ),
        )
        longitude = F.col("_center.coordinates")[0]
        latitude = F.col("_center.coordinates")[1]
    else:
        h3 = _local_h3_module()
        local_center_schema = StructType(
            [
                StructField("latitude", DoubleType(), True),
                StructField("longitude", DoubleType(), True),
            ]
        )

        @F.udf(returnType=local_center_schema)
        def local_center(cell: int | None) -> tuple[float, float] | None:
            if cell is None:
                return None
            latitude_value, longitude_value = h3.cell_to_latlng(format(cell, "x"))
            return float(latitude_value), float(longitude_value)

        with_center = weather_query_cell_df.withColumn(
            "_center",
            local_center(F.col(cell_column)),
        )
        longitude = F.col("_center.longitude")
        latitude = F.col("_center.latitude")

    return (
        with_center
        .withColumn(
            "query_longitude",
            longitude,
        )
        .withColumn(
            "query_latitude",
            latitude,
        )
        .drop("_center")
    )
