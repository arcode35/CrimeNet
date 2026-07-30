from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

CENTER_SCHEMA = """
    type STRING,
    coordinates ARRAY<DOUBLE>
"""

DEFAULT_WEATHER_H3_RESOLUTION = 6


def _databricks_functions() -> Any:
    """Load H3 functions only inside a Databricks runtime."""
    try:
        from pyspark.databricks.sql import functions as dbf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Databricks H3 SQL functions are required for H3 enrichment."
        ) from exc

    return dbf


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

    dbf = _databricks_functions()

    return df.withColumn(
        output_column,
        dbf.h3_longlatash3(
            F.col(longitude_column),
            F.col(latitude_column),
            resolution,
        ),
    )


def extract_h3_centers(
    weather_query_cell_df: DataFrame,
    *,
    cell_column: str = "weather_query_cell_id",
) -> DataFrame:
    dbf = _databricks_functions()

    return (
        weather_query_cell_df
        .withColumn(
            "_center",
            F.from_json(
                dbf.h3_centerasgeojson(
                    F.col(cell_column)
                ),
                CENTER_SCHEMA,
            ),
        )
        .withColumn(
            "query_longitude",
            F.col("_center.coordinates")[0],
        )
        .withColumn(
            "query_latitude",
            F.col("_center.coordinates")[1],
        )
        .drop("_center")
    )
