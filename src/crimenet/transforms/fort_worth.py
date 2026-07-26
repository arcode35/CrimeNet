"""Fort Worth Bronze-to-canonical-Silver transformation."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from crimenet.contracts.silver import SILVER_COLUMNS, assert_silver_contract
from crimenet.transforms.common import (
    null_string,
    timestamp_millis,
)


def to_canonical(dataframe: DataFrame) -> DataFrame:
    result = dataframe.select(
        F.lit("fort_worth").alias("source_city"),
        F.coalesce(
            F.col("case_no_offense").cast("string"),
            F.col("objectid").cast("string"),
        ).alias("source_record_id"),
        F.col("case_no").cast("string").alias("source_incident_id"),
        F.col("offense").cast("string").alias("offense_code"),
        F.col("nature_of_call").cast("string").alias("offense_name"),
        F.col("offense_desc")
        .cast("string")
        .alias("offense_description"),
        timestamp_millis("from_date").alias("occurred_at"),
        timestamp_millis("reported_date").alias("reported_at"),
        timestamp_millis("lastupdated").alias("updated_at"),
        F.lit(1).cast("long").alias("offense_count"),
        F.coalesce(
            F.col("address"),
            F.col("block_address"),
        )
        .cast("string")
        .alias("address"),
        F.col("city").cast("string").alias("city"),
        F.col("state").cast("string").alias("state"),
        null_string().alias("postal_code"),
        F.col("beat").cast("string").alias("beat"),
        F.col("locationtypedescription")
        .cast("string")
        .alias("premise_type"),
        F.col("latitude").cast("double").alias("latitude"),
        F.col("longitude").cast("double").alias("longitude"),
        F.col("alternate_latitude")
        .cast("double")
        .alias("alternate_latitude"),
        F.col("alternate_longitude")
        .cast("double")
        .alias("alternate_longitude"),
        F.col("x_coordinate")
        .cast("double")
        .alias("source_x_coordinate"),
        F.col("y_coordinate")
        .cast("double")
        .alias("source_y_coordinate"),
        F.col("source_file").cast("string").alias("source_file"),
        F.col("source_row_hash").cast("string").alias("source_row_hash"),
    ).select(*SILVER_COLUMNS)

    assert_silver_contract(result)
    return result
