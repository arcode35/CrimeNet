"""Fort Worth Bronze-to-canonical-Silver transformation."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from crimenet.contracts.silver import SILVER_COLUMNS, assert_silver_contract
from crimenet.transforms.common import (
    null_string,
    require_columns,
    timestamp_millis,
    try_cast,
)

_REQUIRED_COLUMNS = (
    "case_no_offense",
    "objectid",
    "case_no",
    "offense",
    "nature_of_call",
    "offense_desc",
    "from_date",
    "reported_date",
    "lastupdated",
    "address",
    "block_address",
    "city",
    "state",
    "beat",
    "locationtypedescription",
    "latitude",
    "longitude",
    "alternate_latitude",
    "alternate_longitude",
    "x_coordinate",
    "y_coordinate",
    "source_file",
    "source_row_hash",
)


def to_canonical(dataframe: DataFrame) -> DataFrame:
    require_columns(
        dataframe,
        _REQUIRED_COLUMNS,
        context="Fort Worth canonical input",
    )
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
        try_cast("latitude", "double").alias("latitude"),
        try_cast("longitude", "double").alias("longitude"),
        try_cast(
            "alternate_latitude",
            "double",
        ).alias("alternate_latitude"),
        try_cast(
            "alternate_longitude",
            "double",
        ).alias("alternate_longitude"),
        try_cast(
            "x_coordinate",
            "double",
        ).alias("source_x_coordinate"),
        try_cast(
            "y_coordinate",
            "double",
        ).alias("source_y_coordinate"),
        F.col("source_file").cast("string").alias("source_file"),
        F.col("source_row_hash").cast("string").alias("source_row_hash"),
    ).select(*SILVER_COLUMNS)

    assert_silver_contract(result)
    return result
