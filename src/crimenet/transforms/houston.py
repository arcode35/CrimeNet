"""Houston Bronze-to-canonical-Silver transformation."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from crimenet.contracts.silver import SILVER_COLUMNS, assert_silver_contract
from crimenet.transforms.common import (
    null_double,
    null_timestamp,
    trimmed_address,
    try_cast,
)


def to_canonical(dataframe: DataFrame) -> DataFrame:
    result = dataframe.select(
        F.lit("houston").alias("source_city"),
        F.col("source_row_hash").cast("string").alias("source_record_id"),
        F.col("incident").cast("string").alias("source_incident_id"),
        F.col("nibrsclass").cast("string").alias("offense_code"),
        F.col("nibrsdescription").cast("string").alias("offense_name"),
        F.col("nibrsdescription")
        .cast("string")
        .alias("offense_description"),
        F.expr(
            "try_to_timestamp("
            "concat(`rmsoccurrencedate`, ' ', "
            "lpad(`rmsoccurrencehour`, 2, '0')), "
            "'M/d/yyyy HH'"
            ")"
        ).alias("occurred_at"),
        null_timestamp().alias("reported_at"),
        null_timestamp().alias("updated_at"),
        try_cast("offensecount", "long").alias("offense_count"),
        trimmed_address(
            "streetno",
            "streetname",
            "streettype",
            "suffix",
        ).alias("address"),
        F.col("city").cast("string").alias("city"),
        F.lit("TX").cast("string").alias("state"),
        F.col("zipcode").cast("string").alias("postal_code"),
        F.col("beat").cast("string").alias("beat"),
        F.col("premise").cast("string").alias("premise_type"),
        try_cast("maplatitude", "double").alias("latitude"),
        try_cast("maplongitude", "double").alias("longitude"),
        null_double().alias("alternate_latitude"),
        null_double().alias("alternate_longitude"),
        null_double().alias("source_x_coordinate"),
        null_double().alias("source_y_coordinate"),
        F.col("source_file").cast("string").alias("source_file"),
        F.col("source_row_hash").cast("string").alias("source_row_hash"),
    ).select(*SILVER_COLUMNS)

    assert_silver_contract(result)
    return result
