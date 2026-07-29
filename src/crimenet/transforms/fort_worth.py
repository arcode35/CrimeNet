"""Fort Worth Bronze-to-canonical-Silver transformation."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from crimenet.contracts.silver import SILVER_COLUMNS, assert_silver_contract
from crimenet.transforms.common import (
    CRIME_TRANSFORMATION_VERSION,
    invalid_nonblank_cast,
    null_string,
    stable_business_identity,
    timestamp_millis,
    try_cast,
)


def to_canonical(dataframe: DataFrame) -> DataFrame:
    incident_id = F.col("case_no").cast("string")
    offense_id = F.coalesce(
        F.col("case_no_offense").cast("string"),
        F.col("objectid").cast("string"),
    )
    numeric_columns = (
        "latitude",
        "longitude",
        "alternate_latitude",
        "alternate_longitude",
        "x_coordinate",
        "y_coordinate",
    )
    numeric_parse_error = F.lit(False)
    for column_name in numeric_columns:
        numeric_parse_error = numeric_parse_error | invalid_nonblank_cast(
            column_name,
            "double",
        )
    source_validation_payload = F.when(
        numeric_parse_error,
        F.to_json(
            F.struct(
                F.lit("INVALID_NUMERIC_TEXT").alias("reason"),
                *(F.col(column_name) for column_name in numeric_columns),
            )
        ),
    )

    result = dataframe.select(
        F.lit("fort_worth").alias("source_system"),
        F.lit("fort_worth").alias("source_city"),
        offense_id.alias("source_record_id"),
        incident_id.alias("source_incident_id"),
        offense_id.alias("source_offense_id"),
        stable_business_identity(
            source_system="fort_worth",
            source_incident_id=incident_id,
            source_offense_id=offense_id,
        ).alias("business_identity"),
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
        try_cast("alternate_latitude", "double").alias("alternate_latitude"),
        try_cast("alternate_longitude", "double").alias("alternate_longitude"),
        try_cast("x_coordinate", "double").alias("source_x_coordinate"),
        try_cast("y_coordinate", "double").alias("source_y_coordinate"),
        F.col("source_file").cast("string").alias("source_file"),
        F.col("source_row_hash").cast("string").alias("source_row_hash"),
        F.col("source_contract_version")
        .cast("string")
        .alias("source_contract_version"),
        F.lit(CRIME_TRANSFORMATION_VERSION).alias(
            "transformation_version"
        ),
        F.col("ingested_at").cast("timestamp").alias("bronze_ingested_at"),
        F.coalesce(
            F.col("corrupt_record").cast("string"),
            source_validation_payload,
        ).alias("source_corrupt_record"),
    ).select(*SILVER_COLUMNS)

    assert_silver_contract(result)
    return result
