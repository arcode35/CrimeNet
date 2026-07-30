"""Canonical Silver offense-level contract."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

SILVER_SCHEMA = StructType(
    [
        StructField("source_city", StringType(), False),
        StructField("source_record_id", StringType(), True),
        StructField("source_incident_id", StringType(), True),
        StructField("offense_code", StringType(), True),
        StructField("offense_name", StringType(), True),
        StructField("offense_description", StringType(), True),
        StructField("occurred_at", TimestampType(), True),
        StructField("reported_at", TimestampType(), True),
        StructField("updated_at", TimestampType(), True),
        StructField("offense_count", LongType(), True),
        StructField("address", StringType(), True),
        StructField("city", StringType(), True),
        StructField("state", StringType(), True),
        StructField("postal_code", StringType(), True),
        StructField("beat", StringType(), True),
        StructField("premise_type", StringType(), True),
        StructField("latitude", DoubleType(), True),
        StructField("longitude", DoubleType(), True),
        StructField("alternate_latitude", DoubleType(), True),
        StructField("alternate_longitude", DoubleType(), True),
        StructField("source_x_coordinate", DoubleType(), True),
        StructField("source_y_coordinate", DoubleType(), True),
        StructField("source_file", StringType(), True),
        StructField("source_row_hash", StringType(), False),
    ]
)

SILVER_COLUMNS = tuple(field.name for field in SILVER_SCHEMA.fields)


def assert_silver_contract(dataframe: DataFrame) -> None:
    """Fail early when a source transform violates the canonical contract."""
    actual_fields = dataframe.schema.fields
    expected_fields = SILVER_SCHEMA.fields

    if len(actual_fields) != len(expected_fields):
        raise ValueError(
            "Silver schema has the wrong number of columns: "
            f"expected={len(expected_fields)}, actual={len(actual_fields)}"
        )

    mismatches: list[str] = []

    for expected, actual in zip(
    expected_fields,
    actual_fields,
    strict=True):
        if expected.name != actual.name or expected.dataType != actual.dataType:
            mismatches.append(
                f"expected {expected.name}:{expected.dataType.simpleString()}, "
                f"got {actual.name}:{actual.dataType.simpleString()}"
            )

    if mismatches:
        raise ValueError(
            "Silver schema contract mismatch:\n- " + "\n- ".join(mismatches)
        )
