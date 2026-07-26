"""Common PySpark expressions used by source transformations."""

from __future__ import annotations

from pyspark.sql import Column
from pyspark.sql import functions as F


def null_string() -> Column:
    return F.lit(None).cast("string")


def null_long() -> Column:
    return F.lit(None).cast("long")


def null_double() -> Column:
    return F.lit(None).cast("double")


def null_timestamp() -> Column:
    return F.lit(None).cast("timestamp")


def try_cast(column_name: str, data_type: str) -> Column:
    """ANSI-safe cast for a trusted internal column name."""
    return F.expr(f"try_cast(`{column_name}` AS {data_type})")


def timestamp_millis(column_name: str) -> Column:
    """Convert a possibly string/long epoch-millisecond field to timestamp."""
    return F.expr(
        f"timestamp_millis(try_cast(`{column_name}` AS BIGINT))"
    )


def trimmed_address(*column_names: str) -> Column:
    """Join nullable address components with one space."""
    return F.trim(
        F.concat_ws(
            " ",
            *(F.col(column_name) for column_name in column_names),
        )
    )
