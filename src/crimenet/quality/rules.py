"""Reusable canonical Silver quality-rule expressions."""

from __future__ import annotations

from pyspark.sql import Column
from pyspark.sql import functions as F


def has_source_identity() -> Column:
    return (
        F.col("source_city").isNotNull()
        & F.col("source_record_id").isNotNull()
    )


def coordinates_are_valid() -> Column:
    return (
        (
            F.col("latitude").isNull()
            & F.col("longitude").isNull()
        )
        |
        (
            F.col("latitude").between(-90.0, 90.0)
            & F.col("longitude").between(-180.0, 180.0)
        )
    )


def occurred_at_is_valid() -> Column:
    return F.col("occurred_at").isNotNull()
