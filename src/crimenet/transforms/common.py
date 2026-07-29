"""Common PySpark expressions used by source transformations."""

from __future__ import annotations

from pyspark.sql import Column
from pyspark.sql import functions as F

CRIME_TRANSFORMATION_VERSION = "canonical_crime_v4"
MUNICIPAL_SOURCE_TIME_ZONE = "America/Chicago"


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


def invalid_nonblank_cast(column_name: str, data_type: str) -> Column:
    """Identify a populated source value that cannot be safely cast."""
    source = F.trim(F.col(column_name).cast("string"))
    return (
        source.isNotNull()
        & (F.length(source) > 0)
        & try_cast(column_name, data_type).isNull()
    )


def timestamp_millis(column_name: str) -> Column:
    """Convert a possibly string/long epoch-millisecond field to timestamp."""
    return F.expr(
        f"timestamp_millis(try_cast(`{column_name}` AS BIGINT))"
    )


def municipal_local_to_utc(timestamp: Column) -> Column:
    """Convert an unambiguous Texas wall clock to UTC.

    Source timestamps do not carry an offset. Times in the spring-forward gap
    never occurred, while times in the fall-back overlap identify two possible
    instants. Return null for both cases so they enter the existing quarantine
    path instead of silently accepting Spark's offset choice.
    """
    utc_candidate = F.to_utc_timestamp(
        timestamp,
        MUNICIPAL_SOURCE_TIME_ZONE,
    )
    round_trip = F.from_utc_timestamp(
        utc_candidate,
        MUNICIPAL_SOURCE_TIME_ZONE,
    )
    previous_hour = F.from_utc_timestamp(
        utc_candidate - F.expr("INTERVAL 1 HOUR"),
        MUNICIPAL_SOURCE_TIME_ZONE,
    )
    next_hour = F.from_utc_timestamp(
        utc_candidate + F.expr("INTERVAL 1 HOUR"),
        MUNICIPAL_SOURCE_TIME_ZONE,
    )
    is_unambiguous = (
        timestamp.isNotNull()
        & round_trip.eqNullSafe(timestamp)
        & ~previous_hour.eqNullSafe(timestamp)
        & ~next_hour.eqNullSafe(timestamp)
    )
    return F.when(is_unambiguous, utc_candidate)


def trimmed_address(*column_names: str) -> Column:
    """Join nullable address components with one space."""
    return F.trim(
        F.concat_ws(
            " ",
            *(F.col(column_name) for column_name in column_names),
        )
    )


def normalized_identifier(column_name: str) -> Column:
    """Normalize blank source identifiers to null."""
    value = F.trim(F.col(column_name).cast("string"))
    return F.when(F.length(value) > 0, value)


def stable_business_identity(
    *,
    source_system: str,
    source_incident_id: Column,
    source_offense_id: Column,
) -> Column:
    """Hash a logical source key without physical ingestion metadata."""
    return F.sha2(
        F.concat_ws(
            "||",
            F.lit(source_system),
            F.coalesce(source_incident_id.cast("string"), F.lit("")),
            F.coalesce(source_offense_id.cast("string"), F.lit("")),
        ),
        256,
    )
