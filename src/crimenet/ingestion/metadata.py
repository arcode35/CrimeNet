"""Reusable ingestion metadata expressions."""

from __future__ import annotations

from collections.abc import Iterable

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

DEFAULT_HASH_EXCLUSIONS = frozenset(
    {
        "_source_file",
        "source_file",
        "source_row_hash",
        "source_system",
        "source_contract_version",
        "retrieved_at",
        "ingested_at",
    }
)


def source_row_hash(
    dataframe: DataFrame,
    excluded_columns: Iterable[str] = DEFAULT_HASH_EXCLUSIONS,
) -> Column:
    """
    Build a deterministic SHA-256 hash from normalized source values.

    The source file is excluded so an identical source record keeps the same
    identity if the file is copied into a different landing subdirectory.
    """
    excluded = set(excluded_columns)
    source_columns = [
        F.col(column_name)
        for column_name in sorted(dataframe.columns)
        if column_name not in excluded
    ]

    if not source_columns:
        raise ValueError("Cannot hash a DataFrame with no eligible columns.")

    return F.sha2(
        F.to_json(
            F.struct(*source_columns),
            options={"ignoreNullFields": "false"},
        ),
        256,
    )


def add_ingestion_metadata(
    dataframe: DataFrame,
    source_system: str,
    *,
    contract_version: str,
) -> DataFrame:
    """Add stable row identity and operational ingestion metadata."""
    return (
        dataframe
        .withColumn("source_row_hash", source_row_hash(dataframe))
        .withColumn("source_system", F.lit(source_system))
        .withColumn("source_contract_version", F.lit(contract_version))
        .withColumn("ingested_at", F.current_timestamp())
    )
