"""Deterministic canonical crime duplicate resolution."""

from __future__ import annotations

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

COMPLETENESS_COLUMNS = (
    "source_incident_id",
    "source_offense_id",
    "offense_code",
    "offense_name",
    "offense_description",
    "occurred_at",
    "reported_at",
    "updated_at",
    "address",
    "city",
    "state",
    "postal_code",
    "latitude",
    "longitude",
)


def record_completeness_score(dataframe: DataFrame) -> Column:
    """Count populated canonical values used by duplicate precedence."""
    available = [
        name for name in COMPLETENESS_COLUMNS if name in dataframe.columns
    ]
    return sum(
        (
            F.when(
                F.col(name).isNotNull()
                & (
                    ~F.col(name).cast("string").isin("", "null")
                ),
                F.lit(1),
            ).otherwise(F.lit(0))
            for name in available
        ),
        F.lit(0),
    )


def deduplicate_crime_offenses(dataframe: DataFrame) -> DataFrame:
    """
    Resolve one row per stable business identity.

    Precedence is newest source update, highest completeness, newest report,
    stable content hash, ingestion timestamp, then lineage path. The last path
    tie-breaker affects lineage only and never the business identity.
    """
    if "business_identity" not in dataframe.columns:
        raise ValueError("business_identity is required for deduplication.")

    ranked = dataframe.withColumn(
        "_completeness_score",
        record_completeness_score(dataframe),
    )
    window = Window.partitionBy("business_identity").orderBy(
        F.col("updated_at").desc_nulls_last(),
        F.col("_completeness_score").desc(),
        F.col("reported_at").desc_nulls_last(),
        F.col("source_row_hash").desc_nulls_last(),
        F.col("bronze_ingested_at").desc_nulls_last(),
        F.col("source_file").asc_nulls_last(),
    )
    return (
        ranked.withColumn("_row_number", F.row_number().over(window))
        .filter(F.col("_row_number") == 1)
        .drop("_row_number", "_completeness_score")
    )
