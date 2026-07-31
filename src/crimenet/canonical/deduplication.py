"""Canonical record deduplication."""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def deduplicate_canonical_records(
    dataframe: DataFrame,
) -> DataFrame:
    identifier = F.trim(
        F.col("canonical_record_id")
    )

    valid = dataframe.filter(
        F.col(
            "canonical_record_id"
        ).isNotNull()
        & (identifier != "")
    )

    invalid = dataframe.filter(
        F.col(
            "canonical_record_id"
        ).isNull()
        | (identifier == "")
    )

    window = (
        Window
        .partitionBy(
            "canonical_record_id"
        )
        .orderBy(
            F.col(
                "updated_at"
            ).desc_nulls_last(),
            F.col(
                "downloaded_at_utc"
            ).desc_nulls_last(),
            F.col(
                "ingested_at_utc"
            ).desc_nulls_last(),
            F.col(
                "source_row_hash"
            ).desc_nulls_last(),
            F.col(
                "source_file"
            ).desc_nulls_last(),
        )
    )

    deduplicated = (
        valid
        .withColumn(
            "_canonical_rank",
            F.row_number().over(window),
        )
        .filter(
            F.col("_canonical_rank") == 1
        )
        .drop("_canonical_rank")
    )

    return deduplicated.unionByName(
        invalid,
        allowMissingColumns=False,
    )