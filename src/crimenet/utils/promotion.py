"""Validated, run-scoped Delta table promotion helpers."""

from __future__ import annotations

from collections.abc import Callable

from pyspark.sql import DataFrame, SparkSession

from crimenet.config.validation import (
    normalize_pipeline_run_id,
    validate_qualified_table_name,
)

CandidateValidator = Callable[[DataFrame], None]


def staging_table_name(
    target_table: str,
    pipeline_run_id: str | None,
) -> str:
    """Build a deterministic run-scoped table beside the final table."""
    validate_qualified_table_name(target_table)
    catalog, schema, table = target_table.split(".")
    run_id = normalize_pipeline_run_id(pipeline_run_id)
    return f"{catalog}.{schema}.{table}__staging__{run_id}"


def quote_table_name(table_name: str) -> str:
    """Quote every component of a qualified Spark table identifier."""
    validate_qualified_table_name(table_name)
    return ".".join(f"`{component}`" for component in table_name.split("."))


def promote_staged_delta_table(
    spark: SparkSession,
    *,
    staging_table: str,
    target_table: str,
) -> None:
    """Replace one final Delta table from a previously validated stage."""
    spark.sql(
        "CREATE OR REPLACE TABLE "
        f"{quote_table_name(target_table)} "
        "DEEP CLONE "
        f"{quote_table_name(staging_table)}"
    )


def drop_staging_table(
    spark: SparkSession,
    staging_table: str,
) -> None:
    """Remove one explicitly named, validated run-scoped staging table."""
    spark.sql(f"DROP TABLE IF EXISTS {quote_table_name(staging_table)}")


def promote_staged_table(
    spark: SparkSession,
    *,
    candidate: DataFrame,
    target_table: str,
    pipeline_run_id: str,
    validate: CandidateValidator,
) -> str:
    """
    Validate a staged Delta candidate before replacing one final table.

    ``CREATE OR REPLACE TABLE`` is one Delta table commit; it is not a
    transaction with any other table. The previous final table is untouched
    when staging or validation fails.
    """
    validate_qualified_table_name(target_table)
    stage = staging_table_name(target_table, pipeline_run_id)
    candidate.write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(stage)

    try:
        validate(spark.table(stage))
        spark.sql(
            f"CREATE OR REPLACE TABLE {target_table} "
            f"USING DELTA AS SELECT * FROM {stage}"
        )
    except Exception:
        spark.sql(f"DROP TABLE IF EXISTS {stage}")
        raise

    spark.sql(f"DROP TABLE IF EXISTS {stage}")
    return target_table
