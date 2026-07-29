"""Python-wheel entry point for ACS Silver processing."""

from __future__ import annotations

import argparse
from uuid import uuid4

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType
from pyspark.sql.window import Window

from crimenet.config.resources import CrimeNetTables
from crimenet.config.validation import validate_qualified_table_name
from crimenet.observability.logging import get_logger
from crimenet.observability.run_context import resolve_pipeline_run_id
from crimenet.quality.external import split_external_quarantine
from crimenet.quality.quarantine import merge_quarantine
from crimenet.silver.socioeconomic import (
    ACS_QUARANTINE_MESSAGES,
    SOCIOECONOMIC_DEFINITION_VERSION,
    annotate_acs_validation,
    transform_acs5_tracts,
)
from crimenet.utils.promotion import promote_staged_table, quote_table_name

LOGGER = get_logger(__name__)

MERGE_KEYS = (
    "geoid",
    "acs_vintage",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transform Bronze ACS tract data into the Silver "
            "tract socioeconomic table."
        )
    )

    parser.add_argument(
        "--catalog",
        required=True,
    )
    parser.add_argument(
        "--bronze-schema",
        default="bronze",
    )
    parser.add_argument(
        "--silver-schema",
        default="silver",
    )
    parser.add_argument(
        "--data-quality-schema",
        default="data_quality",
    )
    parser.add_argument("--pipeline-run-id")
    parser.add_argument(
        "--checkpoint-path",
        required=False,
        help=(
            "Structured Streaming checkpoint path. Required "
            "unless --full-rebuild is specified."
        ),
    )
    parser.add_argument(
        "--full-rebuild",
        action="store_true",
        help=(
            "Reprocess the complete Bronze table and overwrite "
            "the Silver table."
        ),
    )

    args = parser.parse_args()

    if not args.full_rebuild and not args.checkpoint_path:
        parser.error(
            "--checkpoint-path is required unless "
            "--full-rebuild is specified."
        )

    return args


def deduplicate_socioeconomic_records(
    dataframe: DataFrame,
) -> DataFrame:
    latest_record_window = (
        Window
        .partitionBy(*MERGE_KEYS)
        .orderBy(
            F.col("socioeconomic_definition_version")
            .desc_nulls_last(),
            F.col("source_row_hash")
            .desc_nulls_last(),
            F.col("source_file")
            .desc_nulls_last(),
        )
    )

    return (
        dataframe
        .withColumn(
            "_row_number",
            F.row_number().over(
                latest_record_window
            ),
        )
        .filter(
            F.col("_row_number") == 1
        )
        .drop("_row_number")
    )


def ensure_target_table(
    spark: SparkSession,
    *,
    target_table: str,
    schema: StructType,
) -> None:
    validate_qualified_table_name(target_table)
    if not spark.catalog.tableExists(target_table):
        spark.createDataFrame([], schema).write.format("delta").saveAsTable(
            target_table
        )


def rebuild_socioeconomic_table(
    spark: SparkSession,
    *,
    tables: CrimeNetTables,
    pipeline_run_id: str,
) -> None:
    LOGGER.info(
        "Starting full ACS Silver rebuild",
        source_table=tables.acs5_tract_bronze,
        target_table=tables.tract_socioeconomic_silver,
    )

    bronze_dataframe = spark.table(
        tables.acs5_tract_bronze
    )

    annotated = annotate_acs_validation(bronze_dataframe)
    valid_bronze, quarantine = split_external_quarantine(
        annotated,
        reason_codes_column="_quarantine_reason_codes",
        reason_messages=ACS_QUARANTINE_MESSAGES,
        source_system="census_acs5",
        pipeline_run_id=pipeline_run_id,
    )
    merge_quarantine(
        spark,
        quarantine=quarantine,
        target_table=(
            f"{tables.catalog}.{tables.data_quality_schema}.acs_quarantine"
        ),
    )
    transformed_dataframe = transform_acs5_tracts(
        valid_bronze
    )

    rebuilt_dataframe = (
        deduplicate_socioeconomic_records(
            transformed_dataframe
        )
    )

    promote_staged_table(
        spark,
        candidate=rebuilt_dataframe,
        target_table=tables.tract_socioeconomic_silver,
        pipeline_run_id=pipeline_run_id,
        validate=validate_socioeconomic_dataframe,
    )

    rebuilt_count = spark.table(
        tables.tract_socioeconomic_silver
    ).count()

    LOGGER.info(
        "Completed full ACS Silver rebuild",
        target_table=tables.tract_socioeconomic_silver,
        row_count=rebuilt_count,
    )


def merge_socioeconomic_batch(
    batch_dataframe: DataFrame,
    batch_id: int,
    *,
    spark: SparkSession,
    target_table: str,
) -> None:
    quoted_target_table = quote_table_name(target_table)
    if batch_dataframe.isEmpty():
        LOGGER.info(
            "Skipping empty ACS microbatch",
            batch_id=batch_id,
            target_table=target_table,
        )
        return

    deduplicated_dataframe = (
        deduplicate_socioeconomic_records(
            batch_dataframe
        )
    )

    view_name = f"_crimenet_acs_{uuid4().hex}"
    deduplicated_dataframe.createOrReplaceTempView(view_name)
    try:
        spark.sql(
            f"""
            MERGE INTO {quoted_target_table} AS target
            USING {view_name} AS source
              ON target.geoid = source.geoid
             AND target.acs_vintage = source.acs_vintage
            WHEN MATCHED
              AND coalesce(target.socioeconomic_definition_version, '')
                  <> coalesce(source.socioeconomic_definition_version, '')
              THEN UPDATE SET *
            WHEN MATCHED
              AND coalesce(target.socioeconomic_definition_version, '')
                  = coalesce(source.socioeconomic_definition_version, '')
              AND coalesce(target.source_row_hash, '')
                  < coalesce(source.source_row_hash, '')
              THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *
            """
        )
    finally:
        spark.catalog.dropTempView(view_name)

    LOGGER.info(
        "Merged ACS socioeconomic microbatch",
        batch_id=batch_id,
        target_table=target_table,
    )


def run_incremental_stream(
    spark: SparkSession,
    *,
    tables: CrimeNetTables,
    checkpoint_path: str,
    pipeline_run_id: str,
) -> None:
    LOGGER.info(
        "Starting incremental ACS Silver processing",
        source_table=tables.acs5_tract_bronze,
        target_table=tables.tract_socioeconomic_silver,
        checkpoint_path=checkpoint_path,
    )

    bronze_stream = spark.readStream.table(
        tables.acs5_tract_bronze
    )

    schema_stream = transform_acs5_tracts(
        bronze_stream
    )

    ensure_target_table(
        spark,
        target_table=(
            tables.tract_socioeconomic_silver
        ),
        schema=schema_stream.schema,
    )

    def upsert_batch(
        batch_dataframe: DataFrame,
        batch_id: int,
    ) -> None:
        annotated = annotate_acs_validation(batch_dataframe)
        valid_bronze, quarantine = split_external_quarantine(
            annotated,
            reason_codes_column="_quarantine_reason_codes",
            reason_messages=ACS_QUARANTINE_MESSAGES,
            source_system="census_acs5",
            pipeline_run_id=pipeline_run_id,
        )
        merge_quarantine(
            spark,
            quarantine=quarantine,
            target_table=(
                f"{tables.catalog}.{tables.data_quality_schema}."
                "acs_quarantine"
            ),
        )
        merge_socioeconomic_batch(
            transform_acs5_tracts(valid_bronze),
            batch_id,
            spark=spark,
            target_table=(
                tables.tract_socioeconomic_silver
            ),
        )

    query = (
        bronze_stream.writeStream
        .queryName(
            "silver_acs5_tract_socioeconomic"
        )
        .option(
            "checkpointLocation",
            checkpoint_path,
        )
        .trigger(
            availableNow=True,
        )
        .foreachBatch(
            upsert_batch
        )
        .start()
    )

    query.awaitTermination()

    LOGGER.info(
        "Completed incremental ACS Silver processing",
        target_table=tables.tract_socioeconomic_silver,
    )


def validate_rebuilt_table(
    spark: SparkSession,
    *,
    target_table: str,
) -> None:
    validate_qualified_table_name(target_table)
    validate_socioeconomic_dataframe(spark.table(target_table))
    LOGGER.info(
        "Validated Silver ACS table",
        target_table=target_table,
    )


def validate_socioeconomic_dataframe(target: DataFrame) -> None:
    duplicate_keys = (
        target
        .groupBy(*MERGE_KEYS)
        .count()
        .filter(F.col("count") > 1)
        .limit(1)
        .count()
    )

    invalid_median_ages = (
        target
        .filter(
            F.col("median_age").isNotNull()
            & ~F.col("median_age").between(
                0.0,
                120.0,
            )
        )
        .limit(1)
        .count()
    )

    if duplicate_keys:
        raise RuntimeError(
            "Silver ACS table contains duplicate "
            "(geoid, acs_vintage) keys."
        )

    if invalid_median_ages:
        raise RuntimeError(
            "Silver ACS table still contains invalid "
            "median_age values."
        )


def socioeconomic_rebuild_required(
    *,
    bronze_dataframe: DataFrame,
    target_dataframe: DataFrame,
) -> bool:
    """Return true when the incremental checkpoint cannot refresh history."""
    if bronze_dataframe.limit(1).isEmpty():
        return False
    if target_dataframe.limit(1).isEmpty():
        return True
    if "socioeconomic_definition_version" not in target_dataframe.columns:
        return True
    stale_rows = target_dataframe.filter(
        F.col("socioeconomic_definition_version").isNull()
        | (
            F.col("socioeconomic_definition_version")
            != F.lit(SOCIOECONOMIC_DEFINITION_VERSION)
        )
    )
    return not stale_rows.limit(1).isEmpty()


def run(
    spark: SparkSession,
    *,
    catalog: str,
    bronze_schema: str,
    silver_schema: str,
    data_quality_schema: str,
    checkpoint_path: str | None,
    full_rebuild: bool,
    pipeline_run_id: str | None = None,
) -> None:
    run_id = resolve_pipeline_run_id(pipeline_run_id)
    tables = CrimeNetTables(
        catalog=catalog,
        bronze_schema=bronze_schema,
        silver_schema=silver_schema,
        data_quality_schema=data_quality_schema,
    )

    if not full_rebuild and checkpoint_path is None:
        raise ValueError(
            "checkpoint_path is required for incremental processing."
        )

    if (
        not full_rebuild
        and spark.catalog.tableExists(tables.tract_socioeconomic_silver)
        and socioeconomic_rebuild_required(
            bronze_dataframe=spark.table(tables.acs5_tract_bronze),
            target_dataframe=spark.table(tables.tract_socioeconomic_silver),
        )
    ):
        LOGGER.info(
            "Promoting ACS Silver to the current definition with a full rebuild",
            socioeconomic_definition_version=(
                SOCIOECONOMIC_DEFINITION_VERSION
            ),
        )
        full_rebuild = True

    if full_rebuild:
        rebuild_socioeconomic_table(
            spark,
            tables=tables,
            pipeline_run_id=run_id,
        )
    else:
        assert checkpoint_path is not None

        run_incremental_stream(
            spark,
            tables=tables,
            checkpoint_path=checkpoint_path,
            pipeline_run_id=run_id,
        )

    validate_rebuilt_table(
        spark,
        target_table=(
            tables.tract_socioeconomic_silver
        ),
    )


def main() -> None:
    args = parse_args()

    spark = (
        SparkSession.getActiveSession()
        or SparkSession.builder.getOrCreate()
    )

    LOGGER.info(
        "Starting ACS Silver job",
        catalog=args.catalog,
        bronze_schema=args.bronze_schema,
        silver_schema=args.silver_schema,
        full_rebuild=args.full_rebuild,
    )

    try:
        run(
            spark,
            catalog=args.catalog,
            bronze_schema=args.bronze_schema,
            silver_schema=args.silver_schema,
            data_quality_schema=args.data_quality_schema,
            checkpoint_path=args.checkpoint_path,
            full_rebuild=args.full_rebuild,
            pipeline_run_id=args.pipeline_run_id,
        )
    except Exception:
        LOGGER.exception(
            "ACS Silver job failed",
            catalog=args.catalog,
            full_rebuild=args.full_rebuild,
        )
        raise

    LOGGER.info(
        "ACS Silver job completed",
        catalog=args.catalog,
        silver_schema=args.silver_schema,
        full_rebuild=args.full_rebuild,
    )


if __name__ == "__main__":
    main()
