"""Python-wheel entry point for ACS Silver processing."""

from __future__ import annotations

import argparse
from importlib import import_module

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType

from crimenet.config.resources import CrimeNetTables
from crimenet.observability.logging import get_logger
from crimenet.silver.socioeconomic import (
    SOCIOECONOMIC_KEYS,
    deduplicate_socioeconomic_records,
    transform_acs5_tracts,
)

LOGGER = get_logger(__name__)

MERGE_KEYS = SOCIOECONOMIC_KEYS


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


def ensure_target_table(
    spark: SparkSession,
    *,
    target_table: str,
    schema: StructType,
) -> None:
    DeltaTable = import_module(
        "delta.tables"
    ).DeltaTable

    (
        DeltaTable.createIfNotExists(spark)
        .tableName(target_table)
        .addColumns(schema)
        .comment(
            "ACS 5-year tract-level socioeconomic features"
        )
        .execute()
    )


def rebuild_socioeconomic_table(
    spark: SparkSession,
    *,
    tables: CrimeNetTables,
) -> None:
    LOGGER.info(
        "Starting full ACS Silver rebuild",
        source_table=tables.acs5_tract_bronze,
        target_table=tables.tract_socioeconomic_silver,
    )

    bronze_dataframe = spark.table(
        tables.acs5_tract_bronze
    )

    transformed_dataframe = transform_acs5_tracts(
        bronze_dataframe
    )

    rebuilt_dataframe = (
        deduplicate_socioeconomic_records(
            transformed_dataframe
        )
    )

    (
        rebuilt_dataframe.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(
            tables.tract_socioeconomic_silver
        )
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

    DeltaTable = import_module(
        "delta.tables"
    ).DeltaTable

    target = DeltaTable.forName(
        spark,
        target_table,
    )

    merge_condition = """
        target.geoid = source.geoid
        AND target.acs_vintage = source.acs_vintage
    """

    (
        target.alias("target")
        .merge(
            deduplicated_dataframe.alias("source"),
            merge_condition,
        )
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )

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

    silver_stream = transform_acs5_tracts(
        bronze_stream
    )

    ensure_target_table(
        spark,
        target_table=(
            tables.tract_socioeconomic_silver
        ),
        schema=silver_stream.schema,
    )

    def upsert_batch(
        batch_dataframe: DataFrame,
        batch_id: int,
    ) -> None:
        merge_socioeconomic_batch(
            batch_dataframe,
            batch_id,
            spark=spark,
            target_table=(
                tables.tract_socioeconomic_silver
            ),
        )

    query = (
        silver_stream.writeStream
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
    target = spark.table(target_table)

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

    LOGGER.info(
        "Validated Silver ACS table",
        target_table=target_table,
    )


def run(
    spark: SparkSession,
    *,
    catalog: str,
    bronze_schema: str,
    silver_schema: str,
    checkpoint_path: str | None,
    full_rebuild: bool,
) -> None:
    tables = CrimeNetTables(
        catalog=catalog,
        bronze_schema=bronze_schema,
        silver_schema=silver_schema,
    )

    if full_rebuild:
        rebuild_socioeconomic_table(
            spark,
            tables=tables,
        )
    else:
        if checkpoint_path is None:
            raise ValueError(
                "checkpoint_path is required for "
                "incremental processing."
            )

        run_incremental_stream(
            spark,
            tables=tables,
            checkpoint_path=checkpoint_path,
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
            checkpoint_path=args.checkpoint_path,
            full_rebuild=args.full_rebuild,
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
