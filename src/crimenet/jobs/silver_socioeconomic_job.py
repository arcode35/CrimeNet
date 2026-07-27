"""Python-wheel entry point for ACS Silver processing."""

from __future__ import annotations

import argparse
import logging

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType
from pyspark.sql.window import Window

from crimenet.config.resources import CrimeNetTables
from crimenet.silver.socioeconomic import (
    transform_acs5_tracts,
)


LOGGER = logging.getLogger(__name__)

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
            F.col("bronze_ingested_at")
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
        "Starting full rebuild of %s from %s",
        tables.tract_socioeconomic_silver,
        tables.acs5_tract_bronze,
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
        "Completed full rebuild of %s with %s rows",
        tables.tract_socioeconomic_silver,
        rebuilt_count,
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
            "Skipping empty microbatch %s",
            batch_id,
        )
        return

    deduplicated_dataframe = (
        deduplicate_socioeconomic_records(
            batch_dataframe
        )
    )

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
        "Merged ACS socioeconomic microbatch %s",
        batch_id,
    )


def run_incremental_stream(
    spark: SparkSession,
    *,
    tables: CrimeNetTables,
    checkpoint_path: str,
) -> None:
    LOGGER.info(
        "Starting incremental ACS Silver processing "
        "with checkpoint %s",
        checkpoint_path,
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
        "Incremental ACS Silver processing completed"
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
        "Validated Silver ACS table successfully: %s",
        target_table,
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

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s %(levelname)s "
            "%(name)s - %(message)s"
        ),
    )

    spark = (
        SparkSession.getActiveSession()
        or SparkSession.builder.getOrCreate()
    )

    run(
        spark,
        catalog=args.catalog,
        bronze_schema=args.bronze_schema,
        silver_schema=args.silver_schema,
        checkpoint_path=args.checkpoint_path,
        full_rebuild=args.full_rebuild,
    )


if __name__ == "__main__":
    main()