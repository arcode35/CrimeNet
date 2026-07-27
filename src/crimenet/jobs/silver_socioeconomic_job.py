"""Python-wheel entry point for ACS Silver processing."""

from __future__ import annotations

import argparse

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from crimenet.config.resources import CrimeNetTables
from crimenet.silver.socioeconomic import (
    transform_acs5_tracts,
)


MERGE_KEYS = (
    "geoid",
    "acs_vintage",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

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
        required=True,
    )

    return parser.parse_args()


def ensure_target_table(
    spark: SparkSession,
    *,
    target_table: str,
    schema,
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


def merge_socioeconomic_batch(
    batch_dataframe: DataFrame,
    batch_id: int,
    *,
    spark: SparkSession,
    target_table: str,
) -> None:
    del batch_id

    if batch_dataframe.isEmpty():
        return

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

    deduplicated_dataframe = (
        batch_dataframe
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


def run(
    spark: SparkSession,
    *,
    catalog: str,
    bronze_schema: str,
    silver_schema: str,
    checkpoint_path: str,
) -> None:
    tables = CrimeNetTables(
        catalog=catalog,
        bronze_schema=bronze_schema,
        silver_schema=silver_schema,
    )

    bronze_stream = (
        spark.readStream
        .table(
            tables.acs5_tract_bronze
        )
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


def main() -> None:
    args = parse_args()

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
    )


if __name__ == "__main__":
    main()