"""Python-wheel entry point for Bronze ingestion."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Protocol
from uuid import uuid4

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from crimenet.config.resources import CrimeNetTables
from crimenet.config.validation import validate_qualified_table_name
from crimenet.contracts.bronze import (
    get_source_contract,
    validate_contract_columns,
)
from crimenet.ingestion.column_names import normalize_column_names
from crimenet.ingestion.metadata import add_ingestion_metadata
from crimenet.ingestion.readers import (
    read_acs5_tract_raw,
    read_dallas_raw,
    read_fort_worth_raw,
    read_houston_raw,
    read_weather_raw,
)
from crimenet.observability.logging import get_logger
from crimenet.observability.run_context import resolve_pipeline_run_id

LOGGER = get_logger(__name__)


Reader = Callable[
    [SparkSession, str],
    DataFrame,
]


class StreamingReader(Protocol):
    def __call__(
        self,
        spark: SparkSession,
        input_path: str,
        *,
        schema_path: str,
    ) -> DataFrame:
        ...


BATCH_READERS: dict[str, Reader] = {
    "dallas": read_dallas_raw,
    "houston": read_houston_raw,
    "fort_worth": read_fort_worth_raw,
}

STREAMING_READERS: dict[str, StreamingReader] = {
    "open_meteo_weather": read_weather_raw,
    "acs5_tract": read_acs5_tract_raw,
}

SOURCE_SYSTEMS = {
    "open_meteo_weather": "open_meteo",
    "acs5_tract": "census_acs5",
}

SOURCE_CONTRACT_VERSIONS = {
    "open_meteo_weather": "open_meteo_archive_v1",
    "acs5_tract": "census_acs5_tract_v1",
}

SUPPORTED_SOURCES = (
    *BATCH_READERS.keys(),
    *STREAMING_READERS.keys(),
)

COLUMN_OVERRIDES: dict[
    str,
    dict[str, str],
] = {
    "fort_worth": {
        "Latitude": "latitude",
        "latitude": "latitude",
        "_Latitude": "alternate_latitude",
        "_latitude": "alternate_latitude",
        "Longitude": "longitude",
        "longitude": "longitude",
        "_Longitude": "alternate_longitude",
        "_longitude": "alternate_longitude",
    }
}


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
        "--source",
        required=True,
        choices=SUPPORTED_SOURCES,
    )
    parser.add_argument(
        "--input-path",
        required=True,
    )
    parser.add_argument(
        "--schema-path",
    )
    parser.add_argument(
        "--checkpoint-path",
    )
    parser.add_argument(
        "--write-mode",
        default="merge",
        choices=("merge",),
        help=(
            "Bronze always uses an idempotent Delta MERGE. This argument is "
            "retained to make the sink policy explicit."
        ),
    )
    parser.add_argument("--pipeline-run-id")

    args = parser.parse_args()

    if args.source in STREAMING_READERS:
        missing_arguments = [
            argument
            for argument, value in {
                "--schema-path": args.schema_path,
                "--checkpoint-path": args.checkpoint_path,
            }.items()
            if not value
        ]

        if missing_arguments:
            parser.error(
                f"{args.source} requires: "
                + ", ".join(missing_arguments)
            )

    return args


def _run_batch_ingestion(
    spark: SparkSession,
    *,
    tables: CrimeNetTables,
    source: str,
    input_path: str,
    write_mode: str,
) -> None:
    raw_dataframe = BATCH_READERS[source](
        spark,
        input_path,
    )

    normalized_dataframe = normalize_column_names(
        raw_dataframe,
        overrides=COLUMN_OVERRIDES.get(source),
    )
    contract = get_source_contract(source)
    validate_contract_columns(
        normalized_dataframe.columns,
        contract,
    )

    bronze_dataframe = add_ingestion_metadata(
        normalized_dataframe,
        source_system=source,
        contract_version=contract.version,
    )

    if write_mode != "merge":
        raise ValueError("Bronze ingestion supports only write_mode='merge'.")

    merge_bronze_batch(
        bronze_dataframe,
        batch_id=0,
        spark=spark,
        target_table=tables.bronze_for_source(source),
    )


def _deduplicate_bronze_batch(dataframe: DataFrame) -> DataFrame:
    """Choose one deterministic lineage row for each logical raw row."""
    window = (
        Window.partitionBy("source_system", "source_row_hash")
        .orderBy(
            F.col("source_file").asc_nulls_last(),
            F.col("ingested_at").asc_nulls_last(),
        )
    )
    return (
        dataframe.withColumn("_row_number", F.row_number().over(window))
        .filter(F.col("_row_number") == 1)
        .drop("_row_number")
    )


def merge_bronze_batch(
    batch_dataframe: DataFrame,
    batch_id: int,
    *,
    spark: SparkSession,
    target_table: str,
) -> None:
    """
    Insert unseen source rows by stable content identity.

    Matching rows are intentionally not updated, so a replay from a new path
    cannot churn first-seen operational metadata.
    """
    validate_qualified_table_name(target_table)
    if batch_dataframe.isEmpty():
        LOGGER.info(
            "Skipping empty Bronze batch",
            batch_id=batch_id,
            target_table=target_table,
        )
        return

    input_count = batch_dataframe.count()
    source = _deduplicate_bronze_batch(batch_dataframe)
    deduplicated_count = source.count()
    if not spark.catalog.tableExists(target_table):
        source.limit(0).write.format("delta").saveAsTable(target_table)
    target_count_before = spark.table(target_table).count()

    view_name = f"_crimenet_bronze_{uuid4().hex}"
    source.createOrReplaceTempView(view_name)
    try:
        spark.sql(
            f"""
            MERGE INTO {target_table} AS target
            USING {view_name} AS source
              ON target.source_system = source.source_system
             AND target.source_row_hash = source.source_row_hash
            WHEN NOT MATCHED THEN INSERT *
            """
        )
    finally:
        spark.catalog.dropTempView(view_name)

    target_count_after = spark.table(target_table).count()
    LOGGER.info(
        "Merged replay-safe Bronze batch",
        batch_id=batch_id,
        target_table=target_table,
        input_count=input_count,
        output_count=target_count_after,
        insert_count=target_count_after - target_count_before,
        update_count=0,
        duplicate_count=input_count - deduplicated_count,
    )


def _run_streaming_ingestion(
    spark: SparkSession,
    *,
    tables: CrimeNetTables,
    source: str,
    input_path: str,
    schema_path: str,
    checkpoint_path: str,
) -> None:
    raw_dataframe = STREAMING_READERS[source](
        spark,
        input_path,
        schema_path=schema_path,
    )

    normalized_dataframe = normalize_column_names(
        raw_dataframe,
    )

    bronze_dataframe = add_ingestion_metadata(
        normalized_dataframe,
        source_system=SOURCE_SYSTEMS[source],
        contract_version=SOURCE_CONTRACT_VERSIONS[source],
    )

    target_table = tables.bronze_for_source(source)

    def upsert_batch(batch_dataframe: DataFrame, batch_id: int) -> None:
        merge_bronze_batch(
            batch_dataframe,
            batch_id,
            spark=spark,
            target_table=target_table,
        )

    query = (
        bronze_dataframe.writeStream
        .option(
            "checkpointLocation",
            checkpoint_path,
        )
        .trigger(
            availableNow=True,
        )
        .foreachBatch(upsert_batch)
        .start()
    )

    query.awaitTermination()


def run(
    spark: SparkSession,
    *,
    catalog: str,
    bronze_schema: str,
    source: str,
    input_path: str,
    write_mode: str,
    schema_path: str | None = None,
    checkpoint_path: str | None = None,
) -> None:
    tables = CrimeNetTables(
        catalog=catalog,
        bronze_schema=bronze_schema,
    )

    if source in STREAMING_READERS:
        if schema_path is None:
            raise ValueError(
                f"schema_path is required for {source}"
            )

        if checkpoint_path is None:
            raise ValueError(
                f"checkpoint_path is required for {source}"
            )

        _run_streaming_ingestion(
            spark,
            tables=tables,
            source=source,
            input_path=input_path,
            schema_path=schema_path,
            checkpoint_path=checkpoint_path,
        )
        return

    _run_batch_ingestion(
        spark,
        tables=tables,
        source=source,
        input_path=input_path,
        write_mode=write_mode,
    )


def main() -> None:
    args = parse_args()
    run_id = resolve_pipeline_run_id(args.pipeline_run_id)

    spark = (
        SparkSession.getActiveSession()
        or SparkSession.builder.getOrCreate()
    )

    target_table = CrimeNetTables(
        catalog=args.catalog,
        bronze_schema=args.bronze_schema,
    ).bronze_for_source(args.source)

    LOGGER.info(
        "Starting Bronze ingestion",
        source=args.source,
        input_path=args.input_path,
        target_table=target_table,
        write_mode=args.write_mode,
        streaming=args.source in STREAMING_READERS,
        pipeline_run_id=run_id,
    )

    try:
        run(
            spark,
            catalog=args.catalog,
            bronze_schema=args.bronze_schema,
            source=args.source,
            input_path=args.input_path,
            write_mode=args.write_mode,
            schema_path=args.schema_path,
            checkpoint_path=args.checkpoint_path,
        )
    except Exception:
        LOGGER.exception(
            "Bronze ingestion failed",
            source=args.source,
            target_table=target_table,
            pipeline_run_id=run_id,
        )
        raise

    LOGGER.info(
        "Bronze ingestion completed",
        source=args.source,
        target_table=target_table,
        pipeline_run_id=run_id,
    )


if __name__ == "__main__":
    main()
