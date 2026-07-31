"""Python-wheel entry point for Bronze ingestion."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Protocol

from pyspark.sql import DataFrame, SparkSession

from crimenet.config.resources import CrimeNetTables
from crimenet.ingestion.column_names import normalize_column_names
from crimenet.ingestion.metadata import add_ingestion_metadata
from crimenet.ingestion.readers import (
    read_acs5_tract_raw,
    read_city_raw,
    read_weather_raw,
)
from crimenet.observability.logging import get_logger

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

CITY_SOURCES = (
    "dallas",
    "fort_worth",
    "houston",
    "seattle",
    "chicago",
    "new_york",
    "san_francisco",
    "baltimore",
    "washington_dc",
)

STREAMING_READERS: dict[
    str,
    StreamingReader,
] = {
    **{
        source: read_city_raw
        for source in CITY_SOURCES
    },
    "open_meteo_weather": read_weather_raw,
    "acs5_tract": read_acs5_tract_raw,
}


SOURCE_SYSTEMS = {
    "open_meteo_weather": "open_meteo",
    "acs5_tract": "census_acs5",
}


SUPPORTED_SOURCES = tuple(
    STREAMING_READERS.keys()
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
        default="overwrite",
        choices=("overwrite", "append"),
    )

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

    bronze_dataframe = add_ingestion_metadata(
        normalized_dataframe,
        source_system=source,
    )

    writer = (
        bronze_dataframe.write
        .format("delta")
        .mode(write_mode)
    )

    if write_mode == "overwrite":
        writer = writer.option(
            "overwriteSchema",
            "true",
        )

    writer.saveAsTable(
        tables.bronze_for_source(source)
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
        overrides=COLUMN_OVERRIDES.get(source),
    )

    source_system = SOURCE_SYSTEMS.get(
        source,
        source,
    )

    bronze_dataframe = add_ingestion_metadata(
        normalized_dataframe,
        source_system=source_system,
    )

    target_table = tables.bronze_for_source(
        source
    )

    query = (
        bronze_dataframe.writeStream
        .format("delta")
        .option(
            "checkpointLocation",
            checkpoint_path,
        )
        .option(
            "mergeSchema",
            "true",
        )
        .trigger(
            availableNow=True,
        )
        .toTable(target_table)
    )

    query.awaitTermination()

def run(
    spark: SparkSession,
    *,
    catalog: str,
    bronze_schema: str,
    source: str,
    input_path: str,
    schema_path: str | None = None,
    checkpoint_path: str | None = None,
    write_mode: str = "append",
) -> None:
    del write_mode

    if schema_path is None:
        raise ValueError(
            f"schema_path is required for {source}"
        )

    if checkpoint_path is None:
        raise ValueError(
            f"checkpoint_path is required for {source}"
        )

    tables = CrimeNetTables(
        catalog=catalog,
        bronze_schema=bronze_schema,
    )

    _run_streaming_ingestion(
        spark,
        tables=tables,
        source=source,
        input_path=input_path,
        schema_path=schema_path,
        checkpoint_path=checkpoint_path,
    )

def main() -> None:
    args = parse_args()

    spark = (
        SparkSession.getActiveSession()
        or SparkSession.builder.getOrCreate()
    )

    target_table = (
        f"{args.catalog}."
        f"{args.bronze_schema}."
        f"{args.source}"
    )

    LOGGER.info(
        "Starting Bronze ingestion",
        source=args.source,
        input_path=args.input_path,
        target_table=target_table,
        write_mode=args.write_mode,
        streaming=args.source in STREAMING_READERS,
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
        )
        raise

    LOGGER.info(
        "Bronze ingestion completed",
        source=args.source,
        target_table=target_table,
    )


if __name__ == "__main__":
    main()