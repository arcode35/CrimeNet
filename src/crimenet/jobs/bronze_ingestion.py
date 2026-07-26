"""Python-wheel entry point for Bronze ingestion."""

from __future__ import annotations

import argparse
from collections.abc import Callable

from pyspark.sql import DataFrame, SparkSession

from crimenet.config.resources import CrimeNetTables
from crimenet.ingestion.column_names import (
    normalize_column_names,
)
from crimenet.ingestion.metadata import (
    add_ingestion_metadata,
)
from crimenet.ingestion.readers import (
    read_dallas_raw,
    read_fort_worth_raw,
    read_houston_raw,
    read_weather_raw,
)


Reader = Callable[
    [SparkSession, str],
    DataFrame,
]


BATCH_READERS: dict[str, Reader] = {
    "dallas": read_dallas_raw,
    "houston": read_houston_raw,
    "fort_worth": read_fort_worth_raw,
}

SUPPORTED_SOURCES = (
    *BATCH_READERS,
    "open_meteo_weather",
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

    if args.source == "open_meteo_weather":
        missing_arguments = [
            argument
            for argument, value in {
                "--schema-path": args.schema_path,
                "--checkpoint-path": (
                    args.checkpoint_path
                ),
            }.items()
            if not value
        ]

        if missing_arguments:
            parser.error(
                "open_meteo_weather requires: "
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

    normalized_dataframe = (
        normalize_column_names(
            raw_dataframe,
            overrides=COLUMN_OVERRIDES.get(
                source
            ),
        )
    )

    bronze_dataframe = (
        add_ingestion_metadata(
            normalized_dataframe,
            source_system=source,
        )
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
def _run_weather_ingestion(
    spark: SparkSession,
    *,
    tables: CrimeNetTables,
    input_path: str,
    schema_path: str,
    checkpoint_path: str,
) -> None:
    raw_dataframe = read_weather_raw(
        spark,
        input_path,
        schema_path=schema_path,
    )

    normalized_dataframe = (
        normalize_column_names(
            raw_dataframe,
        )
    )

    bronze_dataframe = (
        add_ingestion_metadata(
            normalized_dataframe,
            source_system="open_meteo",
        )
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
        .toTable(
            tables.open_meteo_weather_bronze
        )
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

    if source == "open_meteo_weather":
        if schema_path is None:
            raise ValueError(
                "schema_path is required for "
                "open_meteo_weather"
            )

        if checkpoint_path is None:
            raise ValueError(
                "checkpoint_path is required for "
                "open_meteo_weather"
            )

        _run_weather_ingestion(
            spark,
            tables=tables,
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

    spark = (
        SparkSession.getActiveSession()
        or SparkSession.builder.getOrCreate()
    )

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


if __name__ == "__main__":
    main()