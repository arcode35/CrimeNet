"""Python-wheel entry point for one city's Bronze ingestion."""

from __future__ import annotations

import argparse
from collections.abc import Callable

from pyspark.sql import DataFrame, SparkSession

from crimenet.config.resources import CrimeNetTables
from crimenet.ingestion.column_names import normalize_column_names
from crimenet.ingestion.metadata import add_ingestion_metadata
from crimenet.ingestion.readers import (
    read_dallas_raw,
    read_fort_worth_raw,
    read_houston_raw,
)

Reader = Callable[[SparkSession, str], DataFrame]

READERS: dict[str, Reader] = {
    "dallas": read_dallas_raw,
    "houston": read_houston_raw,
    "fort_worth": read_fort_worth_raw,
}

COLUMN_OVERRIDES: dict[str, dict[str, str]] = {
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
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--bronze-schema", default="bronze")
    parser.add_argument(
        "--city",
        required=True,
        choices=tuple(READERS),
    )
    parser.add_argument("--input-path", required=True)
    parser.add_argument(
        "--write-mode",
        default="overwrite",
        choices=("overwrite", "append"),
    )
    return parser.parse_args()


def run(
    spark: SparkSession,
    *,
    catalog: str,
    bronze_schema: str,
    city: str,
    input_path: str,
    write_mode: str,
) -> None:
    tables = CrimeNetTables(
        catalog=catalog,
        bronze_schema=bronze_schema,
    )

    raw_dataframe = READERS[city](spark, input_path)
    normalized_dataframe = normalize_column_names(
        raw_dataframe,
        overrides=COLUMN_OVERRIDES.get(city),
    )
    bronze_dataframe = add_ingestion_metadata(
        normalized_dataframe,
        source_system=city,
    )

    writer = (
        bronze_dataframe.write
        .format("delta")
        .mode(write_mode)
    )

    if write_mode == "overwrite":
        writer = writer.option("overwriteSchema", "true")

    writer.saveAsTable(tables.bronze_for_city(city))


def main() -> None:
    args = parse_args()
    spark = SparkSession.getActiveSession() or (
        SparkSession.builder.getOrCreate()
    )

    run(
        spark,
        catalog=args.catalog,
        bronze_schema=args.bronze_schema,
        city=args.city,
        input_path=args.input_path,
        write_mode=args.write_mode,
    )


if __name__ == "__main__":
    main()
