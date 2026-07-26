"""Python-wheel entry point for canonical Silver transformation."""

from __future__ import annotations

import argparse

from pyspark.sql import SparkSession

from crimenet.config.resources import CrimeNetTables
from crimenet.transforms.canonical import build_crime_offenses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--bronze-schema", default="bronze")
    parser.add_argument("--silver-schema", default="silver")
    # Accepted for compatibility with the bundle task. Quarantine writes can
    # use this in the next iteration without changing the task interface.
    parser.add_argument(
        "--data-quality-schema",
        default="data_quality",
    )
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
    silver_schema: str,
    write_mode: str,
) -> None:
    tables = CrimeNetTables(
        catalog=catalog,
        bronze_schema=bronze_schema,
        silver_schema=silver_schema,
    )

    silver_dataframe = build_crime_offenses(
        dallas_bronze=spark.table(tables.dallas_bronze),
        houston_bronze=spark.table(tables.houston_bronze),
        fort_worth_bronze=spark.table(tables.fort_worth_bronze),
    )

    writer = (
        silver_dataframe.write
        .format("delta")
        .mode(write_mode)
    )

    if write_mode == "overwrite":
        writer = writer.option("overwriteSchema", "true")

    writer.saveAsTable(tables.crime_offenses_silver)


def main() -> None:
    args = parse_args()
    spark = SparkSession.getActiveSession() or (
        SparkSession.builder.getOrCreate()
    )

    run(
        spark,
        catalog=args.catalog,
        bronze_schema=args.bronze_schema,
        silver_schema=args.silver_schema,
        write_mode=args.write_mode,
    )


if __name__ == "__main__":
    main()
