"""Publish staged canonical city tables into one Silver table."""

from __future__ import annotations

import argparse

from pyspark.sql import DataFrame, SparkSession

from crimenet.canonical.schema import (
    CANONICAL_CRIME_SCHEMA,
)
from crimenet.config.resources import (
    CrimeNetTables,
)
from crimenet.observability.logging import (
    get_logger,
)


LOGGER = get_logger(__name__)


CITIES = (
    "dallas",
    "fort_worth",
    "new_york",
    "chicago",
    "san_francisco",
    "seattle",
    "baltimore",
    "washington_dc",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--catalog",
        required=True,
    )

    parser.add_argument(
        "--silver-schema",
        default="silver",
    )

    return parser.parse_args()


def read_staged_cities(
    spark: SparkSession,
    *,
    catalog: str,
    silver_schema: str,
) -> DataFrame:
    dataframes = [
        spark.read.table(
            f"{catalog}."
            f"{silver_schema}."
            f"_canonical_city_{city}"
        )
        for city in CITIES
    ]

    combined = dataframes[0]

    for dataframe in dataframes[1:]:
        combined = combined.unionByName(
            dataframe,
            allowMissingColumns=False,
        )

    return combined


def run(
    spark: SparkSession,
    *,
    catalog: str,
    silver_schema: str,
) -> None:
    spark.conf.set(
        "spark.sql.session.timeZone",
        "UTC",
    )

    tables = CrimeNetTables(
        catalog=catalog,
        silver_schema=silver_schema,
    )

    target_table = (
        tables.crime_offenses_silver
    )

    combined = read_staged_cities(
        spark,
        catalog=catalog,
        silver_schema=silver_schema,
    )

    (
        combined.write
        .format("delta")
        .mode("overwrite")
        .option(
            "overwriteSchema",
            "true",
        )
        .partitionBy(
            "source_city",
            "occurrence_year",
        )
        .saveAsTable(target_table)
    )

    LOGGER.info(
        "Published canonical Silver table",
        target_table=target_table,
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
        silver_schema=args.silver_schema,
    )


if __name__ == "__main__":
    main()