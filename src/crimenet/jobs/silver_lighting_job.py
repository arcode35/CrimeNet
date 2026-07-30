"""Python-wheel entry point for solar lighting features."""

from __future__ import annotations

import argparse

from pyspark.sql import SparkSession

from crimenet.observability.logging import get_logger
from crimenet.silver.lighting import (
    materialize_lighting_conditions,
)

LOGGER = get_logger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute solar position and lighting conditions "
            "for Silver crime records."
        )
    )

    parser.add_argument(
        "--catalog",
        required=True,
    )
    parser.add_argument(
        "--silver-schema",
        default="silver",
    )
    parser.add_argument(
        "--mode",
        choices=("incremental", "full"),
        default="incremental",
    )

    return parser.parse_args()


def run(
    spark: SparkSession,
    *,
    catalog: str,
    silver_schema: str,
    full_rebuild: bool,
) -> None:
    # Spark-to-pandas timestamp conversion must use UTC because 
    # receives timezone aware UTC timestamps.
    spark.conf.set(
        "spark.sql.session.timeZone",
        "UTC",
    )

    crime_table = (
        f"{catalog}."
        f"{silver_schema}."
        "crime_offenses"
    )

    target_table = (
        f"{catalog}."
        f"{silver_schema}."
        "solar_lighting_conditions"
    )

    materialize_lighting_conditions(
        spark,
        crime_table=crime_table,
        target_table=target_table,
        full_rebuild=full_rebuild,
    )


def main() -> None:
    args = parse_args()

    spark = (
        SparkSession.getActiveSession()
        or SparkSession.builder.getOrCreate()
    )

    full_rebuild = args.mode == "full"

    LOGGER.info(
        "Starting Silver lighting job",
        catalog=args.catalog,
        silver_schema=args.silver_schema,
        mode=args.mode,
        full_rebuild=full_rebuild,
    )

    try:
        run(
            spark,
            catalog=args.catalog,
            silver_schema=args.silver_schema,
            full_rebuild=full_rebuild,
        )
    except Exception:
        LOGGER.exception(
            "Silver lighting job failed",
            catalog=args.catalog,
            silver_schema=args.silver_schema,
            mode=args.mode,
            full_rebuild=full_rebuild,
        )
        raise

    LOGGER.info(
        "Silver lighting job completed",
        catalog=args.catalog,
        silver_schema=args.silver_schema,
        mode=args.mode,
        full_rebuild=full_rebuild,
    )

if __name__ == "__main__":
    main()