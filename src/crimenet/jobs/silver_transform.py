"""Python-wheel entry point for canonical Silver transformation."""

from __future__ import annotations

import argparse

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from crimenet.config.resources import CrimeNetTables
from crimenet.observability.logging import get_logger
from crimenet.spatial.h3 import (
    DEFAULT_WEATHER_H3_RESOLUTION,
    add_weather_query_cell,
)
from crimenet.transforms.canonical import (
    add_crime_offense_id,
    build_crime_offenses
)

LOGGER = get_logger(__name__)


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

    # Retained for compatibility with the bundle task.
    parser.add_argument(
        "--data-quality-schema",
        default="data_quality",
    )

    return parser.parse_args()


def run(
    spark: SparkSession,
    *,
    catalog: str,
    bronze_schema: str,
    silver_schema: str,
) -> None:
    tables = CrimeNetTables(
        catalog=catalog,
        bronze_schema=bronze_schema,
        silver_schema=silver_schema,
    )

    LOGGER.info(
        "Building canonical crime offenses",
        dallas_table=tables.dallas_bronze,
        houston_table=tables.houston_bronze,
        fort_worth_table=tables.fort_worth_bronze,
        target_table=tables.crime_offenses_silver,
        weather_h3_resolution=(
            DEFAULT_WEATHER_H3_RESOLUTION
        ),
    )

    silver_dataframe = build_crime_offenses(
        dallas_bronze=spark.table(
            tables.dallas_bronze
        ),
        houston_bronze=spark.table(
            tables.houston_bronze
        ),
        fort_worth_bronze=spark.table(
            tables.fort_worth_bronze
        ),
    )

    silver_dataframe = add_crime_offense_id(
        silver_dataframe
    )

    missing_ids = silver_dataframe.filter(
        F.col("crime_offense_id").isNull()
    )

    if not missing_ids.isEmpty():
        examples = [
            row.asDict()
            for row in (
                missing_ids
                .select(
                    "source_city",
                    "source_incident_id",
                    "source_record_id",
                    "offense_code",
                    "source_row_hash",
                )
                .limit(20)
                .collect()
            )
        ]

        raise RuntimeError(
            "Some canonical crime records have no "
            f"crime_offense_id. Examples: {examples}"
        )

    deduplication_window = (
        Window
        .partitionBy("crime_offense_id")
        .orderBy(
            F.col("updated_at").desc_nulls_last(),
            F.col("reported_at").desc_nulls_last(),
            F.col("occurred_at").desc_nulls_last(),
            F.col("source_file").desc_nulls_last(),
        )
    )

    silver_dataframe = (
        silver_dataframe
        .withColumn(
            "_deduplication_rank",
            F.row_number().over(
                deduplication_window
            ),
        )
        .filter(
            F.col("_deduplication_rank") == 1
        )
        .drop("_deduplication_rank")
    )

    silver_dataframe = add_weather_query_cell(
        silver_dataframe,
        resolution=DEFAULT_WEATHER_H3_RESOLUTION,
    )

    (
        silver_dataframe.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(
            tables.crime_offenses_silver
        )
    )

    LOGGER.info(
        "Canonical crime offenses materialized",
        target_table=tables.crime_offenses_silver,
    )


def main() -> None:
    args = parse_args()

    spark = (
        SparkSession.getActiveSession()
        or SparkSession.builder.getOrCreate()
    )

    LOGGER.info(
        "Starting canonical Silver transformation",
        catalog=args.catalog,
        bronze_schema=args.bronze_schema,
        silver_schema=args.silver_schema,
    )

    try:
        run(
            spark,
            catalog=args.catalog,
            bronze_schema=args.bronze_schema,
            silver_schema=args.silver_schema,
        )
    except Exception:
        LOGGER.exception(
            "Canonical Silver transformation failed",
            catalog=args.catalog,
            silver_schema=args.silver_schema,
        )
        raise

    LOGGER.info(
        "Canonical Silver transformation completed",
        catalog=args.catalog,
        silver_schema=args.silver_schema,
    )


if __name__ == "__main__":
    main()
