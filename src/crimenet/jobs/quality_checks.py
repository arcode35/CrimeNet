"""Databricks entry point for canonical Silver crime validation."""

from __future__ import annotations

import argparse

from pyspark.sql import SparkSession

from crimenet.observability.logging import get_logger
from crimenet.quality import QualityReport, validate_silver_crime

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the canonical Silver crime table."
    )
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--silver-schema", default="silver")
    parser.add_argument("--data-quality-schema", default="data_quality")
    parser.add_argument(
        "--minimum-occurred-at-coverage",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--maximum-examples",
        type=int,
        default=5,
    )
    return parser.parse_args()


def run(
    spark: SparkSession,
    *,
    catalog: str,
    silver_schema: str,
    minimum_occurred_at_coverage: float | None,
    maximum_examples: int,
) -> QualityReport:
    """Validate materialized Silver crime data and return its report."""
    table_name = f"{catalog}.{silver_schema}.crime_offenses"
    LOGGER.info(
        "Validating canonical Silver crime data",
        table_name=table_name,
    )
    report = validate_silver_crime(
        spark.table(table_name),
        minimum_occurred_at_coverage=minimum_occurred_at_coverage,
        maximum_examples=maximum_examples,
    )
    LOGGER.info(
        "Canonical Silver crime validation passed",
        table_name=table_name,
        check_count=len(report.checks),
    )
    return report


def main() -> None:
    args = parse_args()
    spark = (
        SparkSession.getActiveSession()
        or SparkSession.builder.getOrCreate()
    )
    LOGGER.info(
        "Starting Silver crime quality checks",
        catalog=args.catalog,
        silver_schema=args.silver_schema,
        data_quality_schema=args.data_quality_schema,
    )
    run(
        spark,
        catalog=args.catalog,
        silver_schema=args.silver_schema,
        minimum_occurred_at_coverage=(
            args.minimum_occurred_at_coverage
        ),
        maximum_examples=args.maximum_examples,
    )


if __name__ == "__main__":
    main()
