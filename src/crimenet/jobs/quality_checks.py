"""Databricks entry point for auditable Silver crime validation."""

from __future__ import annotations

import argparse

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from crimenet.config.resources import CrimeNetTables
from crimenet.config.validation import QualityThresholds
from crimenet.observability.logging import get_logger
from crimenet.observability.run_context import resolve_pipeline_run_id
from crimenet.quality.checks import (
    blocking_failures,
    evaluate_crime_quality,
    merge_quality_results,
    quality_results_dataframe,
)

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--bronze-schema", default="bronze")
    parser.add_argument("--silver-schema", required=True)
    parser.add_argument("--data-quality-schema", required=True)
    parser.add_argument("--pipeline-run-id")
    parser.add_argument("--minimum-row-count", type=int, default=1)
    parser.add_argument(
        "--maximum-row-count",
        type=int,
        default=2_000_000_000,
    )
    parser.add_argument(
        "--minimum-silver-to-bronze-ratio",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--maximum-silver-to-bronze-ratio",
        type=float,
        default=1.05,
    )
    parser.add_argument(
        "--maximum-critical-null-rate",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--maximum-coordinate-null-rate",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--maximum-quarantine-rate",
        type=float,
        default=0.05,
    )
    return parser.parse_args()


def run(
    spark: SparkSession,
    *,
    tables: CrimeNetTables,
    pipeline_run_id: str,
    thresholds: QualityThresholds,
) -> None:
    crime = spark.table(tables.crime_offenses_silver)
    bronze_distinct_count = sum(
        spark.table(table_name)
        .select("source_system", "source_row_hash")
        .distinct()
        .count()
        for table_name in (
            tables.dallas_bronze,
            tables.houston_bronze,
            tables.fort_worth_bronze,
        )
    )

    quarantine_count = 0
    reason_counts: dict[str, int] = {}
    observation_table = f"{tables.crime_quarantine}_observations"
    if spark.catalog.tableExists(observation_table):
        quarantine = spark.table(observation_table).filter(
            F.col("pipeline_run_id") == pipeline_run_id
        )
        quarantine_count = quarantine.select("quarantine_id").distinct().count()
        reason_counts = {
            row["quarantine_reason_code"]: row["count"]
            for row in quarantine.groupBy("quarantine_reason_code").count().collect()
        }

    checks = evaluate_crime_quality(
        crime,
        thresholds=thresholds,
        bronze_distinct_count=bronze_distinct_count,
        quarantine_count=quarantine_count,
        quarantine_reason_counts=reason_counts,
    )
    results = quality_results_dataframe(
        spark,
        checks=checks,
        pipeline_run_id=pipeline_run_id,
        table_name=tables.crime_offenses_silver,
    )
    merge_quality_results(
        spark,
        results=results,
        target_table=tables.quality_results,
    )
    failures = blocking_failures(checks)
    LOGGER.info(
        "Persisted Silver crime quality results",
        pipeline_run_id=pipeline_run_id,
        table_name=tables.crime_offenses_silver,
        check_count=len(checks),
        blocking_failure_count=len(failures),
        bronze_distinct_count=bronze_distinct_count,
        silver_count=crime.count(),
        quarantine_count=quarantine_count,
    )
    if failures:
        summary = "; ".join(
            f"{check.check_name} observed={check.observed_value} "
            f"expected={check.expected_threshold}"
            for check in failures
        )
        raise RuntimeError(f"Blocking Silver crime quality checks failed: {summary}")


def main() -> None:
    args = parse_args()
    spark = (
        SparkSession.getActiveSession()
        or SparkSession.builder.getOrCreate()
    )
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    run_id = resolve_pipeline_run_id(args.pipeline_run_id)
    tables = CrimeNetTables(
        catalog=args.catalog,
        bronze_schema=args.bronze_schema,
        silver_schema=args.silver_schema,
        data_quality_schema=args.data_quality_schema,
    )
    thresholds = QualityThresholds(
        minimum_row_count=args.minimum_row_count,
        maximum_row_count=args.maximum_row_count,
        minimum_silver_to_bronze_ratio=(
            args.minimum_silver_to_bronze_ratio
        ),
        maximum_silver_to_bronze_ratio=(
            args.maximum_silver_to_bronze_ratio
        ),
        maximum_critical_null_rate=args.maximum_critical_null_rate,
        maximum_coordinate_null_rate=args.maximum_coordinate_null_rate,
        maximum_quarantine_rate=args.maximum_quarantine_rate,
    ).validate()
    LOGGER.info(
        "Starting Silver crime quality checks",
        pipeline_run_id=run_id,
        table_name=tables.crime_offenses_silver,
    )
    run(
        spark,
        tables=tables,
        pipeline_run_id=run_id,
        thresholds=thresholds,
    )
    LOGGER.info(
        "Silver crime quality checks passed",
        pipeline_run_id=run_id,
        table_name=tables.crime_offenses_silver,
    )


if __name__ == "__main__":
    main()
