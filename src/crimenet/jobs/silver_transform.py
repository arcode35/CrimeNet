"""Python-wheel entry point for canonical Silver transformation."""

from __future__ import annotations

import argparse

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from crimenet.config.resources import CrimeNetTables
from crimenet.config.validation import QualityThresholds
from crimenet.contracts.silver import assert_silver_contract
from crimenet.observability.logging import get_logger
from crimenet.observability.run_context import resolve_pipeline_run_id
from crimenet.quality.checks import (
    QualityCheck,
    blocking_failures,
    evaluate_crime_quality,
    merge_quality_results,
    quality_results_dataframe,
)
from crimenet.quality.quarantine import (
    merge_quarantine,
    split_crime_quarantine,
)
from crimenet.spatial.h3 import (
    DEFAULT_WEATHER_H3_RESOLUTION,
    add_weather_query_cell,
)
from crimenet.transforms.canonical import build_crime_offenses
from crimenet.transforms.deduplication import deduplicate_crime_offenses
from crimenet.utils.promotion import promote_staged_table

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
    catalog: str,
    bronze_schema: str,
    silver_schema: str,
    data_quality_schema: str,
    pipeline_run_id: str | None = None,
    thresholds: QualityThresholds | None = None,
) -> None:
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    run_id = resolve_pipeline_run_id(pipeline_run_id)
    tables = CrimeNetTables(
        catalog=catalog,
        bronze_schema=bronze_schema,
        silver_schema=silver_schema,
        data_quality_schema=data_quality_schema,
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

    quality_thresholds = (thresholds or QualityThresholds()).validate()
    dallas_bronze = spark.table(tables.dallas_bronze)
    houston_bronze = spark.table(tables.houston_bronze)
    fort_worth_bronze = spark.table(tables.fort_worth_bronze)
    canonical_dataframe = build_crime_offenses(
        dallas_bronze=dallas_bronze,
        houston_bronze=houston_bronze,
        fort_worth_bronze=fort_worth_bronze,
    )

    valid_dataframe, quarantine_dataframe = split_crime_quarantine(
        canonical_dataframe,
        pipeline_run_id=run_id,
    )
    merge_quarantine(
        spark,
        quarantine=quarantine_dataframe,
        target_table=tables.crime_quarantine,
    )

    deduplicated_dataframe = deduplicate_crime_offenses(valid_dataframe)
    silver_dataframe = add_weather_query_cell(
        deduplicated_dataframe,
        resolution=DEFAULT_WEATHER_H3_RESOLUTION,
    )

    expected_identities = valid_dataframe.select(
        "business_identity"
    ).distinct()
    bronze_distinct_count = sum(
        dataframe.select("source_system", "source_row_hash")
        .distinct()
        .count()
        for dataframe in (
            dallas_bronze,
            houston_bronze,
            fort_worth_bronze,
        )
    )
    quarantine_count = quarantine_dataframe.select(
        "quarantine_id"
    ).distinct().count()
    quarantine_reason_counts = {
        row["quarantine_reason_code"]: row["count"]
        for row in quarantine_dataframe.groupBy(
            "quarantine_reason_code"
        ).count().collect()
    }

    def evaluate_candidate(dataframe: DataFrame) -> None:
        contract_columns = [
            column
            for column in dataframe.columns
            if column != "weather_query_cell_id"
        ]
        assert_silver_contract(dataframe.select(*contract_columns))
        duplicate_count = (
            dataframe.groupBy("business_identity")
            .count()
            .filter(F.col("count") > 1)
            .limit(1)
            .count()
        )
        if duplicate_count:
            raise RuntimeError(
                "Candidate Silver crime table contains duplicate "
                "business identities."
            )
        candidate_identities = dataframe.select(
            "business_identity"
        ).distinct()
        missing_identity = expected_identities.join(
            candidate_identities,
            on="business_identity",
            how="left_anti",
        ).limit(1).count()
        unexpected_identity = candidate_identities.join(
            expected_identities,
            on="business_identity",
            how="left_anti",
        ).limit(1).count()
        if missing_identity or unexpected_identity:
            raise RuntimeError(
                "Candidate Silver crime business-key set does not match "
                "the valid deduplicated input."
            )
        checks = evaluate_crime_quality(
            dataframe,
            thresholds=quality_thresholds,
            bronze_distinct_count=bronze_distinct_count,
            quarantine_count=quarantine_count,
            quarantine_reason_counts=quarantine_reason_counts,
        )
        results = quality_results_dataframe(
            spark,
            checks=checks,
            pipeline_run_id=run_id,
            table_name=tables.crime_offenses_silver,
        )
        merge_quality_results(
            spark,
            results=results,
            target_table=tables.quality_results,
        )
        failures = blocking_failures(checks)
        if failures:
            summary = "; ".join(
                f"{check.check_name} observed={check.observed_value} "
                f"expected={check.expected_threshold}"
                for check in failures
            )
            raise RuntimeError(
                "Candidate Silver crime quality checks failed before "
                f"promotion: {summary}"
            )

    def validate_candidate(dataframe: DataFrame) -> None:
        try:
            evaluate_candidate(dataframe)
        except Exception as exc:
            failure = quality_results_dataframe(
                spark,
                checks=[
                    QualityCheck(
                        check_name="silver_candidate_validation",
                        severity="BLOCKING",
                        passed=False,
                        observed_value=f"{type(exc).__name__}: {exc}",
                        expected_threshold=(
                            "all pre-promotion Silver invariants pass"
                        ),
                    )
                ],
                pipeline_run_id=run_id,
                table_name=tables.crime_offenses_silver,
            )
            merge_quality_results(
                spark,
                results=failure,
                target_table=tables.quality_results,
            )
            raise

    promote_staged_table(
        spark,
        candidate=silver_dataframe,
        target_table=tables.crime_offenses_silver,
        pipeline_run_id=run_id,
        validate=validate_candidate,
    )

    LOGGER.info(
        "Canonical crime offenses materialized",
        target_table=tables.crime_offenses_silver,
        pipeline_run_id=run_id,
        input_count=bronze_distinct_count,
        output_count=spark.table(tables.crime_offenses_silver).count(),
        quarantine_count=quarantine_count,
        duplicate_count=(
            valid_dataframe.count()
            - expected_identities.count()
        ),
        staging_table_name=(
            f"{tables.crime_offenses_silver}__staging__{run_id}"
        ),
        promotion_status="promoted",
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
        run(
            spark,
            catalog=args.catalog,
            bronze_schema=args.bronze_schema,
            silver_schema=args.silver_schema,
            data_quality_schema=args.data_quality_schema,
            pipeline_run_id=args.pipeline_run_id,
            thresholds=thresholds,
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
