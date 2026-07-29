from __future__ import annotations

from datetime import datetime

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import LongType, StructField, StructType

from crimenet.config.validation import QualityThresholds
from crimenet.contracts.silver import SILVER_SCHEMA
from crimenet.quality.checks import (
    blocking_failures,
    evaluate_crime_quality,
    quality_results_dataframe,
)


def _quality_frame(spark: SparkSession) -> DataFrame:
    schema = StructType(
        [*SILVER_SCHEMA.fields, StructField("weather_query_cell_id", LongType())]
    )
    rows = []
    for index, source in enumerate(("dallas", "houston", "fort_worth")):
        values = {field.name: None for field in schema.fields}
        values.update(
            {
                "source_system": source,
                "source_city": source,
                "source_record_id": f"record-{index}",
                "source_incident_id": f"incident-{index}",
                "source_offense_id": f"offense-{index}",
                "business_identity": f"identity-{index}",
                "occurred_at": datetime(2024, 1, index + 1, 12),
                "source_row_hash": f"hash-{index}",
                "source_contract_version": "contract-v1",
                "transformation_version": "transform-v1",
                "weather_query_cell_id": 123 + index,
            }
        )
        rows.append(values)
    return spark.createDataFrame(rows, schema=schema)


def test_complete_canonical_fixture_passes_blocking_checks(
    spark: SparkSession,
) -> None:
    checks = evaluate_crime_quality(
        _quality_frame(spark),
        thresholds=QualityThresholds(
            minimum_silver_to_bronze_ratio=0.5,
            maximum_coordinate_null_rate=1.0,
        ),
        bronze_distinct_count=3,
        quarantine_count=0,
    )
    assert not blocking_failures(checks)
    assert any(check.check_name == "schema_compatibility" for check in checks)


def test_duplicate_business_identity_is_blocking(
    spark: SparkSession,
) -> None:
    frame = _quality_frame(spark)
    duplicate = frame.unionByName(frame.filter("source_system = 'dallas'"))
    checks = evaluate_crime_quality(
        duplicate,
        thresholds=QualityThresholds(
            maximum_coordinate_null_rate=1.0,
            maximum_silver_to_bronze_ratio=2.0,
        ),
        bronze_distinct_count=4,
    )
    failures = {check.check_name for check in blocking_failures(checks)}
    assert "duplicate_business_identities" in failures


def test_schema_mismatch_is_persistable_and_blocking(
    spark: SparkSession,
) -> None:
    checks = evaluate_crime_quality(
        _quality_frame(spark).drop("source_offense_id"),
        thresholds=QualityThresholds(),
    )
    assert {failure.check_name for failure in blocking_failures(checks)} == {
        "schema_compatibility"
    }
    results = quality_results_dataframe(
        spark,
        checks=checks,
        pipeline_run_id="run-1",
        table_name="catalog.silver.crime_offenses",
    )
    row = results.first()
    assert row is not None
    assert row["pipeline_run_id"] == "run-1"
    assert row["checked_at"] is not None
