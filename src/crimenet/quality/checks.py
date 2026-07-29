"""Auditable quality checks for canonical CrimeNet tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from crimenet.config.validation import (
    QualityThresholds,
    validate_qualified_table_name,
)
from crimenet.contracts.silver import SILVER_SCHEMA
from crimenet.quality.quarantine import SUPPORTED_CRIME_SOURCES

QUALITY_RESULTS_SCHEMA = StructType(
    [
        StructField("pipeline_run_id", StringType(), False),
        StructField("check_name", StringType(), False),
        StructField("severity", StringType(), False),
        StructField("passed", BooleanType(), False),
        StructField("observed_value", StringType(), False),
        StructField("expected_threshold", StringType(), False),
        StructField("table_name", StringType(), False),
        StructField("checked_at", TimestampType(), True),
        StructField("source_system", StringType(), True),
    ]
)


@dataclass(frozen=True)
class QualityCheck:
    check_name: str
    severity: str
    passed: bool
    observed_value: Any
    expected_threshold: str
    source_system: str | None = None

    def as_record(self, *, pipeline_run_id: str, table_name: str) -> tuple[Any, ...]:
        return (
            pipeline_run_id,
            self.check_name,
            self.severity,
            self.passed,
            str(self.observed_value),
            self.expected_threshold,
            table_name,
            None,
            self.source_system,
        )


def _count_where(dataframe: DataFrame, condition: Column) -> int:
    return dataframe.filter(condition).count()


def _schema_mismatches(dataframe: DataFrame) -> list[str]:
    expected = {
        field.name: field.dataType.simpleString()
        for field in SILVER_SCHEMA.fields
    }
    expected["weather_query_cell_id"] = "bigint"
    actual = {
        field.name: field.dataType.simpleString()
        for field in dataframe.schema.fields
    }
    mismatches = [
        f"{name}: expected {data_type}, got {actual.get(name, 'missing')}"
        for name, data_type in expected.items()
        if actual.get(name) != data_type
    ]
    unexpected = sorted(set(actual) - set(expected))
    if unexpected:
        mismatches.append("unexpected columns: " + ", ".join(unexpected))
    return mismatches


def evaluate_crime_quality(
    dataframe: DataFrame,
    *,
    thresholds: QualityThresholds,
    bronze_distinct_count: int | None = None,
    quarantine_count: int = 0,
    quarantine_reason_counts: dict[str, int] | None = None,
) -> list[QualityCheck]:
    """Evaluate blocking and informational canonical crime invariants."""
    thresholds.validate()
    total = dataframe.count()
    checks: list[QualityCheck] = []
    schema_mismatches = _schema_mismatches(dataframe)
    checks.append(
        QualityCheck(
            "schema_compatibility",
            "BLOCKING",
            not schema_mismatches,
            "; ".join(schema_mismatches) or "compatible",
            "canonical Silver schema plus weather_query_cell_id",
        )
    )
    checks.extend(
        [
            QualityCheck(
                "minimum_row_count",
                "BLOCKING",
                total >= thresholds.minimum_row_count,
                total,
                f">={thresholds.minimum_row_count}",
            ),
            QualityCheck(
                "maximum_row_count",
                "BLOCKING",
                total <= thresholds.maximum_row_count,
                total,
                f"<={thresholds.maximum_row_count}",
            ),
        ]
    )

    if schema_mismatches:
        return checks

    duplicate_count = (
        dataframe.groupBy("business_identity")
        .count()
        .filter(F.col("count") > 1)
        .agg(F.coalesce(F.sum(F.col("count") - 1), F.lit(0)).alias("count"))
        .first()
    )
    duplicates = int(duplicate_count["count"]) if duplicate_count else 0
    missing_identifiers = _count_where(
        dataframe,
        F.col("source_incident_id").isNull()
        | (F.trim("source_incident_id") == "")
        | F.col("source_offense_id").isNull()
        | (F.trim("source_offense_id") == "")
        | F.col("business_identity").isNull()
        | F.col("source_row_hash").isNull(),
    )
    invalid_timestamps = _count_where(
        dataframe,
        F.col("occurred_at").isNull()
        | (F.col("occurred_at") < F.lit("1900-01-01").cast("timestamp"))
        | (
            F.col("occurred_at")
            > F.current_timestamp() + F.expr("INTERVAL 1 DAY")
        ),
    )
    invalid_coordinates = _count_where(
        dataframe,
        (F.col("latitude").isNull() != F.col("longitude").isNull())
        | (
            F.col("latitude").isNotNull()
            & (
                F.isnan("latitude")
                | F.isnan("longitude")
                | ~F.col("latitude").between(-90.0, 90.0)
                | ~F.col("longitude").between(-180.0, 180.0)
            )
        ),
    )
    unsupported_sources = _count_where(
        dataframe,
        ~F.col("source_system").isin(*SUPPORTED_CRIME_SOURCES)
        | F.col("source_system").isNull(),
    )

    checks.extend(
        [
            QualityCheck(
                "duplicate_business_identities",
                "BLOCKING",
                duplicates == 0,
                duplicates,
                "0",
            ),
            QualityCheck(
                "missing_required_identifiers",
                "BLOCKING",
                missing_identifiers == 0,
                missing_identifiers,
                "0",
            ),
            QualityCheck(
                "invalid_or_implausible_timestamps",
                "BLOCKING",
                invalid_timestamps == 0,
                invalid_timestamps,
                "0",
            ),
            QualityCheck(
                "invalid_coordinates",
                "BLOCKING",
                invalid_coordinates == 0,
                invalid_coordinates,
                "0",
            ),
            QualityCheck(
                "unsupported_source_systems",
                "BLOCKING",
                unsupported_sources == 0,
                unsupported_sources,
                "0",
            ),
        ]
    )

    critical_columns = (
        "business_identity",
        "source_incident_id",
        "source_offense_id",
        "source_row_hash",
        "occurred_at",
    )
    for column_name in critical_columns:
        null_count = _count_where(dataframe, F.col(column_name).isNull())
        null_rate = null_count / total if total else 1.0
        checks.append(
            QualityCheck(
                f"{column_name}_null_rate",
                "BLOCKING",
                null_rate <= thresholds.maximum_critical_null_rate,
                f"{null_rate:.8f}",
                f"<={thresholds.maximum_critical_null_rate:.8f}",
            )
        )

    coordinate_null_count = _count_where(
        dataframe,
        F.col("latitude").isNull() | F.col("longitude").isNull(),
    )
    coordinate_null_rate = coordinate_null_count / total if total else 1.0
    checks.append(
        QualityCheck(
            "coordinate_null_rate",
            "BLOCKING",
            coordinate_null_rate <= thresholds.maximum_coordinate_null_rate,
            f"{coordinate_null_rate:.8f}",
            f"<={thresholds.maximum_coordinate_null_rate:.8f}",
        )
    )

    for source in SUPPORTED_CRIME_SOURCES:
        source_count = _count_where(
            dataframe,
            F.col("source_system") == source,
        )
        checks.append(
            QualityCheck(
                "source_row_count",
                "BLOCKING",
                source_count > 0,
                source_count,
                ">0",
                source,
            )
        )

    if bronze_distinct_count is not None:
        ratio = total / bronze_distinct_count if bronze_distinct_count else 0.0
        checks.append(
            QualityCheck(
                "silver_to_bronze_row_ratio",
                "BLOCKING",
                thresholds.minimum_silver_to_bronze_ratio
                <= ratio
                <= thresholds.maximum_silver_to_bronze_ratio,
                f"{ratio:.8f}",
                (
                    f"[{thresholds.minimum_silver_to_bronze_ratio:.8f},"
                    f"{thresholds.maximum_silver_to_bronze_ratio:.8f}]"
                ),
            )
        )

    considered = total + quarantine_count
    quarantine_rate = quarantine_count / considered if considered else 0.0
    checks.append(
        QualityCheck(
            "quarantine_rate",
            "BLOCKING",
            quarantine_rate <= thresholds.maximum_quarantine_rate,
            f"{quarantine_rate:.8f}",
            f"<={thresholds.maximum_quarantine_rate:.8f}",
        )
    )
    for reason, count in sorted((quarantine_reason_counts or {}).items()):
        checks.append(
            QualityCheck(
                f"quarantine_reason:{reason}",
                "INFO",
                True,
                count,
                "measured",
            )
        )
    return checks


def quality_results_dataframe(
    spark: SparkSession,
    *,
    checks: list[QualityCheck],
    pipeline_run_id: str,
    table_name: str,
) -> DataFrame:
    records = [
        check.as_record(
            pipeline_run_id=pipeline_run_id,
            table_name=table_name,
        )
        for check in checks
    ]
    dataframe = spark.createDataFrame(records, schema=QUALITY_RESULTS_SCHEMA)
    return dataframe.withColumn("checked_at", F.current_timestamp())


def merge_quality_results(
    spark: SparkSession,
    *,
    results: DataFrame,
    target_table: str,
) -> None:
    """Upsert one result for each run, table, check, and source."""
    validate_qualified_table_name(target_table)
    if not spark.catalog.tableExists(target_table):
        results.limit(0).write.format("delta").saveAsTable(target_table)

    view_name = f"_crimenet_quality_{uuid4().hex}"
    results.createOrReplaceTempView(view_name)
    try:
        spark.sql(
            f"""
            MERGE INTO {target_table} AS target
            USING {view_name} AS source
              ON target.pipeline_run_id = source.pipeline_run_id
             AND target.table_name = source.table_name
             AND target.check_name = source.check_name
             AND coalesce(target.source_system, '')
                 = coalesce(source.source_system, '')
            WHEN MATCHED THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *
            """
        )
    finally:
        spark.catalog.dropTempView(view_name)


def blocking_failures(checks: list[QualityCheck]) -> list[QualityCheck]:
    return [
        check
        for check in checks
        if check.severity == "BLOCKING" and not check.passed
    ]
