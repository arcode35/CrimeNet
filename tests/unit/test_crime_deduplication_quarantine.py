from __future__ import annotations

from datetime import datetime

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from crimenet.quality.quarantine import split_crime_quarantine
from crimenet.transforms.deduplication import deduplicate_crime_offenses


def _dedup_rows(spark: SparkSession) -> DataFrame:
    return spark.createDataFrame(
        [
            (
                "id-1",
                datetime(2024, 1, 2),
                datetime(2024, 1, 1),
                "hash-a",
                datetime(2024, 1, 3),
                "/a.csv",
                "older",
            ),
            (
                "id-1",
                datetime(2024, 2, 2),
                datetime(2024, 2, 1),
                "hash-b",
                datetime(2024, 2, 3),
                "/b.csv",
                "newer",
            ),
            (
                "id-2",
                None,
                None,
                "same-hash",
                datetime(2024, 1, 1),
                "/same.csv",
                "same",
            ),
            (
                "id-2",
                None,
                None,
                "same-hash",
                datetime(2024, 3, 1),
                "/copied.csv",
                "same",
            ),
        ],
        """
        business_identity STRING,
        updated_at TIMESTAMP,
        reported_at TIMESTAMP,
        source_row_hash STRING,
        bronze_ingested_at TIMESTAMP,
        source_file STRING,
        offense_name STRING
        """,
    )


def test_deduplication_handles_same_file_cross_file_replay_and_conflicts(
    spark: SparkSession,
) -> None:
    result = deduplicate_crime_offenses(_dedup_rows(spark))
    rows = {
        row["business_identity"]: row.asDict()
        for row in result.collect()
    }
    assert set(rows) == {"id-1", "id-2"}
    assert rows["id-1"]["offense_name"] == "newer"
    assert rows["id-2"]["source_row_hash"] == "same-hash"


def test_quarantine_has_stable_per_reason_identity(
    spark: SparkSession,
) -> None:
    dataframe = spark.createDataFrame(
        [
            (
                "dallas",
                None,
                None,
                "business",
                "row-hash",
                "1800-01-01 00:00:00",
                95.0,
                -96.0,
                None,
                "/a.csv",
            )
        ],
        """
        source_system STRING,
        source_incident_id STRING,
        source_offense_id STRING,
        business_identity STRING,
        source_row_hash STRING,
        occurred_at STRING,
        latitude DOUBLE,
        longitude DOUBLE,
        source_corrupt_record STRING,
        source_file STRING
        """,
    ).withColumn("occurred_at", F.col("occurred_at").cast("timestamp"))
    valid, quarantine = split_crime_quarantine(
        dataframe,
        pipeline_run_id="run-1",
    )
    assert valid.count() == 0
    reasons = {
        row["quarantine_reason_code"]: row["quarantine_id"]
        for row in quarantine.collect()
    }
    assert {
        "MISSING_SOURCE_INCIDENT_ID",
        "MISSING_SOURCE_OFFENSE_ID",
        "IMPLAUSIBLE_OCCURRED_AT",
        "INVALID_COORDINATES",
    } <= set(reasons)

    _, replay = split_crime_quarantine(
        dataframe,
        pipeline_run_id="run-2",
    )
    replay_ids = {
        row["quarantine_reason_code"]: row["quarantine_id"]
        for row in replay.collect()
    }
    assert reasons == replay_ids


def test_missing_source_hash_uses_payload_identity_without_collapsing_rejects(
    spark: SparkSession,
) -> None:
    dataframe = (
        spark.createDataFrame(
            [
                ("dallas", "INC-1", "OFF-1", "business-1", None),
                ("dallas", "INC-2", "OFF-2", "business-2", ""),
            ],
            """
            source_system STRING,
            source_incident_id STRING,
            source_offense_id STRING,
            business_identity STRING,
            source_row_hash STRING
            """,
        )
        .withColumn("occurred_at", F.lit("2024-01-01").cast("timestamp"))
        .withColumn("latitude", F.lit(32.7))
        .withColumn("longitude", F.lit(-96.8))
        .withColumn("source_corrupt_record", F.lit(None).cast("string"))
        .withColumn("source_file", F.lit("/landing/source.csv"))
    )

    _, first = split_crime_quarantine(
        dataframe,
        pipeline_run_id="run-1",
    )
    _, replay = split_crime_quarantine(
        dataframe,
        pipeline_run_id="run-2",
    )
    first_ids = {
        row["source_incident_id"]: row["quarantine_id"]
        for row in first.select(
            F.get_json_object(
                "validation_fields",
                "$.source_incident_id",
            ).alias("source_incident_id"),
            "quarantine_id",
        ).collect()
    }
    replay_ids = {
        row["source_incident_id"]: row["quarantine_id"]
        for row in replay.select(
            F.get_json_object(
                "validation_fields",
                "$.source_incident_id",
            ).alias("source_incident_id"),
            "quarantine_id",
        ).collect()
    }

    assert len(set(first_ids.values())) == 2
    assert first_ids == replay_ids
