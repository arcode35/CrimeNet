from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from crimenet.ingestion.column_names import normalize_column_names
from crimenet.ingestion.metadata import (
    add_ingestion_metadata,
    source_row_hash,
)
from crimenet.ingestion.readers import read_houston_raw


def test_fixture_bronze_metadata_is_complete_and_source_specific(
    dallas_bronze: DataFrame,
    houston_bronze: DataFrame,
    fort_worth_bronze: DataFrame,
    socioeconomic_bronze: DataFrame,
    weather_bronze: DataFrame,
) -> None:
    expectations = [
        (dallas_bronze, "dallas", 119, 118),
        (houston_bronze, "houston", 157, 156),
        (fort_worth_bronze, "fort_worth", 163, 162),
        (socioeconomic_bronze, "census_acs5", 117, 116),
        (weather_bronze, "open_meteo", 94, 93),
    ]

    for dataframe, source_system, row_count, hash_count in expectations:
        summary = dataframe.agg(
            F.count(F.lit(1)).alias("row_count"),
            F.sum(
                F.col("source_file").isNull().cast("int")
            ).alias("null_source_files"),
            F.sum(
                F.col("source_row_hash").isNull().cast("int")
            ).alias("null_hashes"),
            F.sum(
                (
                    ~F.col("source_row_hash")
                    .rlike("^[0-9a-f]{64}$")
                ).cast("int")
            ).alias("invalid_hashes"),
            F.countDistinct("source_row_hash").alias(
                "distinct_hashes"
            ),
            F.collect_set("source_system").alias("source_systems"),
        ).first()

        assert summary is not None
        assert summary.row_count == row_count
        assert summary.null_source_files == 0
        assert summary.null_hashes == 0
        assert summary.invalid_hashes == 0
        assert summary.distinct_hashes == hash_count
        assert set(summary.source_systems) == {
            source_system
        }


def test_metadata_addition_preserves_fixture_payload_values(
    houston_raw: DataFrame,
    houston_bronze: DataFrame,
) -> None:
    normalized = normalize_column_names(houston_raw)
    payload_columns = normalized.columns
    expected = (
        normalized
        .filter(F.col("incident") == "126259321")
        .select(*payload_columns)
        .first()
    )
    actual = (
        houston_bronze
        .filter(F.col("incident") == "126259321")
        .select(*payload_columns)
        .first()
    )

    assert expected is not None
    assert actual is not None
    assert actual.asDict() == expected.asDict()
    assert set(houston_bronze.columns) == (
        set(payload_columns)
        | {
            "source_row_hash",
            "source_system",
            "ingested_at",
        }
    )


def test_rereading_unchanged_fixture_reproduces_hash(
    spark: SparkSession,
    fixture_path: Callable[[str], Path],
    houston_bronze: DataFrame,
) -> None:
    reread = add_ingestion_metadata(
        normalize_column_names(
            read_houston_raw(
                spark,
                str(fixture_path("houston/houston_fixture.csv")),
            )
        ),
        source_system="houston",
    )
    expected = (
        houston_bronze
        .filter(F.col("incident") == "126259321")
        .select("source_row_hash")
        .first()
    )
    actual = (
        reread
        .filter(F.col("incident") == "126259321")
        .select("source_row_hash")
        .first()
    )

    assert expected is not None
    assert actual is not None
    assert actual.source_row_hash == expected.source_row_hash


def test_operational_metadata_is_excluded_from_content_hash(
    houston_bronze: DataFrame,
) -> None:
    payload = (
        houston_bronze
        .filter(F.col("incident") == "126259321")
        .limit(1)
        .drop("source_row_hash", "source_system", "ingested_at")
    )
    first = (
        payload
        .withColumn("source_file", F.lit("landing/first.csv"))
        .withColumn("source_system", F.lit("houston"))
        .withColumn(
            "ingested_at",
            F.to_timestamp(F.lit("2025-01-01 00:00:00")),
        )
    )
    moved = (
        payload
        .withColumn("source_file", F.lit("landing/moved.csv"))
        .withColumn("source_system", F.lit("renamed_system"))
        .withColumn(
            "ingested_at",
            F.to_timestamp(F.lit("2026-01-01 00:00:00")),
        )
    )

    first_hash = first.select(
        source_row_hash(first).alias("hash")
    ).first()
    moved_hash = moved.select(
        source_row_hash(moved).alias("hash")
    ).first()
    assert first_hash is not None
    assert moved_hash is not None
    assert first_hash.hash == moved_hash.hash


def test_payload_change_and_new_payload_column_change_hash(
    houston_bronze: DataFrame,
) -> None:
    payload = (
        houston_bronze
        .filter(F.col("incident") == "126259321")
        .limit(1)
        .drop("source_row_hash", "source_system", "ingested_at")
    )
    original = payload.select(
        source_row_hash(payload).alias("hash")
    ).first()
    changed_payload = payload.withColumn(
        "premise",
        F.lit("Changed premise"),
    )
    changed = changed_payload.select(
        source_row_hash(changed_payload).alias("hash")
    ).first()
    extended_payload = payload.withColumn(
        "new_source_field",
        F.lit(None).cast("string"),
    )
    extended = extended_payload.select(
        source_row_hash(extended_payload).alias("hash")
    ).first()

    assert original is not None
    assert changed is not None
    assert extended is not None
    assert changed.hash != original.hash
    assert extended.hash != original.hash


def test_hash_is_independent_of_dataframe_column_order(
    houston_bronze: DataFrame,
) -> None:
    payload = (
        houston_bronze
        .filter(F.col("incident") == "126259321")
        .limit(1)
        .drop("source_row_hash", "source_system", "ingested_at")
    )
    reordered = payload.select(*reversed(payload.columns))

    original_hash = payload.select(
        source_row_hash(payload).alias("hash")
    ).first()
    reordered_hash = reordered.select(
        source_row_hash(reordered).alias("hash")
    ).first()
    assert original_hash is not None
    assert reordered_hash is not None
    assert reordered_hash.hash == original_hash.hash
