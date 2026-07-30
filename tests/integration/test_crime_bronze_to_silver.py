from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from crimenet.contracts.silver import SILVER_COLUMNS
from crimenet.ingestion.column_names import normalize_column_names
from crimenet.ingestion.metadata import add_ingestion_metadata
from crimenet.ingestion.readers import (
    read_dallas_raw,
    read_fort_worth_raw,
    read_houston_raw,
)
from crimenet.jobs.bronze_ingestion import COLUMN_OVERRIDES
from crimenet.transforms.canonical import (
    add_crime_offense_id,
    build_crime_offenses,
    deduplicate_crime_offenses,
)


def _fixture_bronze_crimes(
    spark: SparkSession,
    fixture_path: Callable[[str], Path],
) -> tuple[DataFrame, DataFrame, DataFrame]:
    dallas = add_ingestion_metadata(
        normalize_column_names(
            read_dallas_raw(
                spark,
                str(fixture_path("dallas/dallas_fixture.csv")),
            )
        ),
        source_system="dallas",
    )
    houston = add_ingestion_metadata(
        normalize_column_names(
            read_houston_raw(
                spark,
                str(fixture_path("houston/houston_fixture.csv")),
            )
        ),
        source_system="houston",
    )
    fort_worth = add_ingestion_metadata(
        normalize_column_names(
            read_fort_worth_raw(
                spark,
                str(
                    fixture_path(
                        "fort_worth/fort_worth_fixture.json"
                    )
                ),
            ),
            overrides=COLUMN_OVERRIDES["fort_worth"],
        ),
        source_system="fort_worth",
    )
    return dallas, houston, fort_worth


def _build_fixture_silver(
    spark: SparkSession,
    fixture_path: Callable[[str], Path],
) -> DataFrame:
    dallas, houston, fort_worth = _fixture_bronze_crimes(
        spark,
        fixture_path,
    )
    canonical = build_crime_offenses(
        dallas,
        houston,
        fort_worth,
    )
    return deduplicate_crime_offenses(
        add_crime_offense_id(canonical)
    )


def test_real_readers_through_deterministic_silver_pipeline(
    spark: SparkSession,
    fixture_path: Callable[[str], Path],
) -> None:
    dallas, houston, fort_worth = _fixture_bronze_crimes(
        spark,
        fixture_path,
    )
    canonical = build_crime_offenses(
        dallas,
        houston,
        fort_worth,
    )
    identified = add_crime_offense_id(canonical)
    silver = deduplicate_crime_offenses(identified)

    assert canonical.count() == 439
    assert tuple(canonical.columns) == SILVER_COLUMNS
    assert identified.filter(
        F.col("crime_offense_id").isNull()
    ).count() == 0
    assert silver.count() == 436
    assert (
        silver.select("crime_offense_id").distinct().count()
        == 436
    )
    assert {
        row.source_city: row["count"]
        for row in silver.groupBy("source_city").count().collect()
    } == {
        "dallas": 118,
        "houston": 156,
        "fort_worth": 162,
    }

    representative_ids = {
        row.source_city: row.crime_offense_id
        for row in (
            silver
            .filter(
                (
                    (F.col("source_city") == "dallas")
                    & (
                        F.col("source_record_id")
                        == "095681-2022-01"
                    )
                )
                | (
                    (F.col("source_city") == "houston")
                    & (
                        F.col("source_incident_id")
                        == "126259321"
                    )
                )
                | (
                    (F.col("source_city") == "fort_worth")
                    & (
                        F.col("source_record_id")
                        == "190085144-23C"
                    )
                )
            )
            .select("source_city", "crime_offense_id")
            .collect()
        )
    }
    assert representative_ids == {
        "dallas": (
            "53eb340d8a16356b296b96c3c8f77df77f90e1958c93b4c5"
            "c1ef2533ab0607cf"
        ),
        "houston": (
            "c9482fd041a9764fbbd13feb5b0717f49bc7b4e9ab719c009"
            "8d58fcce1c85b7b"
        ),
        "fort_worth": (
            "4a38ca6c9d47577bed88a36776145423631b6562bd7aa99433"
            "86172187bf48d3"
        ),
    }


def test_complete_fixture_pipeline_is_idempotent(
    spark: SparkSession,
    fixture_path: Callable[[str], Path],
) -> None:
    first = _build_fixture_silver(spark, fixture_path)
    second = _build_fixture_silver(spark, fixture_path)
    stable_columns = sorted(
        column_name
        for column_name in first.columns
        if column_name != "source_file"
    )

    assert (
        first.select(*stable_columns)
        .exceptAll(second.select(*stable_columns))
        .count()
        == 0
    )
    assert (
        second.select(*stable_columns)
        .exceptAll(first.select(*stable_columns))
        .count()
        == 0
    )


def test_reordered_and_duplicated_sources_keep_original_silver_keys(
    dallas_bronze: DataFrame,
    houston_bronze: DataFrame,
    fort_worth_bronze: DataFrame,
    deduplicated_crimes: DataFrame,
) -> None:
    stressed = build_crime_offenses(
        dallas_bronze
        .orderBy(F.col("source_row_hash").desc())
        .unionByName(dallas_bronze),
        houston_bronze
        .repartition(3)
        .unionByName(houston_bronze),
        fort_worth_bronze
        .orderBy(F.col("source_row_hash"))
        .unionByName(fort_worth_bronze),
    )
    result = deduplicate_crime_offenses(
        add_crime_offense_id(stressed).repartition(8)
    )

    assert result.count() == 436
    assert (
        result.select("crime_offense_id")
        .exceptAll(
            deduplicated_crimes.select("crime_offense_id")
        )
        .count()
        == 0
    )
