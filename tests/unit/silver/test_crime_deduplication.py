from __future__ import annotations

import pytest
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from crimenet.transforms.canonical import (
    add_crime_offense_id,
    deduplicate_crime_offenses,
)


def test_fixture_duplicates_collapse_to_unique_ids(
    deduplicated_crimes: DataFrame,
) -> None:
    assert deduplicated_crimes.count() == 436
    assert (
        deduplicated_crimes
        .select("crime_offense_id")
        .distinct()
        .count()
        == 436
    )
    assert {
        row.source_city: row["count"]
        for row in (
            deduplicated_crimes
            .groupBy("source_city")
            .count()
            .collect()
        )
    } == {
        "dallas": 118,
        "houston": 156,
        "fort_worth": 162,
    }


def test_duplicating_every_input_row_does_not_change_key_set(
    crime_offenses_with_ids: DataFrame,
    deduplicated_crimes: DataFrame,
) -> None:
    duplicated_input = crime_offenses_with_ids.unionByName(
        crime_offenses_with_ids
    )
    result = deduplicate_crime_offenses(
        duplicated_input.repartition(6)
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


def test_repartitioning_and_reordering_do_not_change_logical_result(
    crime_offenses_with_ids: DataFrame,
) -> None:
    first = deduplicate_crime_offenses(
        crime_offenses_with_ids.repartition(2)
    )
    second = deduplicate_crime_offenses(
        crime_offenses_with_ids
        .orderBy(F.col("source_row_hash").desc())
        .repartition(7)
    )
    logical_columns = sorted(
        column_name
        for column_name in first.columns
        if column_name != "source_file"
    )

    assert (
        first.select(*logical_columns)
        .exceptAll(second.select(*logical_columns))
        .count()
        == 0
    )
    assert (
        second.select(*logical_columns)
        .exceptAll(first.select(*logical_columns))
        .count()
        == 0
    )


def test_latest_record_is_deterministic_survivor(
    dallas_canonical: DataFrame,
) -> None:
    source = dallas_canonical.filter(
        F.col("source_record_id") == "095681-2022-01"
    ).limit(1)
    older = (
        source
        .withColumn(
            "updated_at",
            F.to_timestamp(F.lit("2024-01-01 00:00:00")),
        )
        .withColumn("offense_name", F.lit("older value"))
        .withColumn("source_row_hash", F.lit("a" * 64))
    )
    newer = (
        source
        .withColumn(
            "updated_at",
            F.to_timestamp(F.lit("2025-01-01 00:00:00")),
        )
        .withColumn("offense_name", F.lit("newer value"))
        .withColumn("source_row_hash", F.lit("b" * 64))
    )
    identified = add_crime_offense_id(
        older.unionByName(newer).repartition(2)
    )

    row = deduplicate_crime_offenses(identified).first()
    assert row is not None
    assert row.offense_name == "newer value"
    assert row.source_row_hash == "b" * 64


def test_null_ids_are_rejected_before_window_deduplication(
    crime_offenses_with_ids: DataFrame,
) -> None:
    invalid = (
        crime_offenses_with_ids
        .limit(1)
        .withColumn(
            "crime_offense_id",
            F.lit(None).cast("string"),
        )
    )

    with pytest.raises(
        ValueError,
        match="no crime_offense_id",
    ):
        deduplicate_crime_offenses(
            invalid.unionByName(invalid)
        )


def test_missing_id_column_has_clear_contract_error(
    canonical_crimes: DataFrame,
) -> None:
    with pytest.raises(
        ValueError,
        match="crime_offense_id",
    ):
        deduplicate_crime_offenses(canonical_crimes)


def test_empty_input_is_supported_and_preserves_schema(
    crime_offenses_with_ids: DataFrame,
) -> None:
    empty = crime_offenses_with_ids.limit(0)
    result = deduplicate_crime_offenses(empty)

    assert result.count() == 0
    assert result.schema == empty.schema


def test_distinct_houston_content_is_not_collapsed(
    crime_offenses_with_ids: DataFrame,
) -> None:
    result = deduplicate_crime_offenses(
        crime_offenses_with_ids.filter(
            F.col("source_city") == "houston"
        )
    )

    for incident_id in ("6940023", "59808225"):
        assert result.filter(
            F.col("source_incident_id") == incident_id
        ).count() == 2
