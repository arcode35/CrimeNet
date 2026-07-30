from __future__ import annotations

import pytest
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from crimenet.transforms.canonical import add_crime_offense_id

pytestmark = pytest.mark.unit

def test_fixture_crime_ids_are_non_null_unique_content_hashes(
    crime_offenses_with_ids: DataFrame,
) -> None:
    assert crime_offenses_with_ids.count() == 439
    assert crime_offenses_with_ids.filter(
        F.col("crime_offense_id").isNull()
    ).count() == 0
    assert crime_offenses_with_ids.filter(
        ~F.col("crime_offense_id").rlike("^[0-9a-f]{64}$")
    ).count() == 0
    assert (
        crime_offenses_with_ids
        .select("crime_offense_id")
        .distinct()
        .count()
        == 436
    )


def test_representative_city_ids_match_hard_coded_expected_values(
    crime_offenses_with_ids: DataFrame,
) -> None:
    expectations = {
        ("dallas", "095681-2022-01"): (
            "53eb340d8a16356b296b96c3c8f77df77f90e1958c93b4c5"
            "c1ef2533ab0607cf"
        ),
        ("houston", "126259321"): (
            "c9482fd041a9764fbbd13feb5b0717f49bc7b4e9ab719c009"
            "8d58fcce1c85b7b"
        ),
        ("fort_worth", "190085144-23C"): (
            "4a38ca6c9d47577bed88a36776145423631b6562bd7aa99433"
            "86172187bf48d3"
        ),
    }

    for (city, source_key), expected_id in expectations.items():
        if city == "houston":
            predicate = F.col("source_incident_id") == source_key
        else:
            predicate = F.col("source_record_id") == source_key
        ids = {
            row.crime_offense_id
            for row in (
                crime_offenses_with_ids
                .filter(
                    (F.col("source_city") == city)
                    & predicate
                )
                .select("crime_offense_id")
                .collect()
            )
        }
        assert ids == {expected_id}


def test_ids_are_stable_after_repartition_and_repeated_transform(
    canonical_crimes: DataFrame,
) -> None:
    first = add_crime_offense_id(canonical_crimes)
    second = add_crime_offense_id(
        canonical_crimes
        .orderBy(F.col("source_row_hash").desc())
        .repartition(5)
    )
    identity_columns = [
        "source_city",
        "source_row_hash",
        "source_record_id",
        "crime_offense_id",
    ]

    assert (
        first.select(*identity_columns)
        .exceptAll(second.select(*identity_columns))
        .count()
        == 0
    )
    assert (
        second.select(*identity_columns)
        .exceptAll(first.select(*identity_columns))
        .count()
        == 0
    )


def test_dallas_and_fort_worth_ids_ignore_operational_content_hash(
    canonical_crimes: DataFrame,
) -> None:
    for city, record_id in [
        ("dallas", "095681-2022-01"),
        ("fort_worth", "190085144-23C"),
    ]:
        source = canonical_crimes.filter(
            (F.col("source_city") == city)
            & (F.col("source_record_id") == record_id)
        ).limit(1)
        changed = (
            source
            .withColumn("source_row_hash", F.lit("0" * 64))
            .withColumn("source_file", F.lit("moved/source.file"))
        )
        original_id = add_crime_offense_id(source).first()
        changed_id = add_crime_offense_id(changed).first()
        assert original_id is not None
        assert changed_id is not None
        assert changed_id.crime_offense_id == (
            original_id.crime_offense_id
        )


def test_houston_identity_uses_content_not_physical_file(
    houston_canonical: DataFrame,
) -> None:
    source = houston_canonical.filter(
        F.col("source_incident_id") == "126259321"
    ).limit(1)
    moved = source.withColumn(
        "source_file",
        F.lit("moved/houston.csv"),
    )
    changed_content_hash = source.withColumn(
        "source_row_hash",
        F.lit("0" * 64),
    )

    original = add_crime_offense_id(source).first()
    moved_row = add_crime_offense_id(moved).first()
    changed = add_crime_offense_id(changed_content_hash).first()
    assert original is not None
    assert moved_row is not None
    assert changed is not None
    assert moved_row.crime_offense_id == original.crime_offense_id
    assert changed.crime_offense_id != original.crime_offense_id


def test_nonidentical_houston_rows_with_same_incident_remain_distinct(
    houston_canonical: DataFrame,
) -> None:
    identified = add_crime_offense_id(houston_canonical)

    for incident_id in ("6940023", "59808225"):
        rows = identified.filter(
            F.col("source_incident_id") == incident_id
        ).collect()
        assert len(rows) == 2
        assert len({row.crime_offense_id for row in rows}) == 2


def test_exact_fixture_duplicates_receive_the_same_id(
    crime_offenses_with_ids: DataFrame,
) -> None:
    duplicate_expectations = [
        ("dallas", "140956-2023-01"),
        ("houston", "35512324"),
        ("fort_worth", "190085144-23C"),
    ]
    for city, source_key in duplicate_expectations:
        key_column = (
            "source_incident_id"
            if city == "houston"
            else "source_record_id"
        )
        rows = crime_offenses_with_ids.filter(
            (F.col("source_city") == city)
            & (F.col(key_column) == source_key)
        ).collect()
        assert len(rows) == 2
        assert len({row.crime_offense_id for row in rows}) == 1
