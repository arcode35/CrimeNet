from __future__ import annotations

from datetime import datetime

import pytest
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from crimenet.contracts.silver import (
    SILVER_COLUMNS,
    assert_silver_contract,
)


def test_houston_fixture_maps_to_exact_canonical_contract(
    houston_canonical: DataFrame,
) -> None:
    assert houston_canonical.count() == 157
    assert tuple(houston_canonical.columns) == SILVER_COLUMNS
    assert_silver_contract(houston_canonical)
    assert {
        row.source_city
        for row in (
            houston_canonical
            .select("source_city")
            .distinct()
            .collect()
        )
    } == {"houston"}


def test_houston_representative_row_has_exact_mappings(
    houston_canonical: DataFrame,
) -> None:
    row = houston_canonical.filter(
        F.col("source_incident_id") == "126259321"
    ).first()

    assert row is not None
    assert row.source_record_id == row.source_row_hash
    assert row.source_row_hash == (
        "6cab1910adcfdad24bf4b528b39d9f185c4df079642e7042e"
        "8fbc34b2142bf94"
    )
    assert row.offense_code == "90J"
    assert row.offense_name == "Trespass of real property"
    assert row.offense_description == "Trespass of real property"
    assert row.occurred_at == datetime(2021, 9, 18, 19)
    assert row.reported_at is None
    assert row.updated_at is None
    assert row.offense_count == 1
    assert row.address == "926 WESTHEIMER RD"
    assert row.city == "HOUSTON"
    assert row.state == "TX"
    assert row.postal_code == "77006"
    assert row.beat == "1A20"
    assert row.premise_type == "Service, Gas Station"
    assert row.latitude == pytest.approx(29.744681)
    assert row.longitude == pytest.approx(-95.391398)
    assert row.alternate_latitude is None
    assert row.alternate_longitude is None
    assert row.source_x_coordinate is None
    assert row.source_y_coordinate is None
    assert row.source_file.endswith("houston_fixture.csv")


def test_houston_literal_null_coordinates_are_safely_cast(
    houston_canonical: DataFrame,
) -> None:
    row = houston_canonical.filter(
        F.col("source_incident_id") == "72337425"
    ).first()

    assert row is not None
    assert row.postal_code == "NULL"
    assert row.latitude is None
    assert row.longitude is None
    assert houston_canonical.filter(
        F.col("latitude").isNull()
        | F.col("longitude").isNull()
    ).count() == 20


def test_houston_blank_fixture_record_preserves_row_with_null_payload(
    houston_canonical: DataFrame,
) -> None:
    rows = houston_canonical.filter(
        F.col("source_incident_id").isNull()
    ).collect()

    assert len(rows) == 1
    row = rows[0]
    assert row.source_record_id is not None
    assert row.source_record_id == row.source_row_hash
    assert row.offense_code is None
    assert row.offense_name is None
    assert row.occurred_at is None
    assert row.offense_count is None
    assert row.latitude is None
    assert row.longitude is None


def test_houston_exact_duplicate_has_same_content_hash(
    houston_canonical: DataFrame,
) -> None:
    duplicates = houston_canonical.filter(
        F.col("source_incident_id") == "35512324"
    ).collect()

    assert len(duplicates) == 2
    assert len({row.source_row_hash for row in duplicates}) == 1
    assert len({row.source_record_id for row in duplicates}) == 1
    assert {
        row.occurred_at for row in duplicates
    } == {datetime(2024, 3, 10, 4)}


def test_houston_nonidentical_same_incident_rows_keep_distinct_hashes(
    houston_canonical: DataFrame,
) -> None:
    for incident_id, expected_premises in {
        "6940023": {
            "Residence, Home (Includes Apartment)",
            "Cyberspace",
        },
        "59808225": {
            "Other, Unknown",
            "Residence, Home (Includes Apartment)",
        },
    }.items():
        rows = houston_canonical.filter(
            F.col("source_incident_id") == incident_id
        ).collect()
        assert len(rows) == 2
        assert {row.premise_type for row in rows} == expected_premises
        assert len({row.source_row_hash for row in rows}) == 2
