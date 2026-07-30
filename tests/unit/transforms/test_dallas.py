from __future__ import annotations

from datetime import datetime

import pytest
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from crimenet.contracts.silver import (
    SILVER_COLUMNS,
    assert_silver_contract,
)
from crimenet.transforms.dallas import to_canonical


def test_dallas_fixture_maps_to_exact_canonical_contract(
    dallas_canonical: DataFrame,
) -> None:
    assert dallas_canonical.count() == 119
    assert tuple(dallas_canonical.columns) == SILVER_COLUMNS
    assert_silver_contract(dallas_canonical)
    assert {
        row.source_city
        for row in (
            dallas_canonical
            .select("source_city")
            .distinct()
            .collect()
        )
    } == {"dallas"}
    assert dallas_canonical.filter(
        F.col("alternate_latitude").isNotNull()
        | F.col("alternate_longitude").isNotNull()
    ).count() == 0


def test_dallas_representative_row_has_exact_mappings(
    dallas_canonical: DataFrame,
) -> None:
    row = dallas_canonical.filter(
        F.col("source_record_id") == "095681-2022-01"
    ).first()

    assert row is not None
    assert row.source_incident_id == "095681-2022"
    assert row.offense_code == "999"
    assert row.offense_name == "MISCELLANEOUS"
    assert row.offense_description == "FOUND PROPERTY (NO OFFENSE)"
    assert row.occurred_at == datetime(2022, 5, 29, 5, 11)
    assert row.reported_at == datetime(2022, 5, 29, 5, 11, 10)
    assert row.updated_at == datetime(2022, 9, 29, 11, 30, 30)
    assert row.offense_count == 1
    assert row.address == "7229 HOLLY HILL DR"
    assert row.city == "DALLAS"
    assert row.state == "TX"
    assert row.postal_code == "75231"
    assert row.beat == "212"
    assert row.premise_type == "Single Family Residence - Occupied"
    assert row.latitude == pytest.approx(32.877)
    assert row.longitude == pytest.approx(-96.75812)
    assert row.source_x_coordinate == pytest.approx(2503137.80545)
    assert row.source_y_coordinate == pytest.approx(7006354.73139)
    assert row.source_file.endswith("dallas_fixture.csv")
    assert row.source_row_hash == (
        "a9d0461a1473fa9ea9538ea9fcd1518feeb15b6479e326d1"
        "659f780a44309389"
    )


def test_dallas_offense_fallback_precedence_uses_fixture_edges(
    dallas_canonical: DataFrame,
) -> None:
    ucr_fallback = dallas_canonical.filter(
        F.col("source_record_id") == "073342-2016-01"
    ).first()
    incident_fallback = dallas_canonical.filter(
        F.col("source_record_id") == "403608-1982-01"
    ).first()

    assert ucr_fallback is not None
    assert ucr_fallback.offense_code is None
    assert ucr_fallback.offense_name == "ROBBERY-INDIVIDUAL"
    assert ucr_fallback.offense_description == "ROBBERY"

    assert incident_fallback is not None
    assert incident_fallback.offense_code is None
    assert incident_fallback.offense_name == "LEGACY HOMICIDE CODE"
    assert incident_fallback.offense_description == "LEGACY HOMICIDE CODE"


def test_dallas_missing_fixture_coordinates_become_null(
    dallas_canonical: DataFrame,
) -> None:
    row = dallas_canonical.filter(
        F.col("source_record_id") == "225975-2014-01"
    ).first()

    assert row is not None
    assert row.latitude is None
    assert row.longitude is None
    assert row.source_x_coordinate is None
    assert row.source_y_coordinate is None
    assert dallas_canonical.filter(
        F.col("latitude").isNull()
        | F.col("longitude").isNull()
    ).count() == 6


def test_dallas_invalid_values_are_null_instead_of_throwing(
    dallas_bronze: DataFrame,
) -> None:
    malformed = (
        dallas_bronze
        .filter(F.col("service_number_id") == "095681-2022-01")
        .withColumn("date1_of_occurrence", F.lit("not-a-date"))
        .withColumn("time1_of_occurrence", F.lit("99:99"))
        .withColumn("date_of_report", F.lit("not-a-timestamp"))
        .withColumn("update_date", F.lit("also-invalid"))
        .withColumn("location1", F.lit("invalid coordinates"))
        .withColumn("x_coordinate", F.lit("not-a-number"))
        .withColumn("y_cordinate", F.lit("not-a-number"))
    )

    row = to_canonical(malformed).first()
    assert row is not None
    assert row.occurred_at is None
    assert row.reported_at is None
    assert row.updated_at is None
    assert row.latitude is None
    assert row.longitude is None
    assert row.source_x_coordinate is None
    assert row.source_y_coordinate is None


def test_dallas_supported_timestamp_precisions_parse(
    dallas_bronze: DataFrame,
) -> None:
    source = dallas_bronze.filter(
        F.col("service_number_id") == "095681-2022-01"
    )
    variants = (
        source
        .withColumn(
            "date_of_report",
            F.lit("2022-05-29 05:11:10.123456"),
        )
        .withColumn(
            "update_date",
            F.lit("2022-09-29 11:30:30"),
        )
    )

    row = to_canonical(variants).first()
    assert row is not None
    assert row.reported_at == datetime(
        2022,
        5,
        29,
        5,
        11,
        10,
        123456,
    )
    assert row.updated_at == datetime(2022, 9, 29, 11, 30, 30)
