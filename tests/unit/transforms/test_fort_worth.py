from __future__ import annotations

from datetime import datetime

import pytest
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from crimenet.contracts.silver import (
    SILVER_COLUMNS,
    assert_silver_contract,
)
from crimenet.transforms.fort_worth import to_canonical


def test_fort_worth_fixture_maps_to_exact_canonical_contract(
    fort_worth_canonical: DataFrame,
) -> None:
    assert fort_worth_canonical.count() == 163
    assert tuple(fort_worth_canonical.columns) == SILVER_COLUMNS
    assert_silver_contract(fort_worth_canonical)
    assert {
        row.source_city
        for row in (
            fort_worth_canonical
            .select("source_city")
            .distinct()
            .collect()
        )
    } == {"fort_worth"}


def test_fort_worth_representative_row_has_exact_mappings(
    fort_worth_canonical: DataFrame,
) -> None:
    row = fort_worth_canonical.filter(
        F.col("source_record_id") == "190085144-23C"
    ).first()

    assert row is not None
    assert row.source_incident_id == "190085144"
    assert row.offense_code == "23C"
    assert row.offense_name == "THEFT"
    assert row.offense_description == (
        "GC 085-07 Theft under $100 23C SHOPLIFTING 0000000"
    )
    assert row.occurred_at == datetime(2019, 9, 30, 21, 48, 55)
    assert row.reported_at == datetime(2019, 9, 30, 21, 48, 55)
    assert row.updated_at == datetime(2019, 10, 6, 14, 0, 41)
    assert row.offense_count == 1
    assert row.address == "4200 E LANCASTER AVE EB"
    assert row.city == "FORT WORTH"
    assert row.state == "TX"
    assert row.postal_code is None
    assert row.beat == "G14"
    assert row.premise_type == "07 CONVENIENCE STORE"
    assert row.latitude == pytest.approx(32.740739801747495)
    assert row.longitude == pytest.approx(-97.26158046529801)
    assert row.alternate_latitude == pytest.approx(
        32.740745512247045
    )
    assert row.alternate_longitude == pytest.approx(
        -97.26158810460899
    )
    assert row.source_x_coordinate == pytest.approx(2349259.76736274)
    assert row.source_y_coordinate == pytest.approx(6954670.9532799)
    assert row.source_file.endswith("fort_worth_fixture.json")
    assert row.source_row_hash == (
        "dfbc6f118173676038eab00357c6251fabd65fe119dfa72c33"
        "913199a79f204f"
    )


def test_fort_worth_null_coordinate_fixture_row_stays_null(
    fort_worth_canonical: DataFrame,
) -> None:
    row = fort_worth_canonical.filter(
        F.col("source_record_id") == "250008551-40A"
    ).first()

    assert row is not None
    assert row.address == "CAMP BOWIE BLVD & HORNE ST"
    assert row.state == "Te"
    assert row.latitude is None
    assert row.longitude is None
    assert row.alternate_latitude is None
    assert row.alternate_longitude is None
    assert row.source_x_coordinate is None
    assert row.source_y_coordinate is None


def test_fort_worth_negative_epoch_millis_are_invalid(
    fort_worth_canonical: DataFrame,
) -> None:
    row = fort_worth_canonical.filter(
        F.col("source_record_id") == "180019223-90Z"
    ).first()

    assert row is not None
    assert row.occurred_at is None
    assert row.reported_at == datetime(2018, 3, 2, 19, 3, 27)
    assert fort_worth_canonical.filter(
        F.col("occurred_at").isNull()
    ).count() == 14


def test_fort_worth_invalid_numeric_values_are_safe(
    fort_worth_bronze: DataFrame,
) -> None:
    malformed = (
        fort_worth_bronze
        .filter(F.col("case_no_offense") == "190085144-23C")
        .withColumn("from_date", F.lit("not-an-epoch"))
        .withColumn("latitude", F.lit("not-a-number"))
        .withColumn("longitude", F.lit("not-a-number"))
        .withColumn("alternate_latitude", F.lit("not-a-number"))
        .withColumn("alternate_longitude", F.lit("not-a-number"))
        .withColumn("x_coordinate", F.lit("not-a-number"))
        .withColumn("y_coordinate", F.lit("not-a-number"))
    )

    row = to_canonical(malformed).first()
    assert row is not None
    assert row.occurred_at is None
    assert row.latitude is None
    assert row.longitude is None
    assert row.alternate_latitude is None
    assert row.alternate_longitude is None
    assert row.source_x_coordinate is None
    assert row.source_y_coordinate is None


def test_fort_worth_identifier_and_address_fallbacks(
    fort_worth_bronze: DataFrame,
) -> None:
    fallback = (
        fort_worth_bronze
        .filter(F.col("case_no_offense") == "190085144-23C")
        .withColumn("case_no_offense", F.lit(None).cast("string"))
        .withColumn("objectid", F.lit(816850))
        .withColumn("address", F.lit(None).cast("string"))
        .withColumn("block_address", F.lit("4200 E LANCASTER"))
    )

    row = to_canonical(fallback).first()
    assert row is not None
    assert row.source_record_id == "816850"
    assert row.address == "4200 E LANCASTER"
