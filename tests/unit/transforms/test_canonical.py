from __future__ import annotations

import pytest
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from crimenet.contracts.silver import (
    SILVER_COLUMNS,
    assert_silver_contract,
)
from crimenet.transforms.canonical import build_crime_offenses
from crimenet.transforms.dallas import to_canonical as transform_dallas
from crimenet.transforms.fort_worth import (
    to_canonical as transform_fort_worth,
)
from crimenet.transforms.houston import to_canonical as transform_houston


def test_all_fixture_cities_union_by_name_under_silver_contract(
    canonical_crimes: DataFrame,
) -> None:
    assert canonical_crimes.count() == 439
    assert tuple(canonical_crimes.columns) == SILVER_COLUMNS
    assert_silver_contract(canonical_crimes)
    assert {
        row.source_city: row["count"]
        for row in (
            canonical_crimes
            .groupBy("source_city")
            .count()
            .collect()
        )
    } == {
        "dallas": 119,
        "houston": 157,
        "fort_worth": 163,
    }


def test_repeated_unified_transformation_is_logically_stable(
    dallas_bronze: DataFrame,
    houston_bronze: DataFrame,
    fort_worth_bronze: DataFrame,
) -> None:
    first = build_crime_offenses(
        dallas_bronze,
        houston_bronze,
        fort_worth_bronze,
    )
    second = build_crime_offenses(
        dallas_bronze.repartition(3),
        houston_bronze.repartition(2),
        fort_worth_bronze.repartition(4),
    )
    logical_columns = [
        column_name
        for column_name in SILVER_COLUMNS
        if column_name != "source_file"
    ]

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


def test_empty_fixture_shaped_inputs_produce_empty_canonical_output(
    dallas_bronze: DataFrame,
    houston_bronze: DataFrame,
    fort_worth_bronze: DataFrame,
) -> None:
    result = build_crime_offenses(
        dallas_bronze.limit(0),
        houston_bronze.limit(0),
        fort_worth_bronze.limit(0),
    )

    assert result.count() == 0
    assert_silver_contract(result)


def test_missing_required_bronze_column_fails_clearly(
    dallas_bronze: DataFrame,
) -> None:
    with pytest.raises(
        ValueError,
        match="service_number_id",
    ):
        transform_dallas(
            dallas_bronze.drop("service_number_id")
        )


def test_houston_missing_required_column_fails_clearly(
    houston_bronze: DataFrame,
) -> None:
    with pytest.raises(
        ValueError,
        match="Houston canonical input.*maplatitude",
    ):
        transform_houston(
            houston_bronze.drop("maplatitude")
        )


def test_fort_worth_missing_required_column_fails_clearly(
    fort_worth_bronze: DataFrame,
) -> None:
    with pytest.raises(
        ValueError,
        match="Fort Worth canonical input.*alternate_longitude",
    ):
        transform_fort_worth(
            fort_worth_bronze.drop("alternate_longitude")
        )


def test_unexpected_bronze_column_does_not_change_canonical_schema(
    dallas_bronze: DataFrame,
) -> None:
    result = transform_dallas(
        dallas_bronze.withColumn(
            "new_upstream_field",
            F.lit("new value"),
        )
    )

    assert tuple(result.columns) == SILVER_COLUMNS
    assert result.count() == 119
