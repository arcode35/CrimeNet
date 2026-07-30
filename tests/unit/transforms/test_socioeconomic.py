from __future__ import annotations

from typing import Any

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from crimenet.silver.socioeconomic import (
    SOCIOECONOMIC_KEYS,
    deduplicate_socioeconomic_records,
    transform_acs5_tracts,
)


def _logical_rows(dataframe: DataFrame) -> list[tuple[Any, ...]]:
    columns = [
        column_name
        for column_name in dataframe.columns
        if column_name
        not in {
            "bronze_ingested_at",
            "silver_processed_at",
        }
    ]

    return sorted(
        (
            tuple(
                row[column_name]
                for column_name in columns
            )
            for row in dataframe.select(
                *columns
            ).collect()
        ),
        key=repr,
    )


def test_socioeconomic_fixture_maps_exact_values(
    socioeconomic_bronze: DataFrame,
) -> None:
    transformed = transform_acs5_tracts(
        socioeconomic_bronze
    ).cache()

    try:
        assert transformed.count() == 117
        assert len(transformed.columns) == 45
        assert (
            transformed
            .select(*SOCIOECONOMIC_KEYS)
            .distinct()
            .count()
            == 116
        )

        row = (
            transformed
            .filter(
                (F.col("geoid") == "48085031812")
                & (F.col("acs_vintage") == 2021)
            )
            .first()
        )

        assert row is not None
        assert row["geography_name"] == (
            "Census Tract 318.12, Collin County, Texas"
        )
        assert row["state_fips"] == "48"
        assert row["county_fips"] == "085"
        assert row["tract_code"] == "031812"
        assert row["period_start_year"] == 2017
        assert row["period_end_year"] == 2021
        assert row["population"] == 1416
        assert row["population_moe"] == 194
        assert row["median_age"] == pytest.approx(79.6)
        assert row["median_household_income"] == pytest.approx(
            95257.0
        )
        assert row["poverty_rate"] == pytest.approx(
            0.013418079096045197
        )
        assert row["unemployment_rate"] == pytest.approx(0.0)
        assert row["vacancy_rate"] == pytest.approx(
            0.06912991656734208
        )
        assert row["renter_occupied_rate"] == pytest.approx(
            0.6837387964148528
        )
        assert row["no_vehicle_rate"] == pytest.approx(
            0.19846350832266324
        )
    finally:
        transformed.unpersist()


def test_socioeconomic_fixture_sentinels_become_null(
    socioeconomic_bronze: DataFrame,
) -> None:
    transformed = transform_acs5_tracts(
        socioeconomic_bronze
    )

    income_row = (
        transformed
        .filter(
            (F.col("geoid") == "48157673700")
            & (F.col("acs_vintage") == 2024)
        )
        .select(
            "median_household_income",
            "median_household_income_moe",
        )
        .first()
    )
    population_row = (
        transformed
        .filter(
            (F.col("geoid") == "48191950500")
            & (F.col("acs_vintage") == 2012)
        )
        .select(
            "population",
            "population_moe",
        )
        .first()
    )

    assert income_row is not None
    assert income_row["median_household_income"] is None
    assert income_row["median_household_income_moe"] is None
    assert population_row is not None
    assert population_row["population"] == 3337
    assert population_row["population_moe"] is None


def test_socioeconomic_numeric_casts_are_ansi_safe(
    spark: SparkSession,
    socioeconomic_bronze: DataFrame,
) -> None:
    previous_ansi = spark.conf.get(
        "spark.sql.ansi.enabled"
    )
    spark.conf.set("spark.sql.ansi.enabled", "true")

    malformed = socioeconomic_bronze.withColumn(
        "b01003_001e",
        F.when(
            (F.col("geoid") == "48085031812")
            & (F.col("acs_vintage") == 2021),
            F.lit("not-a-number"),
        ).otherwise(F.col("b01003_001e")),
    )

    try:
        row = (
            transform_acs5_tracts(malformed)
            .filter(
                (F.col("geoid") == "48085031812")
                & (F.col("acs_vintage") == 2021)
            )
            .select("population")
            .first()
        )
    finally:
        spark.conf.set(
            "spark.sql.ansi.enabled",
            previous_ansi,
        )

    assert row is not None
    assert row["population"] is None


def test_socioeconomic_deduplication_is_deterministic(
    socioeconomic_bronze: DataFrame,
) -> None:
    transformed = transform_acs5_tracts(
        socioeconomic_bronze
    )
    first = deduplicate_socioeconomic_records(
        transformed
    )
    repartitioned = deduplicate_socioeconomic_records(
        transformed.repartition(5)
    )

    assert first.count() == 116
    assert (
        first
        .groupBy(*SOCIOECONOMIC_KEYS)
        .count()
        .filter(F.col("count") > 1)
        .isEmpty()
    )
    assert _logical_rows(first) == _logical_rows(
        repartitioned
    )
