from __future__ import annotations

import pytest
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from crimenet.silver.socioeconomic import (
    SOCIOECONOMIC_KEYS,
    deduplicate_socioeconomic_records,
    transform_acs5_tracts,
)

pytestmark = pytest.mark.integration

def test_socioeconomic_fixture_bronze_to_silver_is_idempotent(
    socioeconomic_bronze: DataFrame,
) -> None:
    assert socioeconomic_bronze.count() == 117
    assert (
        socioeconomic_bronze
        .filter(
            F.col("source_system")
            != "census_acs5"
        )
        .isEmpty()
    )
    assert (
        socioeconomic_bronze
        .filter(
            F.col("source_row_hash").isNull()
        )
        .isEmpty()
    )

    first = deduplicate_socioeconomic_records(
        transform_acs5_tracts(
            socioeconomic_bronze
        )
    ).cache()
    second = deduplicate_socioeconomic_records(
        transform_acs5_tracts(
            socioeconomic_bronze.repartition(4)
        )
    ).cache()

    try:
        assert first.count() == 116
        assert (
            first
            .groupBy(*SOCIOECONOMIC_KEYS)
            .count()
            .filter(F.col("count") > 1)
            .isEmpty()
        )

        first_keys = first.select(
            *SOCIOECONOMIC_KEYS
        )
        second_keys = second.select(
            *SOCIOECONOMIC_KEYS
        )

        assert first_keys.exceptAll(
            second_keys
        ).isEmpty()
        assert second_keys.exceptAll(
            first_keys
        ).isEmpty()

        duplicate_fixture_key = (
            first
            .filter(
                (F.col("geoid") == "48245000500")
                & (F.col("acs_vintage") == 2012)
            )
            .first()
        )
        assert duplicate_fixture_key is not None
        assert duplicate_fixture_key["population"] == 2104
        assert duplicate_fixture_key[
            "median_household_income"
        ] == 33482.0
    finally:
        first.unpersist()
        second.unpersist()

