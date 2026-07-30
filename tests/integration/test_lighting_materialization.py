from __future__ import annotations

import pytest
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from crimenet.contracts.lighting import (
    LIGHTING_DEFINITION_VERSION,
    LIGHTING_KEYS,
)
from crimenet.silver.lighting import (
    extract_lighting_key_grain,
    select_missing_lighting_keys,
)

pytestmark = pytest.mark.integration


def test_lighting_full_and_incremental_key_sets_are_equivalent(
    canonical_crimes: DataFrame,
) -> None:
    fixture_crimes = canonical_crimes.withColumn(
        "weather_query_cell_id",
        F.when(
            F.col("source_city") == "dallas",
            F.lit(604164844395954175),
        )
        .when(
            F.col("source_city") == "houston",
            F.lit(604686071426449407),
        )
        .otherwise(
            F.lit(604164855133372415)
        )
        .cast("long"),
    )

    full_key_set = (
        extract_lighting_key_grain(
            fixture_crimes
        )
        .cache()
    )

    try:
        assert not full_key_set.isEmpty()
        assert (
            full_key_set
            .filter(
                F.col(
                    "lighting_definition_version"
                )
                != LIGHTING_DEFINITION_VERSION
            )
            .isEmpty()
        )

        existing = full_key_set.limit(1)
        missing = select_missing_lighting_keys(
            full_key_set,
            existing,
        )
        incremental_key_set = (
            existing
            .unionByName(missing)
            .dropDuplicates(
                list(LIGHTING_KEYS)
            )
            .cache()
        )

        try:
            assert select_missing_lighting_keys(
                full_key_set,
                incremental_key_set,
            ).isEmpty()
            assert (
                full_key_set
                .select(*LIGHTING_KEYS)
                .exceptAll(
                    incremental_key_set.select(
                        *LIGHTING_KEYS
                    )
                )
                .isEmpty()
            )
            assert (
                incremental_key_set
                .select(*LIGHTING_KEYS)
                .exceptAll(
                    full_key_set.select(
                        *LIGHTING_KEYS
                    )
                )
                .isEmpty()
            )
        finally:
            incremental_key_set.unpersist()
    finally:
        full_key_set.unpersist()
