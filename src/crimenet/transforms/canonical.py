"""Composition of all source-specific canonical transformations."""

from __future__ import annotations

from pyspark.sql import DataFrame

from crimenet.contracts.silver import assert_silver_contract
from crimenet.transforms.dallas import to_canonical as transform_dallas
from crimenet.transforms.fort_worth import (
    to_canonical as transform_fort_worth,
)
from crimenet.transforms.houston import to_canonical as transform_houston


def build_crime_offenses(
    dallas_bronze: DataFrame,
    houston_bronze: DataFrame,
    fort_worth_bronze: DataFrame,
) -> DataFrame:
    """Create the unified offense-grain Silver DataFrame."""
    result = (
        transform_dallas(dallas_bronze)
        .unionByName(transform_houston(houston_bronze))
        .unionByName(transform_fort_worth(fort_worth_bronze))
    )

    assert_silver_contract(result)
    return result
