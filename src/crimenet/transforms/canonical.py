"""Composition of all source-specific canonical transformations."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
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
    """Create the unified canonical crime DataFrame."""
    return (
        transform_dallas(dallas_bronze)
        .unionByName(
            transform_houston(houston_bronze)
        )
        .unionByName(
            transform_fort_worth(
                fort_worth_bronze
            )
        )
    )
    assert_silver_contract(silver_dataframe)

def add_crime_offense_id(
    dataframe: DataFrame,
) -> DataFrame:
    city = F.lower(F.trim(F.col("source_city")))

    source_key = (
        F.when(
            (city == "dallas")
            & F.col("source_incident_id").isNotNull()
            & F.col("source_record_id").isNotNull()
            & F.col("offense_code").isNotNull(),
            F.concat_ws(
                "||",
                F.lit("dallas"),
                F.trim("source_incident_id"),
                F.trim("source_record_id"),
                F.trim("offense_code"),
            ),
        )
        .when(
            (city == "fort_worth")
            & F.col("source_record_id").isNotNull(),
            F.concat_ws(
                "||",
                F.lit("fort_worth"),
                F.trim("source_record_id"),
            ),
        )
        .when(
            (city == "houston")
            & F.col("source_row_hash").isNotNull(),
            F.concat_ws(
                "||",
                F.lit("houston"),
                F.col("source_row_hash"),
            ),
        )
        .when(
            F.col("source_row_hash").isNotNull(),
            F.concat_ws(
                "||",
                city,
                F.col("source_row_hash"),
            ),
        )
    )

    return dataframe.withColumn(
        "crime_offense_id",
        F.sha2(source_key, 256),
    )