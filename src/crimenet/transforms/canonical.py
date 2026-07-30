"""Composition of all source-specific canonical transformations."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from crimenet.contracts.silver import assert_silver_contract
from crimenet.transforms.dallas import to_canonical as transform_dallas
from crimenet.transforms.fort_worth import (
    to_canonical as transform_fort_worth,
)
from crimenet.transforms.houston import to_canonical as transform_houston

CRIME_DEDUPLICATION_REQUIRED_COLUMNS = (
    "crime_offense_id",
    "updated_at",
    "reported_at",
    "occurred_at",
    "source_row_hash",
    "source_file",
)


def build_crime_offenses(
    dallas_bronze: DataFrame,
    houston_bronze: DataFrame,
    fort_worth_bronze: DataFrame,
) -> DataFrame:
    """Create the unified canonical crime DataFrame."""
    silver_dataframe = (
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
    return silver_dataframe


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


def _require_columns(
    dataframe: DataFrame,
    required_columns: tuple[str, ...],
) -> None:
    missing_columns = sorted(
        set(required_columns) - set(dataframe.columns)
    )
    if missing_columns:
        raise ValueError(
            "Crime offense DataFrame is missing required columns: "
            + ", ".join(missing_columns)
        )


def assert_crime_offense_ids(dataframe: DataFrame) -> None:
    """Reject null offense IDs before any keyed deduplication occurs."""
    _require_columns(dataframe, ("crime_offense_id",))
    missing_ids = dataframe.filter(
        F.col("crime_offense_id").isNull()
    )

    if missing_ids.isEmpty():
        return

    example_columns = [
        column_name
        for column_name in (
            "source_city",
            "source_incident_id",
            "source_record_id",
            "offense_code",
            "source_row_hash",
        )
        if column_name in dataframe.columns
    ]
    examples = [
        row.asDict()
        for row in (
            missing_ids
            .select(*example_columns)
            .limit(20)
            .collect()
        )
    ]
    raise ValueError(
        "Some canonical crime records have no crime_offense_id. "
        f"Examples: {examples}"
    )


def deduplicate_crime_offenses(
    dataframe: DataFrame,
) -> DataFrame:
    """Select one deterministic survivor for each crime offense ID."""
    _require_columns(
        dataframe,
        CRIME_DEDUPLICATION_REQUIRED_COLUMNS,
    )
    assert_crime_offense_ids(dataframe)

    payload_columns = sorted(
        column_name
        for column_name in dataframe.columns
        if column_name != "source_file"
    )
    ranked_dataframe = dataframe.withColumn(
        "_deduplication_payload_hash",
        F.sha2(
            F.to_json(
                F.struct(
                    *(
                        F.col(column_name)
                        for column_name in payload_columns
                    )
                ),
                options={"ignoreNullFields": "false"},
            ),
            256,
        ),
    )
    deduplication_window = (
        Window
        .partitionBy("crime_offense_id")
        .orderBy(
            F.col("updated_at").desc_nulls_last(),
            F.col("reported_at").desc_nulls_last(),
            F.col("occurred_at").desc_nulls_last(),
            F.col("source_row_hash").desc_nulls_last(),
            F.col(
                "_deduplication_payload_hash"
            ).desc_nulls_last(),
            F.col("source_file").desc_nulls_last(),
        )
    )

    return (
        ranked_dataframe
        .withColumn(
            "_deduplication_rank",
            F.row_number().over(deduplication_window),
        )
        .filter(F.col("_deduplication_rank") == 1)
        .drop(
            "_deduplication_rank",
            "_deduplication_payload_hash",
        )
    )
