"""Dallas Bronze-to-canonical-Silver transformation."""

from __future__ import annotations

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from crimenet.contracts.silver import SILVER_COLUMNS, assert_silver_contract
from crimenet.transforms.common import null_double, try_cast

_COORDINATE_PATTERN = (
    r"\(\s*"
    r"(-?\d+(?:\.\d+)?)"
    r"\s*,\s*"
    r"(-?\d+(?:\.\d+)?)"
    r"\s*\)"
)


def _source_timestamp(column_name: str) -> Column:
    """Parse Dallas timestamps with optional fractional seconds."""
    return F.coalesce(
        F.expr(
            f"try_to_timestamp(`{column_name}`, "
            "'yyyy-MM-dd HH:mm:ss.SSSSSSS')"
        ),
        F.expr(
            f"try_to_timestamp(`{column_name}`, "
            "'yyyy-MM-dd HH:mm:ss.SSSSSS')"
        ),
        F.expr(
            f"try_to_timestamp(`{column_name}`, "
            "'yyyy-MM-dd HH:mm:ss')"
        ),
    )


def _occurrence_timestamp() -> Column:
    return F.expr(
        "try_to_timestamp("
        "concat(substring(`date1_of_occurrence`, 1, 10), "
        "' ', `time1_of_occurrence`), "
        "'yyyy-MM-dd HH:mm'"
        ")"
    )


def to_canonical(dataframe: DataFrame) -> DataFrame:
    latitude_text = F.regexp_extract(
        F.col("location1"),
        _COORDINATE_PATTERN,
        1,
    )
    longitude_text = F.regexp_extract(
        F.col("location1"),
        _COORDINATE_PATTERN,
        2,
    )

    result = dataframe.select(
        F.lit("dallas").alias("source_city"),
        F.col("service_number_id")
        .cast("string")
        .alias("source_record_id"),
        F.col("incident_number_w_year")
        .cast("string")
        .alias("source_incident_id"),
        F.col("nibrs_code").cast("string").alias("offense_code"),
        F.coalesce(
            F.col("nibrs_crime"),
            F.col("ucr_offense_name"),
            F.col("type_of_incident"),
        )
        .cast("string")
        .alias("offense_name"),
        F.coalesce(
            F.col("ucr_offense_description"),
            F.col("type_of_incident"),
        )
        .cast("string")
        .alias("offense_description"),
        _occurrence_timestamp().alias("occurred_at"),
        _source_timestamp("date_of_report").alias("reported_at"),
        _source_timestamp("update_date").alias("updated_at"),
        F.lit(1).cast("long").alias("offense_count"),
        F.col("incident_address").cast("string").alias("address"),
        F.col("city").cast("string").alias("city"),
        F.col("state").cast("string").alias("state"),
        F.col("zip_code").cast("string").alias("postal_code"),
        F.col("beat").cast("string").alias("beat"),
        F.col("type_location").cast("string").alias("premise_type"),
        F.when(
            latitude_text != "",
            latitude_text.cast("double"),
        ).alias("latitude"),
        F.when(
            longitude_text != "",
            longitude_text.cast("double"),
        ).alias("longitude"),
        null_double().alias("alternate_latitude"),
        null_double().alias("alternate_longitude"),
        try_cast("x_coordinate", "double").alias("source_x_coordinate"),
        try_cast("y_cordinate", "double").alias("source_y_coordinate"),
        F.col("source_file").cast("string").alias("source_file"),
        F.col("source_row_hash").cast("string").alias("source_row_hash"),
    ).select(*SILVER_COLUMNS)

    assert_silver_contract(result)
    return result
