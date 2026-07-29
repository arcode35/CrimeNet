"""Houston Bronze-to-canonical-Silver transformation."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from crimenet.contracts.silver import SILVER_COLUMNS, assert_silver_contract
from crimenet.transforms.common import (
    CRIME_TRANSFORMATION_VERSION,
    invalid_nonblank_cast,
    municipal_local_to_utc,
    normalized_identifier,
    null_double,
    null_timestamp,
    stable_business_identity,
    trimmed_address,
    try_cast,
)


def to_canonical(dataframe: DataFrame) -> DataFrame:
    incident_id = normalized_identifier("incident")
    offense_class = F.upper(normalized_identifier("nibrsclass"))
    offense_id = F.when(
        incident_id.isNotNull() & offense_class.isNotNull(),
        F.sha2(
            F.concat_ws(
                "||",
                incident_id,
                offense_class,
            ),
            256,
        ),
    )
    coordinate_parse_error = (
        invalid_nonblank_cast("maplatitude", "double")
        | invalid_nonblank_cast("maplongitude", "double")
    )
    source_validation_payload = F.when(
        coordinate_parse_error,
        F.to_json(
            F.struct(
                F.lit("INVALID_COORDINATE_TEXT").alias("reason"),
                F.col("maplatitude"),
                F.col("maplongitude"),
            )
        ),
    )

    result = dataframe.select(
        F.lit("houston").alias("source_system"),
        F.lit("houston").alias("source_city"),
        offense_id.alias("source_record_id"),
        incident_id.alias("source_incident_id"),
        offense_id.alias("source_offense_id"),
        stable_business_identity(
            source_system="houston",
            source_incident_id=incident_id,
            source_offense_id=offense_id,
        ).alias("business_identity"),
        offense_class.alias("offense_code"),
        F.col("nibrsdescription").cast("string").alias("offense_name"),
        F.col("nibrsdescription")
        .cast("string")
        .alias("offense_description"),
        municipal_local_to_utc(
            F.expr(
                "try_to_timestamp("
                "concat(`rmsoccurrencedate`, ' ', "
                "lpad(`rmsoccurrencehour`, 2, '0')), "
                "'M/d/yyyy HH'"
                ")"
            )
        ).alias("occurred_at"),
        null_timestamp().alias("reported_at"),
        null_timestamp().alias("updated_at"),
        try_cast("offensecount", "long").alias("offense_count"),
        trimmed_address(
            "streetno",
            "streetname",
            "streettype",
            "suffix",
        ).alias("address"),
        F.col("city").cast("string").alias("city"),
        F.lit("TX").cast("string").alias("state"),
        F.col("zipcode").cast("string").alias("postal_code"),
        F.col("beat").cast("string").alias("beat"),
        F.col("premise").cast("string").alias("premise_type"),
        try_cast("maplatitude", "double").alias("latitude"),
        try_cast("maplongitude", "double").alias("longitude"),
        null_double().alias("alternate_latitude"),
        null_double().alias("alternate_longitude"),
        null_double().alias("source_x_coordinate"),
        null_double().alias("source_y_coordinate"),
        F.col("source_file").cast("string").alias("source_file"),
        F.col("source_row_hash").cast("string").alias("source_row_hash"),
        F.col("source_contract_version")
        .cast("string")
        .alias("source_contract_version"),
        F.lit(CRIME_TRANSFORMATION_VERSION).alias(
            "transformation_version"
        ),
        F.col("ingested_at").cast("timestamp").alias("bronze_ingested_at"),
        F.coalesce(
            F.col("corrupt_record").cast("string"),
            source_validation_payload,
        ).alias("source_corrupt_record"),
    ).select(*SILVER_COLUMNS)

    assert_silver_contract(result)
    return result
