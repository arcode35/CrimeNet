from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from crimenet.canonical.common import (
    LOCAL_TIMEZONES,
    canonical_coordinate_expressions,
    clean_column,
    clean_string,
    existing_column,
    float_value,
    integer_or_default,
    namespaced_record_id,
    observed_time_precision,
    parse_epoch_milliseconds,
    with_coordinate_metadata,
)


CITY = "fort_worth"


def build_canonical(
    bronze: DataFrame,
) -> DataFrame:
    occurred_at_start = (
        parse_epoch_milliseconds(
            F.col("from_date")
        )
    )

    coordinates = (
        canonical_coordinate_expressions(
            city=CITY,
            latitude_expression=(
                F.col("latitude")
            ),
            longitude_expression=(
                F.col("longitude")
            ),
        )
    )

    prepared = (
        bronze
        .withColumn(
            "_occurred_at_start",
            occurred_at_start,
        )
        .withColumn(
            "_latitude",
            coordinates["latitude"],
        )
        .withColumn(
            "_longitude",
            coordinates["longitude"],
        )
        .withColumn(
            "_coordinate_status",
            coordinates["status"],
        )
        .withColumn(
            "_within_city",
            coordinates["within_city"],
        )
    )

    return prepared.select(
        namespaced_record_id(
            city=CITY,
            local_key=F.col(
                "case_no_offense"
            ),
        ).alias(
            "canonical_record_id"
        ),

        clean_string(
            existing_column(
                prepared,
                "source_system",
            )
        ).alias("source_system"),

        F.lit(
            "Fort Worth Police crime offenses"
        ).alias("source_dataset"),

        F.lit(
            "offense_records"
        ).alias("source_dataset_kind"),

        clean_column(
            "case_no_offense"
        ).alias("source_record_id"),

        F.lit(False).alias(
            "source_record_id_is_generated"
        ),

        clean_column(
            "case_no"
        ).alias("source_incident_id"),

        clean_string(
            existing_column(
                prepared,
                "source_row_hash",
            )
        ).alias("source_row_hash"),

        F.col(
            "_occurred_at_start"
        ).alias("occurred_at_start"),

        parse_epoch_milliseconds(
            F.col("reported_date")
        ).alias("reported_at"),

        parse_epoch_milliseconds(
            F.col("lastupdated")
        ).alias("updated_at"),

        F.lit(
            LOCAL_TIMEZONES[CITY]
        ).alias("source_timezone"),

        observed_time_precision(
            F.col(
                "_occurred_at_start"
            )
        ).alias(
            "occurrence_time_precision"
        ),

        F.when(
            F.col(
                "_occurred_at_start"
            ).isNull(),
            F.lit("parse_failed"),
        )
        .otherwise(F.lit("exact"))
        .alias(
            "occurrence_time_status"
        ),

        F.lit(False).alias(
            "is_occurrence_interval"
        ),

        clean_column(
            "offense"
        ).alias("offense_name"),

        clean_column(
            "offense_desc"
        ).alias(
            "offense_description"
        ),

        clean_column(
            "attempt_complete"
        ).alias("attempt_status"),

        integer_or_default(
            F.col("inttotal"),
            default=1,
        ).alias("offense_count"),

        F.col("_latitude").alias(
            "latitude"
        ),

        F.col("_longitude").alias(
            "longitude"
        ),

        float_value(
            existing_column(
                prepared,
                "alternate_latitude",
            )
        ).alias(
            "alternate_latitude"
        ),

        float_value(
            existing_column(
                prepared,
                "alternate_longitude",
            )
        ).alias(
            "alternate_longitude"
        ),

        float_value(
            F.col("x_coordinate")
        ).alias(
            "source_x_coordinate"
        ),

        float_value(
            F.col("y_coordinate")
        ).alias(
            "source_y_coordinate"
        ),

        F.col(
            "_coordinate_status"
        ).alias(
            "coordinate_validation_status"
        ),

        F.col("_within_city").alias(
            "is_coordinate_within_city_bounds"
        ),

        clean_column(
            "address"
        ).alias("address"),

        clean_column(
            "block_address"
        ).alias("block_address"),

        clean_column(
            "locationtypedescription"
        ).alias("premise_type"),

        clean_column(
            "location_1"
        ).alias(
            "location_description"
        ),

        clean_column("city").alias("city"),
        clean_column("state").alias("state"),

        clean_column(
            "division"
        ).alias("district"),

        clean_column("beat").alias("beat"),

        clean_column(
            "councildistrict"
        ).alias("council_district"),

        existing_column(
            prepared,
            "_source_file",
            "source_file",
        ).cast("string").alias(
            "source_file"
        ),

        existing_column(
            prepared,
            "_ingested_at",
            "ingested_at",
            "ingested_at_utc",
        ).cast("timestamp").alias(
            "ingested_at_utc"
        ),

        *with_coordinate_metadata(
            latitude=F.col("_latitude"),
            method="source_wgs84",
            precision="source_point",
            source_crs=None,
        ),
    )