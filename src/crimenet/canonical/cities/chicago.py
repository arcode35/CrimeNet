from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from crimenet.canonical.common import (
    ISO_LOCAL_PATTERNS,
    LOCAL_TIMEZONES,
    add_fast_local_datetime,
    canonical_coordinate_expressions,
    clean_column,
    existing_column,
    float_value,
    namespaced_record_id,
    nullable_boolean,
    observed_time_precision,
    parse_exact_utc_download_timestamp,
    with_coordinate_metadata,
)


CITY = "chicago"


def build_canonical(
    bronze: DataFrame,
) -> DataFrame:
    timezone = LOCAL_TIMEZONES[CITY]

    source = add_fast_local_datetime(
        bronze,
        source_column="date",
        output_prefix="occurred_start",
        timezone=timezone,
        patterns=ISO_LOCAL_PATTERNS,
    )

    source = add_fast_local_datetime(
        source,
        source_column="updated_on",
        output_prefix="updated",
        timezone=timezone,
        patterns=ISO_LOCAL_PATTERNS,
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
        source
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
            local_key=F.col("id"),
        ).alias(
            "canonical_record_id"
        ),

        F.lit(
            "chicago_police_department"
        ).alias("source_system"),

        F.lit(
            "Chicago crimes"
        ).alias("source_dataset"),

        clean_column(
            "source_dataset_id"
        ).alias("source_dataset_id"),

        F.lit(
            "offense_records"
        ).alias("source_dataset_kind"),

        clean_column("id").alias(
            "source_record_id"
        ),

        F.lit(False).alias(
            "source_record_id_is_generated"
        ),

        clean_column(
            "case_number"
        ).alias("source_incident_id"),

        F.col(
            "_occurred_start_utc"
        ).alias("occurred_at_start"),

        F.col("_updated_utc").alias(
            "updated_at"
        ),

        F.lit(timezone).alias(
            "source_timezone"
        ),

        observed_time_precision(
            F.col(
                "_occurred_start_utc"
            )
        ).alias(
            "occurrence_time_precision"
        ),

        F.col(
            "_occurred_start_status"
        ).alias(
            "occurrence_time_status"
        ),

        F.lit(False).alias(
            "is_occurrence_interval"
        ),

        clean_column("iucr").alias(
            "offense_code"
        ),

        clean_column(
            "primary_type"
        ).alias("offense_name"),

        clean_column(
            "description"
        ).alias(
            "offense_description"
        ),

        clean_column(
            "primary_type"
        ).alias("offense_category"),

        clean_column(
            "fbi_code"
        ).alias("offense_group"),

        F.lit(1).cast("bigint").alias(
            "offense_count"
        ),

        nullable_boolean(
            F.col("arrest")
        ).alias("is_arrest"),

        nullable_boolean(
            F.col("domestic")
        ).alias("is_domestic"),

        F.col("_latitude").alias(
            "latitude"
        ),

        F.col("_longitude").alias(
            "longitude"
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

        clean_column("block").alias(
            "block_address"
        ),

        clean_column(
            "location_description"
        ).alias("premise_type"),

        F.lit("Chicago").alias("city"),
        F.lit("IL").alias("state"),

        clean_column(
            "community_area"
        ).alias("neighborhood"),

        clean_column(
            "district"
        ).alias("district"),

        clean_column("beat").alias("beat"),
        clean_column("ward").alias("ward"),

        existing_column(
            prepared,
            "_source_file",
            "source_file",
        ).cast("string").alias(
            "source_file"
        ),

        parse_exact_utc_download_timestamp(
            F.col("downloaded_at_utc")
        ).alias("downloaded_at_utc"),

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
            precision="shifted_block",
            source_crs="EPSG:3435",
        ),
    )