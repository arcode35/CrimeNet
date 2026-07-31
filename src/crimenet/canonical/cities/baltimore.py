from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from crimenet.canonical.common import (
    LOCAL_TIMEZONES,
    canonical_coordinate_expressions,
    clean_column,
    coalesce_strings,
    existing_column,
    float_value,
    integer_or_default,
    namespaced_record_id,
    nullable_boolean,
    observed_time_precision,
    parse_epoch_milliseconds,
    parse_exact_utc_download_timestamp,
    with_coordinate_metadata,
)


CITY = "baltimore"


def build_canonical(
    bronze: DataFrame,
) -> DataFrame:
    occurred_at_start = (
        parse_epoch_milliseconds(
            F.col("crimedatetime")
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
                "source_record_id"
            ),
        ).alias(
            "canonical_record_id"
        ),

        F.lit(
            "baltimore_police_department"
        ).alias("source_system"),

        clean_column(
            "source_dataset"
        ).alias("source_dataset"),

        clean_column(
            "source_dataset_id"
        ).alias("source_dataset_id"),

        clean_column(
            "source_dataset_kind"
        ).alias("source_dataset_kind"),

        clean_column(
            "source_record_id"
        ).alias("source_record_id"),

        F.lit(False).alias(
            "source_record_id_is_generated"
        ),

        clean_column(
            "ccnumber"
        ).alias("source_incident_id"),

        F.col(
            "_occurred_at_start"
        ).alias("occurred_at_start"),

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
            "crimecode"
        ).alias("offense_code"),

        clean_column(
            "description"
        ).alias("offense_name"),

        clean_column(
            "description"
        ).alias(
            "offense_description"
        ),

        integer_or_default(
            F.col("total_incidents"),
            default=1,
        ).alias("offense_count"),

        nullable_boolean(
            F.col("shooting")
        ).alias("is_shooting"),

        clean_column(
            "weapon"
        ).alias("weapon"),

        F.col("_latitude").alias(
            "latitude"
        ),

        F.col("_longitude").alias(
            "longitude"
        ),

        float_value(
            F.col("geometry_x")
        ).alias(
            "source_x_coordinate"
        ),

        float_value(
            F.col("geometry_y")
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
            "location"
        ).alias("address"),

        clean_column(
            "premisetype"
        ).alias("premise_type"),

        clean_column(
            "geolocation"
        ).alias(
            "location_description"
        ),

        F.lit("Baltimore").alias(
            "city"
        ),

        F.lit("MD").alias("state"),

        clean_column(
            "neighborhood"
        ).alias("neighborhood"),

        coalesce_strings(
            F.col("new_district"),
            F.col("old_district"),
        ).alias("district"),

        clean_column(
            "post"
        ).alias("reporting_area"),

        existing_column(
            prepared,
            "_source_file",
            "source_file",
        ).cast("string").alias(
            "source_file"
        ),

        clean_column(
            "source_url"
        ).alias("source_url"),

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
            precision="address_geocode",
            source_crs="EPSG:4326",
        ),
    )