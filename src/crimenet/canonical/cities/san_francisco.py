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
    namespaced_record_id,
    observed_time_precision,
    parse_exact_utc_download_timestamp,
    with_coordinate_metadata,
)


CITY = "san_francisco"


def build_canonical(
    bronze: DataFrame,
) -> DataFrame:
    timezone = LOCAL_TIMEZONES[CITY]

    source = bronze

    for source_column, output_prefix in (
        (
            "incident_datetime",
            "occurred_start",
        ),
        (
            "report_datetime",
            "reported",
        ),
        (
            "data_as_of",
            "updated",
        ),
        (
            "data_loaded_at",
            "loaded",
        ),
    ):
        source = add_fast_local_datetime(
            source,
            source_column=source_column,
            output_prefix=output_prefix,
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
            local_key=F.col(
                "source_record_id"
            ),
        ).alias(
            "canonical_record_id"
        ),

        F.lit(
            "san_francisco_police_department"
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
            "incident_number"
        ).alias("source_incident_id"),

        F.col(
            "_occurred_start_utc"
        ).alias("occurred_at_start"),

        F.col("_reported_utc").alias(
            "reported_at"
        ),

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

        clean_column(
            "incident_code"
        ).alias("offense_code"),

        clean_column(
            "incident_category"
        ).alias("offense_name"),

        clean_column(
            "incident_description"
        ).alias(
            "offense_description"
        ),

        clean_column(
            "incident_category"
        ).alias("offense_category"),

        clean_column(
            "incident_subcategory"
        ).alias(
            "offense_subcategory"
        ),

        F.lit(1).cast("bigint").alias(
            "offense_count"
        ),

        clean_column(
            "resolution"
        ).alias("resolution"),

        F.col("_latitude").alias(
            "latitude"
        ),

        F.col("_longitude").alias(
            "longitude"
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
            "intersection"
        ).alias("intersection"),

        F.lit(
            "San Francisco"
        ).alias("city"),

        F.lit("CA").alias("state"),

        clean_column(
            "analysis_neighborhood"
        ).alias("neighborhood"),

        clean_column(
            "police_district"
        ).alias("district"),

        clean_column(
            "supervisor_district"
        ).alias("ward"),

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

        F.coalesce(
            F.col("_loaded_utc"),
            existing_column(
                prepared,
                "_ingested_at",
                "ingested_at",
                "ingested_at_utc",
                data_type="timestamp",
            ).cast("timestamp"),
        ).alias("ingested_at_utc"),

        *with_coordinate_metadata(
            latitude=F.col("_latitude"),
            method="source_wgs84",
            precision="address_geocode",
            source_crs="EPSG:4326",
        ),
    )