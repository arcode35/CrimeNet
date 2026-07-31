from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from crimenet.canonical.common import (
    LOCAL_TIMEZONES,
    SQL_LOCAL_PATTERNS,
    add_fast_local_datetime,
    add_fast_local_datetime_pair,
    canonical_coordinate_expressions,
    clean_column,
    clean_string,
    coalesce_strings,
    existing_column,
    float_value,
    namespaced_record_id,
    nullable_boolean,
    observed_time_precision,
    occurrence_interval_minutes,
    stable_composite_sha256,
    valid_interval_expression,
    with_coordinate_metadata,
)


CITY = "dallas"


def build_canonical(
    bronze: DataFrame,
) -> DataFrame:
    timezone = LOCAL_TIMEZONES[CITY]

    source = add_fast_local_datetime_pair(
        bronze,
        date_column=(
            "date1_of_occurrence"
        ),
        time_column=(
            "time1_of_occurrence"
        ),
        output_prefix="occurred_start",
        timezone=timezone,
        time_width=5,
        patterns=(
            "yyyy-MM-dd HH:mm",
            "yyyy-MM-dd HH:mm:ss",
        ),
    )

    source = add_fast_local_datetime_pair(
        source,
        date_column=(
            "date2_of_occurrence"
        ),
        time_column=(
            "time2_of_occurrence"
        ),
        output_prefix="occurred_end",
        timezone=timezone,
        time_width=5,
        patterns=(
            "yyyy-MM-dd HH:mm",
            "yyyy-MM-dd HH:mm:ss",
        ),
    )

    source = add_fast_local_datetime(
        source,
        source_column="date_of_report",
        output_prefix="reported",
        timezone=timezone,
        patterns=SQL_LOCAL_PATTERNS,
    )

    source = add_fast_local_datetime(
        source,
        source_column="update_date",
        output_prefix="updated",
        timezone=timezone,
        patterns=SQL_LOCAL_PATTERNS,
    )

# Dallas publishes projected coordinates in:
# EPSG:2276 — NAD83 / Texas North Central (ftUS).
#
# Transform them to canonical WGS84 coordinates before
# applying the common coordinate validation logic.

    source = (
        source
        .withColumn(
            "_source_x",
            float_value(
                F.col("x_coordinate")
            ),
        )
        .withColumn(
            "_source_y",
            float_value(
                F.col("y_cordinate")
            ),
        )
        .withColumn(
            "_projected_point",
            F.expr(
                "st_point("
                "_source_x, "
                "_source_y, "
                "2276"
                ")"
            ),
        )
        .withColumn(
            "_wgs84_point",
            F.expr(
                "st_transform("
                "_projected_point, "
                "4326"
                ")"
            ),
        )
        .withColumn(
            "_transformed_longitude",
            F.expr(
                "st_x(_wgs84_point)"
            ),
        )
        .withColumn(
            "_transformed_latitude",
            F.expr(
                "st_y(_wgs84_point)"
            ),
        )
    )

    coordinates = (
        canonical_coordinate_expressions(
            city=CITY,
            latitude_expression=(
                F.col(
                    "_transformed_latitude"
                )
            ),
            longitude_expression=(
                F.col(
                    "_transformed_longitude"
                )
            ),
        )
    )

    generated_row_hash = (
        stable_composite_sha256(
            namespace=(
                "dallas:source_row:v1"
            ),
            components=(
                F.col(
                    "service_number_id"
                ),
                F.col(
                    "incident_number_w_year"
                ),
                F.col(
                    "type_of_incident"
                ),
                F.col(
                    "date1_of_occurrence"
                ),
                F.col(
                    "time1_of_occurrence"
                ),
            ),
        )
    )

    source_row_hash = F.coalesce(
        clean_string(
            existing_column(
                source,
                "source_row_hash",
            )
        ),
        generated_row_hash,
    )

    prepared = (
        source
        .withColumn(
            "_source_row_hash",
            source_row_hash,
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

    provided_coordinate_status = (
        clean_string(
            existing_column(
                prepared,
                "coordinate_validation_status",
            )
        )
    )

    provided_within_city = (
        nullable_boolean(
            existing_column(
                prepared,
                "is_within_dallas_bounds",
                "is_coordinate_within_city_bounds",
            )
        )
    )

    return prepared.select(
        namespaced_record_id(
            city=CITY,
            local_key=F.col(
                "_source_row_hash"
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
            "Dallas Police crime offenses"
        ).alias("source_dataset"),

        F.lit(
            "offense_records"
        ).alias("source_dataset_kind"),

        clean_column(
            "service_number_id"
        ).alias("source_record_id"),

        F.lit(False).alias(
            "source_record_id_is_generated"
        ),

        coalesce_strings(
            F.col(
                "incident_number_w_year"
            ),
            F.col(
                "service_number_id"
            ),
        ).alias("source_incident_id"),

        F.col(
            "_source_row_hash"
        ).alias("source_row_hash"),

        F.col(
            "_occurred_start_utc"
        ).alias("occurred_at_start"),

        F.col(
            "_occurred_end_utc"
        ).alias("occurred_at_end"),

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

        valid_interval_expression(
            F.col(
                "_occurred_start_utc"
            ),
            F.col(
                "_occurred_end_utc"
            ),
        ).alias(
            "is_occurrence_interval"
        ),

        occurrence_interval_minutes(
            F.col(
                "_occurred_start_utc"
            ),
            F.col(
                "_occurred_end_utc"
            ),
        ).alias(
            "occurrence_interval_minutes"
        ),

        coalesce_strings(
            F.col("nibrs_code"),
            F.col("ucr_code"),
            F.col("rms_code"),
        ).alias("offense_code"),

        coalesce_strings(
            F.col("nibrs_crime"),
            F.col("ucr_offense_name"),
            F.col("type_of_incident"),
        ).alias("offense_name"),

        clean_column(
            "ucr_offense_description"
        ).alias(
            "offense_description"
        ),

        clean_column(
            "nibrs_crime_category"
        ).alias("offense_category"),

        clean_column(
            "offense_type"
        ).alias(
            "offense_subcategory"
        ),

        clean_column(
            "nibrs_group"
        ).alias("offense_group"),

        clean_column(
            "nibrs_crime_against"
        ).alias(
            "crime_against_category"
        ),

        F.lit(1).cast("bigint").alias(
            "offense_count"
        ),

        clean_column(
            "offense_status"
        ).alias("resolution"),

        nullable_boolean(
            F.col("family_offense")
        ).alias(
            "is_family_offense"
        ),

        nullable_boolean(
            F.col("hate_crime")
        ).alias("is_hate_crime"),

        nullable_boolean(
            F.col(
                "gang_related_offense"
            )
        ).alias("is_gang_related"),

        nullable_boolean(
            F.col(
                "drug_related_istevencident"
            )
        ).alias("is_drug_related"),

        clean_column(
            "weapon_used"
        ).alias("weapon"),

        F.col("_latitude").alias(
            "latitude"
        ),

        F.col("_longitude").alias(
            "longitude"
        ),

        F.col("_source_x").alias(
            "source_x_coordinate"
        ),

        F.col("_source_y").alias(
            "source_y_coordinate"
        ),

        F.coalesce(
            provided_coordinate_status,
            F.col(
                "_coordinate_status"
            ),
        ).alias(
            "coordinate_validation_status"
        ),

        float_value(
            existing_column(
                prepared,
                "coordinate_error_meters",
            )
        ).alias(
            "coordinate_error_meters"
        ),

        F.coalesce(
            provided_within_city,
            F.col("_within_city"),
        ).alias(
            "is_coordinate_within_city_bounds"
        ),

        clean_column(
            "incident_address"
        ).alias("address"),

        clean_column(
            "type_location"
        ).alias("premise_type"),

        clean_column(
            "location1"
        ).alias(
            "location_description"
        ),

        clean_column("city").alias("city"),
        clean_column("state").alias("state"),

        clean_column(
            "zip_code"
        ).alias("postal_code"),

        clean_column(
            "community"
        ).alias("neighborhood"),

        clean_column(
            "division"
        ).alias("district"),

        clean_column("beat").alias("beat"),

        clean_column(
            "sector"
        ).alias("sector"),

        clean_column(
            "reporting_area"
        ).alias("reporting_area"),

        clean_column(
            "council_district"
        ).alias("council_district"),

        clean_column(
            "watch"
        ).alias("shift"),

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
            method="projected_transform",
            precision=(
                "transformed_source_point"
            ),
            source_crs="EPSG:2276",
        ),
    )