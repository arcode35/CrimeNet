from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from crimenet.canonical.common import (
    ISO_LOCAL_PATTERNS,
    LOCAL_TIMEZONES,
    add_fast_local_datetime,
    add_fast_local_datetime_pair,
    canonical_coordinate_expressions,
    clean_column,
    coalesce_strings,
    existing_column,
    float_value,
    namespaced_record_id,
    observed_time_precision,
    occurrence_interval_minutes,
    parse_exact_utc_download_timestamp,
    stable_composite_sha256,
    valid_interval_expression,
    with_coordinate_metadata,
)


CITY = "new_york"


def build_canonical(
    bronze: DataFrame,
) -> DataFrame:
    timezone = LOCAL_TIMEZONES[CITY]

    source = add_fast_local_datetime_pair(
        bronze,
        date_column="cmplnt_fr_dt",
        time_column="cmplnt_fr_tm",
        output_prefix="occurred_start",
        timezone=timezone,
        time_width=8,
        patterns=(
            "yyyy-MM-dd HH:mm:ss",
        ),
    )

    source = add_fast_local_datetime_pair(
        source,
        date_column="cmplnt_to_dt",
        time_column="cmplnt_to_tm",
        output_prefix="occurred_end",
        timezone=timezone,
        time_width=8,
        patterns=(
            "yyyy-MM-dd HH:mm:ss",
        ),
    )

    source = add_fast_local_datetime(
        source,
        source_column="rpt_dt",
        output_prefix="reported",
        timezone=timezone,
        patterns=ISO_LOCAL_PATTERNS,
    )

    generated_record_id = (
        stable_composite_sha256(
            namespace=CITY,
            components=(
                F.col("cmplnt_num"),
                F.col("ky_cd"),
                F.col("pd_cd"),
                F.col("cmplnt_fr_dt"),
                F.col("cmplnt_fr_tm"),
            ),
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
        source
        .withColumn(
            "_generated_record_id",
            generated_record_id,
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
                "_generated_record_id"
            ),
        ).alias(
            "canonical_record_id"
        ),

        F.lit("nypd").alias(
            "source_system"
        ),

        F.lit(
            "NYPD complaint data"
        ).alias("source_dataset"),

        clean_column(
            "source_dataset_id"
        ).alias("source_dataset_id"),

        clean_column(
            "source_dataset_kind"
        ).alias("source_dataset_kind"),

        F.col(
            "_generated_record_id"
        ).alias("source_record_id"),

        F.lit(True).alias(
            "source_record_id_is_generated"
        ),

        clean_column(
            "cmplnt_num"
        ).alias("source_incident_id"),

        F.col(
            "_occurred_start_utc"
        ).alias("occurred_at_start"),

        F.col(
            "_occurred_end_utc"
        ).alias("occurred_at_end"),

        F.col("_reported_utc").alias(
            "reported_at"
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
            F.col("pd_cd"),
            F.col("ky_cd"),
        ).alias("offense_code"),

        clean_column(
            "ofns_desc"
        ).alias("offense_name"),

        clean_column(
            "pd_desc"
        ).alias(
            "offense_description"
        ),

        clean_column(
            "ofns_desc"
        ).alias("offense_category"),

        clean_column(
            "law_cat_cd"
        ).alias("offense_group"),

        clean_column(
            "crm_atpt_cptd_cd"
        ).alias("attempt_status"),

        F.lit(1).cast("bigint").alias(
            "offense_count"
        ),

        F.col("_latitude").alias(
            "latitude"
        ),

        F.col("_longitude").alias(
            "longitude"
        ),

        float_value(
            F.col("x_coord_cd")
        ).alias(
            "source_x_coordinate"
        ),

        float_value(
            F.col("y_coord_cd")
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
            "prem_typ_desc"
        ).alias("premise_type"),

        clean_column(
            "loc_of_occur_desc"
        ).alias(
            "location_description"
        ),

        F.lit("New York").alias("city"),
        F.lit("NY").alias("state"),

        clean_column(
            "boro_nm"
        ).alias("district"),

        clean_column(
            "addr_pct_cd"
        ).alias("precinct"),

        clean_column(
            "juris_desc"
        ).alias("jurisdiction"),

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
            precision="midblock",
            source_crs="EPSG:2263",
        ),
    )