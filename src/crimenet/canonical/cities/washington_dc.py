from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from crimenet.canonical.common import (
    LOCAL_TIMEZONES,
    canonical_coordinate_expressions,
    clean_column,
    existing_column,
    float_value,
    namespaced_record_id,
    observed_time_precision,
    occurrence_interval_minutes,
    parse_epoch_milliseconds,
    parse_exact_utc_download_timestamp,
    valid_interval_expression,
    with_coordinate_metadata,
)


CITY = "washington_dc"


def build_canonical(
    bronze: DataFrame,
) -> DataFrame:
    occurred_at_start = (
        parse_epoch_milliseconds(
            F.col("start_date")
        )
    )

    occurred_at_end = (
        parse_epoch_milliseconds(
            F.col("end_date")
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
            "_occurred_at_end",
            occurred_at_end,
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
            "metropolitan_police_department_dc"
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

        F.lit(True).alias(
            "source_record_id_is_generated"
        ),

        clean_column(
            "ccn"
        ).alias("source_incident_id"),

        F.col(
            "_occurred_at_start"
        ).alias("occurred_at_start"),

        F.col(
            "_occurred_at_end"
        ).alias("occurred_at_end"),

        parse_epoch_milliseconds(
            F.col("report_dat")
        ).alias("reported_at"),

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

        valid_interval_expression(
            F.col(
                "_occurred_at_start"
            ),
            F.col(
                "_occurred_at_end"
            ),
        ).alias(
            "is_occurrence_interval"
        ),

        occurrence_interval_minutes(
            F.col(
                "_occurred_at_start"
            ),
            F.col(
                "_occurred_at_end"
            ),
        ).alias(
            "occurrence_interval_minutes"
        ),

        clean_column(
            "offense"
        ).alias("offense_name"),

        clean_column(
            "offense"
        ).alias("offense_category"),

        F.lit(1).cast("bigint").alias(
            "offense_count"
        ),

        clean_column(
            "method"
        ).alias("weapon"),

        F.col("_latitude").alias(
            "latitude"
        ),

        F.col("_longitude").alias(
            "longitude"
        ),

        float_value(
            F.col("xblock")
        ).alias(
            "source_x_coordinate"
        ),

        float_value(
            F.col("yblock")
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
            "block"
        ).alias("block_address"),

        F.lit("Washington").alias(
            "city"
        ),

        F.lit("DC").alias("state"),

        clean_column(
            "neighborhood_cluster"
        ).alias("neighborhood"),

        clean_column(
            "district"
        ).alias("district"),

        clean_column(
            "psa"
        ).alias("precinct"),

        clean_column("ward").alias("ward"),

        clean_column(
            "census_tract"
        ).alias("census_tract"),

        clean_column(
            "block_group"
        ).alias("census_block"),

        clean_column(
            "shift"
        ).alias("shift"),

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
            precision="block",
            source_crs=None,
        ),
    )