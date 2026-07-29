"""Auditable and idempotent quarantine records."""

from __future__ import annotations

from uuid import uuid4

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from crimenet.config.validation import (
    validate_identifier,
    validate_qualified_table_name,
)
from crimenet.utils.promotion import quote_table_name

SUPPORTED_CRIME_SOURCES = ("dallas", "houston", "fort_worth")

CRIME_REASON_MESSAGES = {
    "CORRUPT_SOURCE_RECORD": (
        "The source parser or typed-field normalization captured a malformed "
        "record."
    ),
    "MISSING_SOURCE_SYSTEM": "The source system is required.",
    "UNSUPPORTED_SOURCE_SYSTEM": "The source system is not supported.",
    "MISSING_SOURCE_INCIDENT_ID": "The source incident identifier is required.",
    "MISSING_SOURCE_OFFENSE_ID": "The source offense identifier is required.",
    "MISSING_BUSINESS_IDENTITY": "The canonical business identity is required.",
    "MISSING_SOURCE_ROW_HASH": "The stable source-row hash is required.",
    "MISSING_OCCURRED_AT": "The occurrence timestamp could not be parsed.",
    "IMPLAUSIBLE_OCCURRED_AT": (
        "The occurrence timestamp is outside the supported range."
    ),
    "INCOMPLETE_COORDINATES": (
        "Latitude and longitude must either both be present or both be null."
    ),
    "INVALID_COORDINATES": "Latitude or longitude is outside its valid range.",
}


def crime_quarantine_reason_codes(dataframe: DataFrame) -> DataFrame:
    """Annotate canonical crime rows with zero or more stable reason codes."""
    latitude = F.col("latitude")
    longitude = F.col("longitude")
    timestamp = F.col("occurred_at")

    reasons = F.array_compact(
        F.array(
            F.when(
                F.col("source_corrupt_record").isNotNull(),
                F.lit("CORRUPT_SOURCE_RECORD"),
            ),
            F.when(
                F.col("source_system").isNull(),
                F.lit("MISSING_SOURCE_SYSTEM"),
            ),
            F.when(
                F.col("source_system").isNotNull()
                & ~F.col("source_system").isin(*SUPPORTED_CRIME_SOURCES),
                F.lit("UNSUPPORTED_SOURCE_SYSTEM"),
            ),
            F.when(
                F.col("source_incident_id").isNull()
                | (F.trim("source_incident_id") == ""),
                F.lit("MISSING_SOURCE_INCIDENT_ID"),
            ),
            F.when(
                F.col("source_offense_id").isNull()
                | (F.trim("source_offense_id") == ""),
                F.lit("MISSING_SOURCE_OFFENSE_ID"),
            ),
            F.when(
                F.col("business_identity").isNull()
                | (F.trim("business_identity") == ""),
                F.lit("MISSING_BUSINESS_IDENTITY"),
            ),
            F.when(
                F.col("source_row_hash").isNull()
                | (F.trim("source_row_hash") == ""),
                F.lit("MISSING_SOURCE_ROW_HASH"),
            ),
            F.when(timestamp.isNull(), F.lit("MISSING_OCCURRED_AT")),
            F.when(
                timestamp.isNotNull()
                & (
                    (timestamp < F.lit("1900-01-01").cast("timestamp"))
                    | (
                        timestamp
                        > F.current_timestamp() + F.expr("INTERVAL 1 DAY")
                    )
                ),
                F.lit("IMPLAUSIBLE_OCCURRED_AT"),
            ),
            F.when(
                latitude.isNull() != longitude.isNull(),
                F.lit("INCOMPLETE_COORDINATES"),
            ),
            F.when(
                latitude.isNotNull()
                & longitude.isNotNull()
                & (
                    F.isnan(latitude)
                    | F.isnan(longitude)
                    | ~latitude.between(-90.0, 90.0)
                    | ~longitude.between(-180.0, 180.0)
                ),
                F.lit("INVALID_COORDINATES"),
            ),
        )
    )
    return dataframe.withColumn("_quarantine_reason_codes", reasons)


def split_crime_quarantine(
    dataframe: DataFrame,
    *,
    pipeline_run_id: str,
) -> tuple[DataFrame, DataFrame]:
    """Return valid canonical rows and one quarantine row per reason."""
    annotated = crime_quarantine_reason_codes(dataframe)
    valid = annotated.filter(
        F.size("_quarantine_reason_codes") == 0
    ).drop("_quarantine_reason_codes")

    reason_map_items = [
        item
        for code, message in CRIME_REASON_MESSAGES.items()
        for item in (F.lit(code), F.lit(message))
    ]
    messages = F.create_map(*reason_map_items)

    rejected = (
        annotated.filter(F.size("_quarantine_reason_codes") > 0)
        .withColumn(
            "quarantine_reason_code",
            F.explode("_quarantine_reason_codes"),
        )
        .withColumn(
            "quarantine_reason",
            messages[F.col("quarantine_reason_code")],
        )
        .withColumn("pipeline_run_id", F.lit(pipeline_run_id))
        .withColumn("quarantined_at", F.current_timestamp())
        .withColumn(
            "raw_payload",
            F.to_json(
                F.struct(
                    *[
                        F.col(name)
                        for name in dataframe.columns
                        if name not in {"source_file"}
                    ]
                ),
                options={"ignoreNullFields": "false"},
            ),
        )
        .withColumn(
            "validation_fields",
            F.to_json(
                F.struct(
                    "source_incident_id",
                    "source_offense_id",
                    "occurred_at",
                    "latitude",
                    "longitude",
                ),
                options={"ignoreNullFields": "false"},
            ),
        )
        .withColumn(
            "quarantine_id",
            F.sha2(
                F.concat_ws(
                    "||",
                    F.coalesce(F.col("source_system"), F.lit("unknown")),
                    F.coalesce(
                        F.when(
                            F.length(F.trim(F.col("source_row_hash"))) > 0,
                            F.col("source_row_hash"),
                        ),
                        F.sha2(F.col("raw_payload"), 256),
                    ),
                    F.col("quarantine_reason_code"),
                ),
                256,
            ),
        )
        .select(
            "quarantine_id",
            "source_system",
            "source_file",
            "source_row_hash",
            "raw_payload",
            "quarantine_reason_code",
            "quarantine_reason",
            "pipeline_run_id",
            "quarantined_at",
            "validation_fields",
        )
    )
    return valid, rejected


def merge_quarantine(
    spark: SparkSession,
    *,
    quarantine: DataFrame,
    target_table: str,
) -> None:
    """
    Persist stable reject entities plus one idempotent observation per run.

    The entity table does not duplicate replayed bad input. Its companion
    ``<table>_observations`` table preserves that the same reject was seen in
    later runs without producing duplicate observations on a task retry.
    """
    validate_qualified_table_name(target_table)
    if quarantine.isEmpty():
        return
    catalog, schema, table = target_table.split(".")
    observations_table = f"{catalog}.{schema}.{table}_observations"
    source = (
        quarantine.dropDuplicates(["quarantine_id"])
        .withColumnRenamed("pipeline_run_id", "first_seen_pipeline_run_id")
        .withColumnRenamed("quarantined_at", "first_quarantined_at")
    )
    if not spark.catalog.tableExists(target_table):
        source.limit(0).write.format("delta").saveAsTable(target_table)

    _merge_insert_only(
        spark,
        source=source,
        target_table=target_table,
        key_column="quarantine_id",
    )

    observations = (
        quarantine.select(
            "quarantine_id",
            "pipeline_run_id",
            F.col("quarantined_at").alias("observed_at"),
            "quarantine_reason_code",
        )
        .withColumn(
            "quarantine_observation_id",
            F.sha2(
                F.concat_ws(
                    "||",
                    "pipeline_run_id",
                    "quarantine_id",
                ),
                256,
            ),
        )
        .dropDuplicates(["quarantine_observation_id"])
    )
    if not spark.catalog.tableExists(observations_table):
        observations.limit(0).write.format("delta").saveAsTable(
            observations_table
        )
    _merge_insert_only(
        spark,
        source=observations,
        target_table=observations_table,
        key_column="quarantine_observation_id",
    )


def _merge_insert_only(
    spark: SparkSession,
    *,
    source: DataFrame,
    target_table: str,
    key_column: str,
) -> None:
    validate_qualified_table_name(target_table)
    validate_identifier(key_column, label="quarantine merge key")
    invalid_keys = source.filter(
        F.col(key_column).isNull()
        | (F.trim(F.col(key_column).cast("string")) == "")
    )
    if not invalid_keys.isEmpty():
        raise RuntimeError(
            f"Quarantine merge key {key_column!r} must be non-null and non-blank."
        )

    view_name = f"_crimenet_quarantine_{uuid4().hex}"
    source.createOrReplaceTempView(view_name)
    try:
        spark.sql(
            f"""
            MERGE INTO {quote_table_name(target_table)} AS target
            USING `{view_name}` AS source
              ON target.`{key_column}` <=> source.`{key_column}`
            WHEN NOT MATCHED THEN INSERT *
            """
        )
    finally:
        spark.catalog.dropTempView(view_name)
