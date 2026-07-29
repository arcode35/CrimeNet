"""Python-wheel entry point for hourly weather Silver processing."""

from __future__ import annotations

import argparse
from uuid import uuid4

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from crimenet.config.resources import CrimeNetTables
from crimenet.observability.logging import get_logger
from crimenet.observability.run_context import resolve_pipeline_run_id
from crimenet.quality.external import split_external_quarantine
from crimenet.quality.quarantine import merge_quarantine
from crimenet.silver.weather import (
    WEATHER_DEFINITION_VERSION,
    WEATHER_QUARANTINE_MESSAGES,
    annotate_weather_validation,
    transform_open_meteo_weather,
)
from crimenet.utils.promotion import (
    promote_staged_table,
    quote_table_name,
)

LOGGER = get_logger(__name__)


MERGE_KEYS = (
    "provider",
    "model",
    "weather_query_cell_id",
    "weather_timestamp",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--catalog",
        required=True,
    )
    parser.add_argument(
        "--bronze-schema",
        default="bronze",
    )
    parser.add_argument(
        "--silver-schema",
        default="silver",
    )
    parser.add_argument(
        "--data-quality-schema",
        default="data_quality",
    )
    parser.add_argument("--pipeline-run-id")
    parser.add_argument(
        "--checkpoint-path",
        required=True,
    )

    return parser.parse_args()


def ensure_weather_hourly_table(
    spark: SparkSession,
    *,
    table_name: str,
) -> None:
    quoted_table_name = quote_table_name(table_name)
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {quoted_table_name} (
            provider STRING,
            model STRING,
            request_id STRING,
            weather_query_cell_id BIGINT,
            h3_resolution INT,
            query_latitude DOUBLE,
            query_longitude DOUBLE,
            grid_latitude DOUBLE,
            grid_longitude DOUBLE,
            grid_elevation DOUBLE,
            weather_timestamp TIMESTAMP,
            temperature_2m_c DOUBLE,
            temperature_unit STRING,
            timezone STRING,
            utc_offset_seconds INT,
            source_file STRING,
            source_row_hash STRING,
            source_contract_version STRING,
            bronze_ingested_at TIMESTAMP,
            silver_processed_at TIMESTAMP,
            weather_date DATE,
            weather_definition_version STRING
        )
        USING DELTA
        COMMENT 'Hourly weather observations derived from Open-Meteo'
        """
    )


def merge_weather_batch(
    batch_dataframe: DataFrame,
    batch_id: int,
    *,
    spark: SparkSession,
    target_table: str,
) -> None:
    quoted_target_table = quote_table_name(target_table)
    if batch_dataframe.isEmpty():
        LOGGER.info(
            "Skipping empty weather microbatch",
            batch_id=batch_id,
            target_table=target_table,
        )
        return

    deduplicated_dataframe = deduplicate_weather_records(batch_dataframe)

    view_name = f"_crimenet_weather_{uuid4().hex}"
    deduplicated_dataframe.createOrReplaceTempView(view_name)
    merge_condition = " AND ".join(
        f"target.{column} = source.{column}" for column in MERGE_KEYS
    )
    try:
        spark.sql(
            f"""
            MERGE INTO {quoted_target_table} AS target
            USING {view_name} AS source
              ON {merge_condition}
            WHEN MATCHED
              AND (
                coalesce(target.weather_definition_version, '')
                  <> coalesce(source.weather_definition_version, '')
                OR (
                  coalesce(target.weather_definition_version, '')
                    = coalesce(source.weather_definition_version, '')
                  AND coalesce(source.source_row_hash, '')
                    > coalesce(target.source_row_hash, '')
                )
              )
              THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *
            """
        )
    finally:
        spark.catalog.dropTempView(view_name)

    LOGGER.info(
        "Merged weather microbatch",
        batch_id=batch_id,
        target_table=target_table,
    )


def deduplicate_weather_records(dataframe: DataFrame) -> DataFrame:
    """Choose one deterministic observation for every weather business key."""
    latest_record_window = (
        Window
        .partitionBy(*MERGE_KEYS)
        .orderBy(
            F.col(
                "source_row_hash"
            ).desc_nulls_last(),
            F.col(
                "bronze_ingested_at"
            ).desc_nulls_last(),
            F.col(
                "silver_processed_at"
            ).desc_nulls_last(),
        )
    )

    return (
        dataframe
        .withColumn(
            "_row_number",
            F.row_number().over(
                latest_record_window
            ),
        )
        .filter(
            F.col("_row_number") == 1
        )
        .drop("_row_number")
    )


def validate_weather_dataframe(dataframe: DataFrame) -> None:
    duplicate_keys = (
        dataframe.groupBy(*MERGE_KEYS).count().filter(F.col("count") > 1)
    )
    invalid_rows = dataframe.filter(
        F.col("provider").isNull()
        | F.col("model").isNull()
        | F.col("weather_query_cell_id").isNull()
        | F.col("weather_timestamp").isNull()
        | F.col("temperature_2m_c").isNull()
        | F.isnan("temperature_2m_c")
        | F.col("weather_definition_version").isNull()
        | (
            F.col("weather_definition_version")
            != F.lit(WEATHER_DEFINITION_VERSION)
        )
    )

    if not duplicate_keys.isEmpty() or not invalid_rows.isEmpty():
        raise RuntimeError(
            "Silver weather candidate contains duplicate keys, invalid values, "
            "or a stale definition version."
        )


def weather_rebuild_required(
    *,
    bronze_dataframe: DataFrame,
    target_dataframe: DataFrame,
) -> bool:
    """Return true when a checkpoint cannot refresh the complete target."""
    if bronze_dataframe.limit(1).isEmpty():
        return False
    if target_dataframe.limit(1).isEmpty():
        return True
    stale_rows = target_dataframe.filter(
        F.col("weather_definition_version").isNull()
        | (
            F.col("weather_definition_version")
            != F.lit(WEATHER_DEFINITION_VERSION)
        )
    )
    return not stale_rows.limit(1).isEmpty()


def rebuild_weather_table(
    spark: SparkSession,
    *,
    tables: CrimeNetTables,
    pipeline_run_id: str,
    bronze_dataframe: DataFrame,
) -> None:
    annotated = annotate_weather_validation(bronze_dataframe)
    valid_bronze, quarantine = split_external_quarantine(
        annotated,
        reason_codes_column="_quarantine_reason_codes",
        reason_messages=WEATHER_QUARANTINE_MESSAGES,
        source_system="open_meteo",
        pipeline_run_id=pipeline_run_id,
    )
    merge_quarantine(
        spark,
        quarantine=quarantine,
        target_table=(
            f"{tables.catalog}.{tables.data_quality_schema}."
            "weather_quarantine"
        ),
    )
    candidate = deduplicate_weather_records(
        transform_open_meteo_weather(valid_bronze)
    )
    promote_staged_table(
        spark,
        candidate=candidate,
        target_table=tables.weather_hourly_silver,
        pipeline_run_id=pipeline_run_id,
        validate=validate_weather_dataframe,
    )


def run(
    spark: SparkSession,
    *,
    catalog: str,
    bronze_schema: str,
    silver_schema: str,
    data_quality_schema: str,
    checkpoint_path: str,
    pipeline_run_id: str | None = None,
) -> None:
    spark.conf.set(
        "spark.sql.session.timeZone",
        "UTC",
    )

    run_id = resolve_pipeline_run_id(pipeline_run_id)
    tables = CrimeNetTables(
        catalog=catalog,
        bronze_schema=bronze_schema,
        silver_schema=silver_schema,
        data_quality_schema=data_quality_schema,
    )

    LOGGER.info(
        "Starting hourly weather Silver processing",
        source_table=(
            tables.open_meteo_weather_bronze
        ),
        target_table=(
            tables.weather_hourly_silver
        ),
        checkpoint_path=checkpoint_path,
    )

    ensure_weather_hourly_table(
        spark,
        table_name=tables.weather_hourly_silver,
    )

    bronze_dataframe = spark.table(tables.open_meteo_weather_bronze)
    if weather_rebuild_required(
        bronze_dataframe=bronze_dataframe,
        target_dataframe=spark.table(tables.weather_hourly_silver),
    ):
        LOGGER.info(
            "Rebuilding complete weather Silver table for current definition",
            weather_definition_version=WEATHER_DEFINITION_VERSION,
        )
        rebuild_weather_table(
            spark,
            tables=tables,
            pipeline_run_id=run_id,
            bronze_dataframe=bronze_dataframe,
        )
        return

    bronze_stream = spark.readStream.table(tables.open_meteo_weather_bronze)

    def upsert_batch(
        batch_dataframe: DataFrame,
        batch_id: int,
    ) -> None:
        annotated = annotate_weather_validation(batch_dataframe)
        valid_bronze, quarantine = split_external_quarantine(
            annotated,
            reason_codes_column="_quarantine_reason_codes",
            reason_messages=WEATHER_QUARANTINE_MESSAGES,
            source_system="open_meteo",
            pipeline_run_id=run_id,
        )
        merge_quarantine(
            spark,
            quarantine=quarantine,
            target_table=(
                f"{tables.catalog}.{tables.data_quality_schema}."
                "weather_quarantine"
            ),
        )
        merge_weather_batch(
            transform_open_meteo_weather(valid_bronze),
            batch_id,
            spark=spark,
            target_table=(
                tables.weather_hourly_silver
            ),
        )

    query = (
        bronze_stream.writeStream
        .queryName(
            "silver_weather_hourly"
        )
        .option(
            "checkpointLocation",
            checkpoint_path,
        )
        .trigger(
            availableNow=True,
        )
        .foreachBatch(
            upsert_batch
        )
        .start()
    )

    query.awaitTermination()

    LOGGER.info(
        "Completed hourly weather Silver processing",
        target_table=tables.weather_hourly_silver,
    )


def main() -> None:
    args = parse_args()

    spark = (
        SparkSession.getActiveSession()
        or SparkSession.builder.getOrCreate()
    )

    try:
        run(
            spark,
            catalog=args.catalog,
            bronze_schema=args.bronze_schema,
            silver_schema=args.silver_schema,
            data_quality_schema=args.data_quality_schema,
            checkpoint_path=args.checkpoint_path,
            pipeline_run_id=args.pipeline_run_id,
        )
    except Exception:
        LOGGER.exception(
            "Hourly weather Silver processing failed",
            catalog=args.catalog,
            silver_schema=args.silver_schema,
        )
        raise


if __name__ == "__main__":
    main()
