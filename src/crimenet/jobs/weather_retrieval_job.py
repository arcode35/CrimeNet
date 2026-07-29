"""Retrieve a planned Open-Meteo manifest into the persistent raw cache."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from uuid import uuid4

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from crimenet.config.validation import validate_qualified_table_name
from crimenet.observability.logging import get_logger
from crimenet.observability.run_context import resolve_pipeline_run_id
from crimenet.weather.open_meteo_client import OpenMeteoClientConfig
from crimenet.weather.weather_ingestion import (
    WeatherIngestionAuditEvent,
    fetch_weather_manifest,
)

LOGGER = get_logger(__name__)

WEATHER_AUDIT_SCHEMA = StructType(
    [
        StructField("event_type", StringType(), False),
        StructField("request_id", StringType(), False),
        StructField("provider", StringType(), False),
        StructField("model", StringType(), False),
        StructField("weather_query_cell_id", LongType(), False),
        StructField("start_date", StringType(), False),
        StructField("end_date", StringType(), False),
        StructField("cache_path", StringType(), False),
        StructField("error_type", StringType(), False),
        StructField("error_message", StringType(), False),
        StructField("occurred_at", TimestampType(), False),
        StructField("error_category", StringType(), True),
        StructField("retryable", BooleanType(), True),
        StructField("status_code", IntegerType(), True),
        StructField("pipeline_run_id", StringType(), False),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and durably cache planned Open-Meteo responses."
    )
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--ops-schema", default="ops")
    parser.add_argument("--cache-directory", required=True)
    parser.add_argument(
        "--archive-url",
        default="https://archive-api.open-meteo.com/v1/archive",
    )
    parser.add_argument("--connect-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--read-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-backoff-factor", type=float, default=1.0)
    parser.add_argument("--max-concurrent-requests", type=int, default=1)
    parser.add_argument(
        "--minimum-request-interval-seconds",
        type=float,
        default=0.0,
    )
    parser.add_argument("--pipeline-run-id")
    return parser.parse_args()


def _persist_audit_events(
    spark: SparkSession,
    *,
    events: list[WeatherIngestionAuditEvent],
    pipeline_run_id: str,
    target_table: str,
) -> None:
    if not events:
        return
    validate_qualified_table_name(target_table)
    rows = [
        {
            **asdict(event),
            "pipeline_run_id": pipeline_run_id,
        }
        for event in events
    ]
    dataframe = spark.createDataFrame(
        rows,
        schema=WEATHER_AUDIT_SCHEMA,
    ).withColumn(
        "event_id",
        F.sha2(
            F.concat_ws(
                "||",
                "pipeline_run_id",
                "request_id",
                "event_type",
                "error_type",
                F.coalesce("error_category", F.lit("")),
                F.coalesce(F.col("status_code").cast("string"), F.lit("")),
            ),
            256,
        ),
    )
    if not spark.catalog.tableExists(target_table):
        dataframe.limit(0).write.format("delta").saveAsTable(target_table)
    view_name = f"_crimenet_weather_audit_{uuid4().hex}"
    dataframe.createOrReplaceTempView(view_name)
    try:
        spark.sql(
            f"""
            MERGE INTO {target_table} AS target
            USING {view_name} AS source
              ON target.event_id = source.event_id
            WHEN MATCHED THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *
            """
        )
    finally:
        spark.catalog.dropTempView(view_name)


def run(
    spark: SparkSession,
    *,
    catalog: str,
    ops_schema: str,
    cache_directory: str,
    client_config: OpenMeteoClientConfig,
    pipeline_run_id: str | None,
) -> None:
    run_id = resolve_pipeline_run_id(pipeline_run_id)
    manifest_table = f"{catalog}.{ops_schema}.weather_request_manifest"
    audit_table = f"{catalog}.{ops_schema}.weather_request_failures"
    events: list[WeatherIngestionAuditEvent] = []
    try:
        fetch_weather_manifest(
            spark.table(manifest_table),
            cache_directory=cache_directory,
            client_config=client_config,
            audit_hook=events.append,
        )
    finally:
        _persist_audit_events(
            spark,
            events=events,
            pipeline_run_id=run_id,
            target_table=audit_table,
        )

    LOGGER.info(
        "Weather manifest retrieval completed",
        pipeline_run_id=run_id,
        manifest_table=manifest_table,
        audit_event_count=len(events),
    )


def main() -> None:
    args = parse_args()
    spark = (
        SparkSession.getActiveSession()
        or SparkSession.builder.getOrCreate()
    )
    config = OpenMeteoClientConfig(
        archive_url=args.archive_url,
        connect_timeout_seconds=args.connect_timeout_seconds,
        read_timeout_seconds=args.read_timeout_seconds,
        max_retries=args.max_retries,
        retry_backoff_factor=args.retry_backoff_factor,
        max_concurrent_requests=args.max_concurrent_requests,
        minimum_request_interval_seconds=(
            args.minimum_request_interval_seconds
        ),
    )
    run(
        spark,
        catalog=args.catalog,
        ops_schema=args.ops_schema,
        cache_directory=args.cache_directory,
        client_config=config,
        pipeline_run_id=args.pipeline_run_id,
    )


if __name__ == "__main__":
    main()
