"""Python-wheel entry point for hourly weather Silver processing."""

from __future__ import annotations

import argparse

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from crimenet.config.resources import CrimeNetTables
from crimenet.silver.weather import (
    transform_open_meteo_weather,
)


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
        "--checkpoint-path",
        required=True,
    )

    return parser.parse_args()


def ensure_weather_hourly_table(
    spark: SparkSession,
    *,
    table_name: str,
) -> None:
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
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
            bronze_ingested_at TIMESTAMP,
            silver_processed_at TIMESTAMP,
            weather_date DATE
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
    del batch_id

    if batch_dataframe.isEmpty():
        return

    latest_record_window = (
        Window
        .partitionBy(*MERGE_KEYS)
        .orderBy(
            F.col(
                "bronze_ingested_at"
            ).desc_nulls_last(),
            F.col(
                "silver_processed_at"
            ).desc_nulls_last(),
        )
    )

    deduplicated_dataframe = (
        batch_dataframe
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

    target = DeltaTable.forName(
        spark,
        target_table,
    )

    merge_condition = " AND ".join(
        f"target.{column} = source.{column}"
        for column in MERGE_KEYS
    )

    (
        target.alias("target")
        .merge(
            deduplicated_dataframe.alias(
                "source"
            ),
            merge_condition,
        )
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )


def run(
    spark: SparkSession,
    *,
    catalog: str,
    bronze_schema: str,
    silver_schema: str,
    checkpoint_path: str,
) -> None:
    spark.conf.set(
        "spark.sql.session.timeZone",
        "UTC",
    )

    tables = CrimeNetTables(
        catalog=catalog,
        bronze_schema=bronze_schema,
        silver_schema=silver_schema,
    )

    ensure_weather_hourly_table(
        spark,
        table_name=tables.weather_hourly_silver,
    )

    bronze_stream = (
        spark.readStream
        .table(
            tables.open_meteo_weather_bronze
        )
    )

    silver_stream = (
        transform_open_meteo_weather(
            bronze_stream
        )
    )

    def upsert_batch(
        batch_dataframe: DataFrame,
        batch_id: int,
    ) -> None:
        merge_weather_batch(
            batch_dataframe,
            batch_id,
            spark=spark,
            target_table=(
                tables.weather_hourly_silver
            ),
        )

    query = (
        silver_stream.writeStream
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


def main() -> None:
    args = parse_args()

    spark = (
        SparkSession.getActiveSession()
        or SparkSession.builder.getOrCreate()
    )

    run(
        spark,
        catalog=args.catalog,
        bronze_schema=args.bronze_schema,
        silver_schema=args.silver_schema,
        checkpoint_path=args.checkpoint_path,
    )


if __name__ == "__main__":
    main()