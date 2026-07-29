"""Plan deterministic Open-Meteo requests from canonical Silver crime."""

from __future__ import annotations

import argparse
from datetime import date

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from crimenet.observability.logging import get_logger
from crimenet.observability.run_context import resolve_pipeline_run_id
from crimenet.utils.promotion import promote_staged_table
from crimenet.weather.request_planner import build_weather_request_manifest

LOGGER = get_logger(__name__)
WEATHER_REQUEST_DEFINITION_VERSION = "open_meteo_cell_year_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the current deterministic Open-Meteo manifest."
    )
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--silver-schema", default="silver")
    parser.add_argument("--ops-schema", default="ops")
    parser.add_argument("--model", default="era5_land")
    parser.add_argument("--h3-resolution", type=int, default=6)
    parser.add_argument("--hourly-variables", default="temperature_2m")
    parser.add_argument("--availability-cutoff")
    parser.add_argument("--minimum-request-count", type=int, default=0)
    parser.add_argument("--pipeline-run-id")
    return parser.parse_args()


def validate_manifest(
    dataframe: DataFrame,
    *,
    minimum_request_count: int,
) -> None:
    row_count = dataframe.count()
    if row_count < minimum_request_count:
        raise RuntimeError(
            "Weather request manifest is unexpectedly small: "
            f"observed={row_count}, expected>={minimum_request_count}."
        )
    duplicate = (
        dataframe.groupBy("request_id")
        .count()
        .filter(F.col("count") > 1)
        .limit(1)
        .count()
    )
    missing = dataframe.filter(
        F.col("request_id").isNull()
        | F.col("weather_query_cell_id").isNull()
        | F.col("start_date").isNull()
        | F.col("end_date").isNull()
    ).limit(1).count()
    if duplicate:
        raise RuntimeError("Weather manifest contains duplicate request IDs.")
    if missing:
        raise RuntimeError("Weather manifest contains missing request keys.")


def run(
    spark: SparkSession,
    *,
    catalog: str,
    silver_schema: str,
    ops_schema: str,
    model: str,
    h3_resolution: int,
    hourly_variables: tuple[str, ...],
    availability_cutoff: date | None,
    minimum_request_count: int,
    pipeline_run_id: str | None,
) -> None:
    run_id = resolve_pipeline_run_id(pipeline_run_id)
    crime_table = f"{catalog}.{silver_schema}.crime_offenses"
    target_table = f"{catalog}.{ops_schema}.weather_request_manifest"
    manifest = (
        build_weather_request_manifest(
            spark.table(crime_table),
            model=model,
            hourly_variables=hourly_variables,
            h3_resolution=h3_resolution,
            availability_cutoff=availability_cutoff,
        )
        .withColumn("pipeline_run_id", F.lit(run_id))
        .withColumn(
            "request_definition_version",
            F.lit(WEATHER_REQUEST_DEFINITION_VERSION),
        )
        .withColumn("planned_at", F.current_timestamp())
    )

    def validate(candidate: DataFrame) -> None:
        validate_manifest(
            candidate,
            minimum_request_count=minimum_request_count,
        )

    promote_staged_table(
        spark,
        candidate=manifest,
        target_table=target_table,
        pipeline_run_id=run_id,
        validate=validate,
    )
    LOGGER.info(
        "Weather request manifest promoted",
        pipeline_run_id=run_id,
        target_table=target_table,
        request_definition_version=WEATHER_REQUEST_DEFINITION_VERSION,
    )


def main() -> None:
    args = parse_args()
    spark = (
        SparkSession.getActiveSession()
        or SparkSession.builder.getOrCreate()
    )
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    cutoff = (
        date.fromisoformat(args.availability_cutoff)
        if args.availability_cutoff
        else None
    )
    variables = tuple(
        item.strip()
        for item in args.hourly_variables.split(",")
        if item.strip()
    )
    run(
        spark,
        catalog=args.catalog,
        silver_schema=args.silver_schema,
        ops_schema=args.ops_schema,
        model=args.model,
        h3_resolution=args.h3_resolution,
        hourly_variables=variables,
        availability_cutoff=cutoff,
        minimum_request_count=args.minimum_request_count,
        pipeline_run_id=args.pipeline_run_id,
    )


if __name__ == "__main__":
    main()
