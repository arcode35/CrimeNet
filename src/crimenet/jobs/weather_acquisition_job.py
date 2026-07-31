"""Python-wheel entry point for Open-Meteo weather acquisition."""

from __future__ import annotations

import argparse
from datetime import date, timedelta

from pyspark.sql import SparkSession

from crimenet.observability.logging import (
    get_logger,
)
from crimenet.weather.request_planner import (
    DEFAULT_AVAILABILITY_LAG_DAYS,
    build_weather_request_manifest,
)
from crimenet.weather.weather_ingestion import (
    fetch_weather_manifest,
)

LOGGER = get_logger(__name__)


DEFAULT_CITIES = (
    "dallas",
    "fort_worth",
    "new_york",
    "chicago",
    "san_francisco",
    "seattle",
    "baltimore",
    "washington_dc",
)

DEFAULT_HOURLY_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "snowfall",
    "weather_code",
    "cloud_cover",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--catalog",
        required=True,
    )
    parser.add_argument(
        "--silver-schema",
        default="silver",
    )
    parser.add_argument(
        "--ops-schema",
        default="ops",
    )
    parser.add_argument(
        "--crime-table",
        default="crime_offenses",
    )
    parser.add_argument(
        "--raw-landing-path",
        required=True,
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        default=list(DEFAULT_CITIES),
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2013,
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
    )
    parser.add_argument(
        "--model",
        choices=[
            "era5",
            "era5_land",
        ],
        default="era5_land",
    )
    parser.add_argument(
        "--h3-resolution",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--hourly-variables",
        nargs="+",
        default=list(
            DEFAULT_HOURLY_VARIABLES
        ),
    )
    parser.add_argument(
        "--availability-lag-days",
        type=int,
        default=(
            DEFAULT_AVAILABILITY_LAG_DAYS
        ),
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
    )
    parser.add_argument(
        "--force",
        action="store_true",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.25,
    )

    return parser.parse_args()


def run(
    spark: SparkSession,
    *,
    catalog: str,
    silver_schema: str,
    ops_schema: str,
    crime_table: str,
    raw_landing_path: str,
    cities: list[str],
    start_year: int,
    end_year: int,
    model: str,
    h3_resolution: int,
    hourly_variables: list[str],
    availability_lag_days: int,
    manifest_only: bool,
    force: bool,
    maximum_requests: int | None,
    pause_seconds: float,
) -> None:
    if availability_lag_days < 0:
        raise ValueError(
            "availability_lag_days cannot be negative"
        )

    spark.conf.set(
        "spark.sql.session.timeZone",
        "UTC",
    )

    source_table = (
        f"{catalog}."
        f"{silver_schema}."
        f"{crime_table}"
    )

    manifest_table = (
        f"{catalog}."
        f"{ops_schema}."
        "weather_request_manifest"
    )

    cache_directory = (
        raw_landing_path.rstrip("/")
        + "/weather/open_meteo"
    )

    availability_cutoff = (
        date.today()
        - timedelta(
            days=availability_lag_days
        )
    )

    LOGGER.info(
        "Starting Open-Meteo acquisition",
        source_table=source_table,
        manifest_table=manifest_table,
        cache_directory=cache_directory,
        cities=cities,
        start_year=start_year,
        end_year=end_year,
        model=model,
        h3_resolution=h3_resolution,
    )

    crime_dataframe = spark.table(
        source_table
    )

    manifest = build_weather_request_manifest(
        crime_dataframe,
        cities=cities,
        start_year=start_year,
        end_year=end_year,
        model=model,
        hourly_variables=hourly_variables,
        h3_resolution=h3_resolution,
        availability_cutoff=(
            availability_cutoff
        ),
    )

    (
        manifest.write
        .format("delta")
        .mode("overwrite")
        .option(
            "overwriteSchema",
            "true",
        )
        .saveAsTable(
            manifest_table
        )
    )

    # Read the persisted manifest so the Silver aggregation
    # is not recomputed while requests are being executed.
    persisted_manifest = (
        spark.read
        .table(manifest_table)
        .orderBy(
            "start_date",
            "weather_query_cell_id",
        )
    )

    manifest_count = (
        persisted_manifest.count()
    )

    LOGGER.info(
        "Weather request manifest created",
        manifest_table=manifest_table,
        request_count=manifest_count,
    )

    if manifest_only:
        LOGGER.info(
            "Manifest-only mode enabled; "
            "no HTTP requests will be made",
            manifest_table=manifest_table,
        )
        return

    summary = fetch_weather_manifest(
        persisted_manifest,
        cache_directory=cache_directory,
        force=force,
        maximum_requests=maximum_requests,
        pause_seconds=pause_seconds,
    )

    LOGGER.info(
        "Completed Open-Meteo acquisition",
        processed=summary.processed,
        attempted=summary.attempted,
        downloaded=summary.downloaded,
        cached=summary.cached,
        failed=summary.failed,
    )

    if summary.failed:
        failure_preview = "\n".join(
            summary.failures[:10]
        )

        raise RuntimeError(
            f"{summary.failed} weather requests "
            "failed. First failures:\n"
            f"{failure_preview}"
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
            silver_schema=args.silver_schema,
            ops_schema=args.ops_schema,
            crime_table=args.crime_table,
            raw_landing_path=(
                args.raw_landing_path
            ),
            cities=args.cities,
            start_year=args.start_year,
            end_year=args.end_year,
            model=args.model,
            h3_resolution=(
                args.h3_resolution
            ),
            hourly_variables=(
                args.hourly_variables
            ),
            availability_lag_days=(
                args.availability_lag_days
            ),
            manifest_only=(
                args.manifest_only
            ),
            force=args.force,
            maximum_requests=(
                args.max_requests
            ),
            pause_seconds=(
                args.pause_seconds
            ),
        )

    except Exception:
        LOGGER.exception(
            "Open-Meteo acquisition failed",
            catalog=args.catalog,
            silver_schema=args.silver_schema,
            crime_table=args.crime_table,
        )
        raise


if __name__ == "__main__":
    main()