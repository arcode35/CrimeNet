from datetime import UTC, datetime

import dagster as dg
import polars as pl

from crimenet_data.resources.crime_lake import CrimeLakeResources
from crimenet_data.observability.logger import get_logger
from crimenet_data.observability.context import log_context
log = get_logger(__name__)

WEATHER_MODES = ["land", "coastal"]


COASTAL_WEATHER_SCHEMA = pl.Schema(
    {
        "latitude": pl.Float64,
        "longitude": pl.Float64,
        "generationtime_ms": pl.Float64,
        "utc_offset_seconds": pl.Int64,
        "timezone": pl.String,
        "timezone_abbreviation": pl.String,
        "elevation": pl.Float64,
        "hourly_units": pl.Struct(
            {
                "time": pl.String,
                "temperature_2m": pl.String,
            }
        ),
        "hourly": pl.Struct(
            {
                "time": pl.List(pl.String),
                "temperature_2m": pl.List(pl.Float64),
            }
        ),
        "_request_id": pl.String,
        "_weather_query_cell_id": pl.Int64,
        "_request_start_date": pl.String,
        "_request_end_date": pl.String,
        "_requested_latitude": pl.Float64,
        "_requested_longitude": pl.Float64,
        "_provider": pl.String,
        "_model": pl.String,
        "_cell_selection": pl.String,
        "_fetched_at_utc": pl.String,
    }
)


LAND_WEATHER_SCHEMA = pl.Schema(
    {
        "request_id": pl.String,
        "provider": pl.String,
        "model": pl.String,
        "weather_query_cell_id": pl.Int64,
        "h3_resolution": pl.Int64,
        "query_latitude": pl.Float64,
        "query_longitude": pl.Float64,
        "grid_latitude": pl.Float64,
        "grid_longitude": pl.Float64,
        "grid_elevation": pl.Float64,
        "start_date": pl.String,
        "end_date": pl.String,
        "timezone": pl.String,
        "utc_offset_seconds": pl.Int64,
        "cell_selection": pl.String,
        "hourly_variables": pl.List(pl.String),
        "hourly_units": pl.Struct(
            {
                "time": pl.String,
                "cloud_cover": pl.String,
                "precipitation": pl.String,
                "rain": pl.String,
                "relative_humidity_2m": pl.String,
                "snowfall": pl.String,
                "surface_pressure": pl.String,
                "temperature_2m": pl.String,
                "weather_code": pl.String,
                "wind_direction_10m": pl.String,
                "wind_gusts_10m": pl.String,
                "wind_speed_10m": pl.String,
            }
        ),
        "hourly": pl.Struct(
            {
                "time": pl.List(pl.String),
                "cloud_cover": pl.List(pl.Float64),
                "precipitation": pl.List(pl.Float64),
                "rain": pl.List(pl.Float64),
                "relative_humidity_2m": pl.List(pl.Int64),
                "snowfall": pl.List(pl.Float64),
                "surface_pressure": pl.List(pl.Float64),
                "temperature_2m": pl.List(pl.Float64),
                "weather_code": pl.List(pl.Int64),
                "wind_direction_10m": pl.List(pl.Float64),
                "wind_gusts_10m": pl.List(pl.Float64),
                "wind_speed_10m": pl.List(pl.Float64),
            }
        ),
    }
)


WEATHER_SCHEMAS: dict[str, pl.Schema] = {
    "coastal": COASTAL_WEATHER_SCHEMA,
    "land": LAND_WEATHER_SCHEMA,
}
def build_weather_bronze(
    raw_df: pl.LazyFrame,
    *,
    run_id: str,
    weather_mode: str,
) -> pl.LazyFrame:
    ingested_at = datetime.now(UTC)
    raw_df = raw_df.rename({"_request_start_date": "start_date", "_request_end_date": "end_date"}, strict=False)
    raw_df = raw_df.filter(
        (pl.col("start_date").str.to_date(strict=False).dt.year().is_not_null()
        | pl.col("end_date").str.to_date(strict=False).dt.year().is_not_null())
    )
    return raw_df.with_columns(
        pl.coalesce(
            pl.col("start_date").str.to_date(strict=False).dt.year(),
            pl.col("end_date").str.to_date(strict=False).dt.year()
        ).alias("year"),
        pl.lit("open_meteo").alias("_source_system"),
        pl.lit(weather_mode).alias("_weather_mode"),
        pl.lit(run_id).alias("_ingestion_run_id"),
        pl.lit(ingested_at).alias("_ingested_at_utc"),
    )


def build_bronze_weather_asset(
    weather_mode: str,
) -> dg.AssetsDefinition:

    @dg.asset(
        name=f"bronze_weather_{weather_mode}",
        group_name="bronze_weather",
    )
    def _bronze_asset(
        context: dg.AssetExecutionContext,
        crime_lake: CrimeLakeResources,
    ) -> dg.MaterializeResult:

        with log_context(
            run_id=context.run_id,
            asset_key=context.asset_key.to_user_string(),
            weather_mode=weather_mode,
        ):
            log.info("processing_started")

            source_uri = crime_lake.resolve_weather_path(weather_mode)

            target_uri = (
                f"{crime_lake.bronze_root.rstrip('/')}/"
                f"weather/open_meteo/era5_{weather_mode}"
            )

            log.info(
                "source_scan_started",
                source_uri=source_uri,
            )

            raw_lf = pl.scan_ndjson(
                source_uri,
                schema=WEATHER_SCHEMAS[weather_mode],
                include_file_paths="_source_file_uri",
                credential_provider=pl.CredentialProviderGCP(),
            )

            bronze_lf = build_weather_bronze(
                raw_df=raw_lf,
                run_id=context.run_id,
                weather_mode=weather_mode,
            )

            schema = bronze_lf.collect_schema()

            if "year" not in schema:
                raise ValueError(
                    "'year' is required for partitioning. "
                    f"Available columns: {schema.names()}"
                )

            log.info(
                "write_started",
                target_uri=target_uri,
            )

            crime_lake.write_crimenet_table(
                lf=bronze_lf,
                target_uri=target_uri,
                partitioning_columns=["year"],
            )

            log.info(
                "processing_completed",
                target_uri=target_uri,
            )

            return dg.MaterializeResult(
                metadata={
                    "source_system": "open_meteo",
                    "weather_mode": weather_mode,
                    "ingestion_run_id": context.run_id,
                    "source_uri": source_uri,
                    "target_uri": target_uri,
                }
            )

    return _bronze_asset

weather_bronze_assets = [
    build_bronze_weather_asset(mode)
    for mode in WEATHER_MODES
]