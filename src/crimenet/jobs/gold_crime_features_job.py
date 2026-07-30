"""Python-wheel entry point for building Gold crime features."""

from __future__ import annotations

import argparse
import logging

from pyspark.sql import SparkSession

from crimenet.gold.crime_features import (
    FeatureTables,
    attach_eligible_acs_vintage,
    attach_lighting_features,
    attach_socioeconomic_features,
    attach_tracts,
    attach_weather_features,
    build_calendar_ranges,
    build_lighting_lookup,
    build_weather_lookup,
    extract_unique_locations,
    log_coverage_metrics,
    materialize_location_mapping,
    prepare_crimes,
    validate_boundary_inputs,
    validate_crime_identities,
)
from crimenet.quality import validate_gold


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-safe crime features from crime, "
            "ACS, tract, and weather tables."
        )
    )

    parser.add_argument(
        "--catalog",
        required=True,
    )
    parser.add_argument(
        "--silver-schema",
        default="silver",
    )
    parser.add_argument(
        "--gold-schema",
        default="gold",
    )
    parser.add_argument(
        "--weather-provider",
        default="open_meteo",
    )
    parser.add_argument(
        "--weather-model",
        default="era5_land",
    )
    parser.add_argument(
        "--weather-h3-resolution",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--rebuild-location-tract-mapping",
        action="store_true",
    )

    return parser.parse_args()


def run(
    spark: SparkSession,
    *,
    catalog: str,
    silver_schema: str,
    gold_schema: str,
    weather_provider: str,
    weather_model: str,
    weather_h3_resolution: int,
    rebuild_location_tract_mapping: bool,
) -> None:
    if not 0 <= weather_h3_resolution <= 15:
        raise ValueError(
            "H3 resolution must be between 0 and 15."
        )

    tables = FeatureTables.from_schemas(
        catalog=catalog,
        silver_schema=silver_schema,
        gold_schema=gold_schema,
    )

    crime_dataframe = spark.table(
        tables.crime
    )
    calendar_dataframe = spark.table(
        tables.calendar
    )
    boundary_dataframe = spark.table(
        tables.boundaries
    )
    socioeconomic_dataframe = spark.table(
        tables.socioeconomic
    )
    weather_dataframe = spark.table(
        tables.weather
    )
    lighting_dataframe = spark.table(
        tables.lighting
    )
    validate_boundary_inputs(
        calendar_dataframe,
        boundary_dataframe,
    )

    calendar_ranges = build_calendar_ranges(
        calendar_dataframe
    )

    crime_prepared = prepare_crimes(
        crime_dataframe
    )

    validate_crime_identities(crime_prepared)

    crime_with_calendar = (
        attach_eligible_acs_vintage(
            crime_prepared,
            calendar_ranges,
        )
    )

    candidate_locations = extract_unique_locations(
        crime_with_calendar
    )

    location_mapping = materialize_location_mapping(
        spark,
        candidate_locations=candidate_locations,
        boundary_dataframe=boundary_dataframe,
        target_table=(
            tables.location_tract_mapping
        ),
        rebuild=(
            rebuild_location_tract_mapping
        ),
    )

    crime_with_tract = attach_tracts(
        crime_with_calendar,
        location_mapping,
    )

    crime_with_socioeconomic = (
        attach_socioeconomic_features(
            crime_with_tract,
            socioeconomic_dataframe,
        )
    )
        
    lighting_lookup = build_lighting_lookup(
        lighting_dataframe
    )
    crime_with_lighting = (
        attach_lighting_features(
            crime_with_socioeconomic, 
            lighting_lookup
        )
    )
    weather_lookup = build_weather_lookup(
        crime_prepared,
        weather_dataframe,
        provider=weather_provider,
        model=weather_model,
        h3_resolution=weather_h3_resolution,
    )

    crime_features = attach_weather_features(
        crime_with_lighting,
        weather_lookup,
    )

    (
        crime_features.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(
            tables.features
        )
    )

    materialized_features = spark.table(
        tables.features
    )

    source_count = crime_dataframe.count()

    metrics = log_coverage_metrics(
        materialized_features
    )
    quality_report = validate_gold(
        materialized_features,
        source_crime_count=source_count,
    )

    LOGGER.info(
        "Successfully materialized %s with %s quality checks; metrics=%s",
        tables.features,
        len(quality_report.checks),
        metrics,
    )


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s %(levelname)s "
            "%(name)s - %(message)s"
        ),
    )

    spark = (
        SparkSession.getActiveSession()
        or SparkSession.builder.getOrCreate()
    )
    spark.conf.set(
        "spark.sql.session.timeZone",
        "UTC",
    )
    run(
        spark,
        catalog=args.catalog,
        silver_schema=args.silver_schema,
        gold_schema=args.gold_schema,
        weather_provider=args.weather_provider,
        weather_model=args.weather_model,
        weather_h3_resolution=(
            args.weather_h3_resolution
        ),
        rebuild_location_tract_mapping=(
            args.rebuild_location_tract_mapping
        ),
    )


if __name__ == "__main__":
    main()
