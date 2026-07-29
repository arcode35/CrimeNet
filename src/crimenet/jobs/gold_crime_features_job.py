"""Python-wheel entry point for building Gold crime features."""

from __future__ import annotations

import argparse

from pyspark.sql import SparkSession

from crimenet.contracts.gold import GoldCoverageThresholds
from crimenet.contracts.lighting import (
    LIGHTING_DEFINITION_VERSION,
)
from crimenet.gold.crime_features import (
    FeatureTables,
    attach_eligible_acs_vintage,
    attach_lighting_features,
    attach_socioeconomic_features,
    attach_tracts,
    attach_weather_features,
    build_calendar_ranges,
    build_lighting_lookup,
    build_location_mapping_lookup,
    build_weather_lookup,
    extract_unique_locations,
    materialize_gold_features,
    prepare_crimes,
)
from crimenet.observability.logging import get_logger
from crimenet.observability.run_context import (
    resolve_pipeline_run_id,
)

LOGGER = get_logger(__name__)


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
        "--data-quality-schema",
        default="data_quality",
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
        "--lighting-definition-version",
        default=LIGHTING_DEFINITION_VERSION,
    )
    parser.add_argument(
        "--minimum-weather-coverage",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--minimum-lighting-coverage",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--minimum-acs-coverage",
        "--minimum-socioeconomic-coverage",
        dest="minimum_acs_coverage",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--minimum-tract-coverage",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--pipeline-run-id",
    )

    return parser.parse_args()


def run(
    spark: SparkSession,
    *,
    catalog: str,
    silver_schema: str,
    gold_schema: str,
    data_quality_schema: str,
    weather_provider: str,
    weather_model: str,
    weather_h3_resolution: int,
    lighting_definition_version: str = (LIGHTING_DEFINITION_VERSION),
    minimum_weather_coverage: float = 0.0,
    minimum_lighting_coverage: float = 0.0,
    minimum_acs_coverage: float = 0.0,
    minimum_tract_coverage: float = 0.0,
    pipeline_run_id: str | None = None,
) -> None:
    if not 0 <= weather_h3_resolution <= 15:
        raise ValueError("H3 resolution must be between 0 and 15.")
    run_id = resolve_pipeline_run_id(pipeline_run_id)

    tables = FeatureTables.from_schemas(
        catalog=catalog,
        silver_schema=silver_schema,
        gold_schema=gold_schema,
    )
    quality_results_table = f"{catalog}.{data_quality_schema}.quality_results"
    coverage_thresholds = GoldCoverageThresholds(
        weather=minimum_weather_coverage,
        lighting=minimum_lighting_coverage,
        socioeconomic=(minimum_acs_coverage),
        tract=minimum_tract_coverage,
    )

    crime_dataframe = spark.table(tables.crime)
    calendar_dataframe = spark.table(tables.calendar)
    socioeconomic_dataframe = spark.table(tables.socioeconomic)
    weather_dataframe = spark.table(tables.weather)
    lighting_dataframe = spark.table(tables.lighting)
    location_mapping_dataframe = spark.table(tables.location_tract_mapping)

    calendar_ranges = build_calendar_ranges(calendar_dataframe)

    crime_prepared = prepare_crimes(crime_dataframe)

    crime_with_calendar = attach_eligible_acs_vintage(
        crime_prepared,
        calendar_ranges,
    )

    candidate_locations = extract_unique_locations(crime_with_calendar)

    location_mapping = build_location_mapping_lookup(
        candidate_locations,
        location_mapping_dataframe,
    )

    crime_with_tract = attach_tracts(
        crime_with_calendar,
        location_mapping,
    )

    crime_with_socioeconomic = attach_socioeconomic_features(
        crime_with_tract,
        socioeconomic_dataframe,
    )

    lighting_lookup = build_lighting_lookup(
        lighting_dataframe,
        definition_version=(lighting_definition_version),
    )
    crime_with_lighting = attach_lighting_features(
        crime_with_socioeconomic,
        lighting_lookup,
        definition_version=(lighting_definition_version),
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

    metrics = materialize_gold_features(
        spark,
        source_dataframe=crime_prepared,
        candidate_dataframe=crime_features,
        target_table=tables.features,
        thresholds=coverage_thresholds,
        lighting_definition_version=(lighting_definition_version),
        pipeline_run_id=run_id,
        quality_results_table=quality_results_table,
    )

    LOGGER.info(
        "Successfully materialized Gold crime features",
        pipeline_run_id=run_id,
        target_table=tables.features,
        output_count=metrics.get("final_rows"),
        weather_coverage=metrics.get("weather_match_rate"),
        lighting_coverage=metrics.get("lighting_match_rate"),
        tract_coverage=metrics.get("tract_match_rate"),
        acs_coverage=metrics.get("socioeconomic_match_rate"),
        promotion_status="promoted",
    )


def main() -> None:
    args = parse_args()
    pipeline_run_id = resolve_pipeline_run_id(args.pipeline_run_id)

    spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    spark.conf.set(
        "spark.sql.session.timeZone",
        "UTC",
    )
    run(
        spark,
        catalog=args.catalog,
        silver_schema=args.silver_schema,
        gold_schema=args.gold_schema,
        data_quality_schema=args.data_quality_schema,
        weather_provider=args.weather_provider,
        weather_model=args.weather_model,
        weather_h3_resolution=(args.weather_h3_resolution),
        lighting_definition_version=(args.lighting_definition_version),
        minimum_weather_coverage=(args.minimum_weather_coverage),
        minimum_lighting_coverage=(args.minimum_lighting_coverage),
        minimum_acs_coverage=(args.minimum_acs_coverage),
        minimum_tract_coverage=(args.minimum_tract_coverage),
        pipeline_run_id=pipeline_run_id,
    )


if __name__ == "__main__":
    main()
