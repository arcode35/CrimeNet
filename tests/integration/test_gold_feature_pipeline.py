from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from crimenet.contracts.lighting import LIGHTING_DEFINITION_VERSION
from crimenet.gold.crime_features import (
    attach_eligible_acs_vintage,
    attach_lighting_features,
    attach_socioeconomic_features,
    attach_tracts,
    attach_weather_features,
    build_calendar_ranges,
    build_lighting_lookup,
    build_weather_lookup,
    log_coverage_metrics,
    prepare_crimes,
    validate_crime_identities,
)
from crimenet.ingestion.column_names import normalize_column_names
from crimenet.ingestion.metadata import add_ingestion_metadata
from crimenet.quality import GoldCoverageThresholds, validate_gold
from crimenet.silver.lighting import (
    OUTPUT_SCHEMA,
    calculate_solar_positions,
)
from crimenet.silver.socioeconomic import (
    deduplicate_socioeconomic_records,
    transform_acs5_tracts,
)
from crimenet.silver.weather import (
    deduplicate_weather_records,
    transform_open_meteo_weather,
)
from crimenet.transforms.canonical import (
    add_crime_offense_id,
    deduplicate_crime_offenses,
)

pytestmark = pytest.mark.integration

_WEATHER_REQUEST_ID = (
    "c52a2df6ddadcaf4376426a0bf6bc2a03"
    "fa24d08c4b1166adbfa4eee274c220a"
)
_WEATHER_CELL_ID = 604686043911815167
_TRACT_GEOID = "48491020330"


def _calendar(spark: SparkSession) -> DataFrame:
    return spark.createDataFrame(
        [
            (2019, date(2020, 12, 10), 2019, "2019"),
            (2020, date(2022, 3, 17), 2020, "2020"),
            (2021, date(2022, 12, 8), 2021, "2021"),
        ],
        (
            "acs_vintage int, acs_release_date date, "
            "tiger_line_year int, tract_definition_vintage string"
        ),
    )


def _lighting_lookup(spark: SparkSession) -> DataFrame:
    keys = pd.DataFrame(
        [
            {
                "weather_query_cell_id": _WEATHER_CELL_ID,
                "solar_timestamp_hour": pd.Timestamp(
                    "2022-07-17T08:00:00Z"
                ),
                "query_latitude": 29.860216534,
                "query_longitude": -95.086285886,
                "lighting_definition_version": (
                    LIGHTING_DEFINITION_VERSION
                ),
            }
        ]
    )
    calculated = next(calculate_solar_positions(iter([keys])))
    values = calculated.iloc[0].to_dict()
    values["solar_timestamp_hour"] = (
        values["solar_timestamp_hour"].to_pydatetime()
    )
    lighting = spark.createDataFrame(
        [tuple(values[field.name] for field in OUTPUT_SCHEMA.fields)],
        schema=OUTPUT_SCHEMA,
    )
    return build_lighting_lookup(lighting)


def _build_features(
    *,
    spark: SparkSession,
    houston_canonical: DataFrame,
    socioeconomic_bronze: DataFrame,
    weather_raw: DataFrame,
) -> DataFrame:
    crime = (
        houston_canonical
        .filter(
            (F.col("source_city") == "houston")
            & (F.col("source_incident_id") == "94661522")
        )
        .transform(add_crime_offense_id)
        .localCheckpoint(eager=True)
    )
    crime = deduplicate_crime_offenses(crime).withColumn(
        "weather_query_cell_id",
        F.lit(_WEATHER_CELL_ID).cast("long"),
    )
    validate_crime_identities(crime)

    with_calendar = attach_eligible_acs_vintage(
        prepare_crimes(crime),
        build_calendar_ranges(_calendar(spark)),
    )
    mapping = (
        with_calendar
        .select(
            "tiger_line_year",
            "latitude",
            "longitude",
        )
        .withColumn("tract_geoid", F.lit(_TRACT_GEOID))
    )
    with_tract = attach_tracts(with_calendar, mapping)

    socioeconomic = deduplicate_socioeconomic_records(
        transform_acs5_tracts(
            socioeconomic_bronze
        ).localCheckpoint(eager=True)
    )
    with_socioeconomic = attach_socioeconomic_features(
        with_tract,
        socioeconomic,
    )

    with_lighting = attach_lighting_features(
        with_socioeconomic,
        _lighting_lookup(spark),
    )

    weather_bronze = add_ingestion_metadata(
        normalize_column_names(
            weather_raw.filter(
                F.col("request_id") == _WEATHER_REQUEST_ID
            )
        ),
        source_system="open_meteo",
    ).localCheckpoint(eager=True)
    weather = deduplicate_weather_records(
        transform_open_meteo_weather(weather_bronze)
    ).filter(
        F.col("weather_timestamp")
        == F.lit(datetime(2022, 7, 17, 8))
    ).localCheckpoint(eager=True)
    weather_lookup = build_weather_lookup(
        with_lighting,
        weather,
        provider="open_meteo",
        model="era5_land",
        h3_resolution=6,
    )

    return attach_weather_features(
        with_lighting,
        weather_lookup,
    )


def test_fixture_derived_gold_pipeline_is_enriched_and_idempotent(
    spark: SparkSession,
    houston_canonical: DataFrame,
    socioeconomic_bronze: DataFrame,
    weather_raw: DataFrame,
) -> None:
    first = _build_features(
        spark=spark,
        houston_canonical=houston_canonical,
        socioeconomic_bronze=socioeconomic_bronze,
        weather_raw=weather_raw,
    ).cache()
    second = _build_features(
        spark=spark,
        houston_canonical=houston_canonical.repartition(3),
        socioeconomic_bronze=socioeconomic_bronze,
        weather_raw=weather_raw,
    )

    assert first.count() == 1
    row = first.first()
    assert row.source_incident_id == "94661522"
    assert row.selected_acs_vintage == 2020
    assert row.selected_acs_release_date == date(2022, 3, 17)
    assert row.tract_geoid == _TRACT_GEOID
    assert row.population == 3754
    assert row.median_household_income == 84327.0
    assert row.socioeconomic_match_found is True
    assert row.temperature_2m_c == pytest.approx(25.9)
    assert row.grid_elevation == pytest.approx(-4.0)
    assert row.weather_request_id == _WEATHER_REQUEST_ID
    assert row.weather_source_row_hash
    assert row.weather_match_found is True
    assert row.lighting_condition == "night"
    assert row.is_daylight is False
    assert (
        row.lighting_definition_version
        == LIGHTING_DEFINITION_VERSION
    )
    assert row.lighting_match_found is True

    first_ids = {
        item.crime_offense_id
        for item in first.select("crime_offense_id").collect()
    }
    second_ids = {
        item.crime_offense_id
        for item in second.select("crime_offense_id").collect()
    }
    assert second.count() == first.count()
    assert second_ids == first_ids

    metrics = log_coverage_metrics(first)
    assert metrics["final_rows"] == 1
    assert metrics["socioeconomic_match_rate"] == 1.0
    assert metrics["weather_match_rate"] == 1.0
    assert metrics["lighting_match_rate"] == 1.0

    quality_report = validate_gold(
        first,
        source_crime_count=1,
        coverage_thresholds=GoldCoverageThresholds(
            tract=1.0,
            socioeconomic=1.0,
            weather=1.0,
            lighting=1.0,
        ),
    )
    assert quality_report.passed
