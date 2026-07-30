from __future__ import annotations

from datetime import date, datetime

import pytest
from pyspark.sql import SparkSession
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
    extract_unique_locations,
    log_coverage_metrics,
    prepare_crimes,
    validate_crime_identities,
)

pytestmark = pytest.mark.unit


def _crime_dataframe(
    spark: SparkSession,
    rows: list[tuple[object, ...]],
):
    return spark.createDataFrame(
        rows,
        schema=(
            "crime_offense_id string, source_city string, "
            "source_incident_id string, source_record_id string, "
            "offense_code string, source_row_hash string, "
            "occurred_at timestamp, latitude double, longitude double, "
            "weather_query_cell_id long"
        ),
    )


def _calendar_dataframe(spark: SparkSession):
    return spark.createDataFrame(
        [
            (2019, date(2020, 12, 10), 2019, "2019"),
            (2020, date(2022, 3, 17), 2020, "2020"),
            (2021, date(2022, 12, 8), 2021, "2021"),
        ],
        schema=(
            "acs_vintage int, acs_release_date date, "
            "tiger_line_year int, tract_definition_vintage string"
        ),
    )


def test_prepare_crimes_adds_time_grains_without_replacing_identity(
    spark: SparkSession,
) -> None:
    crimes = _crime_dataframe(
        spark,
        [
            (
                "crime-1",
                "houston",
                "94661522",
                "row-1",
                "13A",
                "hash-1",
                datetime(2022, 7, 17, 8, 37, 42),
                29.728988,
                -95.527629,
                604686043911815167,
            )
        ],
    )

    row = prepare_crimes(crimes).first()

    assert row.crime_offense_id == "crime-1"
    assert row.occurred_date == date(2022, 7, 17)
    assert row.occurred_at_hour == datetime(2022, 7, 17, 8)


def test_prepare_crimes_reports_missing_required_columns(
    spark: SparkSession,
) -> None:
    missing_timestamp = spark.createDataFrame(
        [("crime-1",)],
        "crime_offense_id string",
    )

    with pytest.raises(ValueError, match="occurred_at"):
        prepare_crimes(missing_timestamp)


def test_calendar_ranges_are_ordered_non_overlapping_and_leakage_safe(
    spark: SparkSession,
) -> None:
    ranges = build_calendar_ranges(_calendar_dataframe(spark))
    actual = {
        row.acs_vintage: (
            row.eligible_start_date,
            row.eligible_end_date,
        )
        for row in ranges.collect()
    }

    assert actual == {
        2019: (date(2020, 12, 11), date(2022, 3, 17)),
        2020: (date(2022, 3, 18), date(2022, 12, 8)),
        2021: (date(2022, 12, 9), None),
    }

    crimes = _crime_dataframe(
        spark,
        [
            (
                "release-day",
                "houston",
                "1",
                "1",
                "13A",
                "hash-1",
                datetime(2022, 3, 17, 12),
                29.7,
                -95.5,
                1,
            ),
            (
                "next-day",
                "houston",
                "2",
                "2",
                "13A",
                "hash-2",
                datetime(2022, 3, 18, 12),
                29.7,
                -95.5,
                1,
            ),
        ],
    )

    selected = {
        row.crime_offense_id: row.selected_acs_vintage
        for row in attach_eligible_acs_vintage(
            prepare_crimes(crimes),
            ranges,
        ).collect()
    }

    assert selected == {"release-day": 2019, "next-day": 2020}


def test_calendar_rejects_duplicate_or_out_of_order_releases(
    spark: SparkSession,
) -> None:
    duplicate = spark.createDataFrame(
        [
            (2020, date(2022, 3, 17), 2020, "2020"),
            (2021, date(2022, 3, 17), 2021, "2021"),
        ],
        schema=(
            "acs_vintage int, acs_release_date date, "
            "tiger_line_year int, tract_definition_vintage string"
        ),
    )
    out_of_order = spark.createDataFrame(
        [
            (2021, date(2022, 3, 17), 2021, "2021"),
            (2020, date(2022, 12, 8), 2020, "2020"),
        ],
        schema=duplicate.schema,
    )

    with pytest.raises(RuntimeError, match="duplicate keys"):
        build_calendar_ranges(duplicate)

    with pytest.raises(ValueError, match="increase"):
        build_calendar_ranges(out_of_order)


def test_location_extraction_and_mapping_preserve_crime_cardinality(
    spark: SparkSession,
) -> None:
    crimes = spark.createDataFrame(
        [
            ("one", 2020, 29.7, -95.5),
            ("duplicate-location", 2020, 29.7, -95.5),
            ("invalid-latitude", 2020, 91.0, -95.5),
            ("missing-year", None, 30.0, -96.0),
        ],
        "crime_offense_id string, tiger_line_year int, "
        "latitude double, longitude double",
    )

    locations = extract_unique_locations(crimes)
    assert locations.collect()[0].asDict() == {
        "tiger_line_year": 2020,
        "latitude": 29.7,
        "longitude": -95.5,
    }

    mapping = spark.createDataFrame(
        [(2020, 29.7, -95.5, "48491020330")],
        "tiger_line_year int, latitude double, longitude double, "
        "tract_geoid string",
    )
    attached = attach_tracts(crimes, mapping)

    assert attached.count() == crimes.count()
    assert (
        attached.filter("crime_offense_id = 'one'")
        .first()
        .tract_geoid
        == "48491020330"
    )
    assert (
        attached.filter("crime_offense_id = 'invalid-latitude'")
        .first()
        .tract_geoid
        is None
    )


def test_duplicate_tract_mapping_fails_before_join(
    spark: SparkSession,
) -> None:
    crimes = spark.createDataFrame(
        [("one", 2020, 29.7, -95.5)],
        "crime_offense_id string, tiger_line_year int, "
        "latitude double, longitude double",
    )
    mapping = spark.createDataFrame(
        [
            (2020, 29.7, -95.5, "one"),
            (2020, 29.7, -95.5, "two"),
        ],
        "tiger_line_year int, latitude double, longitude double, "
        "tract_geoid string",
    )

    with pytest.raises(RuntimeError, match="duplicate keys"):
        attach_tracts(crimes, mapping)


def _socioeconomic_dataframe(spark: SparkSession):
    return spark.createDataFrame(
        [
            (
                "48491020330",
                2020,
                "Census Tract 203.30, Williamson County, Texas",
                3754,
                446,
                41.2,
                2.5,
                84327.0,
                11086.0,
                114 / 3742,
                75 / 2132,
                85 / 1446,
                292 / 1361,
                12 / 1361,
            )
        ],
        schema=(
            "geoid string, acs_vintage int, geography_name string, "
            "population long, population_moe long, median_age double, "
            "median_age_moe double, median_household_income double, "
            "median_household_income_moe double, poverty_rate double, "
            "unemployment_rate double, vacancy_rate double, "
            "renter_occupied_rate double, no_vehicle_rate double"
        ),
    )


def test_socioeconomic_join_sets_match_flags_without_multiplication(
    spark: SparkSession,
) -> None:
    crimes = spark.createDataFrame(
        [
            ("matched", "48491020330", 2020),
            ("missing", "00000000000", 2020),
        ],
        "crime_offense_id string, tract_geoid string, "
        "selected_acs_vintage int",
    )

    enriched = attach_socioeconomic_features(
        crimes,
        _socioeconomic_dataframe(spark),
    )
    actual = {
        row.crime_offense_id: (
            row.socioeconomic_match_found,
            row.population,
            row.median_household_income,
        )
        for row in enriched.collect()
    }

    assert enriched.count() == 2
    assert actual == {
        "matched": (True, 3754, 84327.0),
        "missing": (False, None, None),
    }


def test_duplicate_socioeconomic_keys_fail_before_join(
    spark: SparkSession,
) -> None:
    crimes = spark.createDataFrame(
        [("one", "48491020330", 2020)],
        "crime_offense_id string, tract_geoid string, "
        "selected_acs_vintage int",
    )
    lookup = _socioeconomic_dataframe(spark)

    with pytest.raises(RuntimeError, match="duplicate keys"):
        attach_socioeconomic_features(
            crimes,
            lookup.unionByName(lookup),
        )


def _weather_dataframe(spark: SparkSession):
    return spark.createDataFrame(
        [
            (
                "open_meteo",
                "era5_land",
                "request-1",
                604686043911815167,
                6,
                datetime(2022, 7, 17, 8),
                date(2022, 7, 17),
                25.9,
                -4.0,
                "weather-hash-1",
            ),
            (
                "other_provider",
                "era5_land",
                "request-2",
                604686043911815167,
                6,
                datetime(2022, 7, 17, 8),
                date(2022, 7, 17),
                99.0,
                -4.0,
                "weather-hash-2",
            ),
        ],
        schema=(
            "provider string, model string, request_id string, "
            "weather_query_cell_id long, h3_resolution int, "
            "weather_timestamp timestamp, weather_date date, "
            "temperature_2m_c double, grid_elevation double, "
            "source_row_hash string"
        ),
    )


def test_weather_lookup_filters_and_joins_at_hourly_grain(
    spark: SparkSession,
) -> None:
    features = spark.createDataFrame(
        [
            (
                "matched",
                604686043911815167,
                datetime(2022, 7, 17, 8),
                date(2022, 7, 17),
            ),
            (
                "missing",
                1,
                datetime(2022, 7, 17, 8),
                date(2022, 7, 17),
            ),
        ],
        "crime_offense_id string, weather_query_cell_id long, "
        "occurred_at_hour timestamp, occurred_date date",
    )

    lookup = build_weather_lookup(
        features,
        _weather_dataframe(spark),
        provider="open_meteo",
        model="era5_land",
        h3_resolution=6,
    )
    enriched = attach_weather_features(features, lookup)
    actual = {
        row.crime_offense_id: (
            row.weather_match_found,
            row.temperature_2m_c,
            row.weather_provider,
            row.weather_request_id,
        )
        for row in enriched.collect()
    }

    assert actual == {
        "matched": (True, 25.9, "open_meteo", "request-1"),
        "missing": (False, None, None, None),
    }


def test_duplicate_weather_keys_fail_before_join(
    spark: SparkSession,
) -> None:
    features = spark.createDataFrame(
        [
            (
                "one",
                604686043911815167,
                datetime(2022, 7, 17, 8),
                date(2022, 7, 17),
            )
        ],
        "crime_offense_id string, weather_query_cell_id long, "
        "occurred_at_hour timestamp, occurred_date date",
    )
    weather = _weather_dataframe(spark).filter(
        "provider = 'open_meteo'"
    )

    with pytest.raises(RuntimeError, match="duplicate keys"):
        build_weather_lookup(
            features,
            weather.unionByName(weather),
            provider="open_meteo",
            model="era5_land",
            h3_resolution=6,
        )


def _lighting_dataframe(spark: SparkSession):
    schema = (
        "weather_query_cell_id long, solar_timestamp_hour timestamp, "
        "lighting_definition_version string, solar_elevation_deg double, "
        "apparent_solar_elevation_deg double, solar_zenith_deg double, "
        "solar_azimuth_deg double, lighting_condition string, "
        "is_daylight boolean, pvlib_version string"
    )
    return spark.createDataFrame(
        [
            (
                604686043911815167,
                datetime(2022, 7, 17, 8),
                LIGHTING_DEFINITION_VERSION,
                -26.0,
                -26.0,
                116.0,
                335.0,
                "night",
                False,
                "0.15.2",
            ),
            (
                604686043911815167,
                datetime(2022, 7, 17, 8),
                "retired-definition",
                10.0,
                10.0,
                80.0,
                120.0,
                "daylight",
                True,
                "0.14.0",
            ),
        ],
        schema=schema,
    )


def test_lighting_lookup_uses_only_active_version_and_preserves_missing_rows(
    spark: SparkSession,
) -> None:
    features = spark.createDataFrame(
        [
            (
                "matched",
                604686043911815167,
                datetime(2022, 7, 17, 8),
            ),
            ("missing", 1, datetime(2022, 7, 17, 8)),
        ],
        "crime_offense_id string, weather_query_cell_id long, "
        "occurred_at_hour timestamp",
    )

    lookup = build_lighting_lookup(_lighting_dataframe(spark))
    enriched = attach_lighting_features(features, lookup)
    actual = {
        row.crime_offense_id: (
            row.lighting_match_found,
            row.lighting_condition,
            row.lighting_definition_version,
            row.lighting_pvlib_version,
        )
        for row in enriched.collect()
    }

    assert actual == {
        "matched": (
            True,
            "night",
            LIGHTING_DEFINITION_VERSION,
            "0.15.2",
        ),
        "missing": (False, None, None, None),
    }


def test_duplicate_active_lighting_keys_fail_before_join(
    spark: SparkSession,
) -> None:
    active = _lighting_dataframe(spark).filter(
        f"lighting_definition_version = '{LIGHTING_DEFINITION_VERSION}'"
    )

    with pytest.raises(RuntimeError, match="duplicate keys"):
        build_lighting_lookup(active.unionByName(active))


def test_crime_identity_validation_rejects_nulls_and_duplicates(
    spark: SparkSession,
) -> None:
    valid = _crime_dataframe(
        spark,
        [
            (
                "one",
                "dallas",
                "incident",
                "record",
                "23F",
                "hash",
                datetime(2022, 1, 1),
                32.7,
                -96.8,
                1,
            )
        ],
    )
    validate_crime_identities(valid)

    with pytest.raises(RuntimeError, match="could not be assigned"):
        validate_crime_identities(
            valid.withColumn(
                "crime_offense_id",
                F.lit(None).cast("string"),
            )
        )

    with pytest.raises(RuntimeError, match="duplicate keys"):
        validate_crime_identities(valid.unionByName(valid))


def test_coverage_metrics_report_exact_counts_and_rates(
    spark: SparkSession,
) -> None:
    features = spark.createDataFrame(
        [
            (2020, "tract", True, 80000.0, True, 25.0, True),
            (None, None, False, None, False, None, False),
        ],
        schema=(
            "selected_acs_vintage int, tract_geoid string, "
            "socioeconomic_match_found boolean, "
            "median_household_income double, weather_match_found boolean, "
            "temperature_2m_c double, lighting_match_found boolean"
        ),
    )

    metrics = log_coverage_metrics(features)

    assert metrics == {
        "final_rows": 2,
        "rows_with_eligible_acs_vintage": 1,
        "rows_with_tract": 1,
        "rows_with_socioeconomic_record": 1,
        "rows_with_non_null_income": 1,
        "rows_with_weather_record": 1,
        "rows_with_non_null_temperature": 1,
        "tract_match_rate": 0.5,
        "socioeconomic_match_rate": 0.5,
        "weather_match_rate": 0.5,
        "rows_with_lighting_record": 1,
        "lighting_match_rate": 0.5,
    }
