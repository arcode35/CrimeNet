from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pytest
from pyspark.sql import SparkSession

from crimenet.contracts.gold import GoldCoverageThresholds
from crimenet.contracts.lighting import LIGHTING_DEFINITION_VERSION
from crimenet.gold import crime_features
from crimenet.gold.crime_features import (
    FeatureTables,
    attach_eligible_acs_vintage,
    attach_lighting_features,
    attach_socioeconomic_features,
    attach_tracts,
    attach_weather_features,
    build_calendar_ranges,
    build_gold_quality_checks,
    build_lighting_lookup,
    build_location_mapping_lookup,
    build_weather_lookup,
    extract_unique_locations,
    log_coverage_metrics,
    materialize_gold_features,
    prepare_crimes,
    validate_gold_candidate,
)


def test_gold_lookup_pipeline_enriches_without_changing_business_keys(
    spark: SparkSession,
) -> None:
    tables = FeatureTables.from_schemas(
        catalog="crime",
        silver_schema="silver",
        gold_schema="gold",
    )
    assert tables.location_tract_mapping == (
        "crime.silver.crime_location_tract_mapping"
    )
    assert tables.features == "crime.gold.crime_features"

    crimes = spark.createDataFrame(
        [
            (
                "identity-1",
                "dallas",
                "incident-1",
                "offense-1",
                datetime(2024, 5, 3, 19, 37),
                32.7767,
                -96.7970,
                123,
            ),
            (
                "identity-2",
                "houston",
                "incident-2",
                "offense-2",
                datetime(2024, 5, 4, 8, 15),
                29.7604,
                -95.3698,
                456,
            ),
        ],
        """
        business_identity string,
        source_system string,
        source_incident_id string,
        source_offense_id string,
        occurred_at timestamp,
        latitude double,
        longitude double,
        weather_query_cell_id long
        """,
    )
    prepared = prepare_crimes(crimes)

    calendar = spark.createDataFrame(
        [
            (2022, date(2023, 12, 7), 2022, 2020),
            (2023, date(2024, 12, 12), 2023, 2020),
        ],
        """
        acs_vintage int,
        acs_release_date date,
        tiger_line_year int,
        tract_definition_vintage int
        """,
    )
    calendar_ranges = build_calendar_ranges(calendar)
    eligible = attach_eligible_acs_vintage(prepared, calendar_ranges)
    assert {
        row["selected_acs_vintage"]
        for row in eligible.select("selected_acs_vintage").collect()
    } == {2022}

    locations = extract_unique_locations(eligible)
    assert locations.count() == 2
    mapping = spark.createDataFrame(
        [
            (
                2022,
                32.7767,
                -96.7970,
                "48113000100",
                "matched_contains",
                1,
                "boundary-v1",
                "archive-a",
                "mapping-v1",
                "location-key-1",
                "run-map",
                datetime(2024, 6, 1),
            ),
            (
                2022,
                29.7604,
                -95.3698,
                "48201000100",
                "matched_contains",
                1,
                "boundary-v1",
                "archive-a",
                "mapping-v1",
                "location-key-2",
                "run-map",
                datetime(2024, 6, 1),
            ),
        ],
        """
        tiger_line_year int,
        latitude double,
        longitude double,
        tract_geoid string,
        match_status string,
        candidate_match_count int,
        boundary_definition_version string,
        source_archive_sha256 string,
        mapping_definition_version string,
        location_tract_key string,
        pipeline_run_id string,
        mapped_at timestamp
        """,
    )
    mapping_lookup = build_location_mapping_lookup(locations, mapping)
    with_tracts = attach_tracts(eligible, mapping_lookup)
    assert {
        row["tract_match_status"]
        for row in with_tracts.select("tract_match_status").collect()
    } == {"matched_contains"}

    socioeconomic = spark.createDataFrame(
        [
            (
                "48113000100",
                2022,
                "Dallas tract",
                1000,
                50,
                35.0,
                1.0,
                72000.0,
                2500.0,
                0.1,
                0.04,
                0.08,
                0.45,
                0.05,
            )
        ],
        """
        geoid string,
        acs_vintage int,
        geography_name string,
        population long,
        population_moe long,
        median_age double,
        median_age_moe double,
        median_household_income double,
        median_household_income_moe double,
        poverty_rate double,
        unemployment_rate double,
        vacancy_rate double,
        renter_occupied_rate double,
        no_vehicle_rate double
        """,
    )
    with_socioeconomic = attach_socioeconomic_features(
        with_tracts,
        socioeconomic,
    )

    weather = spark.createDataFrame(
        [
            (
                "open_meteo",
                "era5_land",
                6,
                456,
                datetime(2024, 5, 4, 8),
                date(2024, 5, 4),
                21.5,
                150.0,
            ),
            (
                "other",
                "other",
                6,
                123,
                datetime(2024, 5, 3, 19),
                date(2024, 5, 3),
                99.0,
                0.0,
            ),
        ],
        """
        provider string,
        model string,
        h3_resolution int,
        weather_query_cell_id long,
        weather_timestamp timestamp,
        weather_date date,
        temperature_2m_c double,
        grid_elevation double
        """,
    )
    weather_lookup = build_weather_lookup(
        with_socioeconomic,
        weather,
        provider="open_meteo",
        model="era5_land",
        h3_resolution=6,
    )
    with_weather = attach_weather_features(
        with_socioeconomic,
        weather_lookup,
    )

    lighting = spark.createDataFrame(
        [
            (
                123,
                datetime(2024, 5, 3, 19),
                LIGHTING_DEFINITION_VERSION,
                30.0,
                30.1,
                60.0,
                180.0,
                "daylight",
                True,
            )
        ],
        """
        lighting_query_cell_id long,
        solar_timestamp_hour timestamp,
        lighting_definition_version string,
        solar_elevation_deg double,
        apparent_solar_elevation_deg double,
        solar_zenith_deg double,
        solar_azimuth_deg double,
        lighting_condition string,
        is_daylight boolean
        """,
    )
    candidate = attach_lighting_features(
        with_weather,
        build_lighting_lookup(lighting),
    )
    metrics = validate_gold_candidate(
        prepared,
        candidate,
        thresholds=GoldCoverageThresholds(),
    )

    assert metrics["source_rows"] == 2
    assert metrics["final_rows"] == 2
    assert metrics["tract_match_rate"] == 1.0
    assert metrics["socioeconomic_match_rate"] == 0.5
    assert metrics["weather_match_rate"] == 0.5
    assert metrics["lighting_match_rate"] == 0.5
    assert len(
        build_gold_quality_checks(
            metrics,
            GoldCoverageThresholds(),
        )
    ) == 17
    assert log_coverage_metrics(candidate)["final_rows"] == 2


class _FakeWriter:
    def __init__(self, calls: list[tuple[str, object]]) -> None:
        self.calls = calls

    def format(self, value: str) -> _FakeWriter:
        self.calls.append(("format", value))
        return self

    def mode(self, value: str) -> _FakeWriter:
        self.calls.append(("mode", value))
        return self

    def option(self, key: str, value: object) -> _FakeWriter:
        self.calls.append((key, value))
        return self

    def saveAsTable(self, table: str) -> None:
        self.calls.append(("save", table))


class _FakeCandidate:
    def __init__(self, calls: list[tuple[str, object]]) -> None:
        self.write = _FakeWriter(calls)


class _FakeSpark:
    def __init__(
        self,
        staged: object,
        calls: list[tuple[str, object]],
    ) -> None:
        self.staged = staged
        self.calls = calls

    def table(self, table: str) -> object:
        self.calls.append(("table", table))
        return self.staged


def test_gold_materialization_audits_success_before_promotion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    candidate = _FakeCandidate(calls)
    staged = object()
    spark = _FakeSpark(staged, calls)
    metrics = {
        "source_rows": 1,
        "final_rows": 1,
        "weather_match_rate": 1.0,
        "lighting_match_rate": 1.0,
        "socioeconomic_match_rate": 1.0,
        "tract_match_rate": 1.0,
    }
    monkeypatch.setattr(
        crime_features,
        "validate_gold_candidate",
        lambda source, actual, **_kwargs: (
            metrics
            if source == "source" and actual is staged
            else pytest.fail("unexpected validation inputs")
        ),
    )
    monkeypatch.setattr(
        crime_features,
        "quality_results_dataframe",
        lambda _spark, **kwargs: ("quality", kwargs),
    )
    monkeypatch.setattr(
        crime_features,
        "merge_quality_results",
        lambda _spark, **kwargs: calls.append(("quality", kwargs)),
    )
    monkeypatch.setattr(
        crime_features,
        "promote_staged_delta_table",
        lambda _spark, **kwargs: calls.append(("promote", kwargs)),
    )
    monkeypatch.setattr(
        crime_features,
        "drop_staging_table",
        lambda _spark, table: calls.append(("drop", table)),
    )

    result = materialize_gold_features(
        spark,  # type: ignore[arg-type]
        source_dataframe="source",  # type: ignore[arg-type]
        candidate_dataframe=candidate,  # type: ignore[arg-type]
        target_table="crime.gold.crime_features",
        thresholds=GoldCoverageThresholds(),
        pipeline_run_id="job/run 1",
        quality_results_table="crime.quality.results",
    )
    assert result == metrics
    assert [name for name, _ in calls][-3:] == [
        "quality",
        "promote",
        "drop",
    ]
    assert ("save", "crime.gold.crime_features__staging__job_run_1") in calls


def test_gold_materialization_audits_failure_and_never_promotes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    candidate = _FakeCandidate(calls)
    spark = _FakeSpark(object(), calls)

    def fail_validation(*_args: Any, **_kwargs: Any) -> dict[str, object]:
        raise RuntimeError("candidate invalid")

    monkeypatch.setattr(
        crime_features,
        "validate_gold_candidate",
        fail_validation,
    )
    monkeypatch.setattr(
        crime_features,
        "quality_results_dataframe",
        lambda _spark, **kwargs: ("failed-quality", kwargs),
    )
    monkeypatch.setattr(
        crime_features,
        "merge_quality_results",
        lambda _spark, **kwargs: calls.append(("quality", kwargs)),
    )
    monkeypatch.setattr(
        crime_features,
        "promote_staged_delta_table",
        lambda *_args, **_kwargs: calls.append(("promote", None)),
    )
    monkeypatch.setattr(
        crime_features,
        "drop_staging_table",
        lambda _spark, table: calls.append(("drop", table)),
    )

    with pytest.raises(RuntimeError, match="candidate invalid"):
        materialize_gold_features(
            spark,  # type: ignore[arg-type]
            source_dataframe=object(),  # type: ignore[arg-type]
            candidate_dataframe=candidate,  # type: ignore[arg-type]
            target_table="crime.gold.crime_features",
            thresholds=GoldCoverageThresholds(),
            pipeline_run_id="failed",
            quality_results_table="crime.quality.results",
        )
    assert any(name == "quality" for name, _ in calls)
    assert not any(name == "promote" for name, _ in calls)
    assert calls[-1][0] == "drop"
