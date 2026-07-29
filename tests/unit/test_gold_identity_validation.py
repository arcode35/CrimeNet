from __future__ import annotations

from datetime import datetime

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from crimenet.contracts.gold import (
    GoldCoverageThresholds,
    coverage_failures,
    crime_offense_id_from_business_identity,
)
from crimenet.contracts.lighting import (
    LIGHTING_DEFINITION_VERSION,
)
from crimenet.gold.crime_features import (
    attach_tracts,
    build_gold_quality_checks,
    build_location_mapping_lookup,
    log_coverage_metrics,
    prepare_crimes,
    source_coverage_metrics,
    validate_gold_candidate,
)


def _valid_gold_candidate(
    spark: SparkSession,
) -> DataFrame:
    occurred_at = datetime(2025, 1, 2, 14, 37, 22)
    hour = datetime(2025, 1, 2, 14)

    return spark.createDataFrame(
        [
            (
                crime_offense_id_from_business_identity(
                    "source-system||incident||offense"
                ),
                "dallas",
                occurred_at,
                hour,
                hour,
                617_000,
                32.78,
                -96.80,
                LIGHTING_DEFINITION_VERSION,
                "daylight",
                True,
                True,
                True,
                True,
                2023,
                "48113000100",
                72_000.0,
                21.0,
                30.0,
                30.1,
                60.0,
                180.0,
                0.15,
                0.04,
                0.08,
                0.45,
                0.05,
            )
        ],
        (
            "crime_offense_id string, source_system string, "
            "occurred_at timestamp, "
            "weather_timestamp timestamp, "
            "solar_timestamp_hour timestamp, "
            "lighting_query_cell_id long, latitude double, "
            "longitude double, "
            "lighting_definition_version string, "
            "lighting_condition string, is_daylight boolean, "
            "weather_match_found boolean, "
            "lighting_match_found boolean, "
            "socioeconomic_match_found boolean, "
            "selected_acs_vintage int, tract_geoid string, "
            "median_household_income double, "
            "temperature_2m_c double, "
            "solar_elevation_deg double, "
            "apparent_solar_elevation_deg double, "
            "solar_zenith_deg double, "
            "solar_azimuth_deg double, poverty_rate double, "
            "unemployment_rate double, vacancy_rate double, "
            "renter_occupied_rate double, "
            "no_vehicle_rate double"
        ),
    )


def test_gold_id_is_a_stable_domain_hash() -> None:
    identity = "dallas||incident-7||offense-2"

    assert crime_offense_id_from_business_identity(
        identity
    ) == crime_offense_id_from_business_identity(identity)
    assert crime_offense_id_from_business_identity(
        identity
    ) != crime_offense_id_from_business_identity("dallas||incident-7||offense-3")


def test_gold_id_fallback_ignores_file_and_row_metadata(
    spark: SparkSession,
) -> None:
    crimes = (
        spark.createDataFrame(
            [
                (
                    "dallas",
                    "incident-7",
                    "offense-2",
                    "dbfs:/landing/original.csv",
                    "raw-row-hash-before-replay",
                    "2025-01-02 14:37:22",
                    617_000,
                ),
                (
                    "dallas",
                    "incident-7",
                    "offense-2",
                    "dbfs:/replay/renamed.csv",
                    "different-physical-replay-hash",
                    "2025-01-02 14:37:22",
                    617_000,
                ),
            ],
            (
                "source_system string, "
                "source_incident_id string, "
                "source_offense_id string, source_file string, "
                "source_row_hash string, occurred_at_text string, "
                "weather_query_cell_id long"
            ),
        )
        .withColumn(
            "occurred_at",
            F.to_timestamp("occurred_at_text"),
        )
        .drop("occurred_at_text")
    )

    identifiers = {
        row["crime_offense_id"]
        for row in (prepare_crimes(crimes).select("crime_offense_id").collect())
    }

    assert len(identifiers) == 1


def test_coverage_threshold_validation_is_configurable() -> None:
    thresholds = GoldCoverageThresholds(
        weather=0.95,
        lighting=0.90,
        socioeconomic=0.80,
        tract=0.85,
    )
    failures = coverage_failures(
        {
            "weather_match_rate": 0.94,
            "lighting_match_rate": 0.90,
            "socioeconomic_match_rate": 0.81,
            "tract_match_rate": 0.84,
        },
        thresholds,
    )

    assert failures == (
        "weather_match_rate=0.94000000 is below minimum=0.95000000",
        "tract_match_rate=0.84000000 is below minimum=0.85000000",
    )

    with pytest.raises(ValueError, match="between 0 and 1"):
        GoldCoverageThresholds(weather=1.01)


def test_gold_quality_checks_capture_validation_evidence() -> None:
    thresholds = GoldCoverageThresholds(
        weather=0.9,
        lighting=0.8,
        socioeconomic=0.7,
        tract=0.6,
    )
    checks = build_gold_quality_checks(
        {
            "source_rows": 10,
            "final_rows": 10,
            "weather_match_rate": 0.95,
            "lighting_match_rate": 0.85,
            "socioeconomic_match_rate": 0.75,
            "tract_match_rate": 0.65,
        },
        thresholds,
    )
    by_name = {check.check_name: check for check in checks}

    assert by_name["gold_exact_business_key_equality"].passed
    assert by_name["gold_row_cardinality"].observed_value == "source=10, candidate=10"
    assert by_name["gold_weather_match_rate"].expected_threshold == ">=0.90000000"


def test_gold_validation_checks_exact_business_key_set(
    spark: SparkSession,
) -> None:
    candidate = _valid_gold_candidate(spark)
    valid_source = candidate.select("crime_offense_id")

    metrics = validate_gold_candidate(
        valid_source,
        candidate,
        thresholds=GoldCoverageThresholds(
            weather=1.0,
            lighting=1.0,
            socioeconomic=1.0,
            tract=1.0,
        ),
    )
    assert metrics["final_rows"] == 1
    assert metrics["source_rows"] == 1

    wrong_source = valid_source.withColumn(
        "crime_offense_id",
        F.lit("different-logical-key"),
    )
    with pytest.raises(
        RuntimeError,
        match="mismatched business-key set",
    ):
        validate_gold_candidate(
            wrong_source,
            candidate,
            thresholds=GoldCoverageThresholds(),
        )


def test_gold_consumes_versioned_prebuilt_location_mapping(
    spark: SparkSession,
) -> None:
    locations = spark.createDataFrame(
        [(2023, 32.78, -96.80)],
        "tiger_line_year int, latitude double, longitude double",
    )
    mapping = spark.createDataFrame(
        [
            (
                2023,
                32.78,
                -96.80,
                "48113000100",
                "tiger_line_normalization_v1",
                "a" * 64,
                "tract_point_in_polygon_v1",
                "stable-location-key",
                "matched_contains",
                1,
                "mapping-run-7",
            )
        ],
        (
            "tiger_line_year int, latitude double, "
            "longitude double, tract_geoid string, "
            "boundary_definition_version string, "
            "source_archive_sha256 string, "
            "mapping_definition_version string, "
            "location_tract_key string, match_status string, "
            "candidate_match_count int, pipeline_run_id string"
        ),
    )

    lookup = build_location_mapping_lookup(
        locations,
        mapping,
    )
    attached = attach_tracts(
        locations,
        lookup,
    ).first()

    assert attached is not None
    assert attached["tract_geoid"] == "48113000100"
    assert attached["mapping_definition_version"] == "tract_point_in_polygon_v1"
    assert attached["boundary_source_archive_sha256"] == "a" * 64
    assert attached["tract_match_status"] == "matched_contains"

    ambiguous = mapping.withColumn("match_status", F.lit("ambiguous")).withColumn(
        "candidate_match_count", F.lit(2)
    )
    with pytest.raises(RuntimeError, match="ambiguous"):
        build_location_mapping_lookup(
            locations,
            ambiguous,
        )

    with pytest.raises(
        RuntimeError,
        match="mismatched business-key set",
    ):
        build_location_mapping_lookup(
            locations,
            mapping.limit(0),
        )


def test_gold_validation_rejects_duplicate_ids_and_low_coverage(
    spark: SparkSession,
) -> None:
    candidate = _valid_gold_candidate(spark)
    source = candidate.select("crime_offense_id")

    with pytest.raises(
        RuntimeError,
        match="duplicate keys",
    ):
        validate_gold_candidate(
            source,
            candidate.unionByName(candidate),
            thresholds=GoldCoverageThresholds(),
        )

    low_coverage = (
        candidate.withColumn("weather_match_found", F.lit(False))
        .withColumn("lighting_match_found", F.lit(False))
        .withColumn(
            "socioeconomic_match_found",
            F.lit(False),
        )
        .withColumn(
            "tract_geoid",
            F.lit(None).cast("string"),
        )
    )
    with pytest.raises(
        RuntimeError,
        match="failed enrichment coverage",
    ):
        validate_gold_candidate(
            source,
            low_coverage,
            thresholds=GoldCoverageThresholds(
                weather=0.5,
                lighting=0.5,
                socioeconomic=0.5,
                tract=0.5,
            ),
        )

    implausible = candidate.withColumn(
        "temperature_2m_c",
        F.lit(500.0),
    )
    with pytest.raises(
        RuntimeError,
        match="feature ranges",
    ):
        validate_gold_candidate(
            source,
            implausible,
            thresholds=GoldCoverageThresholds(),
        )


@pytest.mark.parametrize(
    ("column_name", "data_type"),
    [
        ("temperature_2m_c", "double"),
        ("solar_elevation_deg", "double"),
        ("median_household_income", "double"),
    ],
)
def test_gold_validation_rejects_matched_rows_without_required_features(
    spark: SparkSession,
    column_name: str,
    data_type: str,
) -> None:
    candidate = _valid_gold_candidate(spark).withColumn(
        column_name,
        F.lit(None).cast(data_type),
    )

    with pytest.raises(RuntimeError, match="feature ranges"):
        validate_gold_candidate(
            candidate.select("crime_offense_id"),
            candidate,
            thresholds=GoldCoverageThresholds(),
        )


def test_gold_coverage_counts_only_usable_enrichment_features(
    spark: SparkSession,
) -> None:
    candidate = (
        _valid_gold_candidate(spark)
        .withColumn(
            "temperature_2m_c",
            F.lit(None).cast("double"),
        )
        .withColumn(
            "solar_elevation_deg",
            F.lit(None).cast("double"),
        )
        .withColumn(
            "median_household_income",
            F.lit(None).cast("double"),
        )
    )

    metrics = log_coverage_metrics(candidate)
    assert metrics["rows_with_weather_record"] == 0
    assert metrics["rows_with_lighting_record"] == 0
    assert metrics["rows_with_socioeconomic_record"] == 0
    assert metrics["weather_match_rate"] == 0.0
    assert metrics["lighting_match_rate"] == 0.0
    assert metrics["socioeconomic_match_rate"] == 0.0

    source_metrics = source_coverage_metrics(candidate)["dallas"]
    assert source_metrics["weather_match_rate"] == 0.0
    assert source_metrics["lighting_match_rate"] == 0.0
    assert source_metrics["socioeconomic_match_rate"] == 0.0


def test_gold_validation_blocks_source_level_coverage_regression(
    spark: SparkSession,
) -> None:
    dallas = _valid_gold_candidate(spark)
    houston = (
        dallas.withColumn(
            "crime_offense_id",
            F.lit("houston-stable-offense-id"),
        )
        .withColumn("source_system", F.lit("houston"))
        .withColumn("weather_match_found", F.lit(False))
    )
    candidate = dallas.unionByName(houston)

    with pytest.raises(
        RuntimeError,
        match="source-level enrichment coverage",
    ):
        validate_gold_candidate(
            candidate.select("crime_offense_id"),
            candidate,
            thresholds=GoldCoverageThresholds(weather=0.5),
        )
