from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    BooleanType,
    DateType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from crimenet.contracts.lighting import LIGHTING_DEFINITION_VERSION
from crimenet.contracts.silver import SILVER_SCHEMA
from crimenet.quality import (
    GoldCoverageThresholds,
    QualityValidationError,
    validate_gold,
    validate_lighting,
    validate_silver_crime,
    validate_socioeconomic,
    validate_weather,
)
from crimenet.quality.validators import SOCIOECONOMIC_RATE_COLUMNS

pytestmark = pytest.mark.unit


def _failure_names(error: QualityValidationError) -> set[str]:
    return {
        check.check_name
        for check in error.report.blocking_failures
    }


def _assert_fails(
    validator: Any,
    dataframe: DataFrame,
    expected_check: str,
    **kwargs: Any,
) -> QualityValidationError:
    with pytest.raises(QualityValidationError) as caught:
        validator(
            dataframe,
            maximum_examples=1,
            **kwargs,
        )
    assert expected_check in _failure_names(caught.value)
    failure = next(
        check
        for check in caught.value.report.blocking_failures
        if check.check_name == expected_check
    )
    assert len(failure.examples) <= 1
    return caught.value


def _silver_schema(
    *,
    crime_id_type: Any | None = None,
) -> StructType:
    resolved_crime_id_type = crime_id_type or StringType()
    return StructType(
        [
            *(
                StructField(
                    field.name,
                    field.dataType,
                    nullable=True,
                )
                for field in SILVER_SCHEMA.fields
            ),
            StructField(
                "crime_offense_id",
                resolved_crime_id_type,
                nullable=True,
            ),
        ]
    )


def _silver_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source_city": "dallas",
        "source_record_id": "record-1",
        "source_incident_id": "incident-1",
        "offense_code": "220",
        "offense_name": "Burglary",
        "offense_description": "Burglary",
        "occurred_at": datetime(2024, 1, 2, 3, 0),
        "reported_at": datetime(2024, 1, 2, 4, 0),
        "updated_at": datetime(2024, 1, 3, 4, 0),
        "offense_count": 1,
        "address": "1 Main St",
        "city": "Dallas",
        "state": "TX",
        "postal_code": "75001",
        "beat": "1",
        "premise_type": "Residence",
        "latitude": 32.78,
        "longitude": -96.8,
        "alternate_latitude": None,
        "alternate_longitude": None,
        "source_x_coordinate": None,
        "source_y_coordinate": None,
        "source_file": "fixture.csv",
        "source_row_hash": "a" * 64,
        "crime_offense_id": "crime-1",
    }
    row.update(overrides)
    return row


def _silver_dataframe(
    spark: SparkSession,
    *rows: dict[str, Any],
    schema: StructType | None = None,
) -> DataFrame:
    return spark.createDataFrame(
        list(rows) or [_silver_row()],
        schema=schema or _silver_schema(),
    )


def _weather_schema() -> StructType:
    return StructType(
        [
            StructField("provider", StringType(), True),
            StructField("model", StringType(), True),
            StructField("h3_resolution", IntegerType(), True),
            StructField("weather_query_cell_id", LongType(), True),
            StructField("weather_timestamp", TimestampType(), True),
            StructField("temperature_2m_c", DoubleType(), True),
        ]
    )


def _weather_row(**overrides: Any) -> dict[str, Any]:
    row = {
        "provider": "open_meteo",
        "model": "era5_land",
        "h3_resolution": 6,
        "weather_query_cell_id": 601,
        "weather_timestamp": datetime(2024, 1, 2, 3, 0),
        "temperature_2m_c": 12.5,
    }
    row.update(overrides)
    return row


def _weather_dataframe(
    spark: SparkSession,
    *rows: dict[str, Any],
) -> DataFrame:
    return spark.createDataFrame(
        list(rows) or [_weather_row()],
        schema=_weather_schema(),
    )


def _socioeconomic_schema() -> StructType:
    return StructType(
        [
            StructField("geoid", StringType(), True),
            StructField("acs_vintage", IntegerType(), True),
            *(
                StructField(column, DoubleType(), True)
                for column in SOCIOECONOMIC_RATE_COLUMNS
            ),
        ]
    )


def _socioeconomic_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "geoid": "48113000100",
        "acs_vintage": 2022,
        "poverty_rate": 0.1,
        "unemployment_rate": 0.05,
        "vacancy_rate": 0.08,
        "renter_occupied_rate": 0.4,
        "no_vehicle_rate": 0.03,
    }
    row.update(overrides)
    return row


def _socioeconomic_dataframe(
    spark: SparkSession,
    *rows: dict[str, Any],
) -> DataFrame:
    return spark.createDataFrame(
        list(rows) or [_socioeconomic_row()],
        schema=_socioeconomic_schema(),
    )


def _lighting_schema() -> StructType:
    return StructType(
        [
            StructField("weather_query_cell_id", LongType(), True),
            StructField("solar_timestamp_hour", TimestampType(), True),
            StructField(
                "lighting_definition_version",
                StringType(),
                True,
            ),
            StructField("query_latitude", DoubleType(), True),
            StructField("query_longitude", DoubleType(), True),
            StructField("solar_elevation_deg", DoubleType(), True),
            StructField(
                "apparent_solar_elevation_deg",
                DoubleType(),
                True,
            ),
            StructField("solar_zenith_deg", DoubleType(), True),
            StructField("solar_azimuth_deg", DoubleType(), True),
            StructField("lighting_condition", StringType(), True),
            StructField("is_daylight", BooleanType(), True),
            StructField("pvlib_version", StringType(), True),
        ]
    )


def _lighting_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "weather_query_cell_id": 601,
        "solar_timestamp_hour": datetime(2024, 1, 2, 18, 0),
        "lighting_definition_version": LIGHTING_DEFINITION_VERSION,
        "query_latitude": 32.78,
        "query_longitude": -96.8,
        "solar_elevation_deg": 20.0,
        "apparent_solar_elevation_deg": 20.1,
        "solar_zenith_deg": 70.0,
        "solar_azimuth_deg": 180.0,
        "lighting_condition": "daylight",
        "is_daylight": True,
        "pvlib_version": "0.15.2",
    }
    row.update(overrides)
    return row


def _lighting_dataframe(
    spark: SparkSession,
    *rows: dict[str, Any],
) -> DataFrame:
    return spark.createDataFrame(
        list(rows) or [_lighting_row()],
        schema=_lighting_schema(),
    )


def _gold_schema() -> StructType:
    return StructType(
        [
            StructField("crime_offense_id", StringType(), True),
            StructField("occurred_date", DateType(), True),
            StructField("selected_acs_vintage", IntegerType(), True),
            StructField("selected_acs_release_date", DateType(), True),
            StructField("tract_geoid", StringType(), True),
            StructField(
                "socioeconomic_match_found",
                BooleanType(),
                True,
            ),
            StructField("weather_match_found", BooleanType(), True),
            StructField("lighting_match_found", BooleanType(), True),
            StructField("weather_provider", StringType(), True),
            StructField("weather_model", StringType(), True),
            StructField(
                "weather_h3_resolution",
                IntegerType(),
                True,
            ),
            StructField("weather_request_id", StringType(), True),
            StructField(
                "weather_source_row_hash",
                StringType(),
                True,
            ),
            StructField(
                "lighting_definition_version",
                StringType(),
                True,
            ),
            StructField(
                "lighting_pvlib_version",
                StringType(),
                True,
            ),
        ]
    )


def _gold_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "crime_offense_id": "crime-1",
        "occurred_date": date(2024, 1, 2),
        "selected_acs_vintage": 2022,
        "selected_acs_release_date": date(2023, 12, 7),
        "tract_geoid": "48113000100",
        "socioeconomic_match_found": True,
        "weather_match_found": True,
        "lighting_match_found": True,
        "weather_provider": "open_meteo",
        "weather_model": "era5_land",
        "weather_h3_resolution": 6,
        "weather_request_id": "request-1",
        "weather_source_row_hash": "b" * 64,
        "lighting_definition_version": LIGHTING_DEFINITION_VERSION,
        "lighting_pvlib_version": "0.15.2",
    }
    row.update(overrides)
    return row


def _gold_dataframe(
    spark: SparkSession,
    *rows: dict[str, Any],
) -> DataFrame:
    return spark.createDataFrame(
        list(rows) or [_gold_row()],
        schema=_gold_schema(),
    )


def test_silver_crime_quality_passes_all_checks(
    spark: SparkSession,
) -> None:
    dataframe = _silver_dataframe(
        spark,
        _silver_row(),
        _silver_row(
            source_city="houston",
            source_record_id="record-2",
            source_incident_id="incident-2",
            crime_offense_id="crime-2",
            occurred_at=None,
            latitude=None,
            longitude=None,
        ),
    )

    report = validate_silver_crime(
        dataframe,
        minimum_occurred_at_coverage=0.5,
    )

    assert report.passed
    assert {check.check_name for check in report.checks} == {
        "required_columns",
        "canonical_schema",
        "nonnull_crime_offense_id",
        "recognized_source_city",
        "nonnull_source_row_hash",
        "valid_coordinates",
        "unique_crime_offense_id",
        "occurred_at_coverage",
    }


@pytest.mark.parametrize(
    ("overrides", "expected_check"),
    [
        ({"crime_offense_id": None}, "nonnull_crime_offense_id"),
        ({"source_city": "austin"}, "recognized_source_city"),
        ({"source_row_hash": None}, "nonnull_source_row_hash"),
        ({"latitude": 91.0}, "valid_coordinates"),
        ({"longitude": None}, "valid_coordinates"),
    ],
)
def test_silver_crime_row_checks_fail_clearly(
    spark: SparkSession,
    overrides: dict[str, Any],
    expected_check: str,
) -> None:
    _assert_fails(
        validate_silver_crime,
        _silver_dataframe(spark, _silver_row(**overrides)),
        expected_check,
    )


def test_silver_crime_rejects_duplicate_ids(
    spark: SparkSession,
) -> None:
    dataframe = _silver_dataframe(
        spark,
        _silver_row(),
        _silver_row(source_record_id="record-2"),
    )
    _assert_fails(
        validate_silver_crime,
        dataframe,
        "unique_crime_offense_id",
    )


def test_silver_crime_enforces_occurred_at_coverage(
    spark: SparkSession,
) -> None:
    dataframe = _silver_dataframe(
        spark,
        _silver_row(occurred_at=None),
    )
    error = _assert_fails(
        validate_silver_crime,
        dataframe,
        "occurred_at_coverage",
        minimum_occurred_at_coverage=0.5,
    )
    check = next(
        result
        for result in error.report.checks
        if result.check_name == "occurred_at_coverage"
    )
    assert check.metric_value == 0.0
    assert check.threshold == 0.5


def test_silver_crime_requires_canonical_schema(
    spark: SparkSession,
) -> None:
    dataframe = _silver_dataframe(
        spark,
        _silver_row(crime_offense_id=1),
        schema=_silver_schema(crime_id_type=LongType()),
    )
    _assert_fails(
        validate_silver_crime,
        dataframe,
        "canonical_schema",
    )


def test_silver_crime_reports_missing_columns(
    spark: SparkSession,
) -> None:
    dataframe = spark.createDataFrame([("crime-1",)], ["crime_offense_id"])
    _assert_fails(
        validate_silver_crime,
        dataframe,
        "required_columns",
    )


def test_weather_quality_passes_all_checks(
    spark: SparkSession,
) -> None:
    dataframe = _weather_dataframe(
        spark,
        _weather_row(),
        _weather_row(
            weather_query_cell_id=602,
            temperature_2m_c=None,
        ),
    )
    report = validate_weather(dataframe)
    assert report.passed
    assert {check.check_name for check in report.checks} == {
        "required_columns",
        "nonnull_weather_keys",
        "hour_aligned_weather_timestamp",
        "recognized_weather_provider",
        "recognized_weather_model",
        "recognized_weather_h3_resolution",
        "temperature_bounds",
        "unique_weather_keys",
    }


@pytest.mark.parametrize(
    ("overrides", "expected_check"),
    [
        ({"weather_query_cell_id": None}, "nonnull_weather_keys"),
        (
            {"weather_timestamp": datetime(2024, 1, 2, 3, 30)},
            "hour_aligned_weather_timestamp",
        ),
        ({"provider": "other"}, "recognized_weather_provider"),
        ({"model": "other"}, "recognized_weather_model"),
        (
            {"h3_resolution": 7},
            "recognized_weather_h3_resolution",
        ),
        ({"temperature_2m_c": 71.0}, "temperature_bounds"),
    ],
)
def test_weather_row_checks_fail_clearly(
    spark: SparkSession,
    overrides: dict[str, Any],
    expected_check: str,
) -> None:
    _assert_fails(
        validate_weather,
        _weather_dataframe(spark, _weather_row(**overrides)),
        expected_check,
    )


def test_weather_rejects_duplicate_keys(
    spark: SparkSession,
) -> None:
    dataframe = _weather_dataframe(
        spark,
        _weather_row(),
        _weather_row(temperature_2m_c=13.0),
    )
    _assert_fails(
        validate_weather,
        dataframe,
        "unique_weather_keys",
    )


def test_weather_reports_missing_columns(
    spark: SparkSession,
) -> None:
    _assert_fails(
        validate_weather,
        spark.createDataFrame([("open_meteo",)], ["provider"]),
        "required_columns",
    )


def test_socioeconomic_quality_passes_all_checks(
    spark: SparkSession,
) -> None:
    dataframe = _socioeconomic_dataframe(
        spark,
        _socioeconomic_row(),
        _socioeconomic_row(
            geoid="48113000200",
            acs_vintage=2021,
            poverty_rate=None,
        ),
    )
    report = validate_socioeconomic(dataframe)
    assert report.passed
    expected = {
        "required_columns",
        "nonnull_socioeconomic_keys",
        "valid_tract_geoid",
        "valid_acs_vintage",
        "unique_socioeconomic_keys",
        *(
            f"{column}_domain"
            for column in SOCIOECONOMIC_RATE_COLUMNS
        ),
    }
    assert {check.check_name for check in report.checks} == expected


@pytest.mark.parametrize(
    ("overrides", "expected_check"),
    [
        ({"geoid": None}, "nonnull_socioeconomic_keys"),
        ({"geoid": "48113"}, "valid_tract_geoid"),
        ({"acs_vintage": 2008}, "valid_acs_vintage"),
        ({"poverty_rate": -0.1}, "poverty_rate_domain"),
        (
            {"unemployment_rate": 1.1},
            "unemployment_rate_domain",
        ),
        ({"vacancy_rate": 1.1}, "vacancy_rate_domain"),
        (
            {"renter_occupied_rate": 1.1},
            "renter_occupied_rate_domain",
        ),
        ({"no_vehicle_rate": 1.1}, "no_vehicle_rate_domain"),
    ],
)
def test_socioeconomic_row_checks_fail_clearly(
    spark: SparkSession,
    overrides: dict[str, Any],
    expected_check: str,
) -> None:
    _assert_fails(
        validate_socioeconomic,
        _socioeconomic_dataframe(
            spark,
            _socioeconomic_row(**overrides),
        ),
        expected_check,
    )


def test_socioeconomic_rejects_duplicate_keys(
    spark: SparkSession,
) -> None:
    dataframe = _socioeconomic_dataframe(
        spark,
        _socioeconomic_row(),
        _socioeconomic_row(poverty_rate=0.2),
    )
    _assert_fails(
        validate_socioeconomic,
        dataframe,
        "unique_socioeconomic_keys",
    )


def test_socioeconomic_reports_missing_columns(
    spark: SparkSession,
) -> None:
    _assert_fails(
        validate_socioeconomic,
        spark.createDataFrame([("48113000100",)], ["geoid"]),
        "required_columns",
    )


def test_lighting_quality_passes_all_checks(
    spark: SparkSession,
) -> None:
    dataframe = _lighting_dataframe(
        spark,
        _lighting_row(),
        _lighting_row(
            weather_query_cell_id=602,
            solar_elevation_deg=-7.0,
            apparent_solar_elevation_deg=-6.8,
            solar_zenith_deg=97.0,
            solar_azimuth_deg=220.0,
            lighting_condition="nautical_twilight",
            is_daylight=False,
        ),
    )
    report = validate_lighting(dataframe)
    assert report.passed
    assert {check.check_name for check in report.checks} == {
        "required_columns",
        "nonnull_lighting_keys",
        "active_lighting_definition",
        "valid_lighting_coordinates",
        "solar_elevation_domain",
        "apparent_solar_elevation_domain",
        "solar_zenith_domain",
        "solar_azimuth_domain",
        "recognized_lighting_condition",
        "lighting_classification_consistency",
        "nonnull_pvlib_version",
        "unique_lighting_keys",
    }


@pytest.mark.parametrize(
    ("overrides", "expected_check"),
    [
        (
            {"solar_timestamp_hour": None},
            "nonnull_lighting_keys",
        ),
        (
            {"lighting_definition_version": "old"},
            "active_lighting_definition",
        ),
        ({"query_latitude": 91.0}, "valid_lighting_coordinates"),
        ({"solar_elevation_deg": 91.0}, "solar_elevation_domain"),
        (
            {"apparent_solar_elevation_deg": -91.0},
            "apparent_solar_elevation_domain",
        ),
        ({"solar_zenith_deg": 181.0}, "solar_zenith_domain"),
        ({"solar_azimuth_deg": 361.0}, "solar_azimuth_domain"),
        (
            {"lighting_condition": "unknown"},
            "recognized_lighting_condition",
        ),
        (
            {"lighting_condition": "night"},
            "lighting_classification_consistency",
        ),
        ({"pvlib_version": None}, "nonnull_pvlib_version"),
    ],
)
def test_lighting_row_checks_fail_clearly(
    spark: SparkSession,
    overrides: dict[str, Any],
    expected_check: str,
) -> None:
    _assert_fails(
        validate_lighting,
        _lighting_dataframe(spark, _lighting_row(**overrides)),
        expected_check,
    )


def test_lighting_rejects_duplicate_keys(
    spark: SparkSession,
) -> None:
    dataframe = _lighting_dataframe(
        spark,
        _lighting_row(),
        _lighting_row(solar_azimuth_deg=181.0),
    )
    _assert_fails(
        validate_lighting,
        dataframe,
        "unique_lighting_keys",
    )


def test_lighting_reports_missing_columns(
    spark: SparkSession,
) -> None:
    _assert_fails(
        validate_lighting,
        spark.createDataFrame([(601,)], ["weather_query_cell_id"]),
        "required_columns",
    )


def test_gold_quality_passes_all_checks_and_metrics(
    spark: SparkSession,
) -> None:
    dataframe = _gold_dataframe(
        spark,
        _gold_row(),
        _gold_row(
            crime_offense_id="crime-2",
            selected_acs_vintage=None,
            selected_acs_release_date=None,
            tract_geoid=None,
            socioeconomic_match_found=False,
            weather_match_found=False,
            lighting_match_found=False,
            weather_provider=None,
            weather_model=None,
            weather_h3_resolution=None,
            weather_request_id=None,
            weather_source_row_hash=None,
            lighting_definition_version=None,
            lighting_pvlib_version=None,
        ),
    )
    report = validate_gold(
        dataframe,
        source_crime_count=2,
        coverage_thresholds=GoldCoverageThresholds(
            tract=0.5,
            socioeconomic=0.5,
            weather=0.5,
            lighting=0.5,
        ),
    )
    assert report.passed
    rates = {
        check.check_name: check.metric_value
        for check in report.checks
        if check.check_name.endswith("_coverage")
    }
    assert rates == {
        "tract_coverage": 0.5,
        "socioeconomic_coverage": 0.5,
        "weather_coverage": 0.5,
        "lighting_coverage": 0.5,
    }


@pytest.mark.parametrize(
    ("overrides", "expected_check"),
    [
        (
            {"crime_offense_id": None},
            "nonnull_gold_crime_offense_id",
        ),
        (
            {"weather_match_found": None},
            "calculable_match_metrics",
        ),
        (
            {"selected_acs_release_date": date(2024, 1, 2)},
            "leakage_safe_acs_dates",
        ),
        (
            {"weather_request_id": None},
            "weather_lineage_when_matched",
        ),
        (
            {"weather_source_row_hash": None},
            "weather_lineage_when_matched",
        ),
        (
            {"lighting_pvlib_version": None},
            "lighting_lineage_when_matched",
        ),
        (
            {"lighting_definition_version": "old"},
            "lighting_lineage_when_matched",
        ),
    ],
)
def test_gold_row_checks_fail_clearly(
    spark: SparkSession,
    overrides: dict[str, Any],
    expected_check: str,
) -> None:
    _assert_fails(
        validate_gold,
        _gold_dataframe(spark, _gold_row(**overrides)),
        expected_check,
        source_crime_count=1,
    )


def test_gold_rejects_duplicate_ids_and_join_multiplication(
    spark: SparkSession,
) -> None:
    dataframe = _gold_dataframe(
        spark,
        _gold_row(),
        _gold_row(),
    )
    error = _assert_fails(
        validate_gold,
        dataframe,
        "unique_gold_crime_offense_id",
        source_crime_count=1,
    )
    assert {
        "source_row_cardinality",
        "no_join_multiplication",
    }.issubset(_failure_names(error))


def test_gold_rejects_row_loss(
    spark: SparkSession,
) -> None:
    error = _assert_fails(
        validate_gold,
        _gold_dataframe(spark),
        "source_row_cardinality",
        source_crime_count=2,
    )
    assert "no_join_multiplication" not in _failure_names(error)


def test_gold_enforces_configured_coverage(
    spark: SparkSession,
) -> None:
    dataframe = _gold_dataframe(
        spark,
        _gold_row(
            weather_match_found=False,
            weather_provider=None,
            weather_model=None,
            weather_h3_resolution=None,
            weather_request_id=None,
            weather_source_row_hash=None,
        ),
    )
    _assert_fails(
        validate_gold,
        dataframe,
        "weather_coverage",
        source_crime_count=1,
        coverage_thresholds=GoldCoverageThresholds(weather=0.5),
    )


def test_gold_reports_missing_columns(
    spark: SparkSession,
) -> None:
    _assert_fails(
        validate_gold,
        spark.createDataFrame([("crime-1",)], ["crime_offense_id"]),
        "required_columns",
        source_crime_count=1,
    )


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_gold_coverage_thresholds_validate_rates(value: float) -> None:
    with pytest.raises(ValueError, match="coverage threshold"):
        GoldCoverageThresholds(weather=value)
