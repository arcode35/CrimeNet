from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from crimenet.quality.external import split_external_quarantine
from crimenet.silver.lighting import (
    OUTPUT_SCHEMA,
    calculate_solar_positions,
    classify_lighting_condition,
    compute_lighting_conditions,
    extract_lighting_keys,
)
from crimenet.silver.socioeconomic import (
    ACS_COLUMN_NAMES,
    ACS_QUARANTINE_MESSAGES,
    SOCIOECONOMIC_DEFINITION_VERSION,
    annotate_acs_validation,
    transform_acs5_tracts,
)
from crimenet.silver.weather import (
    WEATHER_DEFINITION_VERSION,
    WEATHER_QUARANTINE_MESSAGES,
    annotate_weather_validation,
    transform_open_meteo_weather,
)
from crimenet.spatial.h3 import add_weather_query_cell


def _weather_frame(spark: SparkSession) -> DataFrame:
    schema = """
        request_id string,
        provider string,
        model string,
        weather_query_cell_id long,
        h3_resolution int,
        query_latitude double,
        query_longitude double,
        grid_latitude double,
        grid_longitude double,
        grid_elevation double,
        timezone string,
        utc_offset_seconds int,
        hourly_units struct<time:string,temperature_2m:string>,
        hourly struct<time:array<string>,temperature_2m:array<double>>,
        rescued_data string,
        source_file string,
        source_row_hash string,
        source_contract_version string,
        ingested_at timestamp
    """
    return spark.createDataFrame(
        [
            (
                "request-ok",
                "open_meteo",
                "era5_land",
                617_000_000_000_000_001,
                6,
                32.78,
                -96.80,
                32.75,
                -96.75,
                150.0,
                "GMT",
                0,
                ("iso8601", "°C"),
                (
                    ["2024-01-01T00:00", "2024-01-01T01:00"],
                    [7.5, 8.0],
                ),
                None,
                "/weather/ok.json",
                "hash-ok",
                "open-meteo-v1",
                datetime(2024, 1, 2),
            ),
            (
                None,
                "other",
                "future_model",
                None,
                6,
                95.0,
                -196.0,
                None,
                None,
                None,
                "GMT",
                0,
                None,
                None,
                '{"new_field": 1}',
                "/weather/bad.json",
                "hash-bad",
                "open-meteo-v1",
                datetime(2024, 1, 2),
            ),
            (
                "request-array-bad",
                "open_meteo",
                "era5",
                617_000_000_000_000_002,
                6,
                32.78,
                -96.80,
                32.75,
                -96.75,
                150.0,
                "GMT",
                0,
                ("iso8601", "°C"),
                (["not-a-time", "2024-01-01T01:00"], [7.5]),
                None,
                "/weather/array-bad.json",
                "hash-array-bad",
                "open-meteo-v1",
                datetime(2024, 1, 2),
            ),
        ],
        schema=schema,
    )


def test_weather_validation_transformation_and_external_quarantine(
    spark: SparkSession,
) -> None:
    bronze = _weather_frame(spark)
    annotated = annotate_weather_validation(bronze)
    reasons = {
        row["request_id"] or "missing": set(row["_quarantine_reason_codes"])
        for row in annotated.select(
            "request_id",
            "_quarantine_reason_codes",
        ).collect()
    }

    assert reasons["request-ok"] == set()
    assert {
        "RESCUED_SCHEMA_DATA",
        "MISSING_REQUEST_KEY",
        "UNSUPPORTED_PROVIDER_OR_MODEL",
        "MISSING_HOURLY_DATA",
        "INVALID_QUERY_COORDINATES",
    }.issubset(reasons["missing"])
    assert reasons["request-array-bad"] == {
        "HOURLY_ARRAY_LENGTH_MISMATCH",
        "INVALID_HOURLY_TIMESTAMP",
    }

    valid, quarantine = split_external_quarantine(
        annotated,
        reason_codes_column="_quarantine_reason_codes",
        reason_messages=WEATHER_QUARANTINE_MESSAGES,
        source_system="open_meteo",
        pipeline_run_id="run-weather-1",
    )
    assert valid.count() == 1
    assert quarantine.count() == sum(len(value) for value in reasons.values())
    assert quarantine.select("quarantine_id").distinct().count() == (
        quarantine.count()
    )
    assert {
        row["pipeline_run_id"]
        for row in quarantine.select("pipeline_run_id").collect()
    } == {"run-weather-1"}
    assert all(
        row["quarantine_reason"]
        for row in quarantine.select("quarantine_reason").collect()
    )

    transformed = transform_open_meteo_weather(bronze)
    rows = transformed.orderBy("weather_timestamp").collect()
    assert [row["temperature_2m_c"] for row in rows] == [7.5, 8.0]
    assert {row["weather_definition_version"] for row in rows} == {
        WEATHER_DEFINITION_VERSION
    }
    assert all(row["weather_date"].isoformat() == "2024-01-01" for row in rows)


def _acs_schema() -> StructType:
    return StructType(
        [
            StructField("name", StringType(), True),
            *[
                StructField(source_name, StringType(), True)
                for source_name in ACS_COLUMN_NAMES
            ],
            StructField("state", StringType(), True),
            StructField("county", StringType(), True),
            StructField("tract", StringType(), True),
            StructField("geoid", StringType(), True),
            StructField("acs_vintage", IntegerType(), True),
            StructField("period_start_year", IntegerType(), True),
            StructField("period_end_year", IntegerType(), True),
            StructField("geography_type", StringType(), True),
            StructField("rescued_data", StringType(), True),
            StructField("source_file", StringType(), True),
            StructField("source_row_hash", StringType(), True),
            StructField("source_contract_version", StringType(), True),
            StructField("ingested_at", TimestampType(), True),
        ]
    )


def _acs_record(**updates: object) -> dict[str, object]:
    record: dict[str, object] = {
        "name": "Census Tract 1, Dallas County, Texas",
        **{source_name: "10" for source_name in ACS_COLUMN_NAMES},
        "b01002_001e": "35.5",
        "b01002_001m": "1.2",
        "b19013_001e": "70000",
        "b19013_001m": "2500",
        "b17001_001e": "100",
        "b17001_002e": "20",
        "b23025_003e": "80",
        "b23025_005e": "4",
        "b25001_001e": "50",
        "b25002_003e": "5",
        "b25003_001e": "45",
        "b25003_003e": "18",
        "b08201_001e": "40",
        "b08201_002e": "8",
        "state": "48",
        "county": "113",
        "tract": "000100",
        "geoid": "48113000100",
        "acs_vintage": 2023,
        "period_start_year": 2019,
        "period_end_year": 2023,
        "geography_type": "tract",
        "rescued_data": None,
        "source_file": "/acs/2023.jsonl",
        "source_row_hash": "acs-hash",
        "source_contract_version": "acs-contract-v1",
        "ingested_at": datetime(2024, 1, 2),
    }
    record.update(updates)
    return record


def test_acs_validation_domain_cleanup_and_rates(
    spark: SparkSession,
) -> None:
    valid = _acs_record(b01003_001m="-666666666")
    invalid = _acs_record(
        geoid="bad",
        acs_vintage=1900,
        b01002_001e="121",
        rescued_data='{"future": true}',
        source_row_hash="acs-bad",
    )
    missing = _acs_record(
        geoid=None,
        acs_vintage=None,
        source_row_hash="acs-missing",
    )
    bronze = spark.createDataFrame(
        [valid, invalid, missing],
        schema=_acs_schema(),
    )

    reasons = {
        row["source_row_hash"]: set(row["_quarantine_reason_codes"])
        for row in annotate_acs_validation(bronze).select(
            "source_row_hash",
            "_quarantine_reason_codes",
        ).collect()
    }
    assert reasons["acs-hash"] == set()
    assert reasons["acs-bad"] == {
        "RESCUED_SCHEMA_DATA",
        "INVALID_TRACT_GEOID",
        "INVALID_ACS_VINTAGE",
        "INVALID_NUMERIC_VALUE",
    }
    assert "MISSING_TRACT_KEY" in reasons["acs-missing"]

    annotated = annotate_acs_validation(bronze)
    valid_rows, quarantine = split_external_quarantine(
        annotated,
        reason_codes_column="_quarantine_reason_codes",
        reason_messages=ACS_QUARANTINE_MESSAGES,
        source_system="census_acs5",
        pipeline_run_id="run-acs-1",
    )
    assert valid_rows.count() == 1
    validation_fields = quarantine.select("validation_fields").first()[0]
    assert "geoid" in validation_fields
    assert "acs_vintage" in validation_fields

    result = transform_acs5_tracts(bronze).first()
    assert result is not None
    assert result["population"] == 10
    assert result["population_moe"] is None
    assert result["poverty_rate"] == pytest.approx(0.2)
    assert result["unemployment_rate"] == pytest.approx(0.05)
    assert result["vacancy_rate"] == pytest.approx(0.1)
    assert result["renter_occupied_rate"] == pytest.approx(0.4)
    assert result["no_vehicle_rate"] == pytest.approx(0.2)
    assert (
        result["socioeconomic_definition_version"]
        == SOCIOECONOMIC_DEFINITION_VERSION
    )


def test_lighting_classification_and_solar_batches(
    spark: SparkSession,
) -> None:
    classified = classify_lighting_condition(
        pd.Series([5.0, -1.0, -7.0, -13.0, -19.0])
    )
    assert classified.tolist() == [
        "daylight",
        "civil_twilight",
        "nautical_twilight",
        "astronomical_twilight",
        "night",
    ]

    input_batch = pd.DataFrame(
        {
            "lighting_query_cell_id": [617_000_000_000_000_001] * 2,
            "solar_timestamp_hour": [
                pd.Timestamp("2024-06-21T12:00:00Z"),
                pd.Timestamp("2024-06-21T18:00:00Z"),
            ],
            "query_latitude": [32.78] * 2,
            "query_longitude": [-96.80] * 2,
            "lighting_definition_version": ["lighting-v-test"] * 2,
        }
    )
    calculated = list(
        calculate_solar_positions(iter([pd.DataFrame(), input_batch]))
    )
    assert calculated[0].empty
    assert calculated[0].columns.tolist() == [
        field.name for field in OUTPUT_SCHEMA.fields
    ]
    assert len(calculated[1]) == 2
    assert set(calculated[1]["lighting_definition_version"]) == {
        "lighting-v-test"
    }
    assert set(calculated[1]["lighting_condition"]).issubset(
        {
            "daylight",
            "civil_twilight",
            "nautical_twilight",
            "astronomical_twilight",
            "night",
        }
    )

    keys = spark.createDataFrame(
        [
            (
                617_000_000_000_000_001,
                datetime(2024, 6, 21, 12),
                32.78,
                -96.80,
                "lighting-v-test",
            )
        ],
        """
        lighting_query_cell_id long,
        solar_timestamp_hour timestamp,
        query_latitude double,
        query_longitude double,
        lighting_definition_version string
        """,
    )
    computed = compute_lighting_conditions(keys).first()
    assert computed is not None
    assert computed["computed_at"] is not None
    assert computed["lighting_definition_version"] == "lighting-v-test"


def test_lighting_keys_are_hour_grained_unique_and_geolocated(
    spark: SparkSession,
) -> None:
    crimes = spark.createDataFrame(
        [
            (32.7767, -96.7970, datetime(2024, 6, 21, 12, 15)),
            (32.7767, -96.7970, datetime(2024, 6, 21, 12, 45)),
            (32.7767, -96.7970, None),
        ],
        "latitude double, longitude double, occurred_at timestamp",
    )
    crimes_with_cells = add_weather_query_cell(crimes, resolution=6)
    keys = extract_lighting_keys(
        crimes_with_cells,
        definition_version="lighting-v-test",
    )
    row = keys.first()
    assert keys.count() == 1
    assert row["solar_timestamp_hour"] == datetime(2024, 6, 21, 12)
    assert row["lighting_definition_version"] == "lighting-v-test"
    assert row["query_latitude"] == pytest.approx(32.7767, abs=0.1)
    assert row["query_longitude"] == pytest.approx(-96.7970, abs=0.1)

    with pytest.raises(ValueError, match="must not be blank"):
        extract_lighting_keys(
            crimes_with_cells,
            definition_version=" ",
        )
