from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest
from pyspark.sql import (
    Column,
    DataFrame,
    SparkSession,
)
from pyspark.sql import functions as F

from crimenet.contracts.lighting import (
    LIGHTING_DEFINITION_VERSION,
    LIGHTING_KEYS,
)
from crimenet.silver.lighting import (
    OUTPUT_SCHEMA,
    calculate_solar_positions,
    classify_lighting_condition,
    extract_lighting_key_grain,
    select_missing_lighting_keys,
    validate_lighting_results,
)

CELL_ID = 604164855133372415


def _valid_lighting(
    spark: SparkSession,
) -> DataFrame:
    return spark.createDataFrame(
        [
            (
                CELL_ID,
                datetime(2024, 6, 21, 18),
                32.7767,
                -96.797,
                78.666651,
                78.670008,
                11.333349,
                143.662815,
                "daylight",
                True,
                "0.15.2",
                LIGHTING_DEFINITION_VERSION,
            )
        ],
        schema=OUTPUT_SCHEMA,
    )


def test_lighting_classification_boundaries() -> None:
    elevations = pd.Series(
        [0.0, -0.01, -6.0, -6.01, -12.0, -12.01, -18.0, -18.01]
    )

    assert list(
        classify_lighting_condition(
            elevations
        )
    ) == [
        "daylight",
        "civil_twilight",
        "civil_twilight",
        "nautical_twilight",
        "nautical_twilight",
        "astronomical_twilight",
        "astronomical_twilight",
        "night",
    ]


def test_pvlib_solar_positions_have_known_dallas_values() -> None:
    batch = pd.DataFrame(
        {
            "weather_query_cell_id": [
                CELL_ID,
                CELL_ID,
            ],
            "solar_timestamp_hour": pd.to_datetime(
                [
                    "2024-06-21T18:00:00Z",
                    "2024-06-21T06:00:00Z",
                ],
                utc=True,
            ),
            "query_latitude": [32.7767, 32.7767],
            "query_longitude": [-96.797, -96.797],
            "lighting_definition_version": [
                LIGHTING_DEFINITION_VERSION,
                LIGHTING_DEFINITION_VERSION,
            ],
        }
    )

    result = next(
        calculate_solar_positions(
            iter([batch])
        )
    )

    day = result.iloc[0]
    night = result.iloc[1]

    assert day["solar_elevation_deg"] == pytest.approx(
        78.666651,
        abs=0.05,
    )
    assert day["solar_azimuth_deg"] == pytest.approx(
        143.662815,
        abs=0.05,
    )
    assert day["lighting_condition"] == "daylight"
    assert bool(day["is_daylight"]) is True

    assert night["solar_elevation_deg"] == pytest.approx(
        -33.361827,
        abs=0.05,
    )
    assert night["solar_azimuth_deg"] == pytest.approx(
        352.016209,
        abs=0.05,
    )
    assert night["lighting_condition"] == "night"
    assert bool(night["is_daylight"]) is False


def test_extract_lighting_key_grain_truncates_once_and_deduplicates(
    spark: SparkSession,
) -> None:
    crimes = spark.createDataFrame(
        [
            (CELL_ID, datetime(2024, 6, 21, 18, 5)),
            (CELL_ID, datetime(2024, 6, 21, 18, 55)),
            (CELL_ID, datetime(2024, 6, 21, 19, 1)),
            (None, datetime(2024, 6, 21, 19, 1)),
            (CELL_ID, None),
        ],
        "weather_query_cell_id long, occurred_at timestamp",
    )

    keys = (
        extract_lighting_key_grain(crimes)
        .orderBy("solar_timestamp_hour")
        .collect()
    )

    assert len(keys) == 2
    assert [
        row["solar_timestamp_hour"].minute
        for row in keys
    ] == [0, 0]
    assert {
        row["lighting_definition_version"]
        for row in keys
    } == {
        LIGHTING_DEFINITION_VERSION
    }


def test_missing_lighting_key_selection_is_idempotent(
    spark: SparkSession,
) -> None:
    candidates = spark.createDataFrame(
        [
            (
                CELL_ID,
                datetime(2024, 6, 21, 18),
                LIGHTING_DEFINITION_VERSION,
                32.7767,
                -96.797,
            ),
            (
                CELL_ID,
                datetime(2024, 6, 21, 19),
                LIGHTING_DEFINITION_VERSION,
                32.7767,
                -96.797,
            ),
        ],
        (
            "weather_query_cell_id long, "
            "solar_timestamp_hour timestamp, "
            "lighting_definition_version string, "
            "query_latitude double, query_longitude double"
        ),
    )
    existing = candidates.limit(1)

    missing = select_missing_lighting_keys(
        candidates,
        existing,
    )
    materialized = existing.unionByName(
        missing
    )

    assert missing.count() == 1
    assert select_missing_lighting_keys(
        candidates,
        materialized,
    ).isEmpty()


def test_lighting_validation_accepts_valid_results(
    spark: SparkSession,
) -> None:
    validate_lighting_results(
        _valid_lighting(spark)
    )


@pytest.mark.parametrize(
    ("column_name", "replacement_value"),
    [
        (
            "weather_query_cell_id",
            None,
        ),
        (
            "lighting_definition_version",
            "obsolete-definition",
        ),
        (
            "solar_azimuth_deg",
            361.0,
        ),
        (
            "lighting_condition",
            "night",
        ),
        (
            "is_daylight",
            False,
        ),
    ],
)
def test_lighting_validation_rejects_invalid_results(
    spark: SparkSession,
    column_name: str,
    replacement_value: object,
) -> None:
    replacement: Column = F.lit(
        replacement_value
    )
    if column_name == "weather_query_cell_id":
        replacement = replacement.cast("long")

    invalid = _valid_lighting(
        spark
    ).withColumn(
        column_name,
        replacement,
    )

    with pytest.raises(RuntimeError):
        validate_lighting_results(invalid)


def test_lighting_validation_rejects_duplicate_keys(
    spark: SparkSession,
) -> None:
    valid = _valid_lighting(spark)

    with pytest.raises(RuntimeError, match="duplicate"):
        validate_lighting_results(
            valid.unionByName(valid)
        )

    assert tuple(LIGHTING_KEYS) == (
        "weather_query_cell_id",
        "solar_timestamp_hour",
        "lighting_definition_version",
    )
