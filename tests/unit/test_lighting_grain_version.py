from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructField, StructType

from crimenet.contracts.gold import (
    crime_offense_id_from_business_identity,
)
from crimenet.contracts.lighting import (
    LIGHTING_DEFINITION_VERSION,
    LIGHTING_KEYS,
    lighting_join_key,
    normalize_solar_timestamp_hour,
)
from crimenet.gold.crime_features import (
    attach_lighting_features,
    build_lighting_lookup,
    prepare_crimes,
)
from crimenet.silver.lighting import (
    OUTPUT_SCHEMA,
    validate_lighting_results,
)


def _valid_lighting_results(spark: SparkSession) -> DataFrame:
    return spark.createDataFrame(
        [
            (
                617_000,
                datetime(2025, 1, 2, 14),
                32.78,
                -96.80,
                30.0,
                30.1,
                60.0,
                180.0,
                "daylight",
                True,
                "0.15.2",
                LIGHTING_DEFINITION_VERSION,
            )
        ],
        OUTPUT_SCHEMA,
    )


def test_lighting_join_key_normalizes_to_utc_hour() -> None:
    central = timezone(timedelta(hours=-6))
    event_timestamp = datetime(
        2025,
        1,
        2,
        14,
        37,
        22,
        123,
        tzinfo=central,
    )

    assert normalize_solar_timestamp_hour(event_timestamp) == datetime(
        2025, 1, 2, 20, tzinfo=UTC
    )
    assert lighting_join_key(
        query_cell_id=617_000,
        timestamp=event_timestamp,
    ) == lighting_join_key(
        query_cell_id=617_000,
        timestamp=datetime(
            2025,
            1,
            2,
            20,
            tzinfo=UTC,
        ),
    )


def test_definition_version_is_part_of_physical_key() -> None:
    assert LIGHTING_KEYS == (
        "lighting_query_cell_id",
        "solar_timestamp_hour",
        "lighting_definition_version",
    )

    timestamp = datetime(2025, 1, 2, 14, tzinfo=UTC)
    assert lighting_join_key(
        query_cell_id=617_000,
        timestamp=timestamp,
        definition_version="v1",
    ) != lighting_join_key(
        query_cell_id=617_000,
        timestamp=timestamp,
        definition_version="v2",
    )


def test_changed_definition_version_is_eligible_for_recompute(
    spark: SparkSession,
) -> None:
    existing = (
        spark.createDataFrame(
            [
                (
                    617_000,
                    "2025-01-02 14:00:00",
                    "v1",
                )
            ],
            (
                "lighting_query_cell_id long, "
                "solar_timestamp_text string, "
                "lighting_definition_version string"
            ),
        )
        .withColumn(
            "solar_timestamp_hour",
            F.to_timestamp("solar_timestamp_text"),
        )
        .drop("solar_timestamp_text")
    )
    requested = existing.withColumn(
        "lighting_definition_version",
        F.lit("v2"),
    )

    keys_to_compute = requested.join(
        existing,
        on=list(LIGHTING_KEYS),
        how="left_anti",
    )

    assert keys_to_compute.count() == 1


def test_non_hour_crime_joins_current_hour_lighting(
    spark: SparkSession,
) -> None:
    crimes = (
        spark.createDataFrame(
            [
                (
                    "stable-logical-identity",
                    "stable-row-hash",
                    "dbfs:/landing/original.csv",
                    617_000,
                    "2025-01-02 14:37:22",
                    32.78,
                    -96.80,
                ),
                (
                    "stable-logical-identity",
                    "stable-row-hash",
                    "dbfs:/replay/renamed.csv",
                    617_000,
                    "2025-01-02 14:37:22",
                    32.78,
                    -96.80,
                ),
            ],
            (
                "business_identity string, source_row_hash string, "
                "source_file string, weather_query_cell_id long, "
                "occurred_at_text string, latitude double, "
                "longitude double"
            ),
        )
        .withColumn(
            "occurred_at",
            F.to_timestamp("occurred_at_text"),
        )
        .drop("occurred_at_text")
    )
    prepared = prepare_crimes(crimes)

    lighting = (
        spark.createDataFrame(
            [
                (
                    617_000,
                    "2025-01-02 14:00:00",
                    LIGHTING_DEFINITION_VERSION,
                    30.0,
                    30.1,
                    60.0,
                    180.0,
                    "daylight",
                    True,
                ),
                (
                    617_000,
                    "2025-01-02 14:00:00",
                    "retired-definition-v0",
                    -5.0,
                    -5.0,
                    95.0,
                    180.0,
                    "civil_twilight",
                    False,
                ),
            ],
            (
                "lighting_query_cell_id long, "
                "solar_timestamp_text string, "
                "lighting_definition_version string, "
                "solar_elevation_deg double, "
                "apparent_solar_elevation_deg double, "
                "solar_zenith_deg double, "
                "solar_azimuth_deg double, "
                "lighting_condition string, is_daylight boolean"
            ),
        )
        .withColumn(
            "solar_timestamp_hour",
            F.to_timestamp("solar_timestamp_text"),
        )
        .drop("solar_timestamp_text")
    )

    lookup = build_lighting_lookup(lighting)
    result = attach_lighting_features(
        prepared,
        lookup,
    )
    rows = result.select(
        "crime_offense_id",
        F.date_format(
            "solar_timestamp_hour",
            "yyyy-MM-dd HH:mm:ss",
        ).alias("solar_hour"),
        "lighting_match_found",
        "solar_elevation_deg",
        "lighting_definition_version",
    ).collect()

    assert len(rows) == 2
    assert {row["crime_offense_id"] for row in rows} == {
        crime_offense_id_from_business_identity("stable-logical-identity")
    }
    assert {row["solar_hour"] for row in rows} == {"2025-01-02 14:00:00"}
    assert all(
        row["lighting_match_found"]
        and row["solar_elevation_deg"] == 30.0
        and row["lighting_definition_version"] == LIGHTING_DEFINITION_VERSION
        for row in rows
    )


def test_lighting_validation_checks_hour_and_exact_keys(
    spark: SparkSession,
) -> None:
    results = _valid_lighting_results(spark)
    expected = results.select(*LIGHTING_KEYS)

    validate_lighting_results(
        results,
        expected_keys=expected,
        definition_version=LIGHTING_DEFINITION_VERSION,
    )

    wrong_hour = results.withColumn(
        "solar_timestamp_hour",
        F.expr("solar_timestamp_hour + INTERVAL 37 MINUTES"),
    )
    with pytest.raises(RuntimeError, match="invalid solar-position"):
        validate_lighting_results(
            wrong_hour,
            enforce_schema_nullability=False,
        )

    with pytest.raises(RuntimeError, match="duplicate"):
        validate_lighting_results(
            results.unionByName(results),
        )


def test_lighting_validation_checks_output_schema_nullability(
    spark: SparkSession,
) -> None:
    results = _valid_lighting_results(spark)
    nullable_schema = StructType(
        [
            StructField(
                field.name,
                field.dataType,
                True,
            )
            for field in OUTPUT_SCHEMA.fields
        ]
    )
    nullable_results = spark.createDataFrame(
        results.collect(),
        nullable_schema,
    )

    with pytest.raises(RuntimeError, match=r"expected nullable=False"):
        validate_lighting_results(nullable_results)


@pytest.mark.parametrize(
    ("column_name", "value", "data_type"),
    [
        ("lighting_condition", None, "string"),
        ("is_daylight", None, "boolean"),
        ("pvlib_version", None, "string"),
        ("pvlib_version", "", "string"),
    ],
)
def test_lighting_validation_rejects_unusable_required_values(
    spark: SparkSession,
    column_name: str,
    value: object,
    data_type: str,
) -> None:
    invalid_results = _valid_lighting_results(spark).withColumn(
        column_name,
        F.lit(value).cast(data_type),
    )

    with pytest.raises(RuntimeError, match="invalid solar-position"):
        validate_lighting_results(
            invalid_results,
            enforce_schema_nullability=False,
        )
