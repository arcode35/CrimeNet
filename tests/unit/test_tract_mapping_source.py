from __future__ import annotations

from datetime import date, datetime

import pytest

from crimenet.spatial.tract_mapping import (
    AMBIGUOUS,
    MAPPING_DEFINITION_VERSION,
    MATCHED_CONTAINS,
    UNMATCHED,
    _spark_location_key,
    attach_release_aware_boundary_year,
    canonical_coordinate,
    coordinate_reason_code,
    location_tract_key,
    mapping_issues_to_quarantine,
    select_stale_or_missing_locations,
    spatially_map_locations,
    split_location_candidates,
    validate_mapping_dataframe,
    validate_spatial_boundary_inputs,
)


def test_coordinate_canonicalization_is_explicit_and_stable() -> None:
    assert canonical_coordinate(32.7767) == "32.776700000000"
    assert canonical_coordinate(-0.0) == "0.000000000000"
    assert canonical_coordinate(1.2345678901235) == ("1.234567890124")

    with pytest.raises(ValueError, match="finite"):
        canonical_coordinate(float("nan"))


@pytest.mark.parametrize(
    ("latitude", "longitude", "reason"),
    [
        (None, -96.8, "MISSING_COORDINATE"),
        (float("inf"), -96.8, "NON_FINITE_COORDINATE"),
        (91.0, -96.8, "LATITUDE_OUT_OF_RANGE"),
        (32.8, -181.0, "LONGITUDE_OUT_OF_RANGE"),
        (32.8, -96.8, None),
    ],
)
def test_coordinate_reason_codes(
    latitude: float | None,
    longitude: float | None,
    reason: str | None,
) -> None:
    assert (
        coordinate_reason_code(
            latitude=latitude,
            longitude=longitude,
        )
        == reason
    )


def test_location_key_is_replay_safe_and_version_aware() -> None:
    arguments = {
        "tiger_line_year": 2024,
        "latitude": 32.7767,
        "longitude": -96.7970,
        "boundary_definition_version": ("tiger_line_tract_wgs84_v1"),
        "source_archive_sha256": "a" * 64,
    }

    first = location_tract_key(**arguments)
    replayed = location_tract_key(**arguments)
    remapped = location_tract_key(
        **arguments,
        mapping_definition_version=("tract_point_in_polygon_v2"),
    )
    corrected_boundary = location_tract_key(
        **{
            **arguments,
            "source_archive_sha256": "b" * 64,
        }
    )

    assert first == replayed
    assert len(first) == 64
    assert remapped != first
    assert corrected_boundary != first


@pytest.fixture(scope="module")
def spark() -> object:
    from pyspark.sql import SparkSession
    from pyspark.sql.types import (
        BooleanType,
        IntegerType,
        StringType,
    )

    def point(
        longitude: float,
        latitude: float,
        _srid: int,
    ) -> str:
        return f"{longitude},{latitude}"

    def covered(
        geometry: str,
        crime_point: str,
        *,
        include_boundary: bool,
    ) -> bool:
        minimum_x, minimum_y, maximum_x, maximum_y = (
            float(value) for value in geometry.split(",")
        )
        x, y = (float(value) for value in crime_point.split(","))
        if include_boundary:
            return minimum_x <= x <= maximum_x and minimum_y <= y <= maximum_y
        return minimum_x < x < maximum_x and minimum_y < y < maximum_y

    session = (
        SparkSession.builder.master("local[1]")
        .appName("test-tract-mapping")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.udf.register("ST_Point", point, StringType())
    session.udf.register(
        "ST_Covers",
        lambda geometry, crime_point: covered(
            geometry,
            crime_point,
            include_boundary=True,
        ),
        BooleanType(),
    )
    session.udf.register(
        "ST_Contains",
        lambda geometry, crime_point: covered(
            geometry,
            crime_point,
            include_boundary=False,
        ),
        BooleanType(),
    )
    session.udf.register(
        "ST_SRID",
        lambda geometry: None if geometry == "UNKNOWN_SRID" else 4326,
        IntegerType(),
    )
    yield session
    session.stop()


def test_release_aware_selection_and_invalid_quarantine(
    spark: object,
) -> None:
    from pyspark.sql import SparkSession
    from pyspark.sql.types import (
        DateType,
        DoubleType,
        IntegerType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )

    assert isinstance(spark, SparkSession)
    calendar_schema = StructType(
        [
            StructField("acs_vintage", IntegerType(), False),
            StructField("acs_release_date", DateType(), False),
            StructField("tiger_line_year", IntegerType(), False),
            StructField(
                "tract_definition_vintage",
                IntegerType(),
                False,
            ),
        ]
    )
    calendar = spark.createDataFrame(
        [
            (2023, date(2024, 12, 12), 2023, 2020),
            (2024, date(2026, 1, 29), 2024, 2020),
        ],
        schema=calendar_schema,
    )
    crime_schema = StructType(
        [
            StructField("source_system", StringType(), False),
            StructField("source_row_hash", StringType(), False),
            StructField("source_file", StringType(), True),
            StructField("occurred_at", TimestampType(), True),
            StructField("latitude", DoubleType(), True),
            StructField("longitude", DoubleType(), True),
        ]
    )
    crimes = spark.createDataFrame(
        [
            (
                "dallas",
                "a" * 64,
                "/landing/a.csv",
                datetime(2026, 1, 29, 12),
                32.8,
                -96.8,
            ),
            (
                "dallas",
                "b" * 64,
                "/landing/b.csv",
                datetime(2026, 1, 30, 0),
                32.8,
                -96.8,
            ),
            (
                "dallas",
                "c" * 64,
                "/landing/c.csv",
                datetime(2026, 1, 30, 0),
                None,
                -96.8,
            ),
        ],
        schema=crime_schema,
    )

    enriched = attach_release_aware_boundary_year(
        crimes,
        calendar,
    )
    vintages = {
        row["source_row_hash"]: row["tiger_line_year"]
        for row in enriched.select(
            "source_row_hash",
            "tiger_line_year",
        ).collect()
    }
    frames = split_location_candidates(
        enriched,
        pipeline_run_id="run_1",
    )

    # The release date itself still uses the previously public vintage.
    assert vintages["a" * 64] == 2023
    assert vintages["b" * 64] == 2024
    assert frames.source_row_count == 3
    assert frames.invalid_row_count == 1
    assert frames.candidates.count() == 2
    quarantined = frames.quarantine.first()
    assert quarantined["quarantine_reason_code"] == "MISSING_COORDINATE"
    assert len(quarantined["quarantine_id"]) == 64


def test_version_change_selects_existing_location_for_remap(
    spark: object,
) -> None:
    from pyspark.sql import SparkSession

    assert isinstance(spark, SparkSession)
    locations = spark.createDataFrame(
        [(2024, 32.8, -96.8)],
        "tiger_line_year int, latitude double, longitude double",
    )
    boundaries = spark.createDataFrame(
        [
            (
                2024,
                "tiger_line_tract_wgs84_v1",
                "b" * 64,
            )
        ],
        (
            "boundary_vintage int, "
            "boundary_definition_version string, "
            "source_archive_sha256 string"
        ),
    )
    existing = spark.createDataFrame(
        [
            (
                2024,
                32.8,
                -96.8,
                "tiger_line_tract_wgs84_v1",
                "a" * 64,
                MAPPING_DEFINITION_VERSION,
            )
        ],
        (
            "tiger_line_year int, latitude double, longitude double, "
            "boundary_definition_version string, "
            "source_archive_sha256 string, "
            "mapping_definition_version string"
        ),
    )

    stale = select_stale_or_missing_locations(
        locations,
        boundaries,
        existing,
        boundary_definition_version=("tiger_line_tract_wgs84_v1"),
        mapping_definition_version=MAPPING_DEFINITION_VERSION,
    )
    assert stale.collect()[0].asDict() == {
        "tiger_line_year": 2024,
        "latitude": 32.8,
        "longitude": -96.8,
    }


def test_python_and_spark_location_key_algorithms_match(
    spark: object,
) -> None:
    from pyspark.sql import SparkSession

    assert isinstance(spark, SparkSession)
    dataframe = spark.createDataFrame(
        [
            (
                2024,
                32.7767,
                -96.797,
                "tiger_line_tract_wgs84_v1",
                "a" * 64,
            )
        ],
        (
            "tiger_line_year int, latitude double, longitude double, "
            "boundary_definition_version string, "
            "source_archive_sha256 string"
        ),
    )
    spark_key = dataframe.select(
        _spark_location_key(
            mapping_definition_version=(MAPPING_DEFINITION_VERSION)
        ).alias("location_tract_key")
    ).first()["location_tract_key"]
    python_key = location_tract_key(
        tiger_line_year=2024,
        latitude=32.7767,
        longitude=-96.797,
        boundary_definition_version=("tiger_line_tract_wgs84_v1"),
        source_archive_sha256="a" * 64,
    )

    assert spark_key == python_key


def test_spatial_mapping_exposes_contains_ambiguity_and_no_match(
    spark: object,
) -> None:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    assert isinstance(spark, SparkSession)
    locations = spark.createDataFrame(
        [
            (2024, 5.0, 5.0),
            (2024, 5.0, 0.0),
            (2024, 20.0, 20.0),
        ],
        "tiger_line_year int, latitude double, longitude double",
    )
    boundaries = spark.createDataFrame(
        [
            (
                2024,
                "48113000100",
                "0,0,10,10",
                "tiger_line_tract_wgs84_v1",
                "a" * 64,
            ),
            (
                2024,
                "48113000200",
                "-10,0,0,10",
                "tiger_line_tract_wgs84_v1",
                "a" * 64,
            ),
        ],
        (
            "boundary_vintage int, geoid string, tract_geometry string, "
            "boundary_definition_version string, "
            "source_archive_sha256 string"
        ),
    )
    validate_spatial_boundary_inputs(
        boundaries,
        boundary_definition_version=("tiger_line_tract_wgs84_v1"),
    )
    for column_name, invalid_value in (
        ("geoid", None),
        ("source_archive_sha256", None),
        ("tract_geometry", "UNKNOWN_SRID"),
    ):
        invalid_boundaries = boundaries.withColumn(
            column_name,
            F.when(
                F.col("geoid") == "48113000100",
                F.lit(invalid_value).cast(
                    boundaries.schema[column_name].dataType
                ),
            ).otherwise(F.col(column_name)),
        )
        with pytest.raises(ValueError, match="invalid keys"):
            validate_spatial_boundary_inputs(
                invalid_boundaries,
                boundary_definition_version=(
                    "tiger_line_tract_wgs84_v1"
                ),
            )

    mapping = spatially_map_locations(
        locations,
        boundaries,
        boundary_definition_version=("tiger_line_tract_wgs84_v1"),
        mapping_definition_version=MAPPING_DEFINITION_VERSION,
        pipeline_run_id="run_1",
    )
    by_longitude = {
        row["longitude"]: row for row in mapping.orderBy("longitude").collect()
    }

    assert by_longitude[5.0]["match_status"] == MATCHED_CONTAINS
    assert by_longitude[5.0]["tract_geoid"] == "48113000100"
    assert by_longitude[0.0]["match_status"] == AMBIGUOUS
    assert by_longitude[0.0]["candidate_match_count"] == 2
    assert by_longitude[20.0]["match_status"] == UNMATCHED
    assert by_longitude[20.0]["candidate_match_count"] == 0

    quarantine = mapping_issues_to_quarantine(mapping)
    assert {row["quarantine_reason_code"] for row in quarantine.collect()} == {
        "AMBIGUOUS_TRACT_MATCH",
        "NO_TRACT_MATCH",
    }
    validate_mapping_dataframe(
        mapping,
        expected_locations=locations,
        maximum_ambiguous_matches=1,
        maximum_unmatched_rate=0.34,
    )
    with pytest.raises(ValueError, match="ambiguous match threshold"):
        validate_mapping_dataframe(
            mapping,
            expected_locations=locations,
            maximum_ambiguous_matches=0,
            maximum_unmatched_rate=0.34,
        )


def test_mapping_validation_checks_exact_keys_and_thresholds(
    spark: object,
) -> None:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    assert isinstance(spark, SparkSession)
    expected = spark.createDataFrame(
        [(2024, 32.8, -96.8)],
        "tiger_line_year int, latitude double, longitude double",
    )
    mapping = spark.createDataFrame(
        [
            (
                2024,
                32.8,
                -96.8,
                "48113000100",
                "tiger_line_tract_wgs84_v1",
                "a" * 64,
                MAPPING_DEFINITION_VERSION,
                "b" * 64,
                MATCHED_CONTAINS,
                1,
            )
        ],
        (
            "tiger_line_year int, latitude double, longitude double, "
            "tract_geoid string, boundary_definition_version string, "
            "source_archive_sha256 string, "
            "mapping_definition_version string, "
            "location_tract_key string, match_status string, "
            "candidate_match_count int"
        ),
    )

    validate_mapping_dataframe(
        mapping,
        expected_locations=expected,
        maximum_ambiguous_matches=0,
        maximum_unmatched_rate=0.0,
    )

    for column_name in (
        "match_status",
        "location_tract_key",
        "candidate_match_count",
    ):
        invalid_mapping = mapping.withColumn(
            column_name,
            F.lit(None).cast(
                mapping.schema[column_name].dataType
            ),
        )
        with pytest.raises(ValueError, match="invalid match status"):
            validate_mapping_dataframe(
                invalid_mapping,
                expected_locations=expected,
                maximum_ambiguous_matches=0,
                maximum_unmatched_rate=0.0,
            )

    with pytest.raises(ValueError, match="missing expected"):
        validate_mapping_dataframe(
            mapping.limit(0),
            expected_locations=expected,
            maximum_ambiguous_matches=0,
            maximum_unmatched_rate=0.0,
        )
