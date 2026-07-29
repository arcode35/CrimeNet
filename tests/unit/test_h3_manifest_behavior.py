from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from crimenet.jobs.weather_request_planner_job import validate_manifest
from crimenet.spatial import h3 as h3_module
from crimenet.spatial.h3 import (
    add_weather_query_cell,
    extract_h3_centers,
)
from crimenet.weather.request_planner import build_weather_request_manifest


def test_local_h3_round_trip_and_null_handling(
    spark: SparkSession,
) -> None:
    source = spark.createDataFrame(
        [
            (32.7767, -96.7970),
            (None, -96.7970),
        ],
        "latitude double, longitude double",
    )
    cells = add_weather_query_cell(source, resolution=6)
    rows = cells.collect()
    assert rows[0]["weather_query_cell_id"] is not None
    assert rows[1]["weather_query_cell_id"] is None

    centers = extract_h3_centers(cells).collect()
    assert centers[0]["query_latitude"] == pytest.approx(32.7767, abs=0.1)
    assert centers[0]["query_longitude"] == pytest.approx(-96.7970, abs=0.1)
    assert centers[1]["query_latitude"] is None
    assert centers[1]["query_longitude"] is None

    with pytest.raises(ValueError, match="between 0 and 15"):
        add_weather_query_cell(source, resolution=16)


def test_databricks_h3_expression_path_is_schema_compatible(
    spark: SparkSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_dbf = SimpleNamespace(
        h3_longlatash3=lambda _longitude, _latitude, _resolution: F.lit(12345),
        h3_centerasgeojson=lambda _cell: F.lit(
            '{"type":"Point","coordinates":[-96.8,32.8]}'
        ),
    )
    monkeypatch.setattr(h3_module, "dbf", fake_dbf)
    source = spark.createDataFrame(
        [(32.8, -96.8)],
        "latitude double, longitude double",
    )
    cell = add_weather_query_cell(source, resolution=6)
    assert cell.first()["weather_query_cell_id"] == 12345
    center = extract_h3_centers(cell).first()
    assert center["query_latitude"] == pytest.approx(32.8)
    assert center["query_longitude"] == pytest.approx(-96.8)


def test_databricks_runtime_never_uses_local_h3_fallback(
    spark: SparkSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(h3_module, "dbf", None)
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "17.1")
    source = spark.createDataFrame(
        [(32.8, -96.8)],
        "latitude double, longitude double",
    )

    with pytest.raises(RuntimeError, match="refusing to use the local"):
        add_weather_query_cell(source, resolution=6)


def test_weather_manifest_is_filtered_deterministic_and_year_grained(
    spark: SparkSession,
) -> None:
    crimes = (
        spark.createDataFrame(
            [
                (32.7767, -96.7970, "2023-04-10 13:15:00"),
                (32.7767, -96.7970, "2023-09-11 03:10:00"),
                (32.7767, -96.7970, "2024-03-10 07:00:00"),
                (95.0, -96.7970, "2023-01-01 00:00:00"),
                (32.7767, -96.7970, "2025-01-01 00:00:00"),
                (None, -96.7970, "2023-01-01 00:00:00"),
            ],
            "latitude double, longitude double, occurred_at_text string",
        )
        .withColumn("occurred_at", F.to_timestamp("occurred_at_text"))
        .drop("occurred_at_text")
    )
    arguments = {
        "model": " ERA5_LAND ",
        "hourly_variables": ["temperature_2m", " temperature_2m "],
        "h3_resolution": 6,
        "availability_cutoff": date(2024, 6, 30),
    }
    manifest = build_weather_request_manifest(crimes, **arguments)
    replay = build_weather_request_manifest(crimes.repartition(2), **arguments)
    rows = manifest.orderBy("start_date").collect()

    assert len(rows) == 2
    assert [row["crime_record_count"] for row in rows] == [2, 1]
    assert rows[0]["start_date"] == date(2023, 1, 1)
    assert rows[0]["end_date"] == date(2023, 12, 31)
    assert rows[1]["start_date"] == date(2024, 1, 1)
    assert rows[1]["end_date"] == date(2024, 6, 30)
    assert all(row["hourly_variables"] == ["temperature_2m"] for row in rows)
    assert {row["request_id"] for row in rows} == {
        row["request_id"] for row in replay.collect()
    }
    validate_manifest(manifest, minimum_request_count=2)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model": "not-a-model"}, "Unsupported model"),
        ({"h3_resolution": -1}, "between 0 and 15"),
    ],
)
def test_weather_manifest_rejects_invalid_configuration(
    spark: SparkSession,
    overrides: dict[str, object],
    message: str,
) -> None:
    crimes = spark.createDataFrame(
        [(32.8, -96.8, datetime(2023, 1, 1))],
        "latitude double, longitude double, occurred_at timestamp",
    )
    arguments: dict[str, object] = {
        "model": "era5",
        "hourly_variables": ("temperature_2m",),
        "h3_resolution": 6,
        "availability_cutoff": date(2023, 12, 31),
        **overrides,
    }
    with pytest.raises(ValueError, match=message):
        build_weather_request_manifest(crimes, **arguments)  # type: ignore[arg-type]


def test_weather_manifest_rejects_missing_input_columns(
    spark: SparkSession,
) -> None:
    with pytest.raises(ValueError, match="longitude, occurred_at"):
        build_weather_request_manifest(
            spark.createDataFrame([(32.8,)], "latitude double"),
            model="era5",
            hourly_variables=("temperature_2m",),
        )


def test_manifest_quality_validation_failure_modes(
    spark: SparkSession,
) -> None:
    valid = spark.createDataFrame(
        [
            ("request-1", 123, date(2023, 1, 1), date(2023, 12, 31)),
        ],
        """
        request_id string,
        weather_query_cell_id long,
        start_date date,
        end_date date
        """,
    )
    with pytest.raises(RuntimeError, match="unexpectedly small"):
        validate_manifest(valid, minimum_request_count=2)

    duplicate = valid.unionByName(valid)
    with pytest.raises(RuntimeError, match="duplicate request IDs"):
        validate_manifest(duplicate, minimum_request_count=0)

    missing = valid.withColumn("start_date", F.lit(None).cast("date"))
    with pytest.raises(RuntimeError, match="missing request keys"):
        validate_manifest(missing, minimum_request_count=0)
