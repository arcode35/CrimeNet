from __future__ import annotations

from datetime import datetime

import pytest
from pyspark.sql import SparkSession

from crimenet.silver.lighting import extract_lighting_keys
from crimenet.spatial.h3 import (
    add_weather_query_cell,
    extract_h3_centers,
)

pytestmark = pytest.mark.databricks


def test_native_h3_enrichment_and_center_extraction(
    spark: SparkSession,
) -> None:
    coordinates = spark.createDataFrame(
        [(32.7767, -96.7970)],
        "latitude double, longitude double",
    )

    with_cells = add_weather_query_cell(
        coordinates,
        resolution=6,
    )
    row = extract_h3_centers(
        with_cells.select("weather_query_cell_id")
    ).first()

    assert isinstance(row.weather_query_cell_id, int)
    assert row.query_latitude == pytest.approx(32.7767, abs=0.2)
    assert row.query_longitude == pytest.approx(-96.7970, abs=0.2)


def test_native_h3_lighting_keys_use_unique_cell_hours(
    spark: SparkSession,
) -> None:
    crimes = spark.createDataFrame(
        [
            (32.7767, -96.7970, datetime(2022, 7, 17, 8, 15)),
            (32.7767, -96.7970, datetime(2022, 7, 17, 8, 55)),
        ],
        "latitude double, longitude double, occurred_at timestamp",
    )
    crimes = add_weather_query_cell(crimes, resolution=6)

    keys = extract_lighting_keys(crimes)

    assert keys.count() == 1
    row = keys.first()
    assert row.solar_timestamp_hour == datetime(2022, 7, 17, 8)
    assert row.query_latitude == pytest.approx(32.7767, abs=0.2)
    assert row.query_longitude == pytest.approx(-96.7970, abs=0.2)
