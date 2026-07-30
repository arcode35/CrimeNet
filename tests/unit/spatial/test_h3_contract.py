from __future__ import annotations

import pytest
from pyspark.sql import SparkSession

from crimenet.spatial.h3 import add_weather_query_cell

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("resolution", [-1, 16])
def test_h3_resolution_contract_fails_before_runtime_dispatch(
    spark: SparkSession,
    resolution: int,
) -> None:
    coordinates = spark.createDataFrame(
        [(32.7767, -96.7970)],
        "latitude double, longitude double",
    )

    with pytest.raises(ValueError, match="between 0 and 15"):
        add_weather_query_cell(
            coordinates,
            resolution=resolution,
        )


def test_h3_enrichment_reports_local_runtime_limitation(
    spark: SparkSession,
) -> None:
    coordinates = spark.createDataFrame(
        [(32.7767, -96.7970)],
        "latitude double, longitude double",
    )

    with pytest.raises(RuntimeError, match="Databricks H3"):
        add_weather_query_cell(coordinates)
