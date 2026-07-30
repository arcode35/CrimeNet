from __future__ import annotations

import pytest
from pyspark.sql import SparkSession

from crimenet.gold.crime_features import (
    spatially_map_locations,
    validate_boundary_inputs,
)

pytestmark = pytest.mark.databricks


def test_native_geometry_maps_inside_boundary_and_outside_points(
    spark: SparkSession,
) -> None:
    calendar = spark.createDataFrame(
        [(2020,)],
        "tiger_line_year int",
    )
    boundaries = spark.sql(
        """
        SELECT
            2020 AS boundary_vintage,
            '48001000100' AS geoid,
            ST_GeomFromText(
                'POLYGON((-96 29, -95 29, -95 30, -96 30, -96 29))',
                4326
            ) AS tract_geometry
        """
    )
    validate_boundary_inputs(calendar, boundaries)

    locations = spark.createDataFrame(
        [
            (2020, 29.5, -95.5),
            (2020, 29.5, -96.0),
            (2020, 31.0, -97.0),
        ],
        "tiger_line_year int, latitude double, longitude double",
    )

    mapped = {
        (row.latitude, row.longitude): row.tract_geoid
        for row in spatially_map_locations(
            locations,
            boundaries,
        ).collect()
    }

    assert mapped == {
        (29.5, -95.5): "48001000100",
        (29.5, -96.0): "48001000100",
        (31.0, -97.0): None,
    }
