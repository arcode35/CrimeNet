from __future__ import annotations

import inspect
from datetime import UTC, datetime

import polars as pl
import pytest

from crimenet_data.assets.crime.canonical.crosswalk import (
    cleanse_canonical_source,
)
from crimenet_data.assets.crime.common import source_bounds as source_bounds_module
from crimenet_data.assets.crime.common.source_bounds import (
    SOURCE_COORDINATE_BOUNDS,
    apply_source_coordinate_bounds,
    globally_valid_coordinate_expr,
    source_bounds_expr,
    source_coordinate_bounds_summary,
)
from crimenet_data.assets.crime.sources import SILVER_SOURCE_KEYS
from crimenet_data.source_bounds_audit import audit_source_frame


def _coordinate_frame(
    latitude: float | None,
    longitude: float | None,
    *,
    include_in_model: bool = True,
) -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "crime_id": ["source:record"],
            "latitude": [latitude],
            "longitude": [longitude],
            "include_in_model": [include_in_model],
        }
    )


@pytest.mark.parametrize("source_city", SILVER_SOURCE_KEYS)
def test_middle_and_inclusive_edges_are_inside_bounds(source_city: str) -> None:
    bounds = SOURCE_COORDINATE_BOUNDS[source_city]
    middle_lat = (bounds.min_lat + bounds.max_lat) / 2
    middle_lon = (bounds.min_lon + bounds.max_lon) / 2
    frame = pl.DataFrame(
        {
            "latitude": [
                middle_lat,
                bounds.min_lat,
                bounds.max_lat,
                middle_lat,
                middle_lat,
            ],
            "longitude": [
                middle_lon,
                middle_lon,
                middle_lon,
                bounds.min_lon,
                bounds.max_lon,
            ],
        }
    )

    assert frame.select(source_bounds_expr(source_city).all()).item()


@pytest.mark.parametrize("source_city", SILVER_SOURCE_KEYS)
def test_each_source_bound_rejects_values_just_outside(source_city: str) -> None:
    bounds = SOURCE_COORDINATE_BOUNDS[source_city]
    epsilon = 1e-7
    middle_lat = (bounds.min_lat + bounds.max_lat) / 2
    middle_lon = (bounds.min_lon + bounds.max_lon) / 2
    frame = pl.DataFrame(
        {
            "latitude": [
                bounds.min_lat - epsilon,
                bounds.max_lat + epsilon,
                middle_lat,
                middle_lat,
            ],
            "longitude": [
                middle_lon,
                middle_lon,
                bounds.min_lon - epsilon,
                bounds.max_lon + epsilon,
            ],
        }
    )

    assert not frame.select(source_bounds_expr(source_city).any()).item()


@pytest.mark.parametrize(
    ("latitude", "longitude"),
    [
        (91.0, -84.4),
        (33.7, -181.0),
        (float("inf"), -84.4),
        (33.7, float("nan")),
        (None, -84.4),
        (33.7, None),
        (0.0, 0.0),
    ],
)
def test_generic_invalid_coordinates_remain_independently_invalid(
    latitude: float | None,
    longitude: float | None,
) -> None:
    frame = pl.DataFrame({"latitude": [latitude], "longitude": [longitude]})

    assert not frame.select(globally_valid_coordinate_expr()).item()
    summary = source_coordinate_bounds_summary(frame.lazy(), "atlanta").collect()
    assert summary["globally_valid_coordinate_rows"].item() == 0
    assert summary["outside_source_bounds_rows"].item() == 0


def test_generic_cleanse_keeps_world_valid_source_outlier_but_drops_zero_zero() -> None:
    frame = pl.LazyFrame(
        {
            "occurrence_year": [2024, 2024],
            "latitude": [35.0, 0.0],
            "longitude": [-84.4, 0.0],
        }
    )

    result = cleanse_canonical_source(frame, "atlanta").collect()
    assert result.height == 1
    assert result.select("latitude", "longitude").row(0) == (35.0, -84.4)


def test_bounds_are_and_only_model_eligibility_gate_without_row_mutation() -> None:
    source_city = "atlanta"
    input_frame = pl.concat(
        [
            _coordinate_frame(33.75, -84.4, include_in_model=False),
            _coordinate_frame(35.0, -84.4, include_in_model=True).with_columns(
                pl.lit("source:outside").alias("crime_id")
            ),
        ]
    )
    first = apply_source_coordinate_bounds(input_frame, source_city).collect()
    second = apply_source_coordinate_bounds(input_frame, source_city).collect()

    assert first.equals(second)
    assert first.height == 2
    assert first["crime_id"].to_list() == ["source:record", "source:outside"]
    assert first["latitude"].to_list() == [33.75, 35.0]
    assert first["longitude"].to_list() == [-84.4, -84.4]
    assert first["source_coordinate_bounds_valid"].to_list() == [True, False]
    assert first["include_in_model"].to_list() == [False, False]


def test_registry_exactly_matches_modeled_sources_and_unknown_fails() -> None:
    assert set(SOURCE_COORDINATE_BOUNDS) == set(SILVER_SOURCE_KEYS)
    with pytest.raises(KeyError, match="No coordinate bounds configured"):
        source_bounds_expr("unknown_source")


def test_bounds_implementation_uses_vectorized_polars_without_row_iteration() -> None:
    implementation = inspect.getsource(source_bounds_module)
    assert "iter_rows" not in implementation
    assert "map_elements" not in implementation
    assert "shapely" not in implementation.lower()
    assert "geopandas" not in implementation.lower()


def test_audit_ranks_outliers_and_applies_five_percent_safety_gate() -> None:
    frame = pl.DataFrame(
        {
            "crime_id": ["atlanta:inside", "atlanta:outside"],
            "source_city": ["atlanta", "atlanta"],
            "occurrence_timestamp": [
                datetime(2024, 1, 1),  # noqa: DTZ001 - source-local Silver time
                datetime(2024, 1, 2),  # noqa: DTZ001 - source-local Silver time
            ],
            "latitude": [33.75, 40.0],
            "longitude": [-84.4, -100.0],
            "source_offense_code": ["A", "B"],
            "source_offense_category": ["A", "B"],
            "source_offense_description": ["A", "B"],
            "include_in_model": [True, True],
            "ingested_at_utc": [datetime(2026, 1, 1, tzinfo=UTC)] * 2,
        }
    )

    summary, outliers = audit_source_frame(frame, "atlanta")
    assert summary["newly_excluded_rows"] == 1
    assert summary["modeled_rows_after"] == 1
    assert summary["safety_threshold_exceeded"] is True
    assert outliers["crime_id"].to_list() == ["atlanta:outside"]
    assert outliers["max_bound_excess_degrees"].item() > 0
