"""Conservative rectangular coordinate bounds for modeled crime sources."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import polars as pl

SOURCE_COORDINATE_BOUNDS_VALID_COLUMN = "source_coordinate_bounds_valid"


@dataclass(frozen=True)
class CoordinateBounds:
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    def __post_init__(self) -> None:
        if self.min_lat > self.max_lat:
            raise ValueError("Coordinate bounds have reversed latitude limits")
        if self.min_lon > self.max_lon:
            raise ValueError("Coordinate bounds have reversed longitude limits")


SOURCE_COORDINATE_BOUNDS: Mapping[str, CoordinateBounds] = MappingProxyType(
    {
        "atlanta": CoordinateBounds(33.55, 34.01, -84.66, -84.14),
        "baltimore": CoordinateBounds(39.14, 39.44, -76.78, -76.47),
        "chandler_az": CoordinateBounds(33.14, 33.43, -112.04, -111.69),
        "chicago": CoordinateBounds(41.58, 42.09, -88.01, -87.46),
        "dallas": CoordinateBounds(32.55, 33.09, -97.07, -96.40),
        "denver": CoordinateBounds(39.54, 39.99, -105.28, -104.51),
        "fort_worth": CoordinateBounds(32.48, 33.12, -97.67, -96.96),
        "los_angeles_county_sheriff": CoordinateBounds(32.69, 34.93, -119.06, -117.54),
        "marin_county_sheriff_ca": CoordinateBounds(37.70, 38.43, -123.14, -122.24),
        "montgomery_county_md": CoordinateBounds(38.88, 39.40, -77.58, -76.84),
        "new_york": CoordinateBounds(40.43, 40.98, -74.33, -73.63),
        "san_francisco": CoordinateBounds(37.60, 37.90, -122.58, -122.30),
        "seattle": CoordinateBounds(47.42, 47.81, -122.50, -122.17),
        "sonoma_county_sheriff_ca": CoordinateBounds(37.95, 38.97, -123.65, -122.24),
        "washington_dc": CoordinateBounds(38.74, 39.06, -77.19, -76.83),
    }
)


def get_source_coordinate_bounds(source_city: str) -> CoordinateBounds:
    try:
        return SOURCE_COORDINATE_BOUNDS[source_city]
    except KeyError as error:
        raise KeyError(
            f"No coordinate bounds configured for source {source_city!r}; "
            f"configured sources={sorted(SOURCE_COORDINATE_BOUNDS)}"
        ) from error


def validate_source_coordinate_bounds(source_cities: Iterable[str]) -> None:
    """Fail when modeled-source registration and bounds configuration diverge."""

    modeled = set(source_cities)
    configured = set(SOURCE_COORDINATE_BOUNDS)
    missing = sorted(modeled - configured)
    extra = sorted(configured - modeled)
    if missing or extra:
        raise ValueError(
            "Modeled crime source coordinate-bounds registry mismatch: "
            f"missing={missing}, extra={extra}"
        )


def globally_valid_coordinate_expr() -> pl.Expr:
    """Return the existing generic finite/world/zero coordinate contract."""

    latitude = pl.col("latitude")
    longitude = pl.col("longitude")
    return (
        latitude.is_not_null()
        & longitude.is_not_null()
        & latitude.is_finite()
        & longitude.is_finite()
        & latitude.is_between(-90.0, 90.0, closed="both")
        & longitude.is_between(-180.0, 180.0, closed="both")
        & ~((latitude == 0.0) & (longitude == 0.0))
    ).fill_null(False)


def source_bounds_expr(source_city: str) -> pl.Expr:
    """Return an inclusive, vectorized source-box validity expression."""

    bounds = get_source_coordinate_bounds(source_city)
    return (
        globally_valid_coordinate_expr()
        & pl.col("latitude").is_between(
            bounds.min_lat,
            bounds.max_lat,
            closed="both",
        )
        & pl.col("longitude").is_between(
            bounds.min_lon,
            bounds.max_lon,
            closed="both",
        )
    ).fill_null(False)


def apply_source_coordinate_bounds(
    lf: pl.LazyFrame,
    source_city: str,
) -> pl.LazyFrame:
    """Add the audit flag and make bounds an AND-only model eligibility gate."""

    bounded = lf.with_columns(
        source_bounds_expr(source_city).alias(SOURCE_COORDINATE_BOUNDS_VALID_COLUMN)
    )
    return bounded.with_columns(
        (
            pl.col("include_in_model").fill_null(False)
            & pl.col(SOURCE_COORDINATE_BOUNDS_VALID_COLUMN)
        ).alias("include_in_model")
    )


def source_coordinate_bounds_summary(
    lf: pl.LazyFrame,
    source_city: str,
) -> pl.LazyFrame:
    """Return one structured source-level coordinate sanity summary."""

    bounds = get_source_coordinate_bounds(source_city)
    globally_valid = globally_valid_coordinate_expr()
    inside = source_bounds_expr(source_city)
    outside = globally_valid & ~inside
    total = pl.len()
    return lf.select(
        pl.lit(source_city).alias("source_city"),
        total.alias("input_rows"),
        globally_valid.sum().alias("globally_valid_coordinate_rows"),
        inside.sum().alias("inside_source_bounds_rows"),
        outside.sum().alias("outside_source_bounds_rows"),
        pl.when(total > 0)
        .then(100.0 * outside.sum() / total)
        .otherwise(0.0)
        .alias("outside_source_bounds_pct"),
        (globally_valid & (pl.col("latitude") < bounds.min_lat))
        .sum()
        .alias("outside_min_lat"),
        (globally_valid & (pl.col("latitude") > bounds.max_lat))
        .sum()
        .alias("outside_max_lat"),
        (globally_valid & (pl.col("longitude") < bounds.min_lon))
        .sum()
        .alias("outside_min_lon"),
        (globally_valid & (pl.col("longitude") > bounds.max_lon))
        .sum()
        .alias("outside_max_lon"),
    )


__all__ = [
    "SOURCE_COORDINATE_BOUNDS",
    "SOURCE_COORDINATE_BOUNDS_VALID_COLUMN",
    "CoordinateBounds",
    "apply_source_coordinate_bounds",
    "get_source_coordinate_bounds",
    "globally_valid_coordinate_expr",
    "source_bounds_expr",
    "source_coordinate_bounds_summary",
    "validate_source_coordinate_bounds",
]
