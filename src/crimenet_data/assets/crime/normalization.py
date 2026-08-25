from __future__ import annotations

from typing import Literal

import duckdb
import polars as pl

from crimenet_data.assets.crime.sources import get_source

NormalizationType = Literal[
    "unix_ms_timestamp",
    "sonoma_location_coordinates",
    "dallas_epsg2276_to_epsg4326",
    "none",
]

_UNIX_MS_COLUMNS = {
    "washington_dc": "occurred_at_raw",
    "baltimore": "crime_date_time",
}
_SONOMA_SOURCE_KEY = "sonoma_county_sheriff_ca"
_DALLAS_SOURCE_KEY = "dallas"


def normalize_unix_ms_timestamp(
    lf: pl.LazyFrame,
    source_column: str,
) -> pl.LazyFrame:
    """Parse one explicitly audited Unix-millisecond source timestamp."""

    if source_column not in lf.collect_schema().names():
        raise KeyError(f"Unix-millisecond source column is missing: {source_column!r}")
    occurrence_timestamp = pl.from_epoch(
        pl.col(source_column).cast(pl.Int64, strict=False),
        time_unit="ms",
    ).cast(pl.Datetime("us"))
    return lf.with_columns(
        occurrence_timestamp.alias("occurrence_timestamp"),
        occurrence_timestamp.dt.year().cast(pl.Int16).alias("occurrence_year"),
    )


def normalize_sonoma_coordinates(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Extract publisher-supplied ``(latitude, longitude)`` coordinates."""

    if "location" not in lf.collect_schema().names():
        raise KeyError("Sonoma source column is missing: 'location'")
    coordinate = r"-?(?:\d+(?:\.\d*)?|\.\d+)"
    location = pl.col("location").cast(pl.String, strict=False)
    return lf.with_columns(
        location.str.extract(rf"\(\s*({coordinate})\s*,", 1)
        .cast(pl.Float64, strict=False)
        .alias("latitude"),
        location.str.extract(rf",\s*({coordinate})\s*\)", 1)
        .cast(pl.Float64, strict=False)
        .alias("longitude"),
    )


def convert_dallas_coordinates(
    lf: pl.LazyFrame,
    connection: duckdb.DuckDBPyConnection,
) -> pl.LazyFrame:
    """Convert Dallas State Plane coordinates from EPSG:2276 to WGS84."""

    coordinate_columns = {"x_coordinate", "y_cordinate"}
    available = set(lf.collect_schema().names())
    missing = coordinate_columns - available
    if missing:
        raise KeyError(f"Dallas coordinate columns are missing: {sorted(missing)}")

    existing_output = [
        column for column in ("latitude", "longitude") if column in available
    ]
    if existing_output:
        lf = lf.drop(existing_output)

    return connection.sql(
        """
        WITH coordinates AS (
            SELECT
                *,
                TRY_CAST(x_coordinate AS DOUBLE) AS state_plane_x,
                TRY_CAST(y_cordinate AS DOUBLE) AS state_plane_y
            FROM lf
        ),
        transformed AS (
            SELECT
                * EXCLUDE (state_plane_x, state_plane_y),
                CASE
                    WHEN state_plane_x IS NULL
                        OR state_plane_y IS NULL
                        OR NOT isfinite(state_plane_x)
                        OR NOT isfinite(state_plane_y)
                    THEN NULL
                    ELSE ST_Transform(
                        ST_Point(state_plane_x, state_plane_y),
                        'EPSG:2276',
                        'EPSG:4326',
                        always_xy := true
                    )
                END AS wgs84_point
            FROM coordinates
        )
        SELECT
            * EXCLUDE (wgs84_point),
            ST_Y(wgs84_point) AS latitude,
            ST_X(wgs84_point) AS longitude
        FROM transformed
        """
    ).pl(lazy=True)


def normalization_type(source_key: str) -> NormalizationType:
    """Describe the configured pre-Silver representation correction."""

    get_source(source_key)
    if source_key in _UNIX_MS_COLUMNS:
        return "unix_ms_timestamp"
    if source_key == _SONOMA_SOURCE_KEY:
        return "sonoma_location_coordinates"
    if source_key == _DALLAS_SOURCE_KEY:
        return "dallas_epsg2276_to_epsg4326"
    return "none"


def normalization_requires_duckdb(source_key: str) -> bool:
    return normalization_type(source_key) == "dallas_epsg2276_to_epsg4326"


def normalize_source(
    lf: pl.LazyFrame,
    source_key: str,
    *,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> pl.LazyFrame:
    """Apply the one source-specific normalization boundary before Silver."""

    kind = normalization_type(source_key)
    if kind == "unix_ms_timestamp":
        return normalize_unix_ms_timestamp(lf, _UNIX_MS_COLUMNS[source_key])
    if kind == "sonoma_location_coordinates":
        return normalize_sonoma_coordinates(lf)
    if kind == "dallas_epsg2276_to_epsg4326":
        if connection is None:
            raise ValueError("Dallas normalization requires a DuckDB connection")
        return convert_dallas_coordinates(lf, connection)
    return lf


__all__ = [
    "NormalizationType",
    "convert_dallas_coordinates",
    "normalization_requires_duckdb",
    "normalization_type",
    "normalize_sonoma_coordinates",
    "normalize_source",
    "normalize_unix_ms_timestamp",
]
