from __future__ import annotations

import polars as pl

from crimenet_data.assets.crime.canonical.projection import project_source_fields
from crimenet_data.assets.crime.common.expressions import snake_case_columns
from crimenet_data.assets.crime.sources.base import SourcePattern

PARQUET = (SourcePattern("**/*.parquet", "parquet"),)
CSV = (
    SourcePattern(
        "**/*.csv",
        "csv",
        {
            "infer_schema": False,
            "ignore_errors": True,
            "truncate_ragged_lines": True,
            "missing_columns": "insert",
        },
    ),
)
GEOJSON = (SourcePattern("**/*.geojson", "geojson"),)


def prepare_snake_case(lf: pl.LazyFrame) -> pl.LazyFrame:
    return snake_case_columns(lf)


def nullable_string() -> pl.Expr:
    return pl.lit(None, dtype=pl.String)


def nullable_float() -> pl.Expr:
    return pl.lit(None, dtype=pl.Float64)


def adapt_standard(
    lf: pl.LazyFrame,
    occurrence: pl.Expr,
    *,
    source_record_id: pl.Expr,
    report_timestamp: pl.Expr,
    source_offense_code: pl.Expr,
    source_offense_category: pl.Expr,
    source_offense_description: pl.Expr,
    latitude: pl.Expr,
    longitude: pl.Expr,
    location_label: pl.Expr,
    location_type: pl.Expr,
    police_district: pl.Expr,
    local_area: pl.Expr,
) -> pl.LazyFrame:
    return project_source_fields(
        lf,
        {
            "source_record_id": source_record_id,
            "report_timestamp": report_timestamp,
            "source_offense_code": source_offense_code,
            "source_offense_category": source_offense_category,
            "source_offense_description": source_offense_description,
            "latitude": latitude,
            "longitude": longitude,
            "location_label": location_label,
            "location_type": location_type,
            "police_district": police_district,
            "local_area": local_area,
        },
        occurrence,
    )
