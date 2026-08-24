from __future__ import annotations

from collections.abc import Mapping

import polars as pl

from crimenet_data.assets.crime.canonical.schema import SOURCE_PROJECTION_SCHEMA
from crimenet_data.assets.crime.common.expressions import numeric_expr


def project_source_fields(
    lf: pl.LazyFrame,
    fields: Mapping[str, pl.Expr],
    occurrence_timestamp: pl.Expr,
) -> pl.LazyFrame:
    """Project source-native Bronze into the stable pre-crosswalk boundary."""

    occurrence_year = pl.coalesce(
        [
            occurrence_timestamp.dt.year(),
            numeric_expr(lf, "occurrence_year", dtype=pl.Int16),
        ]
    )
    expressions = {
        **fields,
        "occurrence_timestamp": occurrence_timestamp,
        "occurrence_year": occurrence_year,
        "source_file_uri": pl.col("source_file_uri"),
        "ingestion_run_id": pl.col("ingestion_run_id"),
        "ingested_at_utc": pl.col("ingested_at_utc"),
    }
    missing = set(SOURCE_PROJECTION_SCHEMA) - set(expressions)
    if missing:
        raise KeyError(f"Source adapter projection is incomplete: {sorted(missing)}")
    return lf.select(
        expression.cast(SOURCE_PROJECTION_SCHEMA[name], strict=False).alias(name)
        for name, expression in expressions.items()
    ).select(SOURCE_PROJECTION_SCHEMA.names())
