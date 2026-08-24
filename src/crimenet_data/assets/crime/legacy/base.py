from __future__ import annotations

from collections.abc import Mapping

import polars as pl

from crimenet_data.assets.crime.common.expressions import numeric_expr


def project_adapter_fields(
    lf: pl.LazyFrame,
    fields: Mapping[str, pl.Expr],
    occurrence_timestamp: pl.Expr,
) -> pl.LazyFrame:
    occurrence_year = pl.coalesce(
        [
            occurrence_timestamp.dt.year(),
            numeric_expr(lf, "occurrence_year", "year", dtype=pl.Int16),
        ]
    )
    return lf.select(
        *[expression.alias(column_name) for column_name, expression in fields.items()],
        occurrence_timestamp.alias("occurrence_timestamp"),
        occurrence_year.alias("occurrence_year"),
    )
