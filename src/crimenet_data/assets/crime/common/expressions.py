from __future__ import annotations

import re
from collections.abc import Iterable

import polars as pl


def snake_case_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Normalize source headings for adapter use, preserving collisions."""

    used: set[str] = set()
    mapping: dict[str, str] = {}
    for source_name in lf.collect_schema().names():
        normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", source_name)
        normalized = re.sub(r"[^A-Za-z0-9]+", "_", normalized).strip("_").lower()
        normalized = normalized or "unnamed"
        candidate = normalized
        suffix = 2
        while candidate in used:
            candidate = f"{normalized}_{suffix}"
            suffix += 1
        used.add(candidate)
        if source_name != candidate:
            mapping[source_name] = candidate
    return lf.rename(mapping)


def existing_columns(lf: pl.LazyFrame, candidates: Iterable[str]) -> list[str]:
    names = set(lf.collect_schema().names())
    return [name for name in candidates if name in names]


def text_expr(lf: pl.LazyFrame, *candidates: str) -> pl.Expr:
    columns = existing_columns(lf, candidates)
    if not columns:
        return pl.lit(None, dtype=pl.String)
    values = [
        pl.col(column).cast(pl.String, strict=False).str.strip_chars()
        for column in columns
    ]
    value = pl.coalesce(values)
    return pl.when(value == "").then(pl.lit(None, dtype=pl.String)).otherwise(value)


def numeric_expr(
    lf: pl.LazyFrame,
    *candidates: str,
    dtype: pl.DataType = pl.Float64,
) -> pl.Expr:
    columns = existing_columns(lf, candidates)
    if not columns:
        return pl.lit(None, dtype=dtype)
    return pl.coalesce([pl.col(column).cast(dtype, strict=False) for column in columns])


def datetime_value(expr: pl.Expr) -> pl.Expr:
    value = expr.cast(pl.String, strict=False).str.strip_chars()
    return pl.coalesce(
        [
            value.str.to_datetime(
                format="%Y-%m-%dT%H:%M:%S%.f",
                strict=False,
            ),
            value.str.to_datetime(
                format="%Y-%m-%d %H:%M:%S%.f",
                strict=False,
            ),
            value.str.to_datetime(format="%m/%d/%Y %I:%M:%S %p", strict=False),
            value.str.to_datetime(format="%m/%d/%Y %H:%M:%S", strict=False),
            value.str.to_datetime(format="%Y-%m-%d %H:%M:%S", strict=False),
            value.str.to_datetime(format="%m/%d/%Y %H%M", strict=False),
            value.str.to_datetime(format="%Y-%m-%d %H%M", strict=False),
            value.str.to_datetime(format="%m/%d/%Y", strict=False),
            value.str.to_datetime(format="%Y-%m-%d", strict=False),
            pl.from_epoch(value.cast(pl.Int64, strict=False), time_unit="ms"),
        ]
    ).cast(pl.Datetime("us"))


def datetime_expr(lf: pl.LazyFrame, *candidates: str) -> pl.Expr:
    columns = existing_columns(lf, candidates)
    if not columns:
        return pl.lit(None, dtype=pl.Datetime("us"))
    return pl.coalesce([datetime_value(pl.col(column)) for column in columns])


def date_time_expr(
    lf: pl.LazyFrame,
    date_candidates: tuple[str, ...],
    time_candidates: tuple[str, ...],
) -> pl.Expr:
    dates = existing_columns(lf, date_candidates)
    times = existing_columns(lf, time_candidates)
    combined: list[pl.Expr] = []
    for date_column in dates:
        for time_column in times:
            combined.append(
                datetime_value(
                    pl.concat_str(
                        [
                            pl.col(date_column)
                            .cast(pl.String, strict=False)
                            .str.slice(0, 10),
                            pl.col(time_column),
                        ],
                        separator=" ",
                        ignore_nulls=False,
                    )
                )
            )
    return pl.coalesce([*combined, datetime_expr(lf, *date_candidates)])


def date_hour_expr(
    lf: pl.LazyFrame,
    date_column: str,
    hour_column: str,
) -> pl.Expr:
    """Represent a source-reported hour without inventing minute precision."""

    columns = set(lf.collect_schema().names())
    if date_column not in columns or hour_column not in columns:
        return datetime_expr(lf, date_column)
    date = datetime_value(pl.col(date_column)).dt.date()
    hour = pl.col(hour_column).cast(pl.Int64, strict=False)
    return date.cast(pl.Datetime("us")) + pl.duration(hours=hour)


def epoch_milliseconds_expr(lf: pl.LazyFrame, column: str) -> pl.Expr:
    """Parse a field explicitly audited as Unix epoch milliseconds."""

    if column not in lf.collect_schema().names():
        return pl.lit(None, dtype=pl.Datetime("us"))
    return pl.from_epoch(
        pl.col(column).cast(pl.Int64, strict=False),
        time_unit="ms",
    ).cast(pl.Datetime("us"))


def composite_identifier(lf: pl.LazyFrame, *candidates: str) -> pl.Expr:
    columns = existing_columns(lf, candidates)
    if not columns:
        return pl.lit(None, dtype=pl.String)
    values = [text_expr(lf, column).alias(column) for column in columns]
    populated = pl.any_horizontal([value.is_not_null() for value in values])
    identifier = pl.concat_str(values, separator=":", ignore_nulls=True)
    return pl.when(populated).then(identifier).otherwise(pl.lit(None, dtype=pl.String))
