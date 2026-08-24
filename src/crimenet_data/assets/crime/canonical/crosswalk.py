from __future__ import annotations

import polars as pl

from crimenet_data.assets.crime.canonical.schema import (
    CANONICAL_CLASSIFICATION_COLUMNS,
    CANONICAL_CRIME_SCHEMA,
    CANONICAL_MAPPING_VERSION,
)


def apply_canonical_crosswalk(
    lf: pl.LazyFrame,
    crosswalk_lf: pl.LazyFrame,
    source_key: str,
) -> pl.LazyFrame:
    from crimenet_data.assets.crime.sources.registry import get_source

    config = get_source(source_key).config
    selected = [*config.crosswalk_keys, *CANONICAL_CLASSIFICATION_COLUMNS]
    source_crosswalk = crosswalk_lf.filter(pl.col("source_city") == source_key).select(
        selected
    )
    return lf.join(
        source_crosswalk,
        on=list(config.crosswalk_keys),
        how="left",
        validate="m:1",
    )


def cleanse_canonical_source(lf: pl.LazyFrame, source_key: str) -> pl.LazyFrame:
    from crimenet_data.assets.crime.sources.registry import get_source

    config = get_source(source_key).config
    lf = lf.with_columns(
        pl.col("latitude").cast(pl.Float64, strict=False),
        pl.col("longitude").cast(pl.Float64, strict=False),
    )
    year_is_valid = pl.col("occurrence_year").is_between(2014, 2026).fill_null(False)
    if config.coordinate_bounds is None:
        return lf.filter(year_is_valid)

    min_latitude, max_latitude, min_longitude, max_longitude = config.coordinate_bounds
    coordinates_are_valid = (
        pl.col("latitude").is_finite()
        & pl.col("longitude").is_finite()
        & pl.col("latitude").is_between(min_latitude, max_latitude)
        & pl.col("longitude").is_between(min_longitude, max_longitude)
    ).fill_null(False)
    coordinates_are_absent = (
        pl.col("latitude").is_null() & pl.col("longitude").is_null()
    )
    location_is_valid = (
        coordinates_are_valid
        if config.coordinates_required
        else coordinates_are_absent | coordinates_are_valid
    )
    return lf.filter(year_is_valid & location_is_valid)


def project_canonical_schema(lf: pl.LazyFrame, source_key: str) -> pl.LazyFrame:
    from crimenet_data.assets.crime.sources.registry import get_source

    expressions: list[pl.Expr] = []
    available = set(lf.collect_schema().names())
    defaults: dict[str, pl.Expr] = {
        "crime_id": pl.concat_str(
            [pl.lit(source_key), pl.col("source_record_id")],
            separator=":",
            ignore_nulls=False,
        ),
        "source_city": pl.lit(source_key),
        "source_timezone": pl.lit(get_source(source_key).config.timezone),
        "mapping_version": pl.col("mapping_version").fill_null(
            CANONICAL_MAPPING_VERSION
        ),
        "canonical_mapping_found": pl.col("canonical_family_code").is_not_null(),
        "mapping_confidence": pl.col("mapping_confidence").fill_null("unmatched"),
        "review_required": pl.col("review_required").fill_null(True),
        "mapping_action": pl.col("mapping_action").fill_null("unmatched"),
        "include_in_model": pl.col("include_in_model").fill_null(False),
    }
    for name, dtype in CANONICAL_CRIME_SCHEMA.items():
        expression = defaults.get(name)
        if expression is None:
            expression = pl.col(name) if name in available else pl.lit(None)
        expressions.append(expression.cast(dtype, strict=False).alias(name))
    return lf.select(expressions)
