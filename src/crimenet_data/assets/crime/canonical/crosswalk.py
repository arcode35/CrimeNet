from __future__ import annotations

import polars as pl

from crimenet_data.assets.crime.canonical.schema import (
    CANONICAL_CLASSIFICATION_COLUMNS,
    CANONICAL_CRIME_SCHEMA,
    CANONICAL_MAPPING_VERSION,
)

SOURCE_TAXONOMY_COLUMNS = (
    "source_offense_code",
    "source_offense_category",
    "source_offense_description",
    "source_auxiliary",
    "source_severity",
)
CROSSWALK_SOURCE_COLUMNS = SOURCE_TAXONOMY_COLUMNS
CROSSWALK_REQUIRED_COLUMNS = (
    "mapping_version",
    "source_city",
    *CROSSWALK_SOURCE_COLUMNS,
    "canonical_family_code",
    "canonical_offense_family",
    "canonical_subtype_code",
    "canonical_offense_subtype",
    "canonical_domain",
    "canonical_target",
    "is_criminal_event",
    "is_violent",
    "is_property",
    "mapping_confidence",
    "review_required",
    "mapping_action",
    "include_in_model",
)


def _inconsistent_code_rows(
    crosswalk: pl.DataFrame,
    *,
    code: str,
    values: tuple[str, ...],
) -> pl.DataFrame:
    return (
        crosswalk.filter(pl.col(code).is_not_null())
        .group_by(code)
        .agg(*(pl.col(value).n_unique().alias(value) for value in values))
        .filter(pl.any_horizontal(*(pl.col(value) > 1 for value in values)))
    )


def normalize_crosswalk_nulls(crosswalk_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Translate the legacy literal-null sentinel in source taxonomy fields."""

    available = set(crosswalk_lf.collect_schema().names())
    columns = [
        column for column in SOURCE_TAXONOMY_COLUMNS if column in available
    ]
    return crosswalk_lf.with_columns(
        pl.when(pl.col(column) == "null")
        .then(pl.lit(None, dtype=pl.String))
        .otherwise(pl.col(column))
        .alias(column)
        for column in columns
    )


def validate_canonical_crosswalk(
    crosswalk_lf: pl.LazyFrame,
    *,
    source_keys: tuple[str, ...] | None = None,
) -> pl.DataFrame:
    """Validate the small finalized crosswalk before scanning Bronze sources."""

    from crimenet_data.assets.crime.sources import SILVER_SOURCE_KEYS, get_source

    enabled_sources = source_keys or SILVER_SOURCE_KEYS
    crosswalk_lf = normalize_crosswalk_nulls(crosswalk_lf)
    available = set(crosswalk_lf.collect_schema().names())
    missing_columns = set(CROSSWALK_REQUIRED_COLUMNS) - available
    if missing_columns:
        raise ValueError(
            "Canonical crosswalk is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    crosswalk = crosswalk_lf.collect()
    invalid_versions = crosswalk.filter(
        (pl.col("mapping_version") != CANONICAL_MAPPING_VERSION).fill_null(True)
    )
    if not invalid_versions.is_empty():
        versions = crosswalk["mapping_version"].unique().to_list()
        raise ValueError(
            "Canonical crosswalk version mismatch: expected "
            f"{CANONICAL_MAPPING_VERSION!r}, found {versions}"
        )

    for source_key in enabled_sources:
        source_crosswalk = crosswalk.filter(pl.col("source_city") == source_key)
        if source_crosswalk.is_empty():
            raise ValueError(
                f"Canonical crosswalk has no rows for Silver source {source_key!r}"
            )
        keys = get_source(source_key).config.crosswalk_keys
        missing_keys = set(keys) - available
        if missing_keys:
            raise ValueError(
                f"Crosswalk keys for {source_key!r} are missing: "
                f"{sorted(missing_keys)}"
            )
        duplicates = (
            source_crosswalk.group_by(list(keys))
            .len()
            .filter(pl.col("len") > 1)
        )
        if not duplicates.is_empty():
            raise ValueError(
                f"Canonical crosswalk is not unique for {source_key!r} on "
                f"keys={keys}; duplicates={duplicates.head(10).to_dicts()}"
            )

    inconsistent_subtypes = _inconsistent_code_rows(
        crosswalk,
        code="canonical_subtype_code",
        values=(
            "canonical_offense_subtype",
            "canonical_family_code",
            "canonical_offense_family",
        ),
    )
    if not inconsistent_subtypes.is_empty():
        raise ValueError(
            "Canonical subtype codes are inconsistent: "
            f"{inconsistent_subtypes.head(10).to_dicts()}"
        )

    for code, name in (
        ("canonical_family_code", "canonical_offense_family"),
        ("canonical_offense_family", "canonical_family_code"),
    ):
        inconsistent_families = _inconsistent_code_rows(
            crosswalk,
            code=code,
            values=(name,),
        )
        if not inconsistent_families.is_empty():
            raise ValueError(
                "Canonical family codes/names are inconsistent: "
                f"{inconsistent_families.head(10).to_dicts()}"
            )

    return crosswalk


def apply_canonical_crosswalk(
    lf: pl.LazyFrame,
    crosswalk_lf: pl.LazyFrame,
    source_key: str,
) -> pl.LazyFrame:
    from crimenet_data.assets.crime.sources.registry import get_source

    config = get_source(source_key).config
    selected = [*config.crosswalk_keys, *CANONICAL_CLASSIFICATION_COLUMNS]
    source_crosswalk = (
        normalize_crosswalk_nulls(crosswalk_lf)
        .filter(pl.col("source_city") == source_key)
        .select(selected)
    )
    return lf.join(
        source_crosswalk,
        on=list(config.crosswalk_keys),
        how="left",
        validate="m:1",
        nulls_equal=True,
    ).with_columns(
        pl.col("mapping_version")
        .is_not_null()
        .alias("canonical_mapping_found")
    )


def cleanse_canonical_source(lf: pl.LazyFrame, source_key: str) -> pl.LazyFrame:
    from crimenet_data.assets.crime.sources.registry import get_source

    config = get_source(source_key).config
    lf = lf.with_columns(
        pl.col("latitude").cast(pl.Float64, strict=False),
        pl.col("longitude").cast(pl.Float64, strict=False),
    )
    year_is_valid = pl.col("occurrence_year").is_between(2014, 2026).fill_null(False)
    min_latitude, max_latitude, min_longitude, max_longitude = (
        config.coordinate_bounds or (-90.0, 90.0, -180.0, 180.0)
    )
    coordinates_are_valid = (
        pl.col("latitude").is_finite()
        & pl.col("longitude").is_finite()
        & pl.col("latitude").is_between(min_latitude, max_latitude)
        & pl.col("longitude").is_between(min_longitude, max_longitude)
        
    ).fill_null(False)
    coordinates_are_absent = (
        pl.col("latitude").is_null() & pl.col("longitude").is_null()
    )
    coordinates_are_zero_zero = (
        (pl.col("latitude") == 0.0) & (pl.col("longitude") == 0.0)
    ).fill_null(False)
    location_is_valid = (
        coordinates_are_valid
        if config.coordinates_required
        else coordinates_are_absent | coordinates_are_valid
    )
    return lf.filter(
        year_is_valid & location_is_valid & ~coordinates_are_zero_zero
    )


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
    }
    for name, dtype in CANONICAL_CRIME_SCHEMA.items():
        expression = defaults.get(name)
        if expression is None:
            expression = pl.col(name) if name in available else pl.lit(None)
        expressions.append(expression.cast(dtype, strict=False).alias(name))
    return lf.select(expressions)
