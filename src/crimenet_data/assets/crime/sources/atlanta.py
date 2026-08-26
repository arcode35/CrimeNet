from datetime import datetime

import polars as pl

from crimenet_data.assets.crime.common.expressions import (
    composite_identifier,
    date_time_expr,
    datetime_expr,
    numeric_expr,
    text_expr,
)
from crimenet_data.assets.crime.sources._shared import (
    CSV,
    PARQUET,
    adapt_standard,
    nullable_string,
    prepare_snake_case,
)
from crimenet_data.assets.crime.sources.base import (
    AdapterContext,
    CrimeSourceConfig,
    SourceDefinition,
)

CURRENT_ERA_START = datetime(2021, 1, 1)  # noqa: DTZ001 - source-local wall time
ATLANTA_TZ = "America/New_York"


def legacy_occurrence(lf: pl.LazyFrame) -> pl.Expr:
    return date_time_expr(
        lf,
        ("occurdate",),
        ("occurtime",),
    )


def current_datetime(column: str) -> pl.Expr:
    # APD's ArcGIS export stores these epoch values as UTC.
    # Convert them back to Atlanta civil time before making them
    # timezone-naive, matching CrimeNet's canonical timestamp convention.
    return (
        pl.from_epoch(
            pl.col(column).cast(pl.Int64, strict=False),
            time_unit="ms",
        )
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone(ATLANTA_TZ)
        .dt.replace_time_zone(None)
    )


def current_occurrence(lf: pl.LazyFrame) -> pl.Expr:
    return current_datetime("occurred_from_date")


def occurrence(lf: pl.LazyFrame) -> pl.Expr:
    """
    Resolve occurrence time for either APD era.

    This must be schema-aware because Bronze ingestion can invoke the source
    occurrence expression on one physical file at a time. A Polars when/then
    expression that mentions both schemas will still try to resolve columns
    from both branches and can fail when processing a single-era file.
    """
    names = set(lf.collect_schema().names())

    has_legacy = {"occurdate", "occurtime"} <= names
    has_current = "occurred_from_date" in names

    if has_legacy and has_current:
        # Mixed-schema scan: only one side should be populated per physical row.
        return pl.coalesce(
            [
                current_occurrence(lf),
                legacy_occurrence(lf),
            ]
        )

    if has_current:
        return current_occurrence(lf)

    if has_legacy:
        return legacy_occurrence(lf)

    raise KeyError(
        "Atlanta source contains neither the historical occurrence columns "
        "('occurdate', 'occurtime') nor the current APD column "
        "'occurred_from_date'."
    )


def current_report_timestamp(lf: pl.LazyFrame) -> pl.Expr:
    report = current_datetime("report_date")

    # Use immutable Bronze ingestion time rather than wall-clock now so replay
    # produces the same validity decision.
    ingested_local = (
        pl.col("ingested_at_utc")
        .dt.convert_time_zone(ATLANTA_TZ)
        .dt.replace_time_zone(None)
    )

    return (
        pl.when((report >= pl.lit(CURRENT_ERA_START)) & (report <= ingested_local))
        .then(report)
        .otherwise(pl.lit(None, dtype=pl.Datetime("us")))
    )


def adapt_legacy(
    lf: pl.LazyFrame,
) -> pl.LazyFrame:
    adapted = adapt_standard(
        lf,
        legacy_occurrence(lf),
        source_record_id=composite_identifier(
            lf,
            "reportnumber",
            "ucrliteral",
        ),
        report_timestamp=datetime_expr(
            lf,
            "reportdate",
        ),
        source_offense_code=nullable_string(),
        source_offense_category=text_expr(
            lf,
            "ucrliteral",
        ),
        source_offense_description=text_expr(
            lf,
            "ucrliteral",
        ),
        latitude=numeric_expr(
            lf,
            "latitude",
        ),
        longitude=numeric_expr(
            lf,
            "longitude",
        ),
        location_label=text_expr(
            lf,
            "location",
        ),
        location_type=nullable_string(),
        police_district=text_expr(
            lf,
            "beat",
        ),
        local_area=text_expr(
            lf,
            "neighborhood",
            "npu",
        ),
    )

    return (
        adapted.filter(pl.col("occurrence_timestamp") < pl.lit(CURRENT_ERA_START))
        .sort(
            [
                "source_record_id",
                "report_timestamp",
                "occurrence_timestamp",
                "latitude",
                "longitude",
                "location_label",
                "police_district",
                "local_area",
            ],
            nulls_last=True,
        )
        .unique(
            subset=["source_record_id"],
            keep="last",
            maintain_order=False,
        )
    )


def adapt_current(
    lf: pl.LazyFrame,
) -> pl.LazyFrame:
    adapted = adapt_standard(
        lf,
        current_occurrence(lf),
        source_record_id=composite_identifier(
            lf,
            "report_number",
            "nibrs_ucr_code",
        ),
        report_timestamp=current_report_timestamp(lf),
        source_offense_code=text_expr(
            lf,
            "nibrs_ucr_code",
        ),
        source_offense_category=text_expr(
            lf,
            "crime_against",
        ),
        source_offense_description=text_expr(
            lf,
            "nibrs_offense",
        ),
        source_auxiliary=text_expr(
            lf,
            "nibrs_bucket",
        ),
        source_severity=text_expr(
            lf,
            "part",
        ),
        latitude=numeric_expr(
            lf,
            "geometry_y",
            "latitude_2",
            "latitude",
        ),
        longitude=numeric_expr(
            lf,
            "geometry_x",
            "longitude_2",
            "longitude",
        ),
        location_label=text_expr(
            lf,
            "street_address",
        ),
        location_type=text_expr(
            lf,
            "location_type",
        ),
        police_district=text_expr(
            lf,
            "beat",
            "zone",
        ),
        local_area=text_expr(
            lf,
            "nhood_name",
            "npu",
        ),
    )

    # The source audit found only 15 duplicate natural keys. After projection
    # their differences are in APD fields CrimeNet does not model (for example
    # firearm/victim annotations), so one canonical offense row is correct.
    return (
        adapted.filter(pl.col("occurrence_timestamp") >= pl.lit(CURRENT_ERA_START))
        .sort(
            [
                "source_record_id",
                "report_timestamp",
                "occurrence_timestamp",
                "latitude",
                "longitude",
                "location_label",
                "police_district",
                "local_area",
            ],
            nulls_last=True,
        )
        .unique(
            subset=["source_record_id"],
            keep="last",
            maintain_order=False,
        )
    )


def adapt(
    lf: pl.LazyFrame,
    _context: AdapterContext,
) -> pl.LazyFrame:
    """
    Adapt whichever APD era(s) are actually present.

    This is deliberately schema-aware so single-era fixtures and physical-file
    scans do not reference columns that do not exist.
    """
    names = set(lf.collect_schema().names())
    frames: list[pl.LazyFrame] = []

    if {"reportnumber", "ucrliteral"} <= names:
        legacy = lf.filter(pl.col("reportnumber").is_not_null())
        frames.append(adapt_legacy(legacy))

    if {"report_number", "nibrs_ucr_code"} <= names:
        current = lf.filter(pl.col("report_number").is_not_null())
        frames.append(adapt_current(current))

    if not frames:
        raise KeyError(
            "Atlanta source does not match either supported schema: "
            "historical (reportnumber/ucrliteral) or "
            "current (report_number/nibrs_ucr_code)."
        )

    if len(frames) == 1:
        return frames[0]

    return pl.concat(
        frames,
        how="diagonal_relaxed",
    )


SOURCE = SourceDefinition(
    config=CrimeSourceConfig(
        key="atlanta",
        source_system="atlanta_police_open_data",
        patterns=CSV + PARQUET,
        timezone=ATLANTA_TZ,
        crosswalk_keys=("source_offense_description",),
    ),
    prepare_bronze=prepare_snake_case,
    occurrence_timestamp=occurrence,
    adapt_to_silver=adapt,
)
