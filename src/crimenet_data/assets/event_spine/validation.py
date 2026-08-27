"""Quality gates for in-memory and persisted event-spine rows."""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

from crimenet_data.assets.event_spine.schema import (
    MAX_UNJOINABLE_EVENT_PCT,
    MIN_HISTORY_COVERAGE_PCT,
)


def event_spine_quality_summary(
    frame: pl.DataFrame | pl.LazyFrame,
) -> dict[str, int]:
    """Compute one compact set of event-grain and temporal correctness metrics."""

    lf = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
    required = {
        "crime_id",
        "source_city",
        "occurrence_timestamp_utc",
        "osm_h3_cell_id",
        "feature_available_at",
    }
    missing = sorted(required - set(lf.collect_schema().names()))
    if missing:
        raise RuntimeError(f"Event spine is missing required columns: {missing}")

    summary = (
        lf.select(
            pl.len().alias("row_count"),
            pl.col("source_city").n_unique().alias("source_count"),
            pl.col("crime_id").n_unique().alias("unique_crime_ids"),
            pl.col("occurrence_timestamp_utc")
            .null_count()
            .alias("null_occurrence_timestamp_utc"),
            pl.col("osm_h3_cell_id").null_count().alias("null_osm_h3_cell_id"),
            pl.col("feature_available_at")
            .null_count()
            .alias("null_feature_available_at"),
            (pl.col("feature_available_at") > pl.col("occurrence_timestamp_utc"))
            .fill_null(False)
            .sum()
            .alias("future_feature_leaks"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    result = {name: int(value) for name, value in summary.items()}
    result["duplicate_crime_ids"] = result["row_count"] - result["unique_crime_ids"]
    return result


def validate_event_spine(
    spine: pl.DataFrame,
    build_summary: Mapping[str, object],
) -> dict[str, object]:
    """Validate the finished join and return a publication-ready summary."""

    quality = event_spine_quality_summary(spine)
    failures = {
        name: quality[name]
        for name in (
            "null_occurrence_timestamp_utc",
            "null_osm_h3_cell_id",
            "future_feature_leaks",
            "duplicate_crime_ids",
        )
        if quality[name] != 0
    }
    if quality["row_count"] == 0:
        failures["row_count"] = 0
    if failures:
        raise RuntimeError(f"Event-spine quality gate failed: {failures}")

    expected_output_rows = int(build_summary["output_rows"])
    if quality["row_count"] != expected_output_rows:
        raise RuntimeError(
            "Event-spine build summary row count mismatch: "
            f"expected={expected_output_rows:,}, actual={quality['row_count']:,}"
        )

    unjoinable_pct = float(build_summary["unjoinable_pct"])
    if unjoinable_pct > MAX_UNJOINABLE_EVENT_PCT:
        raise RuntimeError(
            "Event-spine unjoinable event rate exceeded safety limit: "
            f"{unjoinable_pct:.6f}% > {MAX_UNJOINABLE_EVENT_PCT:.6f}%"
        )
    coverage_pct = float(build_summary["coverage_pct"])
    if coverage_pct < MIN_HISTORY_COVERAGE_PCT:
        raise RuntimeError(
            "Event-spine temporal-history coverage regressed below the production "
            f"threshold: {coverage_pct:.6f}% < {MIN_HISTORY_COVERAGE_PCT:.6f}%"
        )

    return {**dict(build_summary), **quality}


def validate_event_spine_readback(
    frame: pl.LazyFrame,
    *,
    expected_rows: int,
) -> dict[str, int]:
    """Apply persisted-row checks before manifest and pointer publication."""

    quality = event_spine_quality_summary(frame)
    failures = {
        name: quality[name]
        for name in (
            "null_occurrence_timestamp_utc",
            "null_osm_h3_cell_id",
            "future_feature_leaks",
            "duplicate_crime_ids",
        )
        if quality[name] != 0
    }
    if failures:
        raise RuntimeError(f"Event-spine post-write quality gate failed: {failures}")
    if quality["row_count"] != expected_rows:
        raise RuntimeError(
            "Event-spine read-back row count mismatch: "
            f"expected={expected_rows:,}, actual={quality['row_count']:,}"
        )
    return quality


__all__ = [
    "event_spine_quality_summary",
    "validate_event_spine",
    "validate_event_spine_readback",
]
