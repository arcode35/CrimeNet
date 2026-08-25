"""National temporal feature-history loading and validation."""

from __future__ import annotations

import polars as pl

from crimenet_data.assets.event_spine.schema import (
    COMPONENT_AVAILABILITY_COLUMNS,
    HISTORY_ROOT_SUFFIX,
    REQUIRED_HISTORY_COLUMNS,
)
from crimenet_data.observability.logger import get_logger
from crimenet_data.resources.crime_lake import CrimeLakeResources

log = get_logger(__name__)


def history_root(crime_lake: CrimeLakeResources) -> str:
    """Return the immutable temporal-history dataset root."""

    return f"{crime_lake.gold_root.rstrip('/')}/{HISTORY_ROOT_SUFFIX}"


def scan_temporal_history(crime_lake: CrimeLakeResources) -> pl.LazyFrame:
    """Lazily scan every physical temporal-history version."""

    return pl.scan_parquet(
        (
            f"{history_root(crime_lake)}/"
            "feature_available_date=*/version_id=*/part-*.parquet"
        ),
        storage_options=crime_lake.storage_options,
        credential_provider=None,
        hive_partitioning=False,
    )


def validate_temporal_history(history: pl.DataFrame) -> dict[str, object]:
    """Enforce the temporal history's schema, grain, and availability contract."""

    missing = sorted(REQUIRED_HISTORY_COLUMNS - set(history.columns))
    if missing:
        raise RuntimeError(
            "National temporal history is missing required columns: " f"{missing}"
        )
    if history.is_empty():
        raise RuntimeError("National temporal history contains no rows")

    duplicate_history_rows = int(
        history.select(
            pl.struct(["osm_h3_cell_id", "feature_available_at"])
            .is_duplicated()
            .sum()
            .alias("duplicate_history_rows")
        ).item()
    )
    if duplicate_history_rows:
        raise RuntimeError(
            "National temporal history violates unique "
            "(osm_h3_cell_id, feature_available_at) grain: "
            f"duplicate_rows={duplicate_history_rows:,}"
        )

    component_columns = [
        column
        for column in COMPONENT_AVAILABILITY_COLUMNS
        if column in history.columns
    ]
    component_violations: dict[str, int] = {}
    if component_columns:
        component_summary = history.select(
            (
                pl.col(column).is_not_null()
                & (pl.col(column) > pl.col("feature_available_at"))
            )
            .sum()
            .alias(column)
            for column in component_columns
        ).row(0, named=True)
        component_violations = {
            column: int(value) for column, value in component_summary.items()
        }
        bad = {
            column: count
            for column, count in component_violations.items()
            if count != 0
        }
        if bad:
            raise RuntimeError(
                "Temporal feature history contains component availability "
                f"leakage: {bad}"
            )

    return {
        "history_rows": history.height,
        "history_h3_cells": history.get_column("osm_h3_cell_id").n_unique(),
        "history_feature_versions": history.get_column(
            "feature_version_id"
        ).n_unique(),
        "duplicate_history_rows": duplicate_history_rows,
        "min_feature_available_at": history.get_column(
            "feature_available_at"
        ).min(),
        "max_feature_available_at": history.get_column(
            "feature_available_at"
        ).max(),
        "component_availability_violations": component_violations,
    }


def load_temporal_history(
    crime_lake: CrimeLakeResources,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Load the full temporal history once and validate it in memory."""

    history_lf = scan_temporal_history(crime_lake)
    schema = history_lf.collect_schema()
    missing = sorted(REQUIRED_HISTORY_COLUMNS - set(schema.names()))
    if missing:
        raise RuntimeError(
            "National temporal history is missing required columns: " f"{missing}"
        )

    log.info(
        "event_spine_history_load_started",
        history_root=history_root(crime_lake),
        history_columns=len(schema.names()),
    )
    history = (
        history_lf.with_columns(
            pl.col("osm_h3_cell_id").cast(pl.Int64, strict=False),
        ).collect(engine="streaming")
    )
    summary = validate_temporal_history(history)
    log.info("event_spine_history_loaded", **summary)
    return history, summary


__all__ = [
    "history_root",
    "load_temporal_history",
    "scan_temporal_history",
    "validate_temporal_history",
]
