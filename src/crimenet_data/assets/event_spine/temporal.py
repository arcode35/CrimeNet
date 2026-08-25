"""Footprint-pruned national temporal feature-history access."""

from __future__ import annotations

from time import perf_counter

import polars as pl

from crimenet_data.assets.event_spine.schema import (
    COMPONENT_AVAILABILITY_COLUMNS,
    HISTORY_KEY_COLUMNS,
    HISTORY_ROOT_SUFFIX,
    REQUIRED_HISTORY_COLUMNS,
    TEMPORAL_INDEX_BASE_COLUMNS,
)
from crimenet_data.observability.logger import get_logger
from crimenet_data.resources.crime_lake import CrimeLakeResources

log = get_logger(__name__)


def history_root(crime_lake: CrimeLakeResources) -> str:
    """Return the exact temporal-history root used by the event spine."""

    return f"{crime_lake.gold_root.rstrip('/')}/{HISTORY_ROOT_SUFFIX}"


def scan_temporal_history(crime_lake: CrimeLakeResources) -> pl.LazyFrame:
    """Lazily scan history/, never the annual cache."""

    return pl.scan_parquet(
        (
            f"{history_root(crime_lake)}/"
            "feature_available_date=*/version_id=*/part-*.parquet"
        ),
        storage_options=crime_lake.storage_options,
        credential_provider=None,
        hive_partitioning=False,
    )


def _validate_history_schema(schema: pl.Schema) -> None:
    missing = sorted(REQUIRED_HISTORY_COLUMNS - set(schema.names()))
    if missing:
        raise RuntimeError(
            f"National temporal history is missing required columns: {missing}"
        )


def temporal_index_columns(schema: pl.Schema) -> list[str]:
    """Return only columns needed to choose and validate a legal version."""

    _validate_history_schema(schema)
    return [
        *TEMPORAL_INDEX_BASE_COLUMNS,
        *(column for column in COMPONENT_AVAILABILITY_COLUMNS if column in schema),
    ]


def normalize_relevant_h3_cells(relevant_h3_cells: pl.DataFrame) -> pl.DataFrame:
    """Normalize a Polars-native event footprint for history semi-joins."""

    if "osm_h3_cell_id" not in relevant_h3_cells.columns:
        raise RuntimeError("Relevant H3 cells are missing osm_h3_cell_id")
    normalized = (
        relevant_h3_cells.select(pl.col("osm_h3_cell_id").cast(pl.Int64, strict=False))
        .drop_nulls()
        .unique()
    )
    if normalized.is_empty():
        raise RuntimeError("No relevant event H3 cells were available")
    return normalized


def prune_history_to_h3(
    history: pl.LazyFrame,
    *,
    relevant_h3_cells: pl.DataFrame,
    columns: list[str] | None = None,
) -> pl.LazyFrame:
    """Semi-join history to the event footprint before collection."""

    cells = normalize_relevant_h3_cells(relevant_h3_cells)
    projected = history.select(columns) if columns is not None else history
    return projected.with_columns(
        pl.col("osm_h3_cell_id").cast(pl.Int64, strict=False)
    ).join(
        cells.lazy(),
        on="osm_h3_cell_id",
        how="semi",
    )


def validate_temporal_history(history: pl.DataFrame) -> dict[str, object]:
    """Enforce schema, logical grain, and component chronology in scope."""

    missing = sorted(REQUIRED_HISTORY_COLUMNS - set(history.columns))
    if missing:
        raise RuntimeError(
            f"National temporal history is missing required columns: {missing}"
        )
    if history.is_empty():
        raise RuntimeError("National temporal history contains no relevant rows")

    duplicate_history_rows = int(
        history.select(
            pl.struct(HISTORY_KEY_COLUMNS)
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
        column for column in COMPONENT_AVAILABILITY_COLUMNS if column in history.columns
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
        "history_feature_versions": history.get_column("feature_version_id").n_unique(),
        "duplicate_history_rows": duplicate_history_rows,
        "min_feature_available_at": history.get_column("feature_available_at").min(),
        "max_feature_available_at": history.get_column("feature_available_at").max(),
        "component_availability_violations": component_violations,
    }


def load_temporal_index(
    crime_lake: CrimeLakeResources,
    *,
    relevant_h3_cells: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Collect a skinny history index only for event-footprint H3 cells."""

    started = perf_counter()
    history_lf = scan_temporal_history(crime_lake)
    schema = history_lf.collect_schema()
    columns = temporal_index_columns(schema)
    cells = normalize_relevant_h3_cells(relevant_h3_cells)

    log.info(
        "event_spine_temporal_index_load_started",
        history_root=history_root(crime_lake),
        national_history_columns=len(schema.names()),
        skinny_history_columns=columns,
        unique_relevant_h3_cells=cells.height,
    )
    temporal_index = prune_history_to_h3(
        history_lf,
        relevant_h3_cells=cells,
        columns=columns,
    ).collect(engine="streaming")
    summary = validate_temporal_history(temporal_index)
    summary.update(
        {
            "history_scope": "modeled_event_h3_footprint",
            "unique_relevant_h3_cells": cells.height,
            "skinny_history_columns": columns,
            "skinny_history_column_count": len(columns),
            "filtered_skinny_history_rows": temporal_index.height,
            "filtered_history_h3_cells": summary["history_h3_cells"],
            "temporal_index_load_seconds": perf_counter() - started,
        }
    )
    log.info("event_spine_temporal_index_loaded", **summary)
    return temporal_index, summary


def selected_history_keys(matched_event_keys: pl.DataFrame) -> pl.DataFrame:
    """Return unique exact history keys selected by the as-of stage."""

    missing = sorted(set(HISTORY_KEY_COLUMNS) - set(matched_event_keys.columns))
    if missing:
        raise RuntimeError(f"Matched events are missing history keys: {missing}")
    keys = matched_event_keys.select(HISTORY_KEY_COLUMNS).unique()
    if keys.is_empty():
        raise RuntimeError("No temporal history keys were selected")
    return keys


def load_selected_feature_rows(
    crime_lake: CrimeLakeResources,
    *,
    relevant_h3_cells: pl.DataFrame,
    selected_keys: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Stream history again and retain only exact selected full-width rows."""

    started = perf_counter()
    history_lf = scan_temporal_history(crime_lake)
    schema = history_lf.collect_schema()
    _validate_history_schema(schema)
    cells = normalize_relevant_h3_cells(relevant_h3_cells)
    keys = selected_keys.select(
        pl.col("osm_h3_cell_id").cast(pl.Int64, strict=False),
        pl.col("feature_available_at"),
    ).unique()
    if keys.is_empty():
        raise RuntimeError("No selected temporal keys were available for retrieval")

    log.info(
        "event_spine_full_feature_retrieval_started",
        history_root=history_root(crime_lake),
        unique_relevant_h3_cells=cells.height,
        unique_selected_history_keys=keys.height,
        full_history_columns=len(schema.names()),
    )
    full_features = (
        prune_history_to_h3(
            history_lf,
            relevant_h3_cells=cells,
        )
        .join(
            keys.lazy(),
            on=HISTORY_KEY_COLUMNS,
            how="semi",
        )
        .collect(engine="streaming")
    )

    duplicate_rows = (
        full_features.height - full_features.select(HISTORY_KEY_COLUMNS).unique().height
    )
    missing_keys = keys.join(
        full_features.select(HISTORY_KEY_COLUMNS),
        on=HISTORY_KEY_COLUMNS,
        how="anti",
    ).height
    if duplicate_rows or missing_keys or full_features.height != keys.height:
        raise RuntimeError(
            "Exact full-feature retrieval violated selected history-key grain: "
            f"selected_keys={keys.height:,}, rows={full_features.height:,}, "
            f"duplicate_rows={duplicate_rows:,}, missing_keys={missing_keys:,}"
        )

    summary: dict[str, object] = {
        "unique_selected_history_keys": keys.height,
        "full_feature_rows_retrieved": full_features.height,
        "full_feature_column_count": len(full_features.columns),
        "full_feature_retrieval_seconds": perf_counter() - started,
    }
    log.info("event_spine_full_features_retrieved", **summary)
    return full_features, summary


__all__ = [
    "HISTORY_KEY_COLUMNS",
    "history_root",
    "load_selected_feature_rows",
    "load_temporal_index",
    "normalize_relevant_h3_cells",
    "prune_history_to_h3",
    "scan_temporal_history",
    "selected_history_keys",
    "temporal_index_columns",
    "validate_temporal_history",
]
