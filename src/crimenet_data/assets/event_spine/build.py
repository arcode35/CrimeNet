"""Memory-safe event preparation, temporal selection, and spine assembly."""

from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter

import polars as pl
import polars_h3 as plh3

from crimenet_data.assets.event_spine.schema import (
    COMPONENT_AVAILABILITY_COLUMNS,
    H3_RESOLUTION,
    HISTORY_KEY_COLUMNS,
    TEMPORAL_INDEX_BASE_COLUMNS,
    WEATHER_H3_RESOLUTION,
)
from crimenet_data.observability.logger import get_logger
from crimenet_data.resources.crime_lake import CrimeLakeResources

log = get_logger(__name__)

EVENT_INDEX_COLUMNS = [
    "crime_id",
    "occurrence_timestamp_utc",
    "osm_h3_cell_id",
]


def add_spatial_keys(silver: pl.LazyFrame) -> pl.LazyFrame:
    """Derive the H3 keys used by temporal and downstream weather features."""

    return silver.with_columns(
        plh3.latlng_to_cell(
            "latitude",
            "longitude",
            resolution=H3_RESOLUTION,
            return_dtype=pl.UInt64,
        )
        .cast(pl.Int64, strict=False)
        .alias("osm_h3_cell_id"),
        plh3.latlng_to_cell(
            "latitude",
            "longitude",
            resolution=WEATHER_H3_RESOLUTION,
            return_dtype=pl.UInt64,
        )
        .cast(pl.Int64, strict=False)
        .alias("weather_query_cell_id"),
    )


def localize_occurrence_times(events: pl.DataFrame) -> pl.DataFrame:
    """Convert source-local wall-clock occurrence times to deterministic UTC."""

    timezones = (
        events.select("source_timezone")
        .drop_nulls()
        .unique()
        .sort("source_timezone")
        .get_column("source_timezone")
        .to_list()
    )
    parts = [
        events.filter(pl.col("source_timezone") == timezone).with_columns(
            pl.col("occurrence_timestamp")
            .dt.replace_time_zone(
                timezone,
                ambiguous="earliest",
                non_existent="null",
            )
            .dt.convert_time_zone("UTC")
            .alias("occurrence_timestamp_utc")
        )
        for timezone in timezones
    ]
    null_timezone = events.filter(pl.col("source_timezone").is_null())
    if null_timezone.height:
        parts.append(
            null_timezone.with_columns(
                pl.lit(None, dtype=pl.Datetime("us", time_zone="UTC")).alias(
                    "occurrence_timestamp_utc"
                )
            )
        )
    if not parts:
        raise RuntimeError("No event rows were available for timezone conversion")
    return pl.concat(parts, how="vertical_relaxed")


def load_modeled_events(
    *,
    crime_lake: CrimeLakeResources,
    silver_snapshot_uri: str,
    expected_modeled_rows: int,
) -> pl.DataFrame:
    """Scan the immutable Silver snapshot once and prepare modeled events."""

    started = perf_counter()
    modeled = crime_lake.scan_silver_snapshot(snapshot_uri=silver_snapshot_uri).filter(
        pl.col("include_in_model").fill_null(False)
        & pl.col("is_criminal_event").fill_null(False)
    )
    log.info(
        "event_spine_silver_load_started",
        silver_snapshot_uri=silver_snapshot_uri,
        expected_modeled_rows=expected_modeled_rows,
    )
    events = add_spatial_keys(modeled).collect(engine="streaming")
    if events.height != expected_modeled_rows:
        raise RuntimeError(
            "Silver include_in_model contract does not agree with criminal-event "
            f"eligibility: manifest_modeled_rows={expected_modeled_rows:,}, "
            f"eligible_rows={events.height:,}"
        )
    events = localize_occurrence_times(events)
    log.info(
        "event_spine_silver_loaded",
        modeled_event_rows=events.height,
        silver_load_seconds=perf_counter() - started,
    )
    return events


def prepare_event_index(
    events: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, object]]:
    """Create the narrow join input and unique event H3 footprint."""

    missing = sorted(set(EVENT_INDEX_COLUMNS) - set(events.columns))
    if missing:
        raise RuntimeError(f"Modeled events are missing index columns: {missing}")
    input_rows = events.height
    if input_rows == 0:
        raise RuntimeError("Cannot build event spine from zero events")

    invalid_event_utc_rows = events.filter(
        pl.col("occurrence_timestamp_utc").is_null()
    ).height
    null_h3_rows = events.filter(pl.col("osm_h3_cell_id").is_null()).height
    unjoinable_rows = events.filter(
        pl.col("occurrence_timestamp_utc").is_null()
        | pl.col("osm_h3_cell_id").is_null()
    ).height
    event_index = events.filter(
        pl.col("occurrence_timestamp_utc").is_not_null()
        & pl.col("osm_h3_cell_id").is_not_null()
    ).select(EVENT_INDEX_COLUMNS)
    relevant_h3_cells = event_index.select("osm_h3_cell_id").unique()
    unjoinable_pct = 100.0 * unjoinable_rows / input_rows
    summary: dict[str, object] = {
        "input_modeled_rows": input_rows,
        "joinable_rows": event_index.height,
        "invalid_event_utc_rows": invalid_event_utc_rows,
        "null_h3_rows": null_h3_rows,
        "unjoinable_rows": unjoinable_rows,
        "unjoinable_pct": unjoinable_pct,
        "unique_relevant_h3_cells": relevant_h3_cells.height,
    }
    log.info("event_spine_event_index_prepared", **summary)
    return event_index, relevant_h3_cells, summary


def select_temporal_matches(
    *,
    event_index: pl.DataFrame,
    temporal_index: pl.DataFrame,
    event_summary: Mapping[str, object],
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Select the latest legal history key using a skinny backward as-of join."""

    started = perf_counter()
    required_history = set(TEMPORAL_INDEX_BASE_COLUMNS)
    missing = sorted(required_history - set(temporal_index.columns))
    if missing:
        raise RuntimeError(f"Temporal index is missing columns: {missing}")

    left = event_index.sort("occurrence_timestamp_utc")
    right = temporal_index.select(TEMPORAL_INDEX_BASE_COLUMNS).sort(
        "feature_available_at"
    )
    log.info(
        "event_spine_skinny_asof_started",
        joinable_event_rows=left.height,
        filtered_skinny_history_rows=right.height,
        skinny_asof_event_columns=left.columns,
        skinny_asof_history_columns=right.columns,
        join_strategy="backward",
    )
    joined = left.join_asof(
        right,
        left_on="occurrence_timestamp_utc",
        right_on="feature_available_at",
        by="osm_h3_cell_id",
        strategy="backward",
        allow_exact_matches=True,
        check_sortedness=False,
    )
    history_unmatched_rows = joined.filter(
        pl.col("feature_available_at").is_null()
    ).height
    matched = joined.filter(pl.col("feature_available_at").is_not_null())
    matched_rows = matched.height
    input_rows = int(event_summary["input_modeled_rows"])
    joinable_rows = int(event_summary["joinable_rows"])
    coverage_pct = 100.0 * matched_rows / input_rows
    joinable_coverage_pct = (
        100.0 * matched_rows / joinable_rows if joinable_rows else 0.0
    )
    summary: dict[str, object] = {
        **dict(event_summary),
        "history_unmatched_rows": history_unmatched_rows,
        "no_legal_history_match_rows": history_unmatched_rows,
        "selected_temporal_match_rows": matched_rows,
        "output_rows": matched_rows,
        "dropped_rows": input_rows - matched_rows,
        "coverage_pct": coverage_pct,
        "joinable_coverage_pct": joinable_coverage_pct,
        "feature_versions_used": matched.get_column("feature_version_id").n_unique(),
        "min_occurrence_timestamp_utc": matched.get_column(
            "occurrence_timestamp_utc"
        ).min(),
        "max_occurrence_timestamp_utc": matched.get_column(
            "occurrence_timestamp_utc"
        ).max(),
        "min_feature_available_at": matched.get_column("feature_available_at").min(),
        "max_feature_available_at": matched.get_column("feature_available_at").max(),
        "skinny_asof_seconds": perf_counter() - started,
    }
    matched_event_keys = matched.select("crime_id", *HISTORY_KEY_COLUMNS)
    log.info("event_spine_skinny_asof_completed", **summary)
    return matched_event_keys, summary


def attach_selected_features(
    *,
    events: pl.DataFrame,
    matched_event_keys: pl.DataFrame,
    full_feature_rows: pl.DataFrame,
) -> pl.DataFrame:
    """Attach exact full feature rows, then restore the full Silver payload."""

    started = perf_counter()
    collisions = (set(events.columns) & set(full_feature_rows.columns)) - {
        "osm_h3_cell_id"
    }
    if collisions:
        raise RuntimeError(
            "Event/history schemas contain unexpected overlapping columns: "
            f"{sorted(collisions)}"
        )

    event_features = matched_event_keys.join(
        full_feature_rows,
        on=HISTORY_KEY_COLUMNS,
        how="inner",
        validate="m:1",
    )
    if event_features.height != matched_event_keys.height:
        raise RuntimeError(
            "Full-feature reattachment lost temporal matches: "
            f"matches={matched_event_keys.height:,}, "
            f"reattached={event_features.height:,}"
        )

    spine = events.join(
        event_features.drop("osm_h3_cell_id"),
        on="crime_id",
        how="inner",
        validate="1:1",
    )
    if spine.height != matched_event_keys.height:
        raise RuntimeError(
            "Silver payload reattachment changed event grain: "
            f"matches={matched_event_keys.height:,}, spine={spine.height:,}"
        )
    log.info(
        "event_spine_full_payload_reattached",
        selected_temporal_match_rows=matched_event_keys.height,
        full_feature_rows=full_feature_rows.height,
        final_spine_rows=spine.height,
        final_spine_columns=len(spine.columns),
        payload_reattachment_seconds=perf_counter() - started,
    )
    return spine


def build_event_spine(
    *,
    events: pl.DataFrame,
    history: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Build a spine from in-memory inputs using the production two-stage logic."""

    event_index, _, event_summary = prepare_event_index(events)
    index_columns = [
        *TEMPORAL_INDEX_BASE_COLUMNS,
        *(
            column
            for column in COMPONENT_AVAILABILITY_COLUMNS
            if column in history.columns
        ),
    ]
    temporal_index = history.select(index_columns)
    matched, summary = select_temporal_matches(
        event_index=event_index,
        temporal_index=temporal_index,
        event_summary=event_summary,
    )
    selected_keys = matched.select(HISTORY_KEY_COLUMNS).unique()
    full_features = history.join(
        selected_keys,
        on=HISTORY_KEY_COLUMNS,
        how="semi",
    )
    spine = attach_selected_features(
        events=events,
        matched_event_keys=matched,
        full_feature_rows=full_features,
    )
    socioeconomic_matched_rows: int | None = None
    if "_socioeconomic_matched" in spine.columns:
        socioeconomic_matched_rows = int(
            spine.select(pl.col("_socioeconomic_matched").fill_null(False).sum()).item()
        )
    summary["socioeconomic_matched_rows"] = socioeconomic_matched_rows
    return spine, summary


__all__ = [
    "EVENT_INDEX_COLUMNS",
    "add_spatial_keys",
    "attach_selected_features",
    "build_event_spine",
    "load_modeled_events",
    "localize_occurrence_times",
    "prepare_event_index",
    "select_temporal_matches",
]
