"""Build model-eligible events and align them to temporal features."""

from __future__ import annotations

import polars as pl
import polars_h3 as plh3

from crimenet_data.assets.event_spine.schema import (
    H3_RESOLUTION,
    WEATHER_H3_RESOLUTION,
)
from crimenet_data.observability.logger import get_logger
from crimenet_data.resources.crime_lake import CrimeLakeResources

log = get_logger(__name__)


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

    modeled = crime_lake.scan_silver_snapshot(
        snapshot_uri=silver_snapshot_uri
    ).filter(
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
    log.info("event_spine_silver_loaded", rows=events.height)
    return events


def build_event_spine(
    *,
    events: pl.DataFrame,
    history: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Perform the single exact backward as-of join at event grain."""

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
    unjoinable_pct = 100.0 * unjoinable_rows / input_rows
    log.info(
        "event_spine_unjoinable_events",
        input_rows=input_rows,
        invalid_event_utc_rows=invalid_event_utc_rows,
        null_h3_rows=null_h3_rows,
        unjoinable_rows=unjoinable_rows,
        unjoinable_pct=unjoinable_pct,
    )

    joinable = events.filter(
        pl.col("occurrence_timestamp_utc").is_not_null()
        & pl.col("osm_h3_cell_id").is_not_null()
    )
    collisions = (set(joinable.columns) & set(history.columns)) - {
        "osm_h3_cell_id"
    }
    if collisions:
        raise RuntimeError(
            "Event/history schemas contain unexpected overlapping columns: "
            f"{sorted(collisions)}"
        )

    left = joinable.sort("occurrence_timestamp_utc")
    right = history.sort("feature_available_at")
    log.info(
        "event_spine_asof_join_started",
        joinable_events=left.height,
        history_rows=right.height,
        h3_resolution=H3_RESOLUTION,
        join_strategy="backward",
    )
    joined = left.join_asof(
        right,
        left_on="occurrence_timestamp_utc",
        right_on="feature_available_at",
        by="osm_h3_cell_id",
        strategy="backward",
        allow_exact_matches=True,
        # Polars cannot verify sortedness when an as-of join has `by` groups;
        # both inputs are explicitly sorted by their as-of key immediately above.
        check_sortedness=False,
    )
    history_unmatched_rows = joined.filter(
        pl.col("feature_available_at").is_null()
    ).height
    spine = joined.filter(pl.col("feature_available_at").is_not_null())
    output_rows = spine.height
    coverage_pct = 100.0 * output_rows / input_rows
    joinable_coverage_pct = (
        100.0 * output_rows / joinable.height if joinable.height else 0.0
    )

    socioeconomic_matched_rows: int | None = None
    if "_socioeconomic_matched" in spine.columns:
        socioeconomic_matched_rows = int(
            spine.select(pl.col("_socioeconomic_matched").fill_null(False).sum()).item()
        )

    summary: dict[str, object] = {
        "input_modeled_rows": input_rows,
        "joinable_rows": joinable.height,
        "invalid_event_utc_rows": invalid_event_utc_rows,
        "null_h3_rows": null_h3_rows,
        "unjoinable_rows": unjoinable_rows,
        "unjoinable_pct": unjoinable_pct,
        "history_unmatched_rows": history_unmatched_rows,
        "output_rows": output_rows,
        "dropped_rows": input_rows - output_rows,
        "coverage_pct": coverage_pct,
        "joinable_coverage_pct": joinable_coverage_pct,
        "feature_versions_used": spine.get_column(
            "feature_version_id"
        ).n_unique(),
        "min_occurrence_timestamp_utc": spine.get_column(
            "occurrence_timestamp_utc"
        ).min(),
        "max_occurrence_timestamp_utc": spine.get_column(
            "occurrence_timestamp_utc"
        ).max(),
        "min_feature_available_at": spine.get_column("feature_available_at").min(),
        "max_feature_available_at": spine.get_column("feature_available_at").max(),
        "socioeconomic_matched_rows": socioeconomic_matched_rows,
    }
    log.info("event_spine_asof_join_completed", **summary)
    return spine, summary


__all__ = [
    "add_spatial_keys",
    "build_event_spine",
    "load_modeled_events",
    "localize_occurrence_times",
]
