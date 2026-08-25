"""Dagster orchestration for the Gold crime event spine."""

from datetime import UTC, datetime
from uuid import uuid4

import dagster as dg
from dagster import AssetExecutionContext

from crimenet_data.assets.event_spine.build import (
    build_event_spine,
    load_modeled_events,
)
from crimenet_data.assets.event_spine.publishing import (
    publish_event_spine_snapshot,
)
from crimenet_data.assets.event_spine.schema import H3_RESOLUTION, PARTITION_COLUMNS
from crimenet_data.assets.event_spine.temporal import (
    history_root,
    load_temporal_history,
)
from crimenet_data.assets.event_spine.validation import validate_event_spine
from crimenet_data.observability.context import log_context
from crimenet_data.observability.logger import get_logger
from crimenet_data.resources.crime_lake import CrimeLakeResources

log = get_logger(__name__)


@dg.asset(
    name="gold_event_spine",
    group_name="gold_event_spine",
    deps=["silver_crime_offenses"],
    pool="gold_event_spine_writer",
    compute_kind="polars",
    description=(
        "Materialize model-eligible Silver crime events with exact leakage-safe "
        "H3-r9 national temporal features using a backward as-of join."
    ),
)
def gold_event_spine(
    context: AssetExecutionContext,
    crime_lake: CrimeLakeResources,
) -> dg.MaterializeResult:
    """Coordinate immutable inputs, one build, validation, and publication."""

    with log_context(
        run_id=context.run_id,
        asset_key=context.asset_key.to_user_string(),
    ):
        snapshot_id = str(uuid4())
        created_at_utc = datetime.now(UTC)
        silver_snapshot_uri = crime_lake.resolve_current_silver_snapshot()
        silver_manifest = crime_lake.read_silver_manifest(
            snapshot_uri=silver_snapshot_uri
        )
        expected_modeled_rows = int(silver_manifest["include_in_model_rows"])

        log.info(
            "event_spine_build_started",
            snapshot_id=snapshot_id,
            silver_snapshot_uri=silver_snapshot_uri,
            expected_modeled_rows=expected_modeled_rows,
            history_root=history_root(crime_lake),
        )
        events = load_modeled_events(
            crime_lake=crime_lake,
            silver_snapshot_uri=silver_snapshot_uri,
            expected_modeled_rows=expected_modeled_rows,
        )
        history, history_summary = load_temporal_history(crime_lake)
        spine, build_summary = build_event_spine(events=events, history=history)
        join_summary = validate_event_spine(spine, build_summary)
        manifest = publish_event_spine_snapshot(
            crime_lake=crime_lake,
            spine=spine,
            snapshot_id=snapshot_id,
            created_at_utc=created_at_utc,
            silver_snapshot_uri=silver_snapshot_uri,
            silver_manifest=silver_manifest,
            join_summary=join_summary,
            history_summary=history_summary,
        )
        return dg.MaterializeResult(
            metadata={
                "snapshot_id": snapshot_id,
                "snapshot_uri": manifest["snapshot_uri"],
                "silver_snapshot_id": manifest["silver_snapshot_id"],
                "input_modeled_rows": int(join_summary["input_modeled_rows"]),
                "output_rows": int(join_summary["output_rows"]),
                "dropped_rows": int(join_summary["dropped_rows"]),
                "invalid_event_utc_rows": int(
                    join_summary["invalid_event_utc_rows"]
                ),
                "history_unmatched_rows": int(
                    join_summary["history_unmatched_rows"]
                ),
                "history_coverage_pct": float(join_summary["coverage_pct"]),
                "feature_versions_used": int(
                    join_summary["feature_versions_used"]
                ),
                "h3_resolution": H3_RESOLUTION,
                "partitioning_columns": PARTITION_COLUMNS,
                "event_grain": "one model-eligible crime event",
            }
        )


event_spine_gold_assets = [gold_event_spine]

__all__ = ["event_spine_gold_assets", "gold_event_spine"]
