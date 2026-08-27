"""Immutable Parquet publication for the Gold event spine."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime

import polars as pl

from crimenet_data.assets.event_spine.schema import (
    EVENT_SPINE_SCHEMA_VERSION,
    EVENT_SPINE_UNMATCHED_HISTORY_POLICY,
    H3_RESOLUTION,
    PARTITION_COLUMNS,
)
from crimenet_data.assets.event_spine.temporal import history_root
from crimenet_data.assets.event_spine.validation import (
    validate_event_spine_readback,
)
from crimenet_data.observability.logger import get_logger
from crimenet_data.resources.crime_lake import CrimeLakeResources

log = get_logger(__name__)


def event_spine_root(crime_lake: CrimeLakeResources) -> str:
    """Compatibility accessor for the CrimeLake-owned event-spine root."""

    return crime_lake.event_spine_root


def event_spine_snapshot_uri(
    crime_lake: CrimeLakeResources,
    snapshot_id: str,
) -> str:
    return crime_lake.event_spine_snapshot_uri(snapshot_id)


def schema_document(schema: pl.Schema) -> dict[str, str]:
    return {name: str(dtype) for name, dtype in schema.items()}


def schema_sha256(schema: pl.Schema) -> str:
    payload = json.dumps(
        schema_document(schema), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def scan_event_spine_snapshot(
    *,
    crime_lake: CrimeLakeResources,
    snapshot_uri: str,
    expected_schema: pl.Schema,
) -> pl.LazyFrame:
    """Read a just-written snapshot and restore its partition columns."""

    scanned = pl.scan_parquet(
        crime_lake.event_spine_parquet_glob(snapshot_uri),
        storage_options=crime_lake.storage_options,
        credential_provider=None,
        hive_partitioning=True,
        hive_schema={
            "snapshot_id": pl.String,
            "source_city": pl.String,
            "occurrence_year": pl.Int16,
        },
    )
    available = set(scanned.collect_schema().names())
    missing = set(expected_schema.names()) - available
    if missing:
        raise RuntimeError(
            f"Written event-spine snapshot is missing columns: {sorted(missing)}"
        )
    return scanned.select(
        pl.col(name).cast(dtype, strict=False).alias(name)
        for name, dtype in expected_schema.items()
    )


def publish_event_spine_snapshot(
    *,
    crime_lake: CrimeLakeResources,
    spine: pl.DataFrame,
    snapshot_id: str,
    created_at_utc: datetime,
    silver_snapshot_uri: str,
    silver_manifest: Mapping[str, object],
    join_summary: Mapping[str, object],
    history_summary: Mapping[str, object],
) -> dict[str, object]:
    """Write, read back, validate, and atomically publish one Gold snapshot."""

    if created_at_utc.tzinfo is None:
        raise ValueError("created_at_utc must include a timezone")
    snapshot_uri = event_spine_snapshot_uri(crime_lake, snapshot_id)
    if crime_lake._prefix_has_objects(snapshot_uri):
        raise RuntimeError(
            f"Event-spine snapshot prefix already exists: {snapshot_uri}"
        )

    expected_schema = spine.schema
    log.info(
        "event_spine_snapshot_write_started",
        snapshot_id=snapshot_id,
        snapshot_uri=snapshot_uri,
        row_count=spine.height,
        partitioning_columns=PARTITION_COLUMNS,
    )
    spine.lazy().sink_parquet(
        pl.PartitionBy(
            snapshot_uri,
            key=PARTITION_COLUMNS,
            include_key=False,
        ),
        compression="zstd",
        compression_level=3,
        storage_options=crime_lake.storage_options,
        credential_provider=None,
        mkdir=True,
        engine="streaming",
    )
    if not crime_lake._snapshot_has_parquet(snapshot_uri):
        raise RuntimeError(
            f"Event-spine snapshot produced no Parquet files: {snapshot_uri}"
        )
    encoded_paths = crime_lake._encoded_partition_paths(snapshot_uri)
    if encoded_paths:
        raise RuntimeError(
            "Event-spine snapshot contains URL-encoded Hive partition paths: "
            f"{encoded_paths}"
        )

    readback = scan_event_spine_snapshot(
        crime_lake=crime_lake,
        snapshot_uri=snapshot_uri,
        expected_schema=expected_schema,
    )
    postwrite = validate_event_spine_readback(
        readback,
        expected_rows=spine.height,
    )
    feature_version_ids = sorted(
        str(value)
        for value in spine.get_column("feature_version_id")
        .drop_nulls()
        .unique()
        .to_list()
    )

    manifest: dict[str, object] = {
        "snapshot_id": snapshot_id,
        "snapshot_uri": snapshot_uri,
        "created_at_utc": created_at_utc.astimezone(UTC).isoformat(),
        "schema_version": EVENT_SPINE_SCHEMA_VERSION,
        "row_count": postwrite["row_count"],
        "source_count": postwrite["source_count"],
        "partition_columns": PARTITION_COLUMNS,
        "parquet_file_count": crime_lake._parquet_file_count(snapshot_uri),
        "silver_snapshot_uri": silver_snapshot_uri,
        "silver_snapshot_id": silver_manifest.get("snapshot_id"),
        "silver_mapping_version": silver_manifest.get("mapping_version"),
        "silver_schema_version": silver_manifest.get("schema_version"),
        "history_root": history_root(crime_lake),
        "h3_resolution": H3_RESOLUTION,
        "join_contract": (
            "same osm_h3_cell_id AND feature_available_at <= "
            "occurrence_timestamp_utc; choose latest feature_available_at"
        ),
        "join_strategy": "backward_asof",
        "allow_exact_matches": True,
        "ambiguous_local_time_policy": "earliest",
        "nonexistent_local_time_policy": "fail",
        "unmatched_history_policy": EVENT_SPINE_UNMATCHED_HISTORY_POLICY,
        **dict(join_summary),
        "history_rows": history_summary["history_rows"],
        "history_h3_cells": history_summary["history_h3_cells"],
        "history_feature_versions": history_summary["history_feature_versions"],
        "history_scope": history_summary["history_scope"],
        "unique_relevant_h3_cells": history_summary["unique_relevant_h3_cells"],
        "skinny_history_columns": history_summary["skinny_history_columns"],
        "skinny_history_column_count": history_summary["skinny_history_column_count"],
        "filtered_skinny_history_rows": history_summary["filtered_skinny_history_rows"],
        "filtered_history_h3_cells": history_summary["filtered_history_h3_cells"],
        "unique_selected_history_keys": history_summary["unique_selected_history_keys"],
        "full_feature_rows_retrieved": history_summary["full_feature_rows_retrieved"],
        "full_feature_column_count": history_summary["full_feature_column_count"],
        "feature_version_ids_used": feature_version_ids,
        "schema": schema_document(expected_schema),
        "schema_sha256": schema_sha256(expected_schema),
    }
    git_commit_sha = os.environ.get("GIT_COMMIT_SHA") or os.environ.get("GITHUB_SHA")
    if git_commit_sha:
        manifest["git_commit_sha"] = git_commit_sha

    crime_lake._write_object(
        crime_lake.event_spine_manifest_uri(snapshot_uri),
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8"),
        content_type="application/json",
    )
    crime_lake._write_object(
        crime_lake.event_spine_success_uri(snapshot_uri),
        b"",
        content_type="application/octet-stream",
    )

    pointer = {
        "snapshot_id": snapshot_id,
        "snapshot_uri": snapshot_uri,
        "created_at_utc": created_at_utc.astimezone(UTC).isoformat(),
        "schema_version": EVENT_SPINE_SCHEMA_VERSION,
        "silver_snapshot_id": silver_manifest.get("snapshot_id"),
    }
    crime_lake._write_object(
        crime_lake.event_spine_latest_pointer_uri,
        json.dumps(pointer, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        content_type="application/json",
    )
    log.info(
        "event_spine_snapshot_published",
        snapshot_id=snapshot_id,
        snapshot_uri=snapshot_uri,
        row_count=manifest["row_count"],
        parquet_file_count=manifest["parquet_file_count"],
    )
    return manifest


__all__ = [
    "event_spine_root",
    "event_spine_snapshot_uri",
    "publish_event_spine_snapshot",
    "scan_event_spine_snapshot",
    "schema_document",
    "schema_sha256",
]
