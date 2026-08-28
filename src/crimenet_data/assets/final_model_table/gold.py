"""High-throughput Dagster publication for the final model table."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import dagster as dg
import polars as pl

from crimenet_data.assets.environmental.gold import (
    environmental_features,
    published_integration_sampling,
)
from crimenet_data.assets.event_spine.gold import gold_event_spine
from crimenet_data.assets.final_model_table.transformations import (
    FINAL_COLUMNS,
    FINAL_MODEL_TABLE_SCHEMA_VERSION,
    LIGHTING_FEATURE_COLUMNS,
    MODEL_SPLITS,
    FinalModelContractError,
    ModelSupportInterval,
    build_final_model_table,
    split_expression,
)
from crimenet_data.resources.crime_lake import CrimeLakeResources


# Changes to execution strategy should create a new immutable snapshot even when
# the logical schema stays unchanged.
FINAL_MODEL_TABLE_BUILD_VERSION = "optimized_128cpu_v1"
TARGET_FILE_BYTES = 768 * 1024 * 1024
ROW_GROUP_SIZE = 1_048_576


def _scan_parquet(lake: CrimeLakeResources, uris: str | list[str]) -> pl.LazyFrame:
    probe = uris[0] if isinstance(uris, list) else uris
    return pl.scan_parquet(
        uris,
        storage_options=lake.storage_options_for(probe),
        credential_provider=None,
        hive_partitioning=False,
    )


def _stable_snapshot_id(values: list[str]) -> str:
    digest = hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()
    return f"final-model-v3-{digest[:24]}"


def _write_json(
    lake: CrimeLakeResources,
    uri: str,
    document: Mapping[str, object],
) -> None:
    lake._write_object(
        uri,
        json.dumps(
            dict(document),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8"),
        content_type="application/json",
    )


def _snapshot_id(manifest: Mapping[str, object], *, label: str) -> str:
    value = str(manifest.get("snapshot_id", "")).strip()
    if not value:
        raise FinalModelContractError(f"{label} manifest is missing snapshot_id")
    return value


def _parse_utc(value: object, *, label: str) -> datetime:
    text = str(value).strip()
    if not text:
        raise FinalModelContractError(f"{label} is empty")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise FinalModelContractError(
            f"{label} is not an ISO-8601 timestamp: {text!r}"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise FinalModelContractError(f"{label} must include a UTC offset")
    return parsed.astimezone(UTC)


def _frozen_support_from_integration_manifest(
    manifest: Mapping[str, object],
) -> list[ModelSupportInterval]:
    if str(manifest.get("schema_version", "")) != "crime_integration_samples_v4":
        raise FinalModelContractError(
            "final model table requires crime_integration_samples_v4"
        )

    result: list[ModelSupportInterval] = []
    for source_record in manifest["sources"]:
        source = str(source_record["source_city"])
        split_support = source_record["split_support"]
        for split in MODEL_SPLITS:
            for index, record in enumerate(
                split_support[split]["temporal_coverage_intervals"]
            ):
                result.append(
                    ModelSupportInterval(
                        source_city=source,
                        split=split,
                        source_timezone=str(record["source_timezone"]),
                        start_utc=_parse_utc(
                            record["coverage_start_utc"],
                            label=f"{source}/{split}/{index}/start",
                        ),
                        end_utc=_parse_utc(
                            record["coverage_end_utc"],
                            label=f"{source}/{split}/{index}/end",
                        ),
                        coverage_basis=str(record["coverage_basis"]),
                        coverage_reference=str(record["coverage_reference"]),
                    )
                )
    return result


def _expected_integration_rows(manifest: Mapping[str, object]) -> int:
    return sum(
        int(source_record["split_support"][split]["integration_sample_rows"])
        for source_record in manifest["sources"]
        for split in MODEL_SPLITS
    )


def _write_snapshot(
    *,
    lake: CrimeLakeResources,
    table: pl.LazyFrame,
    snapshot_uri: str,
) -> None:
    if lake._prefix_has_objects(snapshot_uri):
        raise RuntimeError(
            "Final model-table snapshot prefix already exists without a published "
            f"success marker: {snapshot_uri}"
        )

    table.sink_parquet(
        pl.PartitionBy(
            snapshot_uri,
            key=["split", "source_city"],
            include_key=False,
            approximate_bytes_per_file=TARGET_FILE_BYTES,
        ),
        compression="zstd",
        compression_level=1,
        statistics=True,
        row_group_size=ROW_GROUP_SIZE,
        maintain_order=False,
        storage_options=lake.storage_options_for(snapshot_uri),
        credential_provider=None,
        mkdir=True,
        engine="streaming",
    )

    if not lake._snapshot_has_parquet(snapshot_uri):
        raise RuntimeError("Final model-table write produced no Parquet files")


def _scan_written_snapshot(
    lake: CrimeLakeResources,
    snapshot_uri: str,
) -> pl.LazyFrame:
    glob = lake.final_model_table_parquet_glob(snapshot_uri)
    return pl.scan_parquet(
        glob,
        storage_options=lake.storage_options_for(glob),
        credential_provider=None,
        hive_partitioning=True,
        hive_schema={
            "snapshot_id": pl.String,
            "split": pl.String,
            "source_city": pl.String,
        },
    )


def _audit_written_snapshot(
    *,
    lake: CrimeLakeResources,
    snapshot_uri: str,
    support_intervals: list[ModelSupportInterval],
    expected_integration_rows: int,
    expected_schema: pl.Schema,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """One persisted audit pass; no re-execution of the build graph."""

    frame = _scan_written_snapshot(lake, snapshot_uri)
    actual_schema = frame.collect_schema()
    missing = sorted(set(expected_schema.names()) - set(actual_schema.names()))
    if missing:
        raise FinalModelContractError(
            f"written final model table is missing columns: {missing}"
        )

    expected_split = split_expression(
        support_intervals,
        timestamp_column="model_timestamp_utc",
    )
    structural = [
        "row_id",
        "row_type",
        "source_city",
        "split",
        "model_timestamp_utc",
        "osm_h3_cell_id",
        "weather_query_cell_id",
    ]

    global_query = frame.select(
        pl.len().alias("row_count"),
        (pl.col("row_type") == "event").sum().alias("event_rows"),
        (pl.col("row_type") == "integration").sum().alias("integration_rows"),
        pl.any_horizontal(*(pl.col(column).is_null() for column in structural))
        .sum()
        .alias("null_structural_rows"),
        (pl.col("split") != expected_split).sum().alias("split_mismatch_rows"),
        (
            pl.col("feature_available_at").is_not_null()
            & (pl.col("feature_available_at") > pl.col("model_timestamp_utc"))
        )
        .sum()
        .alias("future_feature_rows"),
        pl.any_horizontal(
            *(pl.col(column).is_null() for column in LIGHTING_FEATURE_COLUMNS)
        )
        .sum()
        .alias("lighting_missing_rows"),
        pl.any_horizontal(
            pl.col("local_hour").is_null(),
            pl.col("local_day_of_week").is_null(),
        )
        .sum()
        .alias("calendar_missing_rows"),
        pl.any_horizontal(
            pl.col("cell_crime_count_28d").is_null(),
            pl.col("city_crime_count_28d").is_null(),
            pl.col("k1_crime_count_28d").is_null(),
        )
        .sum()
        .alias("history_missing_rows"),
        (
            (pl.col("row_type") == "integration")
            & (
                pl.col("integration_weight_cell_seconds").is_null()
                | ~pl.col("integration_weight_cell_seconds").is_finite()
                | (pl.col("integration_weight_cell_seconds") <= 0)
            )
        )
        .sum()
        .alias("invalid_integration_weight_rows"),
        pl.col("weather_available").sum().alias("weather_available_rows"),
        (~pl.col("weather_available")).sum().alias("weather_unavailable_rows"),
        pl.col("model_timestamp_utc").min().alias("min_timestamp_utc"),
        pl.col("model_timestamp_utc").max().alias("max_timestamp_utc"),
    )

    grouped_query = (
        frame.group_by("source_city", "split", "row_type")
        .agg(
            pl.len().alias("row_count"),
            pl.col("model_timestamp_utc").min().alias("min_timestamp_utc"),
            pl.col("model_timestamp_utc").max().alias("max_timestamp_utc"),
            pl.col("weather_available").sum().alias("weather_available_count"),
            pl.col("feature_available_at").count().alias(
                "national_h3_matched_count"
            ),
            pl.col("integration_weight_cell_seconds").sum().alias(
                "integration_weight_sum"
            ),
        )
        .with_columns(
            (
                100.0 * pl.col("weather_available_count") / pl.col("row_count")
            ).alias("weather_available_pct"),
            (
                100.0 * pl.col("national_h3_matched_count") / pl.col("row_count")
            ).alias("national_h3_matched_pct"),
        )
        .sort("source_city", "split", "row_type")
    )

    global_df, grouped_df = pl.collect_all(
        [global_query, grouped_query],
        engine="streaming",
    )
    summary = global_df.row(0, named=True)
    row_count = int(summary["row_count"])
    available = int(summary["weather_available_rows"] or 0)
    integration_rows = int(summary["integration_rows"] or 0)

    audit = {
        **summary,
        "weather_coverage_pct": (
            100.0 * available / row_count if row_count else 100.0
        ),
        "expected_integration_rows": expected_integration_rows,
        "integration_row_delta": integration_rows - expected_integration_rows,
    }

    failures = {
        name: int(audit[name])
        for name in (
            "null_structural_rows",
            "split_mismatch_rows",
            "future_feature_rows",
            "lighting_missing_rows",
            "calendar_missing_rows",
            "history_missing_rows",
            "invalid_integration_weight_rows",
        )
        if int(audit[name]) != 0
    }
    if row_count <= 0:
        failures["row_count"] = row_count
    if integration_rows != expected_integration_rows:
        failures["integration_rows"] = integration_rows

    if failures:
        raise FinalModelContractError(
            f"persisted final model-table audit failed: {failures}"
        )

    return audit, grouped_df.to_dicts()


def _materialize_final_model_snapshot(
    *,
    lake: CrimeLakeResources,
    snapshot_uri: str,
    event_snapshot_uri: str,
    integration_snapshot_uri: str,
    integration_manifest: Mapping[str, object],
    environmental_snapshot_uri: str,
    history_uris: list[str],
    support_intervals: list[ModelSupportInterval],
) -> tuple[dict[str, object], list[dict[str, object]], pl.Schema, dict[str, object]]:
    event_glob = lake.event_spine_parquet_glob(event_snapshot_uri)
    events = pl.scan_parquet(
        event_glob,
        storage_options=lake.storage_options_for(event_glob),
        credential_provider=None,
        hive_partitioning=True,
    )

    integration_uris = lake.integration_sample_uris_from_manifest(
        integration_snapshot_uri,
        integration_manifest,
    )
    integration = _scan_parquet(lake, integration_uris)

    environmental_glob = lake.environmental_features_parquet_glob(
        environmental_snapshot_uri
    )
    environmental = pl.scan_parquet(
        environmental_glob,
        storage_options=lake.storage_options_for(environmental_glob),
        credential_provider=None,
        hive_partitioning=True,
    )
    temporal_history = _scan_parquet(lake, history_uris)

    table, build_metadata = build_final_model_table(
        events=events,
        integration=integration,
        environmental=environmental,
        temporal_history=temporal_history,
        support_intervals=support_intervals,
    )
    expected_schema = table.collect_schema()

    # The expensive graph executes exactly once here.
    _write_snapshot(
        lake=lake,
        table=table,
        snapshot_uri=snapshot_uri,
    )

    audit, grouped_audit = _audit_written_snapshot(
        lake=lake,
        snapshot_uri=snapshot_uri,
        support_intervals=support_intervals,
        expected_integration_rows=_expected_integration_rows(integration_manifest),
        expected_schema=expected_schema,
    )
    return audit, grouped_audit, expected_schema, build_metadata


@dg.asset(
    name="final_model_table",
    group_name="gold_model",
    deps=[gold_event_spine, published_integration_sampling, environmental_features],
    required_resource_keys={"crime_lake"},
    compute_kind="polars",
    description=(
        "Canonical leakage-safe event and integration table, optimized for a "
        "high-core/high-RAM single host and published as an immutable snapshot."
    ),
)
def final_model_table(context) -> dg.MaterializeResult:
    lake: CrimeLakeResources = context.resources.crime_lake
    polars_threads = pl.thread_pool_size()
    context.log.info(
        "final_model_table: Polars thread pool=%s; build=%s",
        polars_threads,
        FINAL_MODEL_TABLE_BUILD_VERSION,
    )

    integration_uri, integration_manifest = lake.resolve_current_integration_snapshot()
    support_intervals = _frozen_support_from_integration_manifest(
        integration_manifest
    )

    frozen_event_uri = str(integration_manifest["event_spine_snapshot_uri"])
    event_uri, event_manifest = lake.resolve_event_spine_snapshot(
        snapshot_override_uri=frozen_event_uri
    )
    expected_event_id = str(integration_manifest["event_spine_snapshot_id"])
    actual_event_id = _snapshot_id(event_manifest, label="event spine")
    if actual_event_id != expected_event_id:
        raise FinalModelContractError(
            "integration/event-spine lineage mismatch: "
            f"integration={expected_event_id!r}, resolved={actual_event_id!r}"
        )

    environmental_uri, environmental_manifest = (
        lake.resolve_current_environmental_features_snapshot()
    )
    history_uris, history_manifest = lake.resolve_national_temporal_history_files()

    input_ids = {
        "event_spine_snapshot_id": actual_event_id,
        "integration_snapshot_id": _snapshot_id(
            integration_manifest, label="integration"
        ),
        "environmental_snapshot_id": _snapshot_id(
            environmental_manifest, label="environmental"
        ),
        "temporal_history_snapshot_id": _snapshot_id(
            history_manifest, label="temporal history"
        ),
    }

    support_contract = [
        {
            "source_city": interval.source_city,
            "split": interval.split,
            "source_timezone": interval.source_timezone,
            "start_utc": interval.start_utc.isoformat(),
            "end_utc": interval.end_utc.isoformat(),
            "coverage_basis": interval.coverage_basis,
            "coverage_reference": interval.coverage_reference,
        }
        for interval in sorted(
            support_intervals,
            key=lambda value: (value.source_city, value.start_utc, value.split),
        )
    ]
    support_json = json.dumps(
        support_contract,
        sort_keys=True,
        separators=(",", ":"),
    )
    support_sha256 = hashlib.sha256(support_json.encode("utf-8")).hexdigest()

    snapshot_id = _stable_snapshot_id(
        [
            *input_ids.values(),
            FINAL_MODEL_TABLE_SCHEMA_VERSION,
            FINAL_MODEL_TABLE_BUILD_VERSION,
            support_sha256,
        ]
    )
    snapshot_uri = lake.final_model_table_snapshot_uri(snapshot_id)
    success_uri = lake.snapshot_success_uri(snapshot_uri)

    if lake._object_exists(success_uri):
        manifest = json.loads(
            lake._read_object(lake.snapshot_manifest_uri(snapshot_uri))
        )
    else:
        audit, grouped_audit, schema, build_metadata = (
            _materialize_final_model_snapshot(
                lake=lake,
                snapshot_uri=snapshot_uri,
                event_snapshot_uri=event_uri,
                integration_snapshot_uri=integration_uri,
                integration_manifest=integration_manifest,
                environmental_snapshot_uri=environmental_uri,
                history_uris=history_uris,
                support_intervals=support_intervals,
            )
        )
        manifest = {
            "snapshot_id": snapshot_id,
            "snapshot_uri": snapshot_uri,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "schema_version": FINAL_MODEL_TABLE_SCHEMA_VERSION,
            "build_version": FINAL_MODEL_TABLE_BUILD_VERSION,
            "polars_thread_pool_size": polars_threads,
            **input_ids,
            "event_spine_snapshot_uri": event_uri,
            "integration_snapshot_uri": integration_uri,
            "environmental_snapshot_uri": environmental_uri,
            "temporal_history_root_uri": history_manifest["root_uri"],
            "temporal_history_object_count": history_manifest["object_count"],
            "temporal_history_object_set_sha256": history_manifest[
                "object_set_sha256"
            ],
            "model_support_contract_sha256": support_sha256,
            "model_support_intervals": support_contract,
            "split_contract": {
                "authority": "frozen_integration_manifest",
                "interval_semantics": "half_open",
                "source_specific": True,
            },
            "execution_contract": {
                "target_host": "128_vcpu_large_ram",
                "single_expensive_materialization": True,
                "post_write_audit": True,
                "parquet_target_file_bytes": TARGET_FILE_BYTES,
                "parquet_row_group_size": ROW_GROUP_SIZE,
                "parquet_compression": "zstd:1",
                "maintain_output_order": False,
            },
            **build_metadata,
            "partition_columns": ["split", "source_city"],
            "schema": {name: str(dtype) for name, dtype in schema.items()},
            "columns": FINAL_COLUMNS,
            **audit,
            "grouped_audit": grouped_audit,
            "parquet_file_count": lake._parquet_file_count(snapshot_uri),
        }
        if not snapshot_uri.startswith("s3://"):
            manifest["parquet_size_bytes"] = sum(
                path.stat().st_size
                for path in Path(snapshot_uri).rglob("*.parquet")
            )

        _write_json(lake, lake.snapshot_manifest_uri(snapshot_uri), manifest)
        lake._write_object(
            success_uri,
            b"",
            content_type="application/octet-stream",
        )

    _write_json(
        lake,
        lake.final_model_table_latest_pointer_uri,
        {
            "snapshot_id": snapshot_id,
            "snapshot_uri": snapshot_uri,
            "created_at_utc": manifest["created_at_utc"],
            "schema_version": FINAL_MODEL_TABLE_SCHEMA_VERSION,
            "build_version": FINAL_MODEL_TABLE_BUILD_VERSION,
        },
    )

    return dg.MaterializeResult(
        metadata={
            "snapshot_id": snapshot_id,
            "snapshot_uri": snapshot_uri,
            "row_count": int(manifest["row_count"]),
            "event_rows": int(manifest["event_rows"]),
            "integration_rows": int(manifest["integration_rows"]),
            "weather_coverage_pct": float(manifest["weather_coverage_pct"]),
            "future_feature_rows": int(manifest["future_feature_rows"]),
            "lighting_missing_rows": int(manifest["lighting_missing_rows"]),
            "split_mismatch_rows": int(manifest["split_mismatch_rows"]),
            "polars_thread_pool_size": int(manifest["polars_thread_pool_size"]),
            **input_ids,
        }
    )


@dg.asset_check(
    asset=final_model_table,
    name="published_contract",
    blocking=True,
    required_resource_keys={"crime_lake"},
)
def final_model_table_published_contract_check(context) -> dg.AssetCheckResult:
    lake: CrimeLakeResources = context.resources.crime_lake
    _, manifest = lake.resolve_current_final_model_table_snapshot()
    failures = {
        name: int(manifest.get(name, -1))
        for name in (
            "null_structural_rows",
            "split_mismatch_rows",
            "future_feature_rows",
            "lighting_missing_rows",
            "calendar_missing_rows",
            "history_missing_rows",
            "invalid_integration_weight_rows",
            "integration_row_delta",
        )
    }
    return dg.AssetCheckResult(
        passed=int(manifest.get("row_count", 0)) > 0
        and all(value == 0 for value in failures.values()),
        metadata={"row_count": int(manifest.get("row_count", 0)), **failures},
    )


final_model_table_assets = [final_model_table]
final_model_table_asset_checks = [final_model_table_published_contract_check]

__all__ = [
    "final_model_table",
    "final_model_table_asset_checks",
    "final_model_table_assets",
    "final_model_table_published_contract_check",
]
