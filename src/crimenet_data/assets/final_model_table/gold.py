"""High-throughput Dagster publication for the final model table."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import dagster as dg
import polars as pl
from boto3.s3.transfer import TransferConfig

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
FINAL_MODEL_TABLE_BUILD_VERSION = "optimized_local_stage_v2"
TARGET_FILE_BYTES = 768 * 1024 * 1024
ROW_GROUP_SIZE = 1_048_576

# The expensive Polars graph is persisted locally first.  A failed B2 upload can
# then be retried without recomputing the model table.
LOCAL_STAGE_ROOT = Path(
    os.environ.get(
        "CRIMENET_FINAL_MODEL_TABLE_STAGE_ROOT",
        "/tmp/crimenet-final-model-table",
    )
)
LOCAL_READY_FILE = "_LOCAL_READY.json"

# B2 upload tuning.  64 MiB parts keep multipart request counts low while still
# allowing useful parallelism.  Outer file concurrency * inner multipart
# concurrency should stay below CrimeLakeResources.s3_client()'s connection pool.
UPLOAD_WORKERS = int(os.environ.get("CRIMENET_B2_UPLOAD_WORKERS", "4"))
UPLOAD_PART_CONCURRENCY = int(
    os.environ.get("CRIMENET_B2_UPLOAD_PART_CONCURRENCY", "4")
)
UPLOAD_CHUNK_BYTES = int(
    os.environ.get("CRIMENET_B2_UPLOAD_CHUNK_BYTES", str(64 * 1024 * 1024))
)
UPLOAD_FILE_ATTEMPTS = int(os.environ.get("CRIMENET_B2_UPLOAD_FILE_ATTEMPTS", "8"))


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


def _local_snapshot_dir(snapshot_id: str) -> Path:
    return LOCAL_STAGE_ROOT / f"snapshot_id={snapshot_id}"


def _local_parquet_inventory(snapshot_dir: Path) -> dict[str, int]:
    return {
        path.relative_to(snapshot_dir).as_posix(): path.stat().st_size
        for path in sorted(snapshot_dir.rglob("*.parquet"))
        if path.is_file()
    }


def _write_local_json(path: Path, document: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            dict(document),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_local_ready_state(
    *,
    snapshot_id: str,
    snapshot_dir: Path,
) -> dict[str, object] | None:
    ready_path = snapshot_dir / LOCAL_READY_FILE
    if not ready_path.exists():
        return None
    try:
        state = json.loads(ready_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Malformed local final-model-table ready state: {ready_path}"
        ) from error

    if state.get("snapshot_id") != snapshot_id:
        raise RuntimeError(
            "Local final-model-table snapshot identity mismatch: "
            f"expected={snapshot_id!r}, actual={state.get('snapshot_id')!r}"
        )
    if state.get("build_version") != FINAL_MODEL_TABLE_BUILD_VERSION:
        raise RuntimeError(
            "Local final-model-table build-version mismatch: "
            f"expected={FINAL_MODEL_TABLE_BUILD_VERSION!r}, "
            f"actual={state.get('build_version')!r}"
        )

    inventory = _local_parquet_inventory(snapshot_dir)
    expected_count = int(state.get("parquet_file_count", -1))
    expected_size = int(state.get("parquet_size_bytes", -1))
    if not inventory:
        raise RuntimeError(
            f"Local ready marker exists but no Parquet files were found: {snapshot_dir}"
        )
    if len(inventory) != expected_count or sum(inventory.values()) != expected_size:
        raise RuntimeError(
            "Local final-model-table snapshot changed after audit: "
            f"expected_files={expected_count}, actual_files={len(inventory)}, "
            f"expected_bytes={expected_size}, actual_bytes={sum(inventory.values())}"
        )
    return state


def _write_local_snapshot(
    *,
    table: pl.LazyFrame,
    snapshot_dir: Path,
) -> None:
    # A directory without the ready marker is an interrupted local materialization,
    # not a resumable snapshot.  Rebuild it from scratch so partial shards cannot
    # enter publication.
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    snapshot_dir.parent.mkdir(parents=True, exist_ok=True)

    table.sink_parquet(
        pl.PartitionBy(
            str(snapshot_dir),
            key=["split", "source_city"],
            include_key=False,
            approximate_bytes_per_file=TARGET_FILE_BYTES,
        ),
        compression="zstd",
        compression_level=1,
        statistics=True,
        row_group_size=ROW_GROUP_SIZE,
        maintain_order=False,
        mkdir=True,
        engine="streaming",
    )

    if not _local_parquet_inventory(snapshot_dir):
        raise RuntimeError("Final model-table local write produced no Parquet files")


def _scan_written_snapshot(
    lake: CrimeLakeResources,
    snapshot_uri: str,
) -> pl.LazyFrame:
    if snapshot_uri.startswith("s3://"):
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

    glob = f"{Path(snapshot_uri).as_posix().rstrip('/')}/**/*.parquet"
    return pl.scan_parquet(
        glob,
        hive_partitioning=True,
        hive_schema={
            "snapshot_id": pl.String,
            "split": pl.String,
            "source_city": pl.String,
        }
    )


def _remote_parquet_inventory(
    lake: CrimeLakeResources,
    snapshot_uri: str,
) -> dict[str, int]:
    if not snapshot_uri.startswith("s3://"):
        return _local_parquet_inventory(Path(snapshot_uri))

    bucket, prefix = lake._s3_location(f"{snapshot_uri.rstrip('/')}/placeholder")
    prefix = prefix.removesuffix("placeholder")
    paginator = lake.s3_client().get_paginator("list_objects_v2")
    inventory: dict[str, int] = {}
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = str(item.get("Key", ""))
            if not key.endswith(".parquet"):
                continue
            relative = key[len(prefix):].lstrip("/")
            inventory[relative] = int(item["Size"])
    return inventory


def _upload_parquet_file(
    *,
    client,
    local_path: Path,
    bucket: str,
    key: str,
    transfer_config: TransferConfig,
) -> None:
    last_error: Exception | None = None
    for attempt in range(1, UPLOAD_FILE_ATTEMPTS + 1):
        try:
            client.upload_file(
                str(local_path),
                bucket,
                key,
                ExtraArgs={"ContentType": "application/vnd.apache.parquet"},
                Config=transfer_config,
            )
            return
        except Exception as error:
            last_error = error
            if attempt >= UPLOAD_FILE_ATTEMPTS:
                break
            # File-level retries sit above botocore's request/part retries.  This
            # makes a transient 503 restart only one shard, never the Polars graph.
            time.sleep(min(5.0 * (2 ** (attempt - 1)), 120.0))
    assert last_error is not None
    raise RuntimeError(
        f"Failed to upload {local_path} after {UPLOAD_FILE_ATTEMPTS} file attempts"
    ) from last_error


def _publish_local_snapshot(
    *,
    lake: CrimeLakeResources,
    local_snapshot_dir: Path,
    snapshot_uri: str,
    log,
) -> tuple[int, int]:
    local_inventory = _local_parquet_inventory(local_snapshot_dir)
    if not local_inventory:
        raise RuntimeError(
            f"Refusing to publish an empty local snapshot: {local_snapshot_dir}"
        )
    if UPLOAD_WORKERS < 1 or UPLOAD_PART_CONCURRENCY < 1:
        raise ValueError("B2 upload concurrency must be >= 1")
    if UPLOAD_CHUNK_BYTES < 5 * 1024 * 1024:
        raise ValueError("B2 multipart chunk size must be at least 5 MiB")
    if UPLOAD_FILE_ATTEMPTS < 1:
        raise ValueError("B2 upload file attempts must be >= 1")

    if not snapshot_uri.startswith("s3://"):
        destination = Path(snapshot_uri)
        if destination.resolve() != local_snapshot_dir.resolve():
            if destination.exists():
                shutil.rmtree(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(local_snapshot_dir, destination)
        return len(local_inventory), sum(local_inventory.values())

    bucket, prefix_probe = lake._s3_location(
        f"{snapshot_uri.rstrip('/')}/placeholder"
    )
    prefix = prefix_probe.removesuffix("placeholder")
    remote_inventory = _remote_parquet_inventory(lake, snapshot_uri)

    # The snapshot is still unpublished here.  Existing files are from an earlier
    # interrupted upload of this exact deterministic local snapshot.  Matching-size
    # objects are complete S3 objects and can be safely reused.
    extra_remote = sorted(set(remote_inventory) - set(local_inventory))
    if extra_remote:
        raise RuntimeError(
            "Incomplete remote snapshot contains unexpected Parquet objects; "
            "refusing to mutate an ambiguous prefix: "
            f"{extra_remote[:10]}"
        )

    pending = [
        relative
        for relative, size in local_inventory.items()
        if remote_inventory.get(relative) != size
    ]
    reused = len(local_inventory) - len(pending)
    log.info(
        "final_model_table: publishing local snapshot to B2; "
        "files=%s pending=%s reusable_remote=%s bytes=%s",
        len(local_inventory),
        len(pending),
        reused,
        sum(local_inventory.values()),
    )

    client = lake.s3_client()
    transfer_config = TransferConfig(
        multipart_threshold=UPLOAD_CHUNK_BYTES,
        multipart_chunksize=UPLOAD_CHUNK_BYTES,
        max_concurrency=UPLOAD_PART_CONCURRENCY,
        use_threads=True,
    )

    with ThreadPoolExecutor(max_workers=UPLOAD_WORKERS) as executor:
        futures = {
            executor.submit(
                _upload_parquet_file,
                client=client,
                local_path=local_snapshot_dir / relative,
                bucket=bucket,
                key=f"{prefix}{relative}",
                transfer_config=transfer_config,
            ): relative
            for relative in pending
        }
        completed = 0
        for future in as_completed(futures):
            relative = futures[future]
            future.result()
            completed += 1
            if completed == len(pending) or completed % 10 == 0:
                log.info(
                    "final_model_table: B2 upload progress %s/%s pending shards",
                    completed,
                    len(pending),
                )

    remote_after = _remote_parquet_inventory(lake, snapshot_uri)
    if remote_after != local_inventory:
        missing = sorted(set(local_inventory) - set(remote_after))
        extra = sorted(set(remote_after) - set(local_inventory))
        size_mismatch = sorted(
            relative
            for relative in set(local_inventory) & set(remote_after)
            if local_inventory[relative] != remote_after[relative]
        )
        raise RuntimeError(
            "B2 final-model-table verification failed: "
            f"missing={missing[:10]}, extra={extra[:10]}, "
            f"size_mismatch={size_mismatch[:10]}"
        )

    return len(local_inventory), sum(local_inventory.values())


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
    snapshot_id: str,
    local_snapshot_dir: Path,
    event_snapshot_uri: str,
    integration_snapshot_uri: str,
    integration_manifest: Mapping[str, object],
    environmental_snapshot_uri: str,
    history_uris: list[str],
    support_intervals: list[ModelSupportInterval],
    log,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    dict[str, str],
    dict[str, object],
    int,
    int,
]:
    ready_state = _read_local_ready_state(
        snapshot_id=snapshot_id,
        snapshot_dir=local_snapshot_dir,
    )
    if ready_state is not None:
        log.info(
            "final_model_table: reusing audited local snapshot %s; "
            "skipping expensive Polars materialization",
            local_snapshot_dir,
        )
        return (
            dict(ready_state["audit"]),
            list(ready_state["grouped_audit"]),
            dict(ready_state["schema"]),
            dict(ready_state["build_metadata"]),
            int(ready_state["parquet_file_count"]),
            int(ready_state["parquet_size_bytes"]),
        )

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

    local_snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(local_snapshot_dir.parent)
    log.info(
        "final_model_table: materializing locally at %s; free_disk_bytes=%s",
        local_snapshot_dir,
        disk.free,
    )

    # The expensive graph executes exactly once and talks only to local disk here.
    _write_local_snapshot(
        table=table,
        snapshot_dir=local_snapshot_dir,
    )

    audit, grouped_audit = _audit_written_snapshot(
        lake=lake,
        snapshot_uri=str(local_snapshot_dir),
        support_intervals=support_intervals,
        expected_integration_rows=_expected_integration_rows(integration_manifest),
        expected_schema=expected_schema,
    )

    inventory = _local_parquet_inventory(local_snapshot_dir)
    parquet_file_count = len(inventory)
    parquet_size_bytes = sum(inventory.values())
    schema_document = {name: str(dtype) for name, dtype in expected_schema.items()}

    ready_state = {
        "snapshot_id": snapshot_id,
        "build_version": FINAL_MODEL_TABLE_BUILD_VERSION,
        "audit": audit,
        "grouped_audit": grouped_audit,
        "schema": schema_document,
        "build_metadata": build_metadata,
        "parquet_file_count": parquet_file_count,
        "parquet_size_bytes": parquet_size_bytes,
        "audited_at_utc": datetime.now(UTC).isoformat(),
    }
    # Written last and atomically renamed: its presence means the local shards
    # passed the persisted audit and are safe to reuse after a B2 failure.
    _write_local_json(local_snapshot_dir / LOCAL_READY_FILE, ready_state)

    log.info(
        "final_model_table: local snapshot audited and ready; files=%s bytes=%s",
        parquet_file_count,
        parquet_size_bytes,
    )
    return (
        audit,
        grouped_audit,
        schema_document,
        build_metadata,
        parquet_file_count,
        parquet_size_bytes,
    )


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
        "final_model_table: Polars thread pool=%s; build=%s; local_stage_root=%s",
        polars_threads,
        FINAL_MODEL_TABLE_BUILD_VERSION,
        LOCAL_STAGE_ROOT,
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
    local_snapshot_dir = _local_snapshot_dir(snapshot_id)

    if lake._object_exists(success_uri):
        manifest = json.loads(
            lake._read_object(lake.snapshot_manifest_uri(snapshot_uri))
        )
    else:
        (
            audit,
            grouped_audit,
            schema_document,
            build_metadata,
            local_parquet_file_count,
            local_parquet_size_bytes,
        ) = _materialize_final_model_snapshot(
            lake=lake,
            snapshot_id=snapshot_id,
            local_snapshot_dir=local_snapshot_dir,
            event_snapshot_uri=event_uri,
            integration_snapshot_uri=integration_uri,
            integration_manifest=integration_manifest,
            environmental_snapshot_uri=environmental_uri,
            history_uris=history_uris,
            support_intervals=support_intervals,
            log=context.log,
        )

        # Publication is a separate retryable phase.  A 503 here leaves the audited
        # local snapshot intact, so the next Dagster retry skips all Polars compute.
        remote_parquet_file_count, remote_parquet_size_bytes = _publish_local_snapshot(
            lake=lake,
            local_snapshot_dir=local_snapshot_dir,
            snapshot_uri=snapshot_uri,
            log=context.log,
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
                "target_host": "high_core_large_ram_local_nvme",
                "single_expensive_materialization": True,
                "materialization_target": "local_disk",
                "publication_target": "backblaze_b2",
                "local_persisted_audit": True,
                "post_upload_object_verification": True,
                "resumable_publication": True,
                "parquet_target_file_bytes": TARGET_FILE_BYTES,
                "parquet_row_group_size": ROW_GROUP_SIZE,
                "parquet_compression": "zstd:1",
                "maintain_output_order": False,
                "upload_workers": UPLOAD_WORKERS,
                "upload_part_concurrency": UPLOAD_PART_CONCURRENCY,
                "upload_chunk_bytes": UPLOAD_CHUNK_BYTES,
                "upload_file_attempts": UPLOAD_FILE_ATTEMPTS,
            },
            **build_metadata,
            "partition_columns": ["split", "source_city"],
            "schema": schema_document,
            "columns": FINAL_COLUMNS,
            **audit,
            "grouped_audit": grouped_audit,
            "parquet_file_count": remote_parquet_file_count,
            "parquet_size_bytes": remote_parquet_size_bytes,
            "local_parquet_file_count": local_parquet_file_count,
            "local_parquet_size_bytes": local_parquet_size_bytes,
        }

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
