"""Dagster orchestration and immutable publication for environmental features."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import dagster as dg
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from crimenet_data.assets.environmental.transformations import (
    ENVIRONMENTAL_FEATURE_SCHEMA,
    ENVIRONMENTAL_SCHEMA_VERSION,
    REQUIREMENT_KEY_SCHEMA,
    SILVER_WEATHER_SCHEMA,
    SILVER_WEATHER_SCHEMA_VERSION,
    WEATHER_CONTRACT_VERSION,
    archive_available_through_hour,
    build_environmental_features,
    build_required_environmental_keys,
    normalize_weather_envelope,
    validate_environmental_features,
    validate_environmental_features_lazy,
    validate_silver_weather,
    validate_silver_weather_lazy,
)
from crimenet_data.resources.crime_lake import CrimeLakeResources

RAW_YEAR_RE = re.compile(r"(?:^|/)year=(\d{4})(?:/|$)")
DEFAULT_RAW_WEATHER_READ_WORKERS = 16
RAW_WEATHER_PROGRESS_INTERVAL = 100

raw_model_weather_v2 = dg.AssetSpec(
    key=["raw", "model_weather_v2"],
    group_name="raw_environmental",
    description=(
        "Immutable Open-Meteo Best Match model_weather_v2 JSON envelopes at "
        "H3-r6 cell × UTC-year grain."
    ),
    metadata={
        "crime_lake_property": "model_weather_v2_best_match_root",
        "weather_contract_version": WEATHER_CONTRACT_VERSION,
    },
    kinds={"s3", "json"},
)

published_integration_sampling = dg.AssetSpec(
    key="published_integration_sampling",
    group_name="gold_integration",
    description=(
        "Current validated immutable integration-sampling snapshot used to define "
        "the environmental model-point universe."
    ),
    metadata={"crime_lake_property": "integration_latest_pointer_uri"},
    kinds={"s3", "parquet"},
)


def _stable_snapshot_id(prefix: str, values: Sequence[str]) -> str:
    digest = hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:24]}"


def _raw_year(uri: str) -> int:
    match = RAW_YEAR_RE.search(uri)
    if match is None:
        raise RuntimeError(f"model_weather_v2 object has no year partition: {uri}")
    return int(match.group(1))


def _read_json(lake: CrimeLakeResources, uri: str) -> dict[str, Any]:
    try:
        document = json.loads(lake._read_object(uri))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Malformed JSON object: {uri}") from error
    if not isinstance(document, dict):
        raise RuntimeError(f"Expected JSON object: {uri}")
    return document


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


def _publish_latest_pointer(
    *,
    lake: CrimeLakeResources,
    pointer_uri: str,
    snapshot_id: str,
    snapshot_uri: str,
    schema_version: str,
    created_at_utc: str,
) -> None:
    _write_json(
        lake,
        pointer_uri,
        {
            "snapshot_id": snapshot_id,
            "snapshot_uri": snapshot_uri,
            "schema_version": schema_version,
            "created_at_utc": created_at_utc,
        },
    )


def _validate_completed_manifest(
    *,
    lake: CrimeLakeResources,
    manifest: Mapping[str, object],
    snapshot_id: str,
    snapshot_uri: str,
    schema_version: str,
    schema: pl.Schema,
) -> None:
    expected = {
        "snapshot_id": snapshot_id,
        "snapshot_uri": snapshot_uri,
        "schema_version": schema_version,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise RuntimeError(
                f"Completed snapshot manifest has invalid {field}: "
                f"expected={value!r}, actual={manifest.get(field)!r}"
            )
    expected_schema = {name: str(dtype) for name, dtype in schema.items()}
    if manifest.get("schema") != expected_schema:
        raise RuntimeError("Completed snapshot manifest has an unexpected schema")
    years = manifest.get("years")
    if not isinstance(years, list) or not years:
        raise RuntimeError("Completed snapshot manifest has no year partitions")
    recorded_rows = 0
    for record in years:
        if not isinstance(record, Mapping):
            raise RuntimeError("Completed snapshot has a malformed year record")
        parquet_uri = str(record.get("parquet_uri", ""))
        if not parquet_uri.startswith(f"{snapshot_uri.rstrip('/')}/year="):
            raise RuntimeError(
                f"Completed snapshot records an invalid Parquet URI: {parquet_uri!r}"
            )
        if not lake._object_exists(parquet_uri):
            raise RuntimeError(f"Completed snapshot is missing Parquet: {parquet_uri}")
        recorded_rows += int(record.get("row_count", 0) or 0)
    if recorded_rows != int(manifest.get("row_count", -1)):
        raise RuntimeError(
            "Completed snapshot manifest row counts disagree: "
            f"partitions={recorded_rows}, total={manifest.get('row_count')}"
        )


def _scan_parquet(lake: CrimeLakeResources, uri_or_uris: str | list[str]) -> pl.LazyFrame:
    first = uri_or_uris[0] if isinstance(uri_or_uris, list) else uri_or_uris
    return pl.scan_parquet(
        uri_or_uris,
        storage_options=lake.storage_options_for(first),
        credential_provider=None,
        hive_partitioning=False,
    )


def _integration_sample_uris(
    *,
    lake: CrimeLakeResources,
    snapshot_uri: str,
    manifest: Mapping[str, object],
) -> list[str]:
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise RuntimeError("Integration manifest has no source sample metadata")
    uris: list[str] = []
    for record in sources:
        if not isinstance(record, Mapping):
            raise RuntimeError("Malformed integration source manifest record")
        source = str(record.get("source_city", "")).strip()
        part_count = int(record.get("sample_part_count", 0) or 0)
        if not source or part_count <= 0:
            raise RuntimeError(
                f"Malformed integration source sample metadata: {record}"
            )
        uris.extend(
            lake.integration_sample_part_uri(snapshot_uri, source, part_index)
            for part_index in range(part_count)
        )
    return uris


@dataclass(frozen=True)
class _NormalizedWeatherObject:
    source_uri: str
    h3_cell_id: int
    request_id: str
    payload_years: tuple[int, ...]
    row_count: int
    min_hour_utc: datetime
    max_hour_utc: datetime
    table: pa.Table


def _read_normalize_weather_object(
    *,
    lake: CrimeLakeResources,
    source_uri: str,
) -> _NormalizedWeatherObject:
    """Read and validate one raw object without touching shared output state."""

    try:
        envelope = _read_json(lake, source_uri)
        frame = normalize_weather_envelope(
            envelope,
            source_object_uri=source_uri,
        )
        validate_silver_weather(frame)
        min_hour = frame.get_column("hour").min()
        max_hour = frame.get_column("hour").max()
        if min_hour is None or max_hour is None:
            raise RuntimeError("normalized weather object contains no timestamps")
        return _NormalizedWeatherObject(
            source_uri=source_uri,
            h3_cell_id=int(frame.get_column("h3_cell_id")[0]),
            request_id=str(frame.get_column("weather_request_id")[0]),
            payload_years=tuple(
                int(year)
                for year in frame.get_column("hour").dt.year().unique().sort()
            ),
            row_count=frame.height,
            min_hour_utc=min_hour,
            max_hour_utc=max_hour,
            table=frame.to_arrow(),
        )
    except Exception as error:
        raise RuntimeError(
            f"Failed to read/normalize raw weather object {source_uri}: {error}"
        ) from error


def _iter_normalized_weather_objects(
    *,
    lake: CrimeLakeResources,
    source_uris: Sequence[str],
    max_workers: int,
) -> Iterator[_NormalizedWeatherObject]:
    """Prefetch objects concurrently while bounding queued Arrow tables."""

    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    ordered_uris = sorted(source_uris)
    if not ordered_uris:
        return

    max_in_flight = min(len(ordered_uris), max_workers * 2)
    executor = ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="silver-weather-reader",
    )
    futures: dict[Future[_NormalizedWeatherObject], int] = {}
    ready: dict[int, _NormalizedWeatherObject] = {}
    next_submit = 0
    next_yield = 0

    def fill_window() -> None:
        nonlocal next_submit
        while (
            next_submit < len(ordered_uris)
            and len(futures) + len(ready) < max_in_flight
        ):
            index = next_submit
            uri = ordered_uris[index]
            future = executor.submit(
                _read_normalize_weather_object,
                lake=lake,
                source_uri=uri,
            )
            futures[future] = index
            next_submit += 1

    try:
        fill_window()
        while futures:
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                index = futures.pop(future)
                ready[index] = future.result()
            while next_yield in ready:
                yield ready.pop(next_yield)
                next_yield += 1
            fill_window()
        while next_yield in ready:
            yield ready.pop(next_yield)
            next_yield += 1
    finally:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def _write_silver_weather_snapshot(
    *,
    lake: CrimeLakeResources,
    raw_uris: list[str],
    snapshot_uri: str,
    max_workers: int = DEFAULT_RAW_WEATHER_READ_WORKERS,
    progress_interval: int = RAW_WEATHER_PROGRESS_INTERVAL,
    progress_log: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if progress_interval < 1:
        raise ValueError("progress_interval must be at least 1")

    by_year: dict[int, list[str]] = defaultdict(list)
    for uri in raw_uris:
        by_year[_raw_year(uri)].append(uri)

    year_summaries: list[dict[str, object]] = []
    seen_cell_years: set[tuple[int, int]] = set()
    seen_request_ids: set[str] = set()
    total_rows = 0
    all_cells: set[int] = set()
    minimum_hour: datetime | None = None
    maximum_hour: datetime | None = None

    with tempfile.TemporaryDirectory(prefix="crimenet-silver-weather-") as staging:
        staging_root = Path(staging)
        for year, year_uris in sorted(by_year.items()):
            local_part = staging_root / f"year={year}" / "part-00000.parquet"
            local_part.parent.mkdir(parents=True, exist_ok=True)
            writer: pq.ParquetWriter | None = None
            year_rows = 0
            year_cells: set[int] = set()
            year_min: datetime | None = None
            year_max: datetime | None = None
            try:
                normalized_objects = _iter_normalized_weather_objects(
                    lake=lake,
                    source_uris=year_uris,
                    max_workers=max_workers,
                )
                for completed_count, normalized in enumerate(
                    normalized_objects,
                    start=1,
                ):
                    uri = normalized.source_uri
                    cell = normalized.h3_cell_id
                    request_id = normalized.request_id
                    logical_key = (cell, year)
                    if logical_key in seen_cell_years:
                        raise RuntimeError(
                            "Duplicate raw weather logical object for "
                            f"h3_cell_id={cell}, year={year}, source_uri={uri}"
                        )
                    if request_id in seen_request_ids:
                        raise RuntimeError(
                            "Duplicate raw weather request_id: "
                            f"{request_id}, source_uri={uri}"
                        )
                    if normalized.payload_years != (year,):
                        raise RuntimeError(
                            "Raw weather URI year disagrees with payload: "
                            f"source_uri={uri}, uri_year={year}, "
                            f"payload_years={list(normalized.payload_years)}"
                        )
                    seen_cell_years.add(logical_key)
                    seen_request_ids.add(request_id)
                    if writer is None:
                        writer = pq.ParquetWriter(
                            local_part,
                            normalized.table.schema,
                            compression="zstd",
                            use_dictionary=True,
                        )
                    try:
                        writer.write_table(normalized.table)
                    except Exception as error:
                        raise RuntimeError(
                            "Failed to write normalized raw weather object "
                            f"{uri}: {error}"
                        ) from error
                    year_rows += normalized.row_count
                    year_cells.add(cell)
                    frame_min = normalized.min_hour_utc
                    frame_max = normalized.max_hour_utc
                    year_min = frame_min if year_min is None else min(year_min, frame_min)
                    year_max = frame_max if year_max is None else max(year_max, frame_max)
                    if progress_log is not None and (
                        completed_count % progress_interval == 0
                        or completed_count == len(year_uris)
                    ):
                        progress_log(
                            f"year={year} objects={completed_count}/{len(year_uris)} "
                            f"rows_written={year_rows}"
                        )
            finally:
                if writer is not None:
                    writer.close()
            if writer is None:
                raise RuntimeError(f"No raw weather objects were normalized for {year}")

            destination = lake.silver_weather_year_uri(snapshot_uri, year)
            lake.upload_local_file(local_part, destination)
            summary = validate_silver_weather_lazy(_scan_parquet(lake, destination))
            if int(summary["row_count"]) != year_rows:
                raise RuntimeError(
                    f"Silver weather {year} read-back count mismatch: "
                    f"expected={year_rows}, actual={summary['row_count']}"
                )
            year_summaries.append(
                {
                    "year": year,
                    "raw_object_count": len(year_uris),
                    **summary,
                    "parquet_uri": destination,
                }
            )
            total_rows += year_rows
            all_cells.update(year_cells)
            minimum_hour = year_min if minimum_hour is None else min(minimum_hour, year_min)
            maximum_hour = year_max if maximum_hour is None else max(maximum_hour, year_max)

    return year_summaries, {
        "row_count": total_rows,
        "unique_h3_cells": len(all_cells),
        "min_hour_utc": minimum_hour,
        "max_hour_utc": maximum_hour,
        "duplicate_key_count": 0,
        "raw_reader_workers": max_workers,
    }


@dg.asset(
    name="silver_weather_features",
    group_name="silver_environmental",
    deps=[raw_model_weather_v2],
    required_resource_keys={"crime_lake"},
    compute_kind="polars+pyarrow",
    description=(
        "Normalize model_weather_v2 Open-Meteo Best Match cell-year JSON "
        "envelopes to compact H3-r6 × UTC-hour Parquet."
    ),
)
def silver_weather_features(context) -> dg.MaterializeResult:
    lake: CrimeLakeResources = context.resources.crime_lake
    raw_uris = lake.list_object_uris(
        lake.model_weather_v2_best_match_root,
        suffix=".json",
    )
    if not raw_uris:
        raise RuntimeError(
            "No model_weather_v2 Best Match raw JSON objects found at "
            f"{lake.model_weather_v2_best_match_root}"
        )
    raw_root_prefix = f"{lake.model_weather_v2_best_match_root.rstrip('/')}/"
    raw_object_keys = [uri.removeprefix(raw_root_prefix) for uri in raw_uris]
    snapshot_id = _stable_snapshot_id(
        "weather-v2",
        [
            *raw_object_keys,
            WEATHER_CONTRACT_VERSION,
            SILVER_WEATHER_SCHEMA_VERSION,
        ],
    )
    snapshot_uri = lake.silver_weather_snapshot_uri(snapshot_id)
    success_uri = lake.snapshot_success_uri(snapshot_uri)
    if lake._object_exists(success_uri):
        manifest = _read_json(lake, lake.snapshot_manifest_uri(snapshot_uri))
    else:
        year_summaries, summary = _write_silver_weather_snapshot(
            lake=lake,
            raw_uris=raw_uris,
            snapshot_uri=snapshot_uri,
            progress_log=context.log.info,
        )
        created_at = datetime.now(UTC).isoformat()
        manifest = {
            "snapshot_id": snapshot_id,
            "snapshot_uri": snapshot_uri,
            "created_at_utc": created_at,
            "schema_version": SILVER_WEATHER_SCHEMA_VERSION,
            "weather_contract_version": WEATHER_CONTRACT_VERSION,
            "raw_root": lake.model_weather_v2_best_match_root,
            "raw_object_count": len(raw_uris),
            "raw_object_set_sha256": hashlib.sha256(
                "\n".join(sorted(raw_object_keys)).encode("utf-8")
            ).hexdigest(),
            "schema": {name: str(dtype) for name, dtype in SILVER_WEATHER_SCHEMA.items()},
            "partitioning": ["year"],
            **summary,
            "years": year_summaries,
        }
        _write_json(lake, lake.snapshot_manifest_uri(snapshot_uri), manifest)
        lake._write_object(success_uri, b"", content_type="application/octet-stream")

    _validate_completed_manifest(
        lake=lake,
        manifest=manifest,
        snapshot_id=snapshot_id,
        snapshot_uri=snapshot_uri,
        schema_version=SILVER_WEATHER_SCHEMA_VERSION,
        schema=SILVER_WEATHER_SCHEMA,
    )
    _publish_latest_pointer(
        lake=lake,
        pointer_uri=lake.silver_weather_latest_pointer_uri,
        snapshot_id=snapshot_id,
        snapshot_uri=snapshot_uri,
        schema_version=SILVER_WEATHER_SCHEMA_VERSION,
        created_at_utc=str(manifest["created_at_utc"]),
    )

    return dg.MaterializeResult(
        metadata={
            "snapshot_id": snapshot_id,
            "snapshot_uri": snapshot_uri,
            "raw_object_count": int(manifest["raw_object_count"]),
            "row_count": int(manifest["row_count"]),
            "unique_h3_cells": int(manifest["unique_h3_cells"]),
            "min_hour_utc": str(manifest["min_hour_utc"]),
            "max_hour_utc": str(manifest["max_hour_utc"]),
            "duplicate_key_count": int(manifest["duplicate_key_count"]),
            "raw_reader_workers": int(
                manifest.get(
                    "raw_reader_workers",
                    DEFAULT_RAW_WEATHER_READ_WORKERS,
                )
            ),
        }
    )


def _stage_requirement_keys(
    *,
    lake: CrimeLakeResources,
    event_snapshot_uri: str,
    integration_snapshot_uri: str,
    integration_manifest: Mapping[str, object],
    staging_root: Path,
) -> list[int]:
    event_glob = lake.event_spine_parquet_glob(event_snapshot_uri)
    events = pl.scan_parquet(
        event_glob,
        storage_options=lake.storage_options_for(event_glob),
        credential_provider=None,
        hive_partitioning=True,
    )
    integration_uris = _integration_sample_uris(
        lake=lake,
        snapshot_uri=integration_snapshot_uri,
        manifest=integration_manifest,
    )
    integration = _scan_parquet(lake, integration_uris)
    requirements = build_required_environmental_keys(
        events=events,
        integration=integration,
    ).with_columns(pl.col("hour").dt.year().cast(pl.Int32).alias("year"))
    requirements.sink_parquet(
        pl.PartitionBy(str(staging_root), key=["year"], include_key=False),
        compression="zstd",
        compression_level=3,
        mkdir=True,
        engine="streaming",
    )
    years = sorted(
        int(path.name.split("=", 1)[1])
        for path in staging_root.glob("year=*")
        if path.is_dir()
    )
    if not years:
        raise RuntimeError("No environmental requirement keys were staged")
    return years


def _materialize_environmental_snapshot(
    *,
    lake: CrimeLakeResources,
    snapshot_uri: str,
    silver_snapshot_uri: str,
    event_snapshot_uri: str,
    integration_snapshot_uri: str,
    integration_manifest: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    cutoff = archive_available_through_hour()
    year_summaries: list[dict[str, object]] = []
    all_cells: set[int] = set()
    total_rows = 0
    minimum_hour: datetime | None = None
    maximum_hour: datetime | None = None

    with tempfile.TemporaryDirectory(prefix="crimenet-environmental-") as staging:
        staging_root = Path(staging)
        requirements_root = staging_root / "requirements"
        years = _stage_requirement_keys(
            lake=lake,
            event_snapshot_uri=event_snapshot_uri,
            integration_snapshot_uri=integration_snapshot_uri,
            integration_manifest=integration_manifest,
            staging_root=requirements_root,
        )
        for year in years:
            requirement_parts = sorted(
                str(path) for path in (requirements_root / f"year={year}").rglob("*.parquet")
            )
            requirements = pl.read_parquet(requirement_parts).cast(REQUIREMENT_KEY_SCHEMA)
            weather_uri = lake.silver_weather_year_uri(silver_snapshot_uri, year)
            if lake._object_exists(weather_uri):
                weather = _scan_parquet(lake, weather_uri).select(
                    "h3_cell_id",
                    "hour",
                    "weather_temperature_2m_c",
                    "weather_relative_humidity_2m_pct",
                ).collect(engine="streaming")
            else:
                weather = pl.DataFrame(
                    schema={
                        "h3_cell_id": pl.Int64,
                        "hour": pl.Datetime("us", time_zone="UTC"),
                        "weather_temperature_2m_c": pl.Float32,
                        "weather_relative_humidity_2m_pct": pl.Float32,
                    }
                )
            environmental = build_environmental_features(
                requirements=requirements,
                silver_weather=weather,
            )
            summary = validate_environmental_features(
                environmental,
                archive_cutoff_hour=cutoff,
            )
            local_part = staging_root / "output" / f"year={year}" / "part-00000.parquet"
            local_part.parent.mkdir(parents=True, exist_ok=True)
            environmental.write_parquet(
                local_part,
                compression="zstd",
                compression_level=3,
                statistics=True,
            )
            destination = lake.environmental_features_year_uri(snapshot_uri, year)
            lake.upload_local_file(local_part, destination)
            readback_summary = validate_environmental_features_lazy(
                _scan_parquet(lake, destination),
                archive_cutoff_hour=cutoff,
            )
            if readback_summary != summary:
                raise RuntimeError(
                    f"Gold environmental {year} read-back validation mismatch"
                )
            year_summaries.append({"year": year, "parquet_uri": destination, **summary})
            total_rows += int(summary["row_count"])
            all_cells.update(environmental.get_column("h3_cell_id").unique().to_list())
            year_min = summary["min_hour_utc"]
            year_max = summary["max_hour_utc"]
            minimum_hour = year_min if minimum_hour is None else min(minimum_hour, year_min)
            maximum_hour = year_max if maximum_hour is None else max(maximum_hour, year_max)

    aggregate_keys = [
        "missing_lighting_rows",
        "inconsistent_lighting_rows",
        "inconsistent_weather_rows",
        "partial_weather_rows",
        "invalid_reference_count_rows",
        "weather_available_rows",
        "weather_null_rows",
        "event_reference_count",
        "event_weather_covered_references",
        "integration_reference_count",
        "integration_weather_covered_references",
        "point_reference_count",
        "point_weather_covered_references",
        "unexpected_archive_eligible_missing_rows",
    ]
    totals = {
        key: sum(int(summary[key]) for summary in year_summaries)
        for key in aggregate_keys
    }

    def pct(numerator: int, denominator: int) -> float:
        return 100.0 * numerator / denominator if denominator else 100.0

    return year_summaries, {
        "row_count": total_rows,
        "unique_h3_cells": len(all_cells),
        "min_hour_utc": minimum_hour,
        "max_hour_utc": maximum_hour,
        "duplicate_key_count": 0,
        **totals,
        "weather_coverage_pct": pct(totals["weather_available_rows"], total_rows),
        "event_weighted_weather_coverage_pct": pct(
            totals["event_weather_covered_references"],
            totals["event_reference_count"],
        ),
        "integration_weighted_weather_coverage_pct": pct(
            totals["integration_weather_covered_references"],
            totals["integration_reference_count"],
        ),
        "point_weighted_weather_coverage_pct": pct(
            totals["point_weather_covered_references"],
            totals["point_reference_count"],
        ),
        "archive_available_through_hour": cutoff,
    }


@dg.asset(
    name="environmental_features",
    group_name="gold_environmental",
    deps=[
        silver_weather_features,
        "gold_event_spine",
        published_integration_sampling,
    ],
    required_resource_keys={"crime_lake"},
    compute_kind="polars+pvlib",
    description=(
        "Unified H3-r6 × UTC-hour environmental store: nullable Best Match "
        "weather plus deterministic pvlib NREL-SPA lighting."
    ),
)
def environmental_features(context) -> dg.MaterializeResult:
    lake: CrimeLakeResources = context.resources.crime_lake
    event_snapshot_uri, event_manifest = lake.resolve_event_spine_snapshot()
    integration_snapshot_uri, integration_manifest = (
        lake.resolve_current_integration_snapshot()
    )
    silver_snapshot_uri, silver_manifest = (
        lake.resolve_current_silver_weather_snapshot()
    )
    event_snapshot_id = str(event_manifest.get("snapshot_id", ""))
    integration_snapshot_id = str(integration_manifest.get("snapshot_id", ""))
    silver_snapshot_id = str(silver_manifest.get("snapshot_id", ""))
    if not all((event_snapshot_id, integration_snapshot_id, silver_snapshot_id)):
        raise RuntimeError("Environmental input manifests are missing snapshot IDs")

    snapshot_id = _stable_snapshot_id(
        "environmental-v1",
        [
            event_snapshot_id,
            integration_snapshot_id,
            silver_snapshot_id,
            ENVIRONMENTAL_SCHEMA_VERSION,
        ],
    )
    snapshot_uri = lake.environmental_features_snapshot_uri(snapshot_id)
    success_uri = lake.snapshot_success_uri(snapshot_uri)
    if lake._object_exists(success_uri):
        manifest = _read_json(lake, lake.snapshot_manifest_uri(snapshot_uri))
    else:
        year_summaries, summary = _materialize_environmental_snapshot(
            lake=lake,
            snapshot_uri=snapshot_uri,
            silver_snapshot_uri=silver_snapshot_uri,
            event_snapshot_uri=event_snapshot_uri,
            integration_snapshot_uri=integration_snapshot_uri,
            integration_manifest=integration_manifest,
        )
        created_at = datetime.now(UTC).isoformat()
        manifest = {
            "snapshot_id": snapshot_id,
            "snapshot_uri": snapshot_uri,
            "created_at_utc": created_at,
            "schema_version": ENVIRONMENTAL_SCHEMA_VERSION,
            "weather_contract_version": WEATHER_CONTRACT_VERSION,
            "silver_weather_snapshot_id": silver_snapshot_id,
            "silver_weather_snapshot_uri": silver_snapshot_uri,
            "event_spine_snapshot_id": event_snapshot_id,
            "event_spine_snapshot_uri": event_snapshot_uri,
            "integration_sampling_snapshot_id": integration_snapshot_id,
            "integration_sampling_snapshot_uri": integration_snapshot_uri,
            "key_grain": ["h3_cell_id", "hour"],
            "h3_resolution": 6,
            "weather_join_policy": "left_join_nullable",
            "lighting_algorithm": "pvlib_nrel_spa_apparent_solar_position",
            "lighting_thresholds_deg": {
                "day": "[0, +inf)",
                "civil_twilight": "[-6, 0)",
                "nautical_twilight": "[-12, -6)",
                "astronomical_twilight": "[-18, -12)",
                "night": "(-inf, -18)",
            },
            "schema": {
                name: str(dtype) for name, dtype in ENVIRONMENTAL_FEATURE_SCHEMA.items()
            },
            "partitioning": ["year"],
            **summary,
            "years": year_summaries,
        }
        _write_json(lake, lake.snapshot_manifest_uri(snapshot_uri), manifest)
        lake._write_object(success_uri, b"", content_type="application/octet-stream")

    _validate_completed_manifest(
        lake=lake,
        manifest=manifest,
        snapshot_id=snapshot_id,
        snapshot_uri=snapshot_uri,
        schema_version=ENVIRONMENTAL_SCHEMA_VERSION,
        schema=ENVIRONMENTAL_FEATURE_SCHEMA,
    )
    _publish_latest_pointer(
        lake=lake,
        pointer_uri=lake.environmental_features_latest_pointer_uri,
        snapshot_id=snapshot_id,
        snapshot_uri=snapshot_uri,
        schema_version=ENVIRONMENTAL_SCHEMA_VERSION,
        created_at_utc=str(manifest["created_at_utc"]),
    )

    return dg.MaterializeResult(
        metadata={
            key: value
            for key, value in {
                "snapshot_id": snapshot_id,
                "snapshot_uri": snapshot_uri,
                "event_spine_snapshot_id": event_snapshot_id,
                "integration_sampling_snapshot_id": integration_snapshot_id,
                "silver_weather_snapshot_id": silver_snapshot_id,
                "row_count": int(manifest["row_count"]),
                "unique_h3_cells": int(manifest["unique_h3_cells"]),
                "weather_coverage_pct": float(manifest["weather_coverage_pct"]),
                "weather_available_rows": int(manifest["weather_available_rows"]),
                "event_weighted_weather_coverage_pct": float(
                    manifest["event_weighted_weather_coverage_pct"]
                ),
                "integration_weighted_weather_coverage_pct": float(
                    manifest["integration_weighted_weather_coverage_pct"]
                ),
                "weather_null_rows": int(manifest["weather_null_rows"]),
                "duplicate_key_count": int(manifest["duplicate_key_count"]),
            }.items()
        }
    )


@dg.asset_check(
    asset=silver_weather_features,
    name="published_contract",
    blocking=True,
    required_resource_keys={"crime_lake"},
)
def silver_weather_published_contract_check(context) -> dg.AssetCheckResult:
    lake: CrimeLakeResources = context.resources.crime_lake
    _, manifest = lake.resolve_current_silver_weather_snapshot()
    duplicate_keys = int(manifest.get("duplicate_key_count", -1))
    return dg.AssetCheckResult(
        passed=duplicate_keys == 0 and int(manifest.get("row_count", 0)) > 0,
        metadata={
            "row_count": int(manifest.get("row_count", 0)),
            "duplicate_key_count": duplicate_keys,
        },
    )


@dg.asset_check(
    asset=environmental_features,
    name="published_contract",
    blocking=True,
    required_resource_keys={"crime_lake"},
)
def environmental_features_published_contract_check(context) -> dg.AssetCheckResult:
    lake: CrimeLakeResources = context.resources.crime_lake
    _, manifest = lake.resolve_current_environmental_features_snapshot()
    duplicate_keys = int(manifest.get("duplicate_key_count", -1))
    unexpected_missing = int(
        manifest.get("unexpected_archive_eligible_missing_rows", -1)
    )
    missing_lighting = int(manifest.get("missing_lighting_rows", -1))
    return dg.AssetCheckResult(
        passed=(
            int(manifest.get("row_count", 0)) > 0
            and duplicate_keys == 0
            and missing_lighting == 0
        ),
        metadata={
            "row_count": int(manifest.get("row_count", 0)),
            "duplicate_key_count": duplicate_keys,
            "unexpected_archive_eligible_missing_rows": unexpected_missing,
            "missing_lighting_rows": missing_lighting,
        },
    )


environmental_assets = [silver_weather_features, environmental_features]
environmental_asset_checks = [
    silver_weather_published_contract_check,
    environmental_features_published_contract_check,
]

__all__ = [
    "environmental_assets",
    "environmental_asset_checks",
    "environmental_features",
    "published_integration_sampling",
    "raw_model_weather_v2",
    "silver_weather_features",
]
