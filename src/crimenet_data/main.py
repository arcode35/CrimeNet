#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3
import polars as pl


DEFAULT_EVENT_SPINE_ROOT = "s3://crimenet-data/gold/event_spine"
DEFAULT_HISTORY_ROOT = (
    "s3://crimenet-data/gold/national_feature_store/temporal/h3_r9/history"
)

# Current known-good production result. These are reference values, not
# permanently valid invariants for all future snapshots.
CURRENT_REFERENCE = {
    "modeled_rows": 15_955_507,
    "invalid_event_utc_rows": 455,
    "joinable_rows": 15_955_052,
    "history_unmatched_rows": 4_430,
    "event_spine_rows": 15_950_622,
    "coverage_pct": 99.96938361156434,
}

REQUIRED_COLUMNS = {
    "crime_id",
    "source_city",
    "occurrence_timestamp_utc",
    "osm_h3_cell_id",
    "feature_available_at",
    "feature_version_id",
}

HISTORY_KEY = [
    "osm_h3_cell_id",
    "feature_available_at",
]

HISTORY_REQUIRED_COLUMNS = {
    "osm_h3_cell_id",
    "feature_available_at",
    "feature_version_id",
}

COMPONENT_AVAILABILITY_COLUMNS = [
    "osm_available_at",
    "acs_release_date",
    "tiger_release_date",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class AuditFailure(RuntimeError):
    pass


def log(message: str) -> None:
    now = datetime.now(UTC).strftime("%H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def human_int(value: int | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:,}"


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def derive_region(endpoint_url: str) -> str:
    match = re.search(
        r"https?://s3\.([^.]+)\.backblazeb2\.com",
        endpoint_url,
    )
    if match:
        return match.group(1)

    return os.getenv("AWS_REGION", "us-east-1")


def b2_config() -> dict[str, str]:
    endpoint = require_env("B2_ENDPOINT_URL")

    return {
        "key_id": require_env("B2_KEY_ID"),
        "application_key": require_env("B2_APPLICATION_KEY"),
        "endpoint_url": endpoint,
        "region": derive_region(endpoint),
    }


def polars_storage_options(config: dict[str, str]) -> dict[str, str]:
    return {
        "aws_access_key_id": config["key_id"],
        "aws_secret_access_key": config["application_key"],
        "aws_endpoint_url": config["endpoint_url"],
        "aws_region": config["region"],
    }


def s3_client(config: dict[str, str]):
    return boto3.client(
        "s3",
        endpoint_url=config["endpoint_url"],
        aws_access_key_id=config["key_id"],
        aws_secret_access_key=config["application_key"],
        region_name=config["region"],
    )


def split_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)

    if parsed.scheme != "s3":
        raise ValueError(f"Expected s3:// URI, got: {uri}")

    return parsed.netloc, parsed.path.lstrip("/")


def object_exists(client, uri: str) -> bool:
    bucket, key = split_s3_uri(uri)

    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def read_object_text(client, uri: str) -> str:
    bucket, key = split_s3_uri(uri)
    response = client.get_object(Bucket=bucket, Key=key)
    return response["Body"].read().decode("utf-8").strip()


def resolve_latest_snapshot(
    client,
    event_spine_root: str,
) -> str:
    pointer_uri = f"{event_spine_root.rstrip('/')}/_latest.json"
    
    raw = read_object_text(client, pointer_uri)

    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw

    if isinstance(value, str):
        if value.startswith("s3://"):
            return value.rstrip("/")

        if value.startswith("snapshot_id="):
            return f"{event_spine_root.rstrip('/')}/{value}"

        return (
            f"{event_spine_root.rstrip('/')}/"
            f"snapshot_id={value}"
        )

    if isinstance(value, dict):
        if value.get("snapshot_uri"):
            return str(value["snapshot_uri"]).rstrip("/")

        if value.get("snapshot_id"):
            return (
                f"{event_spine_root.rstrip('/')}/"
                f"snapshot_id={value['snapshot_id']}"
            )

    raise RuntimeError(
        f"Could not understand _latest pointer contents: {raw!r}"
    )


def scan_parquet_root(
    uri: str,
    storage_options: dict[str, str],
) -> pl.LazyFrame:
    return pl.scan_parquet(
        f"{uri.rstrip('/')}/**/*.parquet",
        storage_options=storage_options,
        hive_partitioning=True,
    )


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}

    if isinstance(value, list):
        return [to_jsonable(v) for v in value]

    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]

    if isinstance(value, datetime):
        return value.isoformat()

    return value


def collect_one(lf: pl.LazyFrame) -> dict[str, Any]:
    return lf.collect(engine="streaming").row(0, named=True)


def fail_if(condition: bool, message: str, failures: list[str]) -> None:
    if condition:
        failures.append(message)
        log(f"FAIL: {message}")


def pass_log(message: str) -> None:
    log(f"PASS: {message}")


# ---------------------------------------------------------------------------
# Publication audit
# ---------------------------------------------------------------------------


def audit_publication(
    client,
    snapshot_uri: str,
    event_spine_root: str,
) -> dict[str, Any]:
    log("AUDIT 1: publication / snapshot integrity")

    success_uri = f"{snapshot_uri}/_SUCCESS"
    manifest_uri = f"{snapshot_uri}/manifest.json"

    success_exists = object_exists(client, success_uri)
    manifest_exists = object_exists(client, manifest_uri)

    latest_snapshot = resolve_latest_snapshot(
        client,
        event_spine_root,
    )

    manifest = None

    if manifest_exists:
        try:
            manifest = json.loads(
                read_object_text(client, manifest_uri)
            )
        except Exception as exc:
            manifest = {
                "_parse_error": repr(exc),
            }

    result = {
        "snapshot_uri": snapshot_uri,
        "_SUCCESS_exists": success_exists,
        "manifest_exists": manifest_exists,
        "latest_snapshot_uri": latest_snapshot,
        "is_current_latest": (
            latest_snapshot.rstrip("/") == snapshot_uri.rstrip("/")
        ),
        "manifest": manifest,
    }

    if success_exists:
        pass_log("_SUCCESS exists")
    else:
        log("FAIL: _SUCCESS missing")

    if manifest_exists:
        pass_log("manifest.json exists")
    else:
        log("FAIL: manifest.json missing")

    if result["is_current_latest"]:
        pass_log("snapshot matches _latest")
    else:
        log(
            "WARN: audited snapshot is not the snapshot currently "
            "referenced by _latest"
        )

    return result


# ---------------------------------------------------------------------------
# Basic schema / grain / null audit
# ---------------------------------------------------------------------------


def audit_basic_integrity(
    spine: pl.LazyFrame,
) -> tuple[dict[str, Any], list[str], list[str]]:
    log("AUDIT 2: schema, grain, uniqueness, and required nulls")

    schema = spine.collect_schema()
    columns = schema.names()

    missing = sorted(REQUIRED_COLUMNS - set(columns))

    if missing:
        raise AuditFailure(
            f"Event spine missing required columns: {missing}"
        )

    exprs: list[pl.Expr] = [
        pl.len().alias("row_count"),
        pl.col("crime_id").n_unique().alias("unique_crime_ids"),
    ]

    for column in sorted(REQUIRED_COLUMNS):
        exprs.append(
            pl.col(column)
            .null_count()
            .alias(f"null__{column}")
        )

    stats = collect_one(spine.select(exprs))

    stats["duplicate_crime_ids"] = (
        stats["row_count"] - stats["unique_crime_ids"]
    )

    failures: list[str] = []

    fail_if(
        stats["duplicate_crime_ids"] != 0,
        f"crime_id grain violation: "
        f"{stats['duplicate_crime_ids']:,} duplicate rows",
        failures,
    )

    for column in sorted(REQUIRED_COLUMNS):
        count = stats[f"null__{column}"]

        fail_if(
            count != 0,
            f"{column} contains {count:,} null rows",
            failures,
        )

    if not failures:
        pass_log(
            f"one row per crime_id across "
            f"{stats['row_count']:,} rows"
        )
        pass_log("all required publication columns are non-null")

    return stats, columns, failures


# ---------------------------------------------------------------------------
# Temporal leakage / chronology audit
# ---------------------------------------------------------------------------


def audit_temporal_integrity(
    spine: pl.LazyFrame,
    columns: list[str],
) -> tuple[dict[str, Any], list[str]]:
    log("AUDIT 3: temporal leakage and feature chronology")

    exprs: list[pl.Expr] = [
        (
            pl.col("feature_available_at")
            > pl.col("occurrence_timestamp_utc")
        )
        .fill_null(False)
        .sum()
        .alias("future_feature_leaks"),
        pl.col("occurrence_timestamp_utc")
        .min()
        .alias("min_occurrence_timestamp_utc"),
        pl.col("occurrence_timestamp_utc")
        .max()
        .alias("max_occurrence_timestamp_utc"),
        pl.col("feature_available_at")
        .min()
        .alias("min_feature_available_at"),
        pl.col("feature_available_at")
        .max()
        .alias("max_feature_available_at"),
        pl.col("feature_version_id")
        .n_unique()
        .alias("feature_versions_used"),
    ]

    component_columns = [
        c for c in COMPONENT_AVAILABILITY_COLUMNS if c in columns
    ]

    for component in component_columns:
        exprs.append(
            (
                pl.col(component)
                > pl.col("feature_available_at")
            )
            .fill_null(False)
            .sum()
            .alias(f"future_component__{component}")
        )

    stats = collect_one(spine.select(exprs))

    failures: list[str] = []

    fail_if(
        stats["future_feature_leaks"] != 0,
        f"{stats['future_feature_leaks']:,} rows use a feature "
        f"version from the future",
        failures,
    )

    for component in component_columns:
        count = stats[f"future_component__{component}"]

        fail_if(
            count != 0,
            f"{component}: {count:,} chronology violations",
            failures,
        )

    if not failures:
        pass_log("zero future feature leaks")
        pass_log("zero component availability chronology violations")

    return stats, failures


# ---------------------------------------------------------------------------
# Timestamp reconstruction audit
# ---------------------------------------------------------------------------


def audit_timezone_reconstruction(
    spine: pl.LazyFrame,
    columns: list[str],
) -> tuple[dict[str, Any], list[str]]:
    log("AUDIT 4: source-local timestamp → UTC reconstruction")

    required = {
        "occurrence_timestamp",
        "source_timezone",
        "occurrence_timestamp_utc",
    }

    if not required.issubset(columns):
        log(
            "SKIP: occurrence_timestamp/source_timezone not all "
            "present in published spine"
        )
        return {"skipped": True}, []

    timezones_df = (
        spine.select("source_timezone")
        .drop_nulls()
        .unique()
        .sort("source_timezone")
        .collect(engine="streaming")
    )

    timezones = timezones_df["source_timezone"].to_list()

    expected: pl.Expr = pl.lit(
        None,
        dtype=pl.Datetime("us", time_zone="UTC"),
    )

    for timezone in reversed(timezones):
        reconstructed = (
            pl.col("occurrence_timestamp")
            .dt.replace_time_zone(
                timezone,
                ambiguous="earliest",
                non_existent="null",
            )
            .dt.convert_time_zone("UTC")
        )

        expected = (
            pl.when(pl.col("source_timezone") == timezone)
            .then(reconstructed)
            .otherwise(expected)
        )

    result = collect_one(
        spine.select(
            (~expected.eq_missing(
                pl.col("occurrence_timestamp_utc")
            ))
            .sum()
            .alias("utc_reconstruction_mismatch_rows"),
            pl.col("source_timezone")
            .n_unique()
            .alias("source_timezone_count"),
        )
    )

    result["timezones"] = timezones

    failures: list[str] = []

    fail_if(
        result["utc_reconstruction_mismatch_rows"] != 0,
        f"{result['utc_reconstruction_mismatch_rows']:,} UTC "
        f"timestamps do not reconstruct from source-local time",
        failures,
    )

    if not failures:
        pass_log(
            "all event UTC timestamps reproduce exactly from "
            "source-local timestamps"
        )

    return result, failures


# ---------------------------------------------------------------------------
# Spatial / occurrence-year audit
# ---------------------------------------------------------------------------


def audit_event_sanity(
    spine: pl.LazyFrame,
    columns: list[str],
) -> tuple[dict[str, Any], list[str]]:
    log("AUDIT 5: coordinate and occurrence-year sanity")

    exprs: list[pl.Expr] = []
    failures: list[str] = []

    if {"latitude", "longitude"}.issubset(columns):
        exprs.extend(
            [
                (
                    pl.col("latitude").is_null()
                    | (pl.col("latitude") < -90)
                    | (pl.col("latitude") > 90)
                )
                .sum()
                .alias("invalid_latitude_rows"),
                (
                    pl.col("longitude").is_null()
                    | (pl.col("longitude") < -180)
                    | (pl.col("longitude") > 180)
                )
                .sum()
                .alias("invalid_longitude_rows"),
            ]
        )

    if {
        "occurrence_year",
        "occurrence_timestamp",
    }.issubset(columns):
        exprs.append(
            (
                pl.col("occurrence_year")
                != pl.col("occurrence_timestamp").dt.year()
            )
            .fill_null(True)
            .sum()
            .alias("occurrence_year_mismatch_rows")
        )

    if not exprs:
        log("SKIP: no applicable coordinate/year fields")
        return {"skipped": True}, []

    stats = collect_one(spine.select(exprs))

    for field, value in stats.items():
        fail_if(
            value != 0,
            f"{field} = {value:,}",
            failures,
        )

    if not failures:
        pass_log("coordinate/year sanity checks clean")

    return stats, failures


# ---------------------------------------------------------------------------
# Distribution / null-profile audit
# ---------------------------------------------------------------------------


def audit_distributions(
    spine: pl.LazyFrame,
    columns: list[str],
) -> dict[str, Any]:
    log("AUDIT 6: source/year distributions and null profile")

    group_columns = ["source_city"]

    if "occurrence_year" in columns:
        group_columns.append("occurrence_year")

    counts_df = (
        spine.group_by(group_columns)
        .agg(pl.len().alias("rows"))
        .sort(group_columns)
        .collect(engine="streaming")
    )

    null_exprs = [
        pl.col(c).null_count().alias(c)
        for c in columns
    ]

    null_counts = collect_one(
        spine.select(null_exprs)
    )

    row_count = int(
        spine.select(pl.len())
        .collect(engine="streaming")
        .item()
    )

    null_profile = []

    for column, null_count in null_counts.items():
        null_count = int(null_count)

        null_profile.append(
            {
                "column": column,
                "null_count": null_count,
                "null_pct": (
                    100.0 * null_count / row_count
                    if row_count
                    else 0.0
                ),
            }
        )

    null_profile.sort(
        key=lambda x: x["null_count"],
        reverse=True,
    )

    log(
        f"distribution groups: {counts_df.height:,}; "
        f"columns profiled: {len(columns):,}"
    )

    return {
        "source_year_counts": counts_df.to_dicts(),
        "null_profile": null_profile,
    }


# ---------------------------------------------------------------------------
# Feature staleness
# ---------------------------------------------------------------------------


def audit_feature_staleness(
    spine: pl.LazyFrame,
) -> dict[str, Any]:
    log("AUDIT 7: selected feature staleness")

    age_days = (
        (
            pl.col("occurrence_timestamp_utc")
            - pl.col("feature_available_at")
        )
        .dt.total_seconds()
        / 86_400
    )

    result = collect_one(
        spine.select(
            age_days.min().alias("min_days"),
            age_days.quantile(0.50).alias("p50_days"),
            age_days.quantile(0.90).alias("p90_days"),
            age_days.quantile(0.95).alias("p95_days"),
            age_days.quantile(0.99).alias("p99_days"),
            age_days.max().alias("max_days"),
            (age_days > 365).sum().alias("rows_gt_1y"),
            (age_days > 730).sum().alias("rows_gt_2y"),
        )
    )

    log(
        "feature age: "
        f"p50={result['p50_days']:.2f}d "
        f"p95={result['p95_days']:.2f}d "
        f"p99={result['p99_days']:.2f}d"
    )

    return result


# ---------------------------------------------------------------------------
# Independent exact temporal as-of audit
# ---------------------------------------------------------------------------


def audit_exact_asof(
    spine: pl.LazyFrame,
    history_root: str,
    storage_options: dict[str, str],
) -> tuple[
    dict[str, Any],
    pl.DataFrame,
    pl.DataFrame,
    list[str],
]:
    log("AUDIT 8: independent exact temporal as-of reconstruction")

    started = time.perf_counter()

    events = (
        spine.select(
            "crime_id",
            "osm_h3_cell_id",
            "occurrence_timestamp_utc",
            pl.col("feature_available_at")
            .alias("spine_feature_available_at"),
            pl.col("feature_version_id")
            .alias("spine_feature_version_id"),
        )
        .collect(engine="streaming")
    )

    relevant_cells = (
        events.select("osm_h3_cell_id")
        .unique()
    )

    log(
        f"independent audit event rows={events.height:,}; "
        f"relevant H3 cells={relevant_cells.height:,}"
    )

    history_scan = scan_parquet_root(
        history_root,
        storage_options,
    )

    history_schema = history_scan.collect_schema()
    history_columns = history_schema.names()

    missing = sorted(
        HISTORY_REQUIRED_COLUMNS - set(history_columns)
    )

    if missing:
        raise AuditFailure(
            f"History missing required columns: {missing}"
        )

    skinny_columns = [
        "osm_h3_cell_id",
        "feature_available_at",
        "feature_version_id",
    ]

    skinny_columns += [
        c
        for c in COMPONENT_AVAILABILITY_COLUMNS
        if c in history_columns
    ]

    history = (
        history_scan
        .select(skinny_columns)
        .join(
            relevant_cells.lazy(),
            on="osm_h3_cell_id",
            how="semi",
        )
        .collect(engine="streaming")
    )

    log(
        f"filtered temporal history rows={history.height:,}; "
        f"H3 cells="
        f"{history['osm_h3_cell_id'].n_unique():,}"
    )

    duplicate_key_rows = (
        history.group_by(HISTORY_KEY)
        .len()
        .filter(pl.col("len") > 1)
        .select(pl.col("len").sum())
        .item()
    )

    duplicate_key_rows = int(duplicate_key_rows or 0)

    chronology_violations: dict[str, int] = {}

    for component in COMPONENT_AVAILABILITY_COLUMNS:
        if component not in history.columns:
            continue

        violations = (
            history.select(
                (
                    pl.col(component)
                    > pl.col("feature_available_at")
                )
                .fill_null(False)
                .sum()
            )
            .item()
        )

        chronology_violations[component] = int(violations)

    right = (
        history.select(
            "osm_h3_cell_id",
            pl.col("feature_available_at")
            .alias("expected_feature_available_at"),
            pl.col("feature_version_id")
            .alias("expected_feature_version_id"),
        )
        .sort("expected_feature_available_at")
    )

    left = events.sort("occurrence_timestamp_utc")

    reconstructed = left.join_asof(
        right,
        left_on="occurrence_timestamp_utc",
        right_on="expected_feature_available_at",
        by="osm_h3_cell_id",
        strategy="backward",
        allow_exact_matches=True,
    )

    comparison = reconstructed.select(
        pl.len().alias("rows"),
        pl.col("expected_feature_available_at")
        .null_count()
        .alias("history_missing_rows"),
        (
            ~pl.col("spine_feature_available_at")
            .eq_missing(
                pl.col("expected_feature_available_at")
            )
        )
        .sum()
        .alias("not_latest_legal_timestamp_rows"),
        (
            ~pl.col("spine_feature_version_id")
            .eq_missing(
                pl.col("expected_feature_version_id")
            )
        )
        .sum()
        .alias("feature_version_mismatch_rows"),
    ).row(0, named=True)

    selected_keys = (
        events.select(
            "osm_h3_cell_id",
            pl.col("spine_feature_available_at")
            .alias("feature_available_at"),
        )
        .unique()
    )

    comparison["unique_selected_history_keys"] = (
        selected_keys.height
    )

    comparison["filtered_history_rows"] = history.height
    comparison["filtered_history_h3_cells"] = (
        history["osm_h3_cell_id"].n_unique()
    )
    comparison["duplicate_history_key_rows"] = (
        duplicate_key_rows
    )
    comparison["component_chronology_violations"] = (
        chronology_violations
    )
    comparison["runtime_seconds"] = (
        time.perf_counter() - started
    )

    failures: list[str] = []

    fail_if(
        comparison["history_missing_rows"] != 0,
        f"{comparison['history_missing_rows']:,} published Gold "
        f"events no longer have a legal history match",
        failures,
    )

    fail_if(
        comparison["not_latest_legal_timestamp_rows"] != 0,
        f"{comparison['not_latest_legal_timestamp_rows']:,} events "
        f"did not select the latest legal feature timestamp",
        failures,
    )

    fail_if(
        comparison["feature_version_mismatch_rows"] != 0,
        f"{comparison['feature_version_mismatch_rows']:,} events "
        f"have the wrong feature_version_id",
        failures,
    )

    fail_if(
        duplicate_key_rows != 0,
        f"{duplicate_key_rows:,} duplicate relevant history-key rows",
        failures,
    )

    for component, violations in chronology_violations.items():
        fail_if(
            violations != 0,
            f"{component}: {violations:,} history chronology violations",
            failures,
        )

    if not failures:
        pass_log(
            "every Gold event independently selects the exact "
            "latest legal temporal history version"
        )
        pass_log("zero relevant history duplicate keys")

    return comparison, selected_keys, relevant_cells, failures


# ---------------------------------------------------------------------------
# Full feature payload equality audit
# ---------------------------------------------------------------------------


def audit_full_feature_payload(
    spine: pl.LazyFrame,
    spine_columns: list[str],
    history_root: str,
    storage_options: dict[str, str],
    selected_keys: pl.DataFrame,
    relevant_cells: pl.DataFrame,
) -> tuple[dict[str, Any], list[str]]:
    log("AUDIT 9: full feature payload equality against source history")

    started = time.perf_counter()

    history_scan = scan_parquet_root(
        history_root,
        storage_options,
    )

    history_columns = history_scan.collect_schema().names()

    shared_columns = [
        c
        for c in history_columns
        if c in spine_columns
        and c not in HISTORY_KEY
    ]

    if not shared_columns:
        raise AuditFailure(
            "No history payload columns are shared with the event spine"
        )

    log(
        f"comparing {len(shared_columns):,} history payload columns"
    )

    selected_history = (
        history_scan
        .join(
            relevant_cells.lazy(),
            on="osm_h3_cell_id",
            how="semi",
        )
        .join(
            selected_keys.lazy(),
            on=HISTORY_KEY,
            how="semi",
        )
        .select(
            *HISTORY_KEY,
            *[
                pl.col(c).alias(f"hist__{c}")
                for c in shared_columns
            ],
        )
        .collect(engine="streaming")
    )

    selected_history_unique = (
        selected_history.select(HISTORY_KEY).n_unique()
    )

    expected_keys = selected_keys.height

    log(
        f"full selected history rows={selected_history.height:,}; "
        f"expected keys={expected_keys:,}"
    )

    joined = (
        spine.select(
            *HISTORY_KEY,
            *shared_columns,
        )
        .join(
            selected_history.lazy(),
            on=HISTORY_KEY,
            how="left",
        )
    )

    mismatch_exprs = [
        (
            ~pl.col(column)
            .eq_missing(pl.col(f"hist__{column}"))
        )
        .sum()
        .alias(column)
        for column in shared_columns
    ]

    mismatches = collect_one(
        joined.select(mismatch_exprs)
    )

    nonzero_mismatches = {
        column: int(count)
        for column, count in mismatches.items()
        if int(count) != 0
    }

    result = {
        "expected_selected_keys": expected_keys,
        "full_history_rows_retrieved": selected_history.height,
        "full_history_unique_keys": selected_history_unique,
        "payload_columns_compared": len(shared_columns),
        "columns_compared": shared_columns,
        "nonzero_mismatch_columns": nonzero_mismatches,
        "runtime_seconds": time.perf_counter() - started,
    }

    failures: list[str] = []

    fail_if(
        selected_history.height != expected_keys,
        f"selected full history returned "
        f"{selected_history.height:,} rows for "
        f"{expected_keys:,} expected keys",
        failures,
    )

    fail_if(
        bool(nonzero_mismatches),
        "published feature payload differs from canonical temporal "
        f"history: {nonzero_mismatches}",
        failures,
    )

    if not failures:
        pass_log(
            f"all {len(shared_columns):,} shared history columns "
            f"match canonical history exactly"
        )

    return result, failures


# ---------------------------------------------------------------------------
# Current-reference regression check
# ---------------------------------------------------------------------------


def audit_current_reference(
    basic: dict[str, Any],
) -> dict[str, Any]:
    log("AUDIT 10: current production reference regression")

    actual_rows = int(basic["row_count"])
    expected_rows = CURRENT_REFERENCE["event_spine_rows"]

    matches = actual_rows == expected_rows

    if matches:
        pass_log(
            f"row count exactly matches audited reference: "
            f"{expected_rows:,}"
        )
    else:
        log(
            "WARN: row count differs from the current reference. "
            "This may be legitimate for a newer Silver snapshot."
        )

    return {
        "actual_event_spine_rows": actual_rows,
        "reference_event_spine_rows": expected_rows,
        "matches_current_reference": matches,
        "reference": CURRENT_REFERENCE,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Thorough independent audit of the CrimeNet Gold event spine."
        )
    )

    parser.add_argument(
        "--snapshot-uri",
        help=(
            "Exact event-spine snapshot URI. "
            "Default: resolve gold/event_spine/_latest."
        ),
    )

    parser.add_argument(
        "--event-spine-root",
        default=DEFAULT_EVENT_SPINE_ROOT,
    )

    parser.add_argument(
        "--history-root",
        default=DEFAULT_HISTORY_ROOT,
    )

    parser.add_argument(
        "--skip-deep-asof",
        action="store_true",
        help="Skip independent national-history as-of reconstruction.",
    )

    parser.add_argument(
        "--skip-payload",
        action="store_true",
        help="Skip full 40-column history payload equality audit.",
    )

    parser.add_argument(
        "--output-dir",
        default="artifacts/audits/event_spine",
    )

    args = parser.parse_args()

    started = time.perf_counter()

    config = b2_config()
    storage_options = polars_storage_options(config)
    client = s3_client(config)

    snapshot_uri = (
        args.snapshot_uri.rstrip("/")
        if args.snapshot_uri
        else resolve_latest_snapshot(
            client,
            args.event_spine_root,
        )
    )

    log("=" * 79)
    log("CrimeNet Gold Event Spine Audit")
    log(f"snapshot: {snapshot_uri}")
    log(f"history:  {args.history_root}")
    log("=" * 79)

    report: dict[str, Any] = {
        "audit_started_at": datetime.now(UTC).isoformat(),
        "snapshot_uri": snapshot_uri,
        "history_root": args.history_root,
    }

    all_failures: list[str] = []

    # Publication
    publication = audit_publication(
        client,
        snapshot_uri,
        args.event_spine_root,
    )
    report["publication"] = publication

    if not publication["_SUCCESS_exists"]:
        all_failures.append("snapshot missing _SUCCESS")

    if not publication["manifest_exists"]:
        all_failures.append("snapshot missing manifest.json")

    # Scan once as LazyFrame definition. Individual audits execute their
    # own projected streaming plans.
    spine = scan_parquet_root(
        snapshot_uri,
        storage_options,
    )

    # Basic
    basic, columns, failures = audit_basic_integrity(spine)
    report["basic_integrity"] = basic
    report["schema"] = {
        "column_count": len(columns),
        "columns": columns,
    }
    all_failures.extend(failures)

    # Compare manifest count when available.
    manifest = publication.get("manifest")

    if isinstance(manifest, dict) and manifest.get("row_count") is not None:
        manifest_count = int(manifest["row_count"])
        actual_count = int(basic["row_count"])

        report["publication"]["manifest_row_count_matches"] = (
            manifest_count == actual_count
        )

        fail_if(
            manifest_count != actual_count,
            f"manifest row_count={manifest_count:,}, "
            f"actual={actual_count:,}",
            all_failures,
        )

    # Temporal leakage
    temporal, failures = audit_temporal_integrity(
        spine,
        columns,
    )
    report["temporal_integrity"] = temporal
    all_failures.extend(failures)

    # Local -> UTC
    timezone_audit, failures = audit_timezone_reconstruction(
        spine,
        columns,
    )
    report["timezone_reconstruction"] = timezone_audit
    all_failures.extend(failures)

    # Coordinates / occurrence year
    sanity, failures = audit_event_sanity(
        spine,
        columns,
    )
    report["event_sanity"] = sanity
    all_failures.extend(failures)

    # Distributions
    report["distributions"] = audit_distributions(
        spine,
        columns,
    )

    # Staleness
    report["feature_staleness"] = audit_feature_staleness(
        spine,
    )

    # Reference
    report["current_reference"] = audit_current_reference(
        basic,
    )

    # Deep exact temporal audit
    if not args.skip_deep_asof:
        asof_result, selected_keys, relevant_cells, failures = (
            audit_exact_asof(
                spine,
                args.history_root,
                storage_options,
            )
        )

        report["independent_exact_asof"] = asof_result
        all_failures.extend(failures)

        if not args.skip_payload:
            payload_result, failures = audit_full_feature_payload(
                spine,
                columns,
                args.history_root,
                storage_options,
                selected_keys,
                relevant_cells,
            )

            report["full_feature_payload"] = payload_result
            all_failures.extend(failures)
    else:
        report["independent_exact_asof"] = {
            "skipped": True,
        }

    elapsed = time.perf_counter() - started

    report["runtime_seconds"] = elapsed
    report["audit_completed_at"] = datetime.now(UTC).isoformat()
    report["failure_count"] = len(all_failures)
    report["failures"] = all_failures
    report["verdict"] = "PASS" if not all_failures else "FAIL"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_id = snapshot_uri.rstrip("/").split("/")[-1]
    output_path = output_dir / f"{snapshot_id}_audit.json"

    output_path.write_text(
        json.dumps(
            to_jsonable(report),
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )

    print()
    print("=" * 79)
    print("FINAL EVENT SPINE AUDIT")
    print("=" * 79)
    print(f"Snapshot:              {snapshot_uri}")
    print(
        f"Rows:                  "
        f"{human_int(int(basic['row_count']))}"
    )
    print(
        f"Unique crime IDs:      "
        f"{human_int(int(basic['unique_crime_ids']))}"
    )
    print(
        f"Future feature leaks:  "
        f"{human_int(int(temporal['future_feature_leaks']))}"
    )

    if "independent_exact_asof" in report:
        exact = report["independent_exact_asof"]

        if not exact.get("skipped"):
            print(
                f"Wrong temporal version: "
                f"{human_int(int(exact['not_latest_legal_timestamp_rows']))}"
            )
            print(
                f"Version-ID mismatches: "
                f"{human_int(int(exact['feature_version_mismatch_rows']))}"
            )
            print(
                f"Relevant history rows: "
                f"{human_int(int(exact['filtered_history_rows']))}"
            )

    if "full_feature_payload" in report:
        payload = report["full_feature_payload"]

        print(
            f"Payload columns checked: "
            f"{payload['payload_columns_compared']}"
        )
        print(
            f"Payload mismatches:      "
            f"{len(payload['nonzero_mismatch_columns'])}"
        )

    print(f"Failures:              {len(all_failures)}")
    print(f"Runtime:               {elapsed:.2f}s")
    print(f"Report:                {output_path}")
    print()
    print(
        "HARD EVENT SPINE QUALITY GATE: "
        + ("PASS" if not all_failures else "FAIL")
    )
    print("=" * 79)

    if all_failures:
        print()
        print("Failures:")
        for failure in all_failures:
            print(f"  - {failure}")

        return 1

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nAudit interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(
            f"\nAUDIT CRASHED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise