#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
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

LATEST_POINTER_CANDIDATES = ("_latest.json", "latest.json", "_latest")
MANIFEST_CANDIDATES = ("manifest.json", "_manifest.json")

REQUIRED_SPINE_COLUMNS = {
    "crime_id",
    "source_city",
    "occurrence_timestamp_utc",
    "osm_h3_cell_id",
    "feature_available_at",
    "feature_version_id",
}

HISTORY_KEY = ["osm_h3_cell_id", "feature_available_at"]

STATE_COLUMN_CANDIDATES = (
    "state_fips",
    "source_state_fips",
    "statefp",
    "state_code",
    "state_abbr",
    "state_name",
    "state",
)

TAXONOMY_CANDIDATES = (
    "canonical_crime_family",
    "canonical_crime_type",
    "canonical_crime_subtype",
    "crime_family",
    "crime_type",
    "crime_subtype",
    "offense_family",
    "offense_type",
    "offense_subtype",
    "canonical_offense_family",
    "canonical_offense_type",
    "canonical_offense_subtype",
)

LOW_CARDINALITY_CANDIDATES = (
    "source_city",
    "source_timezone",
    "lighting_condition",
    "lighting",
    "weather_condition",
    "weather",
    "acs_vintage",
    "osm_snapshot_year",
    "tiger_line_year",
    "h3_resolution",
)

COMPONENT_AVAILABILITY_COLUMNS = (
    "osm_available_at",
    "acs_release_date",
    "tiger_release_date",
)

FEATURE_META_COLUMNS = {
    "osm_h3_cell_id",
    "osm_h3_cell_id_hex",
    "h3_resolution",
    "feature_available_at",
    "feature_version_id",
}

INTEGER_PREFIXES = ("Int", "UInt")
FLOAT_PREFIXES = ("Float", "Decimal")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def log(message: str) -> None:
    print(
        f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {message}",
        flush=True,
    )


def pct(num: int | float, den: int | float) -> float | None:
    if not den:
        return None
    return 100.0 * float(num) / float(den)


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True, default=str) + "\n"
    )


def dataframe_rows(df: pl.DataFrame) -> list[dict[str, Any]]:
    return [jsonable(row) for row in df.to_dicts()]


def md_escape(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ").replace("|", r"\|")
    return text


def md_table(
    df: pl.DataFrame,
    *,
    max_rows: int | None = None,
    columns: list[str] | None = None,
) -> str:
    if df.height == 0:
        return "_No rows._"

    if columns is None:
        columns = df.columns

    data = df.select(columns)
    truncated = False

    if max_rows is not None and data.height > max_rows:
        data = data.head(max_rows)
        truncated = True

    lines = [
        "| " + " | ".join(md_escape(c) for c in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]

    for row in data.iter_rows(named=True):
        lines.append(
            "| "
            + " | ".join(md_escape(row[c]) for c in columns)
            + " |"
        )

    if truncated:
        lines.append("")
        lines.append(
            f"_Showing first {max_rows:,} of {df.height:,} rows; "
            "see the CSV artifact for the full table._"
        )

    return "\n".join(lines)


def fmt_int(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{int(value):,}"


def fmt_pct(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}%"


def fmt_gib(num_bytes: int | float | None) -> str:
    if num_bytes is None:
        return "N/A"
    return f"{float(num_bytes) / (1024 ** 3):,.3f} GiB"


def find_recursive(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = find_recursive(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_recursive(value, key)
            if found is not None:
                return found
    return None


def first_present(columns: list[str] | set[str], candidates: tuple[str, ...]) -> str | None:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def dtype_string(dtype: Any) -> str:
    return str(dtype)


def is_numeric_dtype(dtype: Any) -> bool:
    name = dtype_string(dtype)
    return name.startswith(INTEGER_PREFIXES + FLOAT_PREFIXES)


def is_float_dtype(dtype: Any) -> bool:
    return dtype_string(dtype).startswith(FLOAT_PREFIXES)


def is_datetime_dtype(dtype: Any) -> bool:
    return dtype_string(dtype).startswith(("Datetime", "Date"))


def collect_one(lf: pl.LazyFrame) -> dict[str, Any]:
    df = lf.collect(engine="streaming")
    if df.height != 1:
        raise RuntimeError(f"Expected one row, got {df.height}")
    return df.row(0, named=True)


# ---------------------------------------------------------------------------
# B2 / S3 helpers
# ---------------------------------------------------------------------------


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


def storage_config() -> dict[str, str]:
    endpoint_url = require_env("B2_ENDPOINT_URL")
    return {
        "key_id": require_env("B2_KEY_ID"),
        "application_key": require_env("B2_APPLICATION_KEY"),
        "endpoint_url": endpoint_url,
        "region": derive_region(endpoint_url),
    }


def polars_storage_options(config: dict[str, str]) -> dict[str, str]:
    return {
        "aws_access_key_id": config["key_id"],
        "aws_secret_access_key": config["application_key"],
        "aws_endpoint_url": config["endpoint_url"],
        "aws_region": config["region"],
    }


def make_s3_client(config: dict[str, str]):
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
        raise ValueError(f"Expected s3:// URI, got {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def read_text_object(client: Any, uri: str) -> str:
    bucket, key = split_s3_uri(uri)
    response = client.get_object(Bucket=bucket, Key=key)
    return response["Body"].read().decode("utf-8").strip()


def object_exists(client: Any, uri: str) -> bool:
    bucket, key = split_s3_uri(uri)
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def resolve_latest_snapshot(client: Any, root: str) -> tuple[str, str]:
    root = root.rstrip("/")

    for filename in LATEST_POINTER_CANDIDATES:
        pointer_uri = f"{root}/{filename}"
        if not object_exists(client, pointer_uri):
            continue

        raw = read_text_object(client, pointer_uri)

        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw

        if isinstance(value, str):
            if value.startswith("s3://"):
                return value.rstrip("/"), pointer_uri
            if value.startswith("snapshot_id="):
                return f"{root}/{value}", pointer_uri
            return f"{root}/snapshot_id={value}", pointer_uri

        if isinstance(value, dict):
            for key in ("snapshot_uri", "uri"):
                if value.get(key):
                    return str(value[key]).rstrip("/"), pointer_uri

            for key in ("snapshot_id", "id"):
                if value.get(key):
                    return f"{root}/snapshot_id={value[key]}", pointer_uri

        raise RuntimeError(
            f"Could not parse latest pointer {pointer_uri}: {raw!r}"
        )

    raise RuntimeError(
        f"No latest event-spine pointer found under {root}; checked "
        + ", ".join(LATEST_POINTER_CANDIDATES)
    )


def read_manifest(client: Any, snapshot_uri: str) -> tuple[dict[str, Any] | None, str | None]:
    for name in MANIFEST_CANDIDATES:
        uri = f"{snapshot_uri.rstrip('/')}/{name}"
        if object_exists(client, uri):
            return json.loads(read_text_object(client, uri)), uri
    return None, None


def snapshot_storage_metrics(client: Any, snapshot_uri: str) -> dict[str, Any]:
    bucket, prefix = split_s3_uri(snapshot_uri.rstrip("/") + "/")

    paginator = client.get_paginator("list_objects_v2")

    object_count = 0
    total_bytes = 0
    parquet_count = 0
    parquet_bytes = 0
    parquet_sizes: list[int] = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            size = int(obj.get("Size", 0))
            key = obj["Key"]
            object_count += 1
            total_bytes += size

            if key.endswith(".parquet"):
                parquet_count += 1
                parquet_bytes += size
                parquet_sizes.append(size)

    return {
        "object_count": object_count,
        "total_bytes": total_bytes,
        "parquet_file_count": parquet_count,
        "parquet_bytes": parquet_bytes,
        "avg_parquet_bytes": (
            parquet_bytes / parquet_count if parquet_count else None
        ),
        "min_parquet_bytes": min(parquet_sizes) if parquet_sizes else None,
        "max_parquet_bytes": max(parquet_sizes) if parquet_sizes else None,
    }


def latest_history_partition_uri(client: Any, history_root: str) -> str | None:
    bucket, prefix = split_s3_uri(history_root.rstrip("/") + "/")
    paginator = client.get_paginator("list_objects_v2")

    date_prefixes: list[tuple[str, str]] = []

    for page in paginator.paginate(
        Bucket=bucket,
        Prefix=prefix,
        Delimiter="/",
    ):
        for item in page.get("CommonPrefixes", []):
            full_prefix = item["Prefix"]
            tail = full_prefix[len(prefix):].rstrip("/")
            match = re.fullmatch(r"feature_available_date=(\d{4}-\d{2}-\d{2})", tail)
            if match:
                date_prefixes.append((match.group(1), full_prefix.rstrip("/")))

    if not date_prefixes:
        return None

    _, latest_prefix = max(date_prefixes, key=lambda item: item[0])
    return f"s3://{bucket}/{latest_prefix}"


def scan_parquet_root(
    uri: str,
    storage_options: dict[str, str],
) -> pl.LazyFrame:
    return pl.scan_parquet(
        f"{uri.rstrip('/')}/**/*.parquet",
        storage_options=storage_options,
        hive_partitioning=True,
    )


# ---------------------------------------------------------------------------
# Schema / global profile
# ---------------------------------------------------------------------------


def profile_schema(
    spine: pl.LazyFrame,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    schema = spine.collect_schema()

    df = pl.DataFrame(
        {
            "column": schema.names(),
            "dtype": [dtype_string(schema[c]) for c in schema.names()],
        }
    )

    return df, {
        "column_count": len(schema),
        "columns": schema.names(),
        "dtypes": {
            c: dtype_string(schema[c])
            for c in schema.names()
        },
    }


def global_dataset_metrics(
    spine: pl.LazyFrame,
    columns: list[str],
) -> dict[str, Any]:
    missing_required = sorted(REQUIRED_SPINE_COLUMNS - set(columns))
    if missing_required:
        raise RuntimeError(
            f"Missing required event-spine columns: {missing_required}"
        )

    exprs: list[pl.Expr] = [
        pl.len().alias("row_count"),
        pl.col("crime_id").n_unique().alias("unique_crime_ids"),
        pl.col("source_city").n_unique().alias("source_city_count"),
        pl.col("osm_h3_cell_id").n_unique().alias("unique_h3_cells"),
        pl.col("feature_version_id").n_unique().alias("feature_versions_used"),
        pl.col("occurrence_timestamp_utc").min().alias("min_occurrence_timestamp_utc"),
        pl.col("occurrence_timestamp_utc").max().alias("max_occurrence_timestamp_utc"),
        pl.col("feature_available_at").min().alias("min_feature_available_at"),
        pl.col("feature_available_at").max().alias("max_feature_available_at"),
        (
            pl.col("feature_available_at") > pl.col("occurrence_timestamp_utc")
        )
        .fill_null(False)
        .sum()
        .alias("future_feature_leak_rows"),
    ]

    for column in REQUIRED_SPINE_COLUMNS:
        exprs.append(
            pl.col(column).null_count().alias(f"null__{column}")
        )

    if "occurrence_year" in columns:
        exprs.append(
            pl.col("occurrence_year").n_unique().alias("occurrence_year_count")
        )

    if "source_timezone" in columns:
        exprs.append(
            pl.col("source_timezone").n_unique().alias("source_timezone_count")
        )

    if {"latitude", "longitude"}.issubset(columns):
        exprs.extend(
            [
                pl.col("latitude").min().alias("min_latitude"),
                pl.col("latitude").max().alias("max_latitude"),
                pl.col("longitude").min().alias("min_longitude"),
                pl.col("longitude").max().alias("max_longitude"),
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

    for component in COMPONENT_AVAILABILITY_COLUMNS:
        if component in columns:
            exprs.append(
                (
                    pl.col(component) > pl.col("feature_available_at")
                )
                .fill_null(False)
                .sum()
                .alias(f"future_component__{component}")
            )

    result = collect_one(spine.select(exprs))
    result["duplicate_crime_ids"] = (
        int(result["row_count"]) - int(result["unique_crime_ids"])
    )
    return jsonable(result)


def null_coverage_profile(
    spine: pl.LazyFrame,
    columns: list[str],
) -> pl.DataFrame:
    exprs = [pl.len().alias("__rows")]
    exprs.extend(
        pl.col(c).null_count().alias(c)
        for c in columns
    )

    row = collect_one(spine.select(exprs))
    total_rows = int(row.pop("__rows"))

    rows = []
    for column in columns:
        null_count = int(row[column])
        nonnull = total_rows - null_count
        rows.append(
            {
                "column": column,
                "null_count": null_count,
                "non_null_count": nonnull,
                "coverage_pct": pct(nonnull, total_rows),
            }
        )

    return pl.DataFrame(rows).sort(
        ["coverage_pct", "column"],
        descending=[False, False],
    )


def numeric_profile(
    spine: pl.LazyFrame,
    schema_info: dict[str, Any],
    *,
    deep: bool,
) -> pl.DataFrame:
    numeric_columns = [
        c
        for c, dtype in schema_info["dtypes"].items()
        if is_numeric_dtype(dtype)
        and c not in {"osm_h3_cell_id"}
    ]

    if not numeric_columns:
        return pl.DataFrame()

    expressions: list[pl.Expr] = []

    for column in numeric_columns:
        expressions.extend(
            [
                pl.col(column).count().alias(f"{column}__count"),
                pl.col(column).min().alias(f"{column}__min"),
                pl.col(column).max().alias(f"{column}__max"),
                pl.col(column).mean().alias(f"{column}__mean"),
                pl.col(column).std().alias(f"{column}__std"),
            ]
        )

        if deep:
            expressions.extend(
                [
                    pl.col(column).quantile(0.01).alias(f"{column}__p01"),
                    pl.col(column).median().alias(f"{column}__p50"),
                    pl.col(column).quantile(0.99).alias(f"{column}__p99"),
                    pl.col(column).n_unique().alias(f"{column}__n_unique"),
                ]
            )

    row = collect_one(spine.select(expressions))
    result_rows: list[dict[str, Any]] = []

    for column in numeric_columns:
        entry = {
            "column": column,
            "dtype": schema_info["dtypes"][column],
            "count": row.get(f"{column}__count"),
            "min": row.get(f"{column}__min"),
            "max": row.get(f"{column}__max"),
            "mean": row.get(f"{column}__mean"),
            "std": row.get(f"{column}__std"),
        }

        if deep:
            entry.update(
                {
                    "p01": row.get(f"{column}__p01"),
                    "p50": row.get(f"{column}__p50"),
                    "p99": row.get(f"{column}__p99"),
                    "n_unique": row.get(f"{column}__n_unique"),
                }
            )

        result_rows.append(jsonable(entry))

    return pl.DataFrame(result_rows)


# ---------------------------------------------------------------------------
# Feature coverage
# ---------------------------------------------------------------------------


def feature_family(column: str) -> str:
    if column.startswith("osm_"):
        return "osm"
    if column.startswith(("socio_", "acs_")) or column in {
        "tract_geoid",
        "state_fips",
        "county_fips",
    }:
        return "socioeconomic"
    if column.startswith("tiger_"):
        return "tiger"
    if column.startswith(("weather_", "meteo_")):
        return "weather"
    if column.startswith(("light_", "lighting_", "solar_")):
        return "lighting"
    return "national_feature_store"


def identify_history_feature_columns(
    spine_columns: list[str],
    history_columns: list[str],
) -> list[str]:
    return [
        c
        for c in history_columns
        if c in spine_columns
        and c not in FEATURE_META_COLUMNS
        and c not in HISTORY_KEY
    ]


def feature_column_coverage(
    spine: pl.LazyFrame,
    feature_columns: list[str],
) -> pl.DataFrame:
    if not feature_columns:
        return pl.DataFrame()

    exprs = [pl.len().alias("__rows")]
    exprs.extend(
        pl.col(c).count().alias(c)
        for c in feature_columns
    )

    row = collect_one(spine.select(exprs))
    total = int(row.pop("__rows"))

    rows = [
        {
            "family": feature_family(column),
            "column": column,
            "non_null_count": int(row[column]),
            "null_count": total - int(row[column]),
            "coverage_pct": pct(int(row[column]), total),
        }
        for column in feature_columns
    ]

    return pl.DataFrame(rows).sort(
        ["family", "coverage_pct", "column"],
        descending=[False, False, False],
    )


def feature_family_coverage(
    spine: pl.LazyFrame,
    feature_columns: list[str],
    group_columns: list[str] | None = None,
) -> pl.DataFrame:
    if not feature_columns:
        return pl.DataFrame()

    families: dict[str, list[str]] = defaultdict(list)
    for column in feature_columns:
        families[feature_family(column)].append(column)

    expressions: list[pl.Expr] = [pl.len().alias("rows")]

    for family, columns in sorted(families.items()):
        nonnull_exprs = [pl.col(c).is_not_null() for c in columns]

        expressions.extend(
            [
                pl.all_horizontal(*nonnull_exprs)
                .sum()
                .alias(f"{family}__complete_rows"),
                pl.any_horizontal(*nonnull_exprs)
                .sum()
                .alias(f"{family}__any_rows"),
            ]
        )

    if group_columns:
        df = (
            spine.group_by(group_columns)
            .agg(expressions)
            .collect(engine="streaming")
        )
    else:
        df = spine.select(expressions).collect(engine="streaming")

    for family in sorted(families):
        df = df.with_columns(
            (
                100
                * pl.col(f"{family}__complete_rows")
                / pl.col("rows")
            ).alias(f"{family}__complete_pct"),
            (
                100
                * pl.col(f"{family}__any_rows")
                / pl.col("rows")
            ).alias(f"{family}__any_pct"),
        )

    return df


# ---------------------------------------------------------------------------
# Temporal / geographic summaries
# ---------------------------------------------------------------------------


def feature_age_days_expr() -> pl.Expr:
    return (
        (
            pl.col("occurrence_timestamp_utc")
            - pl.col("feature_available_at")
        )
        .dt.total_seconds()
        / 86_400
    )


def component_age_days_expr(column: str) -> pl.Expr:
    return (
        (
            pl.col("occurrence_timestamp_utc")
            - pl.col(column)
        )
        .dt.total_seconds()
        / 86_400
    )


def temporal_freshness(
    spine: pl.LazyFrame,
    columns: list[str],
) -> dict[str, Any]:
    age = feature_age_days_expr()

    expressions: list[pl.Expr] = [
        age.min().alias("feature_age_min_days"),
        age.quantile(0.50).alias("feature_age_p50_days"),
        age.quantile(0.90).alias("feature_age_p90_days"),
        age.quantile(0.95).alias("feature_age_p95_days"),
        age.quantile(0.99).alias("feature_age_p99_days"),
        age.max().alias("feature_age_max_days"),
        (age > 365).sum().alias("feature_age_rows_gt_1y"),
        (age > 730).sum().alias("feature_age_rows_gt_2y"),
        (age < 0).sum().alias("negative_feature_age_rows"),
    ]

    for component in COMPONENT_AVAILABILITY_COLUMNS:
        if component not in columns:
            continue

        component_age = component_age_days_expr(component)
        expressions.extend(
            [
                component_age.quantile(0.50).alias(f"{component}__age_p50_days"),
                component_age.quantile(0.95).alias(f"{component}__age_p95_days"),
                component_age.quantile(0.99).alias(f"{component}__age_p99_days"),
                component_age.max().alias(f"{component}__age_max_days"),
                (component_age < 0).sum().alias(f"{component}__negative_age_rows"),
            ]
        )

    return jsonable(collect_one(spine.select(expressions)))


def city_summary(
    spine: pl.LazyFrame,
    columns: list[str],
) -> pl.DataFrame:
    age = feature_age_days_expr()

    expressions: list[pl.Expr] = [
        pl.len().alias("rows"),
        pl.col("crime_id").n_unique().alias("unique_crime_ids"),
        pl.col("osm_h3_cell_id").n_unique().alias("unique_h3_cells"),
        pl.col("occurrence_timestamp_utc").min().alias("min_event_utc"),
        pl.col("occurrence_timestamp_utc").max().alias("max_event_utc"),
        pl.col("occurrence_timestamp_utc").dt.date().n_unique().alias("active_days"),
        pl.col("occurrence_timestamp_utc").dt.year().n_unique().alias("years_present"),
        pl.col("feature_version_id").n_unique().alias("feature_versions_used"),
        age.quantile(0.50).alias("feature_age_p50_days"),
        age.quantile(0.95).alias("feature_age_p95_days"),
        age.quantile(0.99).alias("feature_age_p99_days"),
    ]

    if "source_timezone" in columns:
        expressions.append(
            pl.col("source_timezone").n_unique().alias("timezones")
        )

    if {"latitude", "longitude"}.issubset(columns):
        expressions.extend(
            [
                pl.col("latitude").min().alias("min_latitude"),
                pl.col("latitude").max().alias("max_latitude"),
                pl.col("longitude").min().alias("min_longitude"),
                pl.col("longitude").max().alias("max_longitude"),
            ]
        )

    df = (
        spine.group_by("source_city")
        .agg(expressions)
        .sort("source_city")
        .collect(engine="streaming")
    )

    return df.with_columns(
        (
            (
                pl.col("max_event_utc") - pl.col("min_event_utc")
            )
            .dt.total_days()
            + 1
        ).alias("calendar_span_days"),
        (
            pl.col("rows") / pl.col("active_days")
        ).alias("events_per_active_day"),
        (
            pl.col("rows") / pl.col("unique_h3_cells")
        ).alias("events_per_h3_cell"),
    ).with_columns(
        (
            100
            * pl.col("active_days")
            / pl.col("calendar_span_days")
        ).alias("active_day_pct")
    )


def city_year_summary(spine: pl.LazyFrame) -> pl.DataFrame:
    return (
        spine.with_columns(
            pl.col("occurrence_timestamp_utc")
            .dt.year()
            .alias("__year")
        )
        .group_by(["source_city", "__year"])
        .agg(
            pl.len().alias("rows"),
            pl.col("crime_id").n_unique().alias("unique_crime_ids"),
            pl.col("osm_h3_cell_id").n_unique().alias("unique_h3_cells"),
            pl.col("occurrence_timestamp_utc").dt.date().n_unique().alias("active_days"),
            pl.col("occurrence_timestamp_utc").min().alias("min_event_utc"),
            pl.col("occurrence_timestamp_utc").max().alias("max_event_utc"),
        )
        .sort(["source_city", "__year"])
        .collect(engine="streaming")
        .rename({"__year": "occurrence_year"})
    )


def monthly_summary(spine: pl.LazyFrame) -> pl.DataFrame:
    return (
        spine.with_columns(
            pl.col("occurrence_timestamp_utc")
            .dt.strftime("%Y-%m")
            .alias("__month")
        )
        .group_by(["source_city", "__month"])
        .agg(
            pl.len().alias("rows"),
            pl.col("osm_h3_cell_id").n_unique().alias("unique_h3_cells"),
            pl.col("occurrence_timestamp_utc").dt.date().n_unique().alias("active_days"),
        )
        .sort(["source_city", "__month"])
        .collect(engine="streaming")
        .rename({"__month": "month"})
    )


def h3_density_summary(spine: pl.LazyFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    cell_counts = (
        spine.group_by(["source_city", "osm_h3_cell_id"])
        .agg(pl.len().alias("events"))
        .collect(engine="streaming")
    )

    def summarize(df: pl.DataFrame, label: str) -> dict[str, Any]:
        if df.height == 0:
            return {"source_city": label}

        values = df["events"]
        total_events = int(values.sum())
        n_cells = df.height
        sorted_values = values.sort(descending=True)

        def top_share(frac: float) -> float:
            n = max(1, math.ceil(n_cells * frac))
            return 100.0 * int(sorted_values.head(n).sum()) / total_events

        return {
            "source_city": label,
            "unique_h3_cells": n_cells,
            "events": total_events,
            "events_per_h3_mean": float(values.mean()),
            "events_per_h3_p50": float(values.quantile(0.50, interpolation="nearest")),
            "events_per_h3_p90": float(values.quantile(0.90, interpolation="nearest")),
            "events_per_h3_p95": float(values.quantile(0.95, interpolation="nearest")),
            "events_per_h3_p99": float(values.quantile(0.99, interpolation="nearest")),
            "events_per_h3_max": int(values.max()),
            "top_1pct_h3_event_share_pct": top_share(0.01),
            "top_5pct_h3_event_share_pct": top_share(0.05),
            "top_10pct_h3_event_share_pct": top_share(0.10),
        }

    rows = []

    overall = (
        cell_counts.group_by("osm_h3_cell_id")
        .agg(pl.col("events").sum().alias("events"))
    )
    rows.append(summarize(overall, "__ALL__"))

    for city in sorted(cell_counts["source_city"].unique().to_list()):
        rows.append(
            summarize(
                cell_counts.filter(pl.col("source_city") == city)
                .select("osm_h3_cell_id", "events"),
                str(city),
            )
        )

    return pl.DataFrame(rows), cell_counts


def feature_version_summary(spine: pl.LazyFrame) -> pl.DataFrame:
    age = feature_age_days_expr()

    return (
        spine.group_by(["feature_version_id", "feature_available_at"])
        .agg(
            pl.len().alias("event_rows"),
            pl.col("osm_h3_cell_id").n_unique().alias("unique_h3_cells"),
            pl.col("occurrence_timestamp_utc").min().alias("min_event_utc"),
            pl.col("occurrence_timestamp_utc").max().alias("max_event_utc"),
            age.quantile(0.50).alias("feature_age_p50_days"),
            age.quantile(0.95).alias("feature_age_p95_days"),
        )
        .sort("feature_available_at")
        .collect(engine="streaming")
    )


# ---------------------------------------------------------------------------
# State coverage
# ---------------------------------------------------------------------------


def build_state_context(
    spine: pl.LazyFrame,
    spine_columns: list[str],
    client: Any,
    history_root: str,
    storage_options: dict[str, str],
) -> tuple[
    pl.LazyFrame,
    str | None,
    pl.DataFrame,
    dict[str, Any],
]:
    """
    Returns:
      state-enriched spine lazy frame,
      state column name,
      statewide H3 coverage table,
      metadata.
    """

    latest_partition = latest_history_partition_uri(client, history_root)
    if latest_partition is None:
        return spine, None, pl.DataFrame(), {
            "available": False,
            "reason": "Could not discover feature_available_date partitions.",
        }

    history_latest = scan_parquet_root(
        latest_partition,
        storage_options,
    )
    history_schema = history_latest.collect_schema()
    history_columns = history_schema.names()

    history_state_col = first_present(
        history_columns,
        STATE_COLUMN_CANDIDATES,
    )

    if history_state_col is None:
        return spine, None, pl.DataFrame(), {
            "available": False,
            "latest_history_partition": latest_partition,
            "reason": "No state column found in latest national history partition.",
        }

    spine_state_col = first_present(
        spine_columns,
        STATE_COLUMN_CANDIDATES,
    )

    event_cells = (
        spine.select("osm_h3_cell_id")
        .unique()
        .collect(engine="streaming")
    )

    # Denominator: H3 support in the latest national feature-store snapshot.
    state_denominator = (
        history_latest
        .select(
            pl.col(history_state_col).alias("__state"),
            "osm_h3_cell_id",
        )
        .drop_nulls()
        .group_by("__state")
        .agg(
            pl.col("osm_h3_cell_id")
            .n_unique()
            .alias("latest_feature_store_h3_cells")
        )
        .collect(engine="streaming")
    )

    # Relevant event-cell -> state mapping from the same latest snapshot.
    relevant_state_map = (
        history_latest
        .select(
            pl.col(history_state_col).alias("__state"),
            "osm_h3_cell_id",
        )
        .drop_nulls()
        .join(
            event_cells.lazy(),
            on="osm_h3_cell_id",
            how="semi",
        )
        .unique()
        .collect(engine="streaming")
    )

    mapping_ambiguity = (
        relevant_state_map
        .group_by("osm_h3_cell_id")
        .agg(pl.col("__state").n_unique().alias("state_count"))
        .filter(pl.col("state_count") > 1)
        .height
    )

    # Keep a deterministic one-state mapping only after recording ambiguity.
    relevant_state_map = (
        relevant_state_map
        .sort(["osm_h3_cell_id", "__state"])
        .unique(subset=["osm_h3_cell_id"], keep="first")
    )

    if spine_state_col is not None:
        state_spine = spine.with_columns(
            pl.col(spine_state_col).alias("__state")
        )
        state_source = f"event_spine.{spine_state_col}"
    else:
        state_spine = spine.join(
            relevant_state_map.lazy(),
            on="osm_h3_cell_id",
            how="left",
        )
        state_source = (
            f"latest_history_partition.{history_state_col} "
            "mapped by osm_h3_cell_id"
        )

    state_numerator = (
        state_spine
        .group_by("__state")
        .agg(
            pl.len().alias("event_rows"),
            pl.col("crime_id").n_unique().alias("unique_crime_ids"),
            pl.col("osm_h3_cell_id").n_unique().alias("event_h3_cells"),
            pl.col("source_city").n_unique().alias("source_cities"),
            pl.col("occurrence_timestamp_utc").min().alias("min_event_utc"),
            pl.col("occurrence_timestamp_utc").max().alias("max_event_utc"),
        )
        .collect(engine="streaming")
    )

    coverage = (
        state_numerator
        .join(
            state_denominator,
            on="__state",
            how="left",
        )
        .with_columns(
            (
                100
                * pl.col("event_h3_cells")
                / pl.col("latest_feature_store_h3_cells")
            ).alias("state_h3_footprint_coverage_pct")
        )
        .sort("__state")
        .rename({"__state": "state"})
    )

    state_spine = state_spine

    metadata = {
        "available": True,
        "latest_history_partition": latest_partition,
        "history_state_column": history_state_col,
        "state_source": state_source,
        "event_h3_cells_mapped": relevant_state_map.height,
        "ambiguous_event_h3_state_mappings": mapping_ambiguity,
        "definition": (
            "state_h3_footprint_coverage_pct = distinct event-spine H3-r9 "
            "cells in the state / distinct H3-r9 cells supported by the "
            "latest national feature-store partition in that state. "
            "This measures modeled geographic footprint, not completeness "
            "of statewide crime reporting."
        ),
    }

    return state_spine, "__state", coverage, metadata


def state_summary(
    state_spine: pl.LazyFrame,
    state_column: str,
) -> pl.DataFrame:
    age = feature_age_days_expr()

    return (
        state_spine
        .filter(pl.col(state_column).is_not_null())
        .group_by(state_column)
        .agg(
            pl.len().alias("rows"),
            pl.col("crime_id").n_unique().alias("unique_crime_ids"),
            pl.col("source_city").n_unique().alias("source_cities"),
            pl.col("osm_h3_cell_id").n_unique().alias("unique_h3_cells"),
            pl.col("occurrence_timestamp_utc").min().alias("min_event_utc"),
            pl.col("occurrence_timestamp_utc").max().alias("max_event_utc"),
            pl.col("occurrence_timestamp_utc").dt.date().n_unique().alias("active_days"),
            pl.col("feature_version_id").n_unique().alias("feature_versions_used"),
            age.quantile(0.50).alias("feature_age_p50_days"),
            age.quantile(0.95).alias("feature_age_p95_days"),
            age.quantile(0.99).alias("feature_age_p99_days"),
        )
        .sort(state_column)
        .collect(engine="streaming")
        .rename({state_column: "state"})
    )


# ---------------------------------------------------------------------------
# Categorical distributions
# ---------------------------------------------------------------------------


def categorical_columns(columns: list[str]) -> list[str]:
    found: list[str] = []

    for candidate in LOW_CARDINALITY_CANDIDATES + TAXONOMY_CANDIDATES:
        if candidate in columns and candidate not in found:
            found.append(candidate)

    # Catch likely canonical taxonomy naming variants without profiling
    # high-cardinality source identifiers/URIs.
    for column in columns:
        lower = column.lower()
        if column in found:
            continue

        if (
            any(token in lower for token in ("family", "subtype"))
            and not any(
                skip in lower
                for skip in ("source_file", "uri", "id")
            )
        ):
            found.append(column)

    return found


def categorical_distribution(
    spine: pl.LazyFrame,
    column: str,
) -> pl.DataFrame:
    df = (
        spine.group_by(column)
        .agg(pl.len().alias("rows"))
        .sort("rows", descending=True)
        .collect(engine="streaming")
    )

    total = int(df["rows"].sum()) if df.height else 0

    return df.with_columns(
        (
            100 * pl.col("rows") / total
        ).alias("pct")
        if total
        else pl.lit(None).cast(pl.Float64).alias("pct")
    )


# ---------------------------------------------------------------------------
# Manifest/build coverage extraction
# ---------------------------------------------------------------------------


def extract_build_coverage(
    manifest: dict[str, Any] | None,
    event_spine_rows: int,
) -> dict[str, Any]:
    if not manifest:
        return {
            "available": False,
            "event_spine_rows": event_spine_rows,
        }

    keys = (
        "input_modeled_rows",
        "expected_modeled_rows",
        "joinable_rows",
        "joinable_event_rows",
        "invalid_event_utc_rows",
        "null_h3_rows",
        "unjoinable_rows",
        "history_unmatched_rows",
        "no_legal_history_match_rows",
        "selected_temporal_match_rows",
        "output_rows",
        "dropped_rows",
        "coverage_pct",
        "joinable_coverage_pct",
        "unique_relevant_h3_cells",
        "unique_selected_history_keys",
        "filtered_skinny_history_rows",
        "filtered_history_h3_cells",
        "full_feature_rows_retrieved",
    )

    result = {"available": True}

    for key in keys:
        value = find_recursive(manifest, key)
        if value is not None:
            result[key] = value

    modeled = (
        result.get("input_modeled_rows")
        or result.get("expected_modeled_rows")
    )

    if modeled:
        result["published_vs_modeled_pct"] = pct(
            event_spine_rows,
            modeled,
        )

    joinable = (
        result.get("joinable_rows")
        or result.get("joinable_event_rows")
    )

    if joinable:
        result["published_vs_joinable_pct"] = pct(
            event_spine_rows,
            joinable,
        )

    result["event_spine_rows"] = event_spine_rows
    return jsonable(result)


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def write_report(
    path: Path,
    *,
    snapshot_uri: str,
    pointer_uri: str | None,
    history_root: str,
    manifest_uri: str | None,
    storage: dict[str, Any],
    schema_info: dict[str, Any],
    global_metrics: dict[str, Any],
    build_coverage: dict[str, Any],
    temporal: dict[str, Any],
    state_meta: dict[str, Any],
    state_coverage_df: pl.DataFrame,
    city_df: pl.DataFrame,
    state_df: pl.DataFrame,
    feature_coverage_df: pl.DataFrame,
    feature_family_df: pl.DataFrame,
    h3_density_df: pl.DataFrame,
    version_df: pl.DataFrame,
    null_df: pl.DataFrame,
    categorical_tables: dict[str, pl.DataFrame],
    output_files: list[str],
    runtime_seconds: float,
) -> None:
    lines: list[str] = []

    lines.append("# CrimeNet Gold Event Spine Profile")
    lines.append("")
    lines.append(f"Generated: `{datetime.now(UTC).isoformat()}`")
    lines.append("")
    lines.append(f"- Snapshot: `{snapshot_uri}`")
    lines.append(f"- Latest pointer: `{pointer_uri or 'explicit snapshot / unavailable'}`")
    lines.append(f"- Manifest: `{manifest_uri or 'not found'}`")
    lines.append(f"- National temporal history: `{history_root}`")
    lines.append(f"- Profiling runtime: `{runtime_seconds:.2f}s`")
    lines.append("")

    lines.append("## Executive summary")
    lines.append("")
    lines.append(
        f"- Rows: **{fmt_int(global_metrics.get('row_count'))}**"
    )
    lines.append(
        f"- Unique crime IDs: **{fmt_int(global_metrics.get('unique_crime_ids'))}**"
    )
    lines.append(
        f"- Duplicate crime IDs: **{fmt_int(global_metrics.get('duplicate_crime_ids'))}**"
    )
    lines.append(
        f"- Source cities: **{fmt_int(global_metrics.get('source_city_count'))}**"
    )
    lines.append(
        f"- Distinct H3-r9 cells: **{fmt_int(global_metrics.get('unique_h3_cells'))}**"
    )
    lines.append(
        f"- Feature versions used: **{fmt_int(global_metrics.get('feature_versions_used'))}**"
    )
    lines.append(
        f"- Event time range: **{global_metrics.get('min_occurrence_timestamp_utc')} → "
        f"{global_metrics.get('max_occurrence_timestamp_utc')}**"
    )
    lines.append(
        f"- Future feature leaks: **{fmt_int(global_metrics.get('future_feature_leak_rows'))}**"
    )
    lines.append(
        f"- Snapshot Parquet size: **{fmt_gib(storage.get('parquet_bytes'))}** "
        f"across **{fmt_int(storage.get('parquet_file_count'))}** files"
    )

    if build_coverage.get("published_vs_modeled_pct") is not None:
        lines.append(
            f"- Published event coverage vs modeled Silver: "
            f"**{fmt_pct(build_coverage['published_vs_modeled_pct'], 6)}**"
        )

    lines.append("")

    lines.append("## Publication and storage")
    lines.append("")
    lines.append(f"- Objects: {fmt_int(storage.get('object_count'))}")
    lines.append(f"- Total object bytes: {fmt_gib(storage.get('total_bytes'))}")
    lines.append(f"- Parquet files: {fmt_int(storage.get('parquet_file_count'))}")
    lines.append(f"- Parquet bytes: {fmt_gib(storage.get('parquet_bytes'))}")
    lines.append(f"- Average Parquet file: {fmt_gib(storage.get('avg_parquet_bytes'))}")
    lines.append(f"- Smallest Parquet file: {fmt_gib(storage.get('min_parquet_bytes'))}")
    lines.append(f"- Largest Parquet file: {fmt_gib(storage.get('max_parquet_bytes'))}")
    lines.append("")

    lines.append("## Build / temporal join coverage")
    lines.append("")
    if build_coverage.get("available"):
        for key, value in build_coverage.items():
            if key == "available":
                continue
            if key.endswith("_pct"):
                display = fmt_pct(value, 6)
            elif isinstance(value, int):
                display = fmt_int(value)
            else:
                display = str(value)
            lines.append(f"- `{key}`: **{display}**")
    else:
        lines.append("_Build coverage metrics were not available from the manifest._")
    lines.append("")

    lines.append("## Temporal freshness")
    lines.append("")
    for key, value in temporal.items():
        if isinstance(value, float):
            lines.append(f"- `{key}`: **{value:,.4f}**")
        else:
            lines.append(f"- `{key}`: **{value}**")
    lines.append("")

    lines.append("## Statewide geographic footprint")
    lines.append("")
    lines.append(
        "**Definition:** this is modeled H3 footprint coverage relative to the "
        "latest national feature-store H3 support in each represented state. "
        "It is **not** a claim that municipal crime feeds provide complete "
        "statewide crime-reporting coverage."
    )
    lines.append("")
    if state_meta.get("available") and state_coverage_df.height:
        lines.append(f"- Latest history partition: `{state_meta.get('latest_history_partition')}`")
        lines.append(f"- State source: `{state_meta.get('state_source')}`")
        lines.append(
            f"- Ambiguous H3→state mappings: "
            f"**{fmt_int(state_meta.get('ambiguous_event_h3_state_mappings'))}**"
        )
        lines.append("")
        lines.append(md_table(state_coverage_df, max_rows=100))
    else:
        lines.append(
            f"_Statewide coverage unavailable: {state_meta.get('reason', 'unknown reason')}_"
        )
    lines.append("")

    lines.append("## City coverage")
    lines.append("")
    lines.append(md_table(city_df, max_rows=100))
    lines.append("")

    if state_df.height:
        lines.append("## State summary")
        lines.append("")
        lines.append(md_table(state_df, max_rows=100))
        lines.append("")

    lines.append("## Feature coverage")
    lines.append("")
    if feature_coverage_df.height:
        lines.append(md_table(feature_coverage_df, max_rows=200))
    else:
        lines.append("_No canonical history feature columns detected in the spine._")
    lines.append("")

    lines.append("## Feature-family completeness")
    lines.append("")
    if feature_family_df.height:
        lines.append(md_table(feature_family_df, max_rows=100))
    else:
        lines.append("_No feature families detected._")
    lines.append("")

    lines.append("## H3 event density / concentration")
    lines.append("")
    lines.append(md_table(h3_density_df, max_rows=100))
    lines.append("")

    lines.append("## Feature versions")
    lines.append("")
    lines.append(md_table(version_df, max_rows=100))
    lines.append("")

    lines.append("## Highest-missingness columns")
    lines.append("")
    lines.append(md_table(null_df.head(30), max_rows=30))
    lines.append("")

    if categorical_tables:
        lines.append("## Categorical / taxonomy distributions")
        lines.append("")
        for column, table in categorical_tables.items():
            lines.append(f"### `{column}`")
            lines.append("")
            lines.append(md_table(table, max_rows=30))
            lines.append("")

    lines.append("## Schema")
    lines.append("")
    lines.append(f"- Total columns: **{schema_info['column_count']}**")
    lines.append("")
    schema_df = pl.DataFrame(
        {
            "column": list(schema_info["dtypes"].keys()),
            "dtype": list(schema_info["dtypes"].values()),
        }
    )
    lines.append(md_table(schema_df, max_rows=200))
    lines.append("")

    lines.append("## Generated artifacts")
    lines.append("")
    for filename in output_files:
        lines.append(f"- `{filename}`")
    lines.append("")

    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a comprehensive human-readable and machine-readable "
            "profile of the CrimeNet Gold event spine."
        )
    )

    parser.add_argument(
        "--snapshot-uri",
        help=(
            "Exact snapshot URI. If omitted, resolve "
            "s3://.../gold/event_spine/_latest.json."
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
        "--output-dir",
        default="artifacts/event_spine_profile",
    )
    parser.add_argument(
        "--skip-statewide-history",
        action="store_true",
        help=(
            "Skip latest national-history state H3 denominator metrics. "
            "Useful for a faster spine-only profile."
        ),
    )
    parser.add_argument(
        "--skip-deep-numeric-profile",
        action="store_true",
        help=(
            "Skip p01/p50/p99/n_unique numeric-column profiling. "
            "Min/max/mean/std are still computed."
        ),
    )

    args = parser.parse_args()
    started = time.perf_counter()

    config = storage_config()
    storage_options = polars_storage_options(config)
    client = make_s3_client(config)

    pointer_uri: str | None = None

    if args.snapshot_uri:
        snapshot_uri = args.snapshot_uri.rstrip("/")
    else:
        snapshot_uri, pointer_uri = resolve_latest_snapshot(
            client,
            args.event_spine_root,
        )

    snapshot_id = snapshot_uri.rstrip("/").split("/")[-1]
    output_dir = Path(args.output_dir) / snapshot_id
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("CrimeNet Gold Event Spine Comprehensive Profile")
    log(f"Snapshot: {snapshot_uri}")
    log(f"Output:   {output_dir}")
    log("=" * 80)

    manifest, manifest_uri = read_manifest(client, snapshot_uri)

    success_exists = any(
        object_exists(client, f"{snapshot_uri}/{name}")
        for name in ("_SUCCESS", "_SUCCESS.json")
    )

    log("Reading snapshot object/storage metrics")
    storage = snapshot_storage_metrics(client, snapshot_uri)

    log("Scanning event spine")
    spine = scan_parquet_root(snapshot_uri, storage_options)

    log("Profiling schema")
    schema_df, schema_info = profile_schema(spine)
    spine_columns = schema_info["columns"]

    log("Computing global dataset metrics")
    global_metrics = global_dataset_metrics(
        spine,
        spine_columns,
    )

    log("Computing all-column null / coverage profile")
    null_df = null_coverage_profile(
        spine,
        spine_columns,
    )

    log("Computing numeric-column profile")
    numeric_df = numeric_profile(
        spine,
        schema_info,
        deep=not args.skip_deep_numeric_profile,
    )

    log("Inspecting canonical national-history schema")
    history_scan = scan_parquet_root(
        args.history_root,
        storage_options,
    )
    history_columns = history_scan.collect_schema().names()

    feature_columns = identify_history_feature_columns(
        spine_columns,
        history_columns,
    )

    log(
        f"Detected {len(feature_columns):,} canonical history feature columns "
        "present in the event spine"
    )

    log("Computing per-feature coverage")
    feature_coverage_df = feature_column_coverage(
        spine,
        feature_columns,
    )

    log("Computing global feature-family completeness")
    feature_family_df = feature_family_coverage(
        spine,
        feature_columns,
    )

    log("Computing feature-family completeness by city")
    feature_family_city_df = feature_family_coverage(
        spine,
        feature_columns,
        ["source_city"],
    ).sort("source_city")

    log("Computing city coverage summary")
    city_df = city_summary(
        spine,
        spine_columns,
    )

    log("Computing city × year temporal coverage")
    city_year_df = city_year_summary(spine)

    log("Computing city × month temporal coverage")
    monthly_df = monthly_summary(spine)

    log("Computing H3 density and concentration metrics")
    h3_density_df, cell_counts_df = h3_density_summary(spine)

    log("Computing feature-version usage")
    version_df = feature_version_summary(spine)

    log("Computing temporal feature freshness")
    temporal = temporal_freshness(
        spine,
        spine_columns,
    )

    log("Extracting publication/build coverage from manifest")
    build_coverage = extract_build_coverage(
        manifest,
        int(global_metrics["row_count"]),
    )

    state_meta: dict[str, Any]
    state_coverage_df = pl.DataFrame()
    state_df = pl.DataFrame()
    feature_family_state_df = pl.DataFrame()

    if args.skip_statewide_history:
        state_meta = {
            "available": False,
            "reason": "Skipped by --skip-statewide-history.",
        }
    else:
        log("Computing statewide H3 footprint coverage")
        state_spine, state_col, state_coverage_df, state_meta = (
            build_state_context(
                spine,
                spine_columns,
                client,
                args.history_root,
                storage_options,
            )
        )

        if state_col is not None:
            log("Computing state summary")
            state_df = state_summary(
                state_spine,
                state_col,
            )

            log("Computing feature-family completeness by state")
            feature_family_state_df = feature_family_coverage(
                state_spine,
                feature_columns,
                [state_col],
            ).rename({state_col: "state"}).sort("state")

    categorical_tables: dict[str, pl.DataFrame] = {}
    for column in categorical_columns(spine_columns):
        log(f"Computing categorical distribution: {column}")
        categorical_tables[column] = categorical_distribution(
            spine,
            column,
        )

    # ------------------------------------------------------------------
    # Write artifacts
    # ------------------------------------------------------------------

    log("Writing report artifacts")

    csv_tables: dict[str, pl.DataFrame] = {
        "schema.csv": schema_df,
        "column_coverage.csv": null_df,
        "numeric_profile.csv": numeric_df,
        "feature_coverage.csv": feature_coverage_df,
        "feature_family_coverage.csv": feature_family_df,
        "feature_family_by_city.csv": feature_family_city_df,
        "city_summary.csv": city_df,
        "city_year_summary.csv": city_year_df,
        "monthly_summary.csv": monthly_df,
        "h3_density_summary.csv": h3_density_df,
        "h3_cell_event_counts.csv": cell_counts_df,
        "feature_version_summary.csv": version_df,
    }

    if state_coverage_df.height:
        csv_tables["state_h3_coverage.csv"] = state_coverage_df

    if state_df.height:
        csv_tables["state_summary.csv"] = state_df

    if feature_family_state_df.height:
        csv_tables["feature_family_by_state.csv"] = (
            feature_family_state_df
        )

    for column, table in categorical_tables.items():
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", column)
        csv_tables[f"distribution__{safe_name}.csv"] = table

    for filename, table in csv_tables.items():
        if table.width == 0:
            continue
        table.write_csv(output_dir / filename)

    metrics = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "snapshot_uri": snapshot_uri,
        "latest_pointer_uri": pointer_uri,
        "manifest_uri": manifest_uri,
        "success_marker_exists": success_exists,
        "history_root": args.history_root,
        "storage": storage,
        "schema": schema_info,
        "global": global_metrics,
        "build_coverage": build_coverage,
        "temporal_freshness": temporal,
        "statewide_coverage_metadata": state_meta,
        "statewide_coverage": dataframe_rows(state_coverage_df),
        "city_summary": dataframe_rows(city_df),
        "state_summary": dataframe_rows(state_df),
        "feature_coverage": dataframe_rows(feature_coverage_df),
        "feature_family_coverage": dataframe_rows(feature_family_df),
        "h3_density": dataframe_rows(h3_density_df),
        "feature_version_summary": dataframe_rows(version_df),
        "categorical_distributions": {
            column: dataframe_rows(table)
            for column, table in categorical_tables.items()
        },
        "manifest": manifest,
    }

    metrics_path = output_dir / "metrics.json"
    write_json(metrics_path, metrics)

    output_files = sorted(
        [p.name for p in output_dir.iterdir()]
        + ["report.md"]
    )

    runtime = time.perf_counter() - started

    report_path = output_dir / "report.md"
    write_report(
        report_path,
        snapshot_uri=snapshot_uri,
        pointer_uri=pointer_uri,
        history_root=args.history_root,
        manifest_uri=manifest_uri,
        storage=storage,
        schema_info=schema_info,
        global_metrics=global_metrics,
        build_coverage=build_coverage,
        temporal=temporal,
        state_meta=state_meta,
        state_coverage_df=state_coverage_df,
        city_df=city_df,
        state_df=state_df,
        feature_coverage_df=feature_coverage_df,
        feature_family_df=feature_family_df,
        h3_density_df=h3_density_df,
        version_df=version_df,
        null_df=null_df,
        categorical_tables=categorical_tables,
        output_files=output_files,
        runtime_seconds=runtime,
    )

    log("=" * 80)
    log("PROFILE COMPLETE")
    log(f"Rows:             {int(global_metrics['row_count']):,}")
    log(f"Unique crimes:    {int(global_metrics['unique_crime_ids']):,}")
    log(f"Unique H3 cells:  {int(global_metrics['unique_h3_cells']):,}")
    log(f"Cities:           {int(global_metrics['source_city_count']):,}")
    log(
        f"Feature columns:  {len(feature_columns):,} "
        "canonical temporal-history columns"
    )
    log(
        f"Future leaks:     {int(global_metrics['future_feature_leak_rows']):,}"
    )
    if build_coverage.get("published_vs_modeled_pct") is not None:
        log(
            "Modeled coverage:  "
            f"{build_coverage['published_vs_modeled_pct']:.6f}%"
        )
    log(f"Runtime:          {runtime:.2f}s")
    log(f"Report:           {report_path}")
    log(f"Metrics JSON:     {metrics_path}")
    log("=" * 80)

    print()
    print(report_path.read_text())

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nProfile interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(
            f"\nPROFILE CRASHED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise
