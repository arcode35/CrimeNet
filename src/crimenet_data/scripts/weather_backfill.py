#!/usr/bin/env python3
"""Backfill missing Open-Meteo ERA5-Land weather required by CrimeNet.

Workflow
--------
1. Read current integration samples from Delta/GCS and derive the exact required
   (weather_query_cell_id, UTC hour) keys.
2. Read existing Silver land + coastal weather and treat any non-null
   temperature from either source as currently available to integration.
3. Anti-join required keys against available keys.
4. Write:
     - missing_weather_hours.csv: one row per missing required cell/hour
     - weather_backfill_requests.csv: one row per affected H3-6 cell/year
5. Fetch complete affected cell-years from Open-Meteo ERA5-Land using the same
   request identity and raw JSON envelope as CrimeNet's historical downloader.
6. Validate that every originally missing required hour is now present and has
   non-null temperature_2m.
7. Cache each response atomically and upload the schema-compatible JSON object
   to CrimeNet's raw landing prefix.

The complete-cell-year request is intentional. It reproduces the canonical raw
object identity used by the original downloader, so an incomplete/corrupt object
can be replaced instead of creating overlapping partial-range raw objects.
"""

from __future__ import annotations

import argparse
import email.utils
import hashlib
import json
import logging
import os
import random
import shutil
import subprocess
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import h3.api.basic_int as h3
import polars as pl
import requests
from requests import Session
from requests.adapters import HTTPAdapter


# =============================================================================
# CrimeNet defaults
# =============================================================================

INTEGRATION_ROOT = "gs://crimenet/gold/integration_samples"
LAND_WEATHER_ROOT = "gs://crimenet/silver/weather/land_hourly"
COASTAL_WEATHER_ROOT = "gs://crimenet/silver/weather/coastal_hourly"
RAW_LANDING_ROOT = "gs://crimenet/raw_files/landing/weather/open_meteo"

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

PROVIDER = "open_meteo"
MODEL = "era5_land"
TIMEZONE = "GMT"
CELL_SELECTION = "land"
H3_RESOLUTION = 6
DEFAULT_AVAILABILITY_LAG_DAYS = 7

DEFAULT_HOURLY_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "snowfall",
    "weather_code",
    "cloud_cover",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
)

RETRYABLE_HTTP_STATUS_CODES = frozenset(
    {408, 425, 429, 500, 502, 503, 504}
)
DEFAULT_MAX_ATTEMPTS = 8
DEFAULT_RETRY_BASE_SECONDS = 2.0
DEFAULT_RETRY_MAX_SECONDS = 120.0
DEFAULT_RETRY_JITTER_SECONDS = 1.0

logger = logging.getLogger("weather_backfill")


# =============================================================================
# Request model / identity
# =============================================================================

@dataclass(frozen=True)
class WeatherRequest:
    request_sequence: int
    request_id: str
    weather_query_cell_id: int
    latitude: float
    longitude: float
    feature_year: int
    start_date: date
    end_date: date
    missing_required_hours: int
    affected_integration_samples: int
    first_missing_hour_utc: datetime
    last_missing_hour_utc: datetime
    hourly_variables: tuple[str, ...]

    @property
    def provider(self) -> str:
        return PROVIDER

    @property
    def model(self) -> str:
        return MODEL

    @property
    def timezone(self) -> str:
        return TIMEZONE

    @property
    def cell_selection(self) -> str:
        return CELL_SELECTION


class WeatherBackfillError(RuntimeError):
    pass


class PermanentWeatherFetchError(WeatherBackfillError):
    pass


def normalized_hourly_variables(
    values: Sequence[str] = DEFAULT_HOURLY_VARIABLES,
) -> tuple[str, ...]:
    result = tuple(sorted({value.strip() for value in values if value.strip()}))
    if not result:
        raise ValueError("At least one hourly variable is required")
    return result


def request_identifier(
    *,
    weather_query_cell_id: int,
    start_date: date,
    end_date: date,
    hourly_variables: Sequence[str],
) -> str:
    """Match CrimeNet's original stable raw weather request identity."""
    identity = "|".join(
        [
            PROVIDER,
            MODEL,
            str(weather_query_cell_id),
            start_date.isoformat(),
            end_date.isoformat(),
            ",".join(hourly_variables),
            TIMEZONE,
            CELL_SELECTION,
            str(H3_RESOLUTION),
        ]
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


# =============================================================================
# Delta scans / missing-key discovery
# =============================================================================

def scan_delta(
    path: str,
    *,
    credentials: pl.CredentialProviderGCP,
) -> pl.LazyFrame:
    return pl.scan_delta(
        path,
        credential_provider=credentials,
    )


def normalize_weather_hour(column: str) -> pl.Expr:
    """Normalize a weather timestamp to the exact UTC hour key."""
    return pl.col(column).dt.truncate("1h")


def build_required_weather_keys(
    *,
    integration_root: str,
    credentials: pl.CredentialProviderGCP,
) -> pl.LazyFrame:
    """One row per required H3-6 cell/hour, with integration impact count."""
    return (
        scan_delta(
            integration_root,
            credentials=credentials,
        )
        .select(
            pl.col("weather_query_cell_id")
            .cast(pl.Int64)
            .alias("weather_query_cell_id"),
            normalize_weather_hour("sample_timestamp_utc")
            .alias("weather_timestamp"),
        )
        .filter(
            pl.col("weather_query_cell_id").is_not_null()
            & pl.col("weather_timestamp").is_not_null()
        )
        .group_by(
            "weather_query_cell_id",
            "weather_timestamp",
        )
        .agg(
            pl.len()
            .cast(pl.Int64)
            .alias("affected_integration_samples")
        )
    )


def build_available_weather_keys(
    *,
    land_weather_root: str,
    coastal_weather_root: str,
    credentials: pl.CredentialProviderGCP,
) -> pl.LazyFrame:
    """Effective available temperature keys used by integration.

    A key is available when either the normal land table or the coastal fallback
    has non-null temperature_2m_c. This mirrors the current integration coalesce.
    """

    def usable(root: str) -> pl.LazyFrame:
        return (
            scan_delta(root, credentials=credentials)
            .select(
                pl.col("weather_query_cell_id")
                .cast(pl.Int64)
                .alias("weather_query_cell_id"),
                normalize_weather_hour("weather_timestamp")
                .alias("weather_timestamp"),
                pl.col("temperature_2m_c")
                .cast(pl.Float64, strict=False)
                .alias("temperature_2m_c"),
            )
            .filter(
                pl.col("weather_query_cell_id").is_not_null()
                & pl.col("weather_timestamp").is_not_null()
                & pl.col("temperature_2m_c").is_not_null()
                & pl.col("temperature_2m_c").is_finite()
            )
            .select(
                "weather_query_cell_id",
                "weather_timestamp",
            )
        )

    return (
        pl.concat(
            [
                usable(land_weather_root),
                usable(coastal_weather_root),
            ],
            how="vertical",
        )
        .unique(
            subset=[
                "weather_query_cell_id",
                "weather_timestamp",
            ]
        )
    )


def add_h3_centers(frame: pl.DataFrame) -> pl.DataFrame:
    cells = frame.get_column("weather_query_cell_id").unique().to_list()
    rows: list[dict[str, Any]] = []

    for cell in cells:
        cell_id = int(cell)
        resolution = h3.get_resolution(cell_id)
        if resolution != H3_RESOLUTION:
            raise WeatherBackfillError(
                f"H3 cell {cell_id} has resolution {resolution}; "
                f"expected {H3_RESOLUTION}"
            )

        latitude, longitude = h3.cell_to_latlng(cell_id)
        rows.append(
            {
                "weather_query_cell_id": cell_id,
                "query_latitude": float(latitude),
                "query_longitude": float(longitude),
            }
        )

    centers = pl.DataFrame(rows)
    return frame.join(
        centers,
        on="weather_query_cell_id",
        how="left",
        validate="m:1",
    )


def discover_missing_weather_hours(
    *,
    integration_root: str,
    land_weather_root: str,
    coastal_weather_root: str,
    credentials: pl.CredentialProviderGCP,
) -> pl.DataFrame:
    required = build_required_weather_keys(
        integration_root=integration_root,
        credentials=credentials,
    )
    available = build_available_weather_keys(
        land_weather_root=land_weather_root,
        coastal_weather_root=coastal_weather_root,
        credentials=credentials,
    )

    missing = (
        required
        .join(
            available,
            on=[
                "weather_query_cell_id",
                "weather_timestamp",
            ],
            how="anti",
        )
        .with_columns(
            pl.col("weather_timestamp")
            .dt.year()
            .cast(pl.Int32)
            .alias("feature_year")
        )
        .sort(
            "feature_year",
            "weather_query_cell_id",
            "weather_timestamp",
        )
        .collect(engine="streaming")
    )

    if missing.is_empty():
        return missing.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("query_latitude"),
            pl.lit(None, dtype=pl.Float64).alias("query_longitude"),
        )

    return add_h3_centers(missing).select(
        "weather_query_cell_id",
        "query_latitude",
        "query_longitude",
        "weather_timestamp",
        "feature_year",
        "affected_integration_samples",
    )


# =============================================================================
# CSV manifests
# =============================================================================

def build_request_manifest(
    missing_hours: pl.DataFrame,
    *,
    availability_cutoff: date,
    hourly_variables: tuple[str, ...],
    landing_root: str,
) -> pl.DataFrame:
    if missing_hours.is_empty():
        return pl.DataFrame(
            schema={
                "request_sequence": pl.Int64,
                "request_id": pl.String,
                "provider": pl.String,
                "model": pl.String,
                "weather_query_cell_id": pl.Int64,
                "h3_resolution": pl.Int8,
                "query_latitude": pl.Float64,
                "query_longitude": pl.Float64,
                "feature_year": pl.Int32,
                "request_start_date": pl.Date,
                "request_end_date": pl.Date,
                "first_missing_hour_utc": pl.Datetime("us", "UTC"),
                "last_missing_hour_utc": pl.Datetime("us", "UTC"),
                "missing_required_hours": pl.Int64,
                "affected_integration_samples": pl.Int64,
                "timezone": pl.String,
                "cell_selection": pl.String,
                "raw_object_uri": pl.String,
            }
        )

    grouped = (
        missing_hours
        .group_by(
            "weather_query_cell_id",
            "query_latitude",
            "query_longitude",
            "feature_year",
        )
        .agg(
            pl.col("weather_timestamp")
            .min()
            .alias("first_missing_hour_utc"),
            pl.col("weather_timestamp")
            .max()
            .alias("last_missing_hour_utc"),
            pl.len()
            .cast(pl.Int64)
            .alias("missing_required_hours"),
            pl.col("affected_integration_samples")
            .sum()
            .cast(pl.Int64)
            .alias("affected_integration_samples"),
        )
        .sort(
            "feature_year",
            "weather_query_cell_id",
        )
    )

    records: list[dict[str, Any]] = []
    normalized_root = landing_root.rstrip("/")

    for sequence, row in enumerate(
        grouped.iter_rows(named=True),
        start=1,
    ):
        feature_year = int(row["feature_year"])
        request_start = date(feature_year, 1, 1)
        request_end = min(
            date(feature_year, 12, 31),
            availability_cutoff,
        )

        first_missing = row["first_missing_hour_utc"]
        last_missing = row["last_missing_hour_utc"]

        if first_missing.date() > availability_cutoff:
            raise WeatherBackfillError(
                "Missing required weather is newer than the configured "
                f"Open-Meteo availability cutoff: cell="
                f"{row['weather_query_cell_id']}, first_missing={first_missing}, "
                f"cutoff={availability_cutoff}"
            )

        if last_missing.date() > request_end:
            raise WeatherBackfillError(
                "At least one required missing hour is outside the fetchable "
                f"archive window: cell={row['weather_query_cell_id']}, "
                f"last_missing={last_missing}, request_end={request_end}"
            )

        cell_id = int(row["weather_query_cell_id"])
        request_id = request_identifier(
            weather_query_cell_id=cell_id,
            start_date=request_start,
            end_date=request_end,
            hourly_variables=hourly_variables,
        )

        raw_object_uri = (
            f"{normalized_root}/{MODEL}/year={feature_year}/"
            f"{request_id}.json"
        )

        records.append(
            {
                "request_sequence": sequence,
                "request_id": request_id,
                "provider": PROVIDER,
                "model": MODEL,
                "weather_query_cell_id": cell_id,
                "h3_resolution": H3_RESOLUTION,
                "query_latitude": float(row["query_latitude"]),
                "query_longitude": float(row["query_longitude"]),
                "feature_year": feature_year,
                "request_start_date": request_start,
                "request_end_date": request_end,
                "first_missing_hour_utc": first_missing,
                "last_missing_hour_utc": last_missing,
                "missing_required_hours": int(row["missing_required_hours"]),
                "affected_integration_samples": int(
                    row["affected_integration_samples"]
                ),
                "timezone": TIMEZONE,
                "cell_selection": CELL_SELECTION,
                "raw_object_uri": raw_object_uri,
            }
        )

    return pl.DataFrame(records).sort("request_sequence")


def write_manifests(
    *,
    missing_hours: pl.DataFrame,
    requests_manifest: pl.DataFrame,
    output_directory: Path,
) -> tuple[Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)

    missing_path = output_directory / "missing_weather_hours.csv"
    request_path = output_directory / "weather_backfill_requests.csv"

    missing_hours.write_csv(missing_path)
    requests_manifest.write_csv(request_path)

    return missing_path, request_path


# =============================================================================
# HTTP acquisition
# =============================================================================

def build_open_meteo_session() -> Session:
    adapter = HTTPAdapter(
        max_retries=0,
        pool_connections=1,
        pool_maxsize=1,
    )
    session = Session()
    session.mount("https://", adapter)
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "CrimeNet-weather-backfill/2.0",
        }
    )
    return session


def retry_after_seconds(response: requests.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if not value:
        return None

    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = email.utils.parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=UTC)
            return max(
                0.0,
                (retry_at.astimezone(UTC) - datetime.now(UTC)).total_seconds(),
            )
        except (TypeError, ValueError, OverflowError):
            return None


def retry_delay_seconds(
    *,
    attempt: int,
    base_seconds: float,
    maximum_seconds: float,
    jitter_seconds: float,
    response: requests.Response | None,
) -> float:
    exponential = min(
        maximum_seconds,
        base_seconds * (2 ** (attempt - 1)),
    )
    server_delay = (
        retry_after_seconds(response)
        if response is not None
        else None
    )
    return max(exponential, server_delay or 0.0) + random.uniform(
        0.0,
        jitter_seconds,
    )


def http_error_message(
    request: WeatherRequest,
    response: requests.Response,
) -> str:
    try:
        payload = response.json()
        reason = payload.get("reason", payload)
    except requests.JSONDecodeError:
        reason = response.text

    return (
        f"Open-Meteo HTTP {response.status_code}: "
        f"request_id={request.request_id}, reason={str(reason)[:1000]}"
    )


def validate_hourly_payload(
    *,
    request: WeatherRequest,
    payload: Mapping[str, Any],
) -> None:
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        raise WeatherBackfillError("Open-Meteo returned no hourly object")

    times = hourly.get("time")
    if not isinstance(times, list) or not times:
        raise WeatherBackfillError("Open-Meteo returned no hourly timestamps")

    if len(times) != len(set(map(str, times))):
        raise WeatherBackfillError("Open-Meteo returned duplicate timestamps")

    expected_hours = (
        (request.end_date - request.start_date).days + 1
    ) * 24
    if len(times) != expected_hours:
        raise WeatherBackfillError(
            "Open-Meteo returned an incomplete date range: "
            f"expected_hours={expected_hours}, returned_hours={len(times)}, "
            f"request_id={request.request_id}"
        )

    for variable in request.hourly_variables:
        values = hourly.get(variable)
        if not isinstance(values, list):
            raise WeatherBackfillError(
                f"Open-Meteo response is missing {variable!r}"
            )
        if len(values) != len(times):
            raise WeatherBackfillError(
                f"Length mismatch for {variable!r}: "
                f"time={len(times)}, values={len(values)}"
            )


def fetch_historical_weather(
    request: WeatherRequest,
    *,
    session: Session,
    maximum_attempts: int,
    retry_base_seconds: float,
    retry_max_seconds: float,
    retry_jitter_seconds: float,
) -> dict[str, Any]:
    params = {
        "latitude": request.latitude,
        "longitude": request.longitude,
        "start_date": request.start_date.isoformat(),
        "end_date": request.end_date.isoformat(),
        "hourly": ",".join(request.hourly_variables),
        "models": request.model,
        "timezone": request.timezone,
        "cell_selection": request.cell_selection,
    }

    last_error: BaseException | None = None

    for attempt in range(1, maximum_attempts + 1):
        response: requests.Response | None = None

        try:
            response = session.get(
                OPEN_METEO_ARCHIVE_URL,
                params=params,
                timeout=(30, 180),
            )

            if response.status_code >= 400:
                message = http_error_message(request, response)
                if response.status_code not in RETRYABLE_HTTP_STATUS_CODES:
                    raise PermanentWeatherFetchError(message)
                raise WeatherBackfillError(message)

            try:
                payload = response.json()
            except requests.JSONDecodeError as exc:
                raise WeatherBackfillError(
                    "Open-Meteo returned invalid JSON: "
                    f"request_id={request.request_id}"
                ) from exc

            if payload.get("error") is True:
                raise PermanentWeatherFetchError(
                    "Open-Meteo rejected request: "
                    f"request_id={request.request_id}, "
                    f"reason={payload.get('reason')}"
                )

            validate_hourly_payload(
                request=request,
                payload=payload,
            )

            # IMPORTANT: keep this envelope exactly compatible with
            # LAND_WEATHER_SCHEMA in bronze.py. Do not add operational fields.
            return {
                "request_id": request.request_id,
                "provider": request.provider,
                "model": request.model,
                "weather_query_cell_id": request.weather_query_cell_id,
                "h3_resolution": H3_RESOLUTION,
                "query_latitude": request.latitude,
                "query_longitude": request.longitude,
                "grid_latitude": payload.get("latitude"),
                "grid_longitude": payload.get("longitude"),
                "grid_elevation": payload.get("elevation"),
                "start_date": request.start_date.isoformat(),
                "end_date": request.end_date.isoformat(),
                "timezone": payload.get("timezone", request.timezone),
                "utc_offset_seconds": payload.get("utc_offset_seconds"),
                "cell_selection": request.cell_selection,
                "hourly_variables": list(request.hourly_variables),
                "hourly_units": payload.get("hourly_units", {}),
                "hourly": payload["hourly"],
            }

        except PermanentWeatherFetchError:
            raise
        except (
            requests.Timeout,
            requests.ConnectionError,
            requests.RequestException,
            WeatherBackfillError,
        ) as exc:
            last_error = exc
            if attempt >= maximum_attempts:
                break

            delay = retry_delay_seconds(
                attempt=attempt,
                base_seconds=retry_base_seconds,
                maximum_seconds=retry_max_seconds,
                jitter_seconds=retry_jitter_seconds,
                response=response,
            )
            logger.warning(
                "attempt %s/%s failed request_id=%s: %s; retrying in %.1fs",
                attempt,
                maximum_attempts,
                request.request_id,
                exc,
                delay,
            )
            time.sleep(delay)

    raise WeatherBackfillError(
        "Open-Meteo request failed after "
        f"{maximum_attempts} attempts: request_id={request.request_id}, "
        f"last_error={last_error}"
    ) from last_error


# =============================================================================
# Exact required-hour validation
# =============================================================================

def utc_hour_key(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    else:
        value = value.astimezone(UTC)
    return value.strftime("%Y-%m-%dT%H:00")


def required_hours_by_cell_year(
    missing_hours: pl.DataFrame,
) -> dict[tuple[int, int], set[str]]:
    result: dict[tuple[int, int], set[str]] = defaultdict(set)

    for row in missing_hours.iter_rows(named=True):
        result[
            (
                int(row["weather_query_cell_id"]),
                int(row["feature_year"]),
            )
        ].add(utc_hour_key(row["weather_timestamp"]))

    return result


def validate_required_hours_repaired(
    *,
    request: WeatherRequest,
    response: Mapping[str, Any],
    required_hours: set[str],
) -> None:
    hourly = response["hourly"]
    times = hourly["time"]
    temperatures = hourly["temperature_2m"]

    temperatures_by_hour = dict(zip(times, temperatures, strict=True))

    still_missing = [
        hour
        for hour in sorted(required_hours)
        if hour not in temperatures_by_hour
        or temperatures_by_hour[hour] is None
    ]

    if still_missing:
        sample = ", ".join(still_missing[:10])
        raise WeatherBackfillError(
            "Backfill response did not repair every required hour: "
            f"cell={request.weather_query_cell_id}, "
            f"year={request.feature_year}, "
            f"still_missing={len(still_missing)}, sample=[{sample}]"
        )


# =============================================================================
# Local cache + cloud upload
# =============================================================================

def cache_path_for_request(
    cache_directory: Path,
    request: WeatherRequest,
) -> Path:
    return (
        cache_directory
        / request.model
        / f"year={request.feature_year}"
        / f"{request.request_id}.json"
    )


def write_json_atomic(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{uuid4().hex}.tmp"
    )

    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(
                payload,
                file,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_cached_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise WeatherBackfillError(
            f"Cached object is not a JSON object: {path}"
        )
    return payload


def validate_cached_response(
    *,
    request: WeatherRequest,
    payload: Mapping[str, Any],
    required_hours: set[str],
) -> None:
    if str(payload.get("request_id")) != request.request_id:
        raise WeatherBackfillError("Cached request_id does not match")
    if int(payload.get("weather_query_cell_id", -1)) != (
        request.weather_query_cell_id
    ):
        raise WeatherBackfillError("Cached H3 cell does not match")
    if str(payload.get("start_date")) != request.start_date.isoformat():
        raise WeatherBackfillError("Cached start_date does not match")
    if str(payload.get("end_date")) != request.end_date.isoformat():
        raise WeatherBackfillError("Cached end_date does not match")

    validate_hourly_payload(
        request=request,
        payload=payload,
    )
    validate_required_hours_repaired(
        request=request,
        response=payload,
        required_hours=required_hours,
    )


def parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a gs:// URI: {uri}")

    without_scheme = uri[5:]
    bucket, separator, object_name = without_scheme.partition("/")
    if not bucket or not separator or not object_name:
        raise ValueError(f"Invalid GCS object URI: {uri}")
    return bucket, object_name


def upload_file_to_gcs(
    *,
    local_path: Path,
    destination_uri: str,
) -> None:
    """Upload with google-cloud-storage when available, else gcloud CLI."""
    try:
        from google.cloud import storage  # type: ignore
    except ImportError:
        storage = None

    if storage is not None:
        bucket_name, object_name = parse_gs_uri(destination_uri)
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(object_name)
        blob.upload_from_filename(
            str(local_path),
            content_type="application/x-ndjson",
        )
        return

    gcloud = shutil.which("gcloud")
    if gcloud is None:
        raise RuntimeError(
            "Uploading gs:// objects requires either google-cloud-storage "
            "(`pip install google-cloud-storage`) or the `gcloud` CLI."
        )

    subprocess.run(
        [
            gcloud,
            "storage",
            "cp",
            "--quiet",
            str(local_path),
            destination_uri,
        ],
        check=True,
    )


def copy_to_local_landing(
    *,
    local_path: Path,
    landing_root: str,
    request: WeatherRequest,
) -> Path:
    root = Path(landing_root).expanduser().resolve()
    destination = (
        root
        / request.model
        / f"year={request.feature_year}"
        / f"{request.request_id}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary = destination.with_name(
        f".{destination.name}.{uuid4().hex}.tmp"
    )
    try:
        shutil.copyfile(local_path, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    return destination


def publish_raw_object(
    *,
    cache_path: Path,
    landing_root: str,
    request: WeatherRequest,
) -> str:
    if landing_root.startswith("gs://"):
        destination = (
            f"{landing_root.rstrip('/')}/{request.model}/"
            f"year={request.feature_year}/{request.request_id}.json"
        )
        upload_file_to_gcs(
            local_path=cache_path,
            destination_uri=destination,
        )
        return destination

    return str(
        copy_to_local_landing(
            local_path=cache_path,
            landing_root=landing_root,
            request=request,
        )
    )


# =============================================================================
# Manifest row conversion / execution
# =============================================================================

def request_from_row(
    row: Mapping[str, Any],
    *,
    hourly_variables: tuple[str, ...],
) -> WeatherRequest:
    return WeatherRequest(
        request_sequence=int(row["request_sequence"]),
        request_id=str(row["request_id"]),
        weather_query_cell_id=int(row["weather_query_cell_id"]),
        latitude=float(row["query_latitude"]),
        longitude=float(row["query_longitude"]),
        feature_year=int(row["feature_year"]),
        start_date=row["request_start_date"],
        end_date=row["request_end_date"],
        missing_required_hours=int(row["missing_required_hours"]),
        affected_integration_samples=int(
            row["affected_integration_samples"]
        ),
        first_missing_hour_utc=row["first_missing_hour_utc"],
        last_missing_hour_utc=row["last_missing_hour_utc"],
        hourly_variables=hourly_variables,
    )


def execute_backfill(
    *,
    requests_manifest: pl.DataFrame,
    missing_hours: pl.DataFrame,
    cache_directory: Path,
    landing_root: str,
    hourly_variables: tuple[str, ...],
    maximum_requests: int | None,
    pause_seconds: float,
    maximum_attempts: int,
    retry_base_seconds: float,
    retry_max_seconds: float,
    retry_jitter_seconds: float,
    force_download: bool,
    no_upload: bool,
) -> None:
    required_lookup = required_hours_by_cell_year(missing_hours)
    selected = (
        requests_manifest.head(maximum_requests)
        if maximum_requests is not None
        else requests_manifest
    )

    session = build_open_meteo_session()
    downloaded = 0
    cache_reused = 0
    uploaded = 0
    failures: list[str] = []

    try:
        for processed, row in enumerate(
            selected.iter_rows(named=True),
            start=1,
        ):
            request = request_from_row(
                row,
                hourly_variables=hourly_variables,
            )
            required_hours = required_lookup[
                (
                    request.weather_query_cell_id,
                    request.feature_year,
                )
            ]
            cache_path = cache_path_for_request(
                cache_directory,
                request,
            )

            print(
                f"[{processed}/{selected.height}] "
                f"year={request.feature_year} "
                f"cell={request.weather_query_cell_id} "
                f"missing_hours={request.missing_required_hours} "
                f"request_id={request.request_id[:12]}"
            )

            response: dict[str, Any] | None = None

            if cache_path.exists() and not force_download:
                try:
                    cached = read_cached_json(cache_path)
                    validate_cached_response(
                        request=request,
                        payload=cached,
                        required_hours=required_hours,
                    )
                    response = cached
                    cache_reused += 1
                    print(f"  valid cache: {cache_path}")
                except Exception as exc:
                    quarantine = cache_path.with_name(
                        f"{cache_path.name}.corrupt-{uuid4().hex}"
                    )
                    os.replace(cache_path, quarantine)
                    logger.warning(
                        "quarantined invalid cache %s -> %s: %s",
                        cache_path,
                        quarantine,
                        exc,
                    )

            if response is None:
                try:
                    response = fetch_historical_weather(
                        request,
                        session=session,
                        maximum_attempts=maximum_attempts,
                        retry_base_seconds=retry_base_seconds,
                        retry_max_seconds=retry_max_seconds,
                        retry_jitter_seconds=retry_jitter_seconds,
                    )
                    validate_required_hours_repaired(
                        request=request,
                        response=response,
                        required_hours=required_hours,
                    )
                    write_json_atomic(cache_path, response)
                    downloaded += 1
                    print(
                        "  downloaded: "
                        f"{len(response['hourly']['time']):,} hours"
                    )
                except Exception as exc:
                    message = (
                        f"request_id={request.request_id} "
                        f"cell={request.weather_query_cell_id} "
                        f"year={request.feature_year}: {exc}"
                    )
                    failures.append(message)
                    logger.exception("backfill request failed: %s", message)
                    continue

            if not no_upload:
                try:
                    destination = publish_raw_object(
                        cache_path=cache_path,
                        landing_root=landing_root,
                        request=request,
                    )
                    uploaded += 1
                    print(f"  published: {destination}")
                except Exception as exc:
                    message = (
                        f"upload request_id={request.request_id} "
                        f"cell={request.weather_query_cell_id} "
                        f"year={request.feature_year}: {exc}"
                    )
                    failures.append(message)
                    logger.exception("backfill upload failed: %s", message)
                    continue

            if pause_seconds > 0:
                time.sleep(pause_seconds)

    finally:
        session.close()

    print()
    print("Backfill pass complete")
    print(f"Requests selected:      {selected.height:,}")
    print(f"Downloaded:             {downloaded:,}")
    print(f"Valid cache reused:     {cache_reused:,}")
    print(f"Published raw objects:  {uploaded:,}")
    print(f"Failures:               {len(failures):,}")

    if failures:
        print("\nFirst failures:")
        for failure in failures[:10]:
            print(f"  - {failure}")
        raise WeatherBackfillError(
            f"{len(failures)} backfill operations failed. "
            "Successful cached/uploaded objects are preserved; rerun to resume."
        )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover exact weather cell-hours missing from current CrimeNet "
            "integration coverage, export CSV manifests, fetch Open-Meteo "
            "ERA5-Land, and publish schema-compatible raw JSON back to landing."
        )
    )

    parser.add_argument(
        "--integration-root",
        default=INTEGRATION_ROOT,
    )
    parser.add_argument(
        "--land-weather-root",
        default=LAND_WEATHER_ROOT,
    )
    parser.add_argument(
        "--coastal-weather-root",
        default=COASTAL_WEATHER_ROOT,
    )
    parser.add_argument(
        "--landing-root",
        default=RAW_LANDING_ROOT,
        help=(
            "Raw Open-Meteo root. Accepts gs://... or a local/FUSE directory. "
            "Objects are written below era5_land/year=YYYY/."
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/weather/backfill"),
    )
    parser.add_argument(
        "--cache-directory",
        type=Path,
        default=Path(".cache/weather_backfill"),
    )
    parser.add_argument(
        "--availability-lag-days",
        type=int,
        default=DEFAULT_AVAILABILITY_LAG_DAYS,
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Download/cache responses but do not publish them to landing.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Ignore valid local cache files and refetch Open-Meteo.",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Limit the number of requests for a smoke test.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
    )
    parser.add_argument(
        "--retry-base-seconds",
        type=float,
        default=DEFAULT_RETRY_BASE_SECONDS,
    )
    parser.add_argument(
        "--retry-max-seconds",
        type=float,
        default=DEFAULT_RETRY_MAX_SECONDS,
    )
    parser.add_argument(
        "--retry-jitter-seconds",
        type=float,
        default=DEFAULT_RETRY_JITTER_SECONDS,
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.availability_lag_days < 0:
        raise ValueError("--availability-lag-days cannot be negative")
    if args.max_requests is not None and args.max_requests <= 0:
        raise ValueError("--max-requests must be positive")
    if args.pause_seconds < 0:
        raise ValueError("--pause-seconds cannot be negative")
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts must be positive")
    if args.retry_base_seconds < 0:
        raise ValueError("--retry-base-seconds cannot be negative")
    if args.retry_max_seconds < 0:
        raise ValueError("--retry-max-seconds cannot be negative")
    if args.retry_jitter_seconds < 0:
        raise ValueError("--retry-jitter-seconds cannot be negative")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = parse_args()
    validate_args(args)

    credentials = pl.CredentialProviderGCP()
    hourly_variables = normalized_hourly_variables()
    availability_cutoff = date.today() - timedelta(
        days=args.availability_lag_days
    )

    print("Discovering exact missing effective-weather keys...")
    missing_hours = discover_missing_weather_hours(
        integration_root=args.integration_root,
        land_weather_root=args.land_weather_root,
        coastal_weather_root=args.coastal_weather_root,
        credentials=credentials,
    )

    if missing_hours.is_empty():
        print("No required weather cell-hours are missing.")
        return

    request_manifest = build_request_manifest(
        missing_hours,
        availability_cutoff=availability_cutoff,
        hourly_variables=hourly_variables,
        landing_root=args.landing_root,
    )

    output_directory = args.output_directory.expanduser().resolve()
    missing_path, request_path = write_manifests(
        missing_hours=missing_hours,
        requests_manifest=request_manifest,
        output_directory=output_directory,
    )

    print()
    print(f"Missing cell-hours:      {missing_hours.height:,}")
    print(
        "Affected H3 cells:       "
        f"{missing_hours['weather_query_cell_id'].n_unique():,}"
    )
    print(f"Cell/year API requests:  {request_manifest.height:,}")
    print(
        "Affected integration samples: "
        f"{missing_hours['affected_integration_samples'].sum():,}"
    )
    print(f"Missing-hours CSV:       {missing_path}")
    print(f"Request manifest CSV:    {request_path}")

    if args.manifest_only:
        print("\nManifest-only mode; no API requests or uploads were made.")
        return

    execute_backfill(
        requests_manifest=request_manifest,
        missing_hours=missing_hours,
        cache_directory=args.cache_directory.expanduser().resolve(),
        landing_root=args.landing_root,
        hourly_variables=hourly_variables,
        maximum_requests=args.max_requests,
        pause_seconds=args.pause_seconds,
        maximum_attempts=args.max_attempts,
        retry_base_seconds=args.retry_base_seconds,
        retry_max_seconds=args.retry_max_seconds,
        retry_jitter_seconds=args.retry_jitter_seconds,
        force_download=args.force_download,
        no_upload=args.no_upload,
    )

    print()
    print("Next: rematerialize bronze_weather_land, then the land weather Silver asset.")
    print("Finally rerun this script with --manifest-only to verify the gap closed.")


if __name__ == "__main__":
    main()
