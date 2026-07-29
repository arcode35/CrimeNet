from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from crimenet.weather.open_meteo_client import (
    WeatherRequest,
    expected_gmt_hourly_timestamps,
    expected_weather_request_id,
    validate_weather_request,
)

logger = logging.getLogger(__name__)


class WeatherCacheError(RuntimeError):
    """Raised when an existing weather cache entry is unreadable or invalid."""


def get_weather_cache_path(
    cache_directory: str | Path,
    request: WeatherRequest,
) -> Path:
    validate_weather_request(request)
    expected_request_id = expected_weather_request_id(request)
    if request.request_id != expected_request_id:
        raise ValueError(
            "Cannot build a cache path for a non-deterministic request_id"
        )

    return (
        Path(cache_directory)
        / request.model
        / f"year={request.start_date.year}"
        / f"{request.request_id}.json"
    )


def read_weather_cache(
    *,
    cache_path: Path,
    request: WeatherRequest,
) -> dict[str, Any] | None:
    """Load a cache hit only after its JSON and request metadata validate."""

    validate_weather_request(request)

    try:
        with cache_path.open(mode="r", encoding="utf-8") as file:
            response = json.load(file)
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WeatherCacheError(
            "Weather cache entry is unreadable: "
            f"request_id={request.request_id}, "
            f"error_type={type(exc).__name__}"
        ) from exc

    if not isinstance(response, dict):
        raise WeatherCacheError(
            "Weather cache entry must contain a JSON object: "
            f"request_id={request.request_id}"
        )

    _validate_cached_response(
        request=request,
        response=response,
    )
    return response


def write_weather_cache(
    *,
    cache_path: Path,
    response: dict[str, Any],
) -> None:
    """Atomically replace one cache entry with durable JSON."""

    if not isinstance(response, dict):
        raise TypeError("response must be a dictionary")

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = cache_path.with_name(
        f".{cache_path.name}.{uuid4().hex}.tmp"
    )

    try:
        with temporary_path.open(
            mode="x",
            encoding="utf-8",
        ) as file:
            json.dump(
                response,
                file,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

        os.replace(
            temporary_path,
            cache_path,
        )
        _fsync_directory(cache_path.parent)
    finally:
        temporary_path.unlink(
            missing_ok=True,
        )


def _validate_cached_response(
    *,
    request: WeatherRequest,
    response: Mapping[str, Any],
) -> None:
    expected_metadata: dict[str, object] = {
        "request_id": request.request_id,
        "provider": request.provider,
        "model": request.model,
        "weather_query_cell_id": request.weather_query_cell_id,
        "h3_resolution": request.h3_resolution,
        "query_latitude": request.latitude,
        "query_longitude": request.longitude,
        "start_date": request.start_date.isoformat(),
        "end_date": request.end_date.isoformat(),
        "timezone": request.timezone,
        "cell_selection": request.cell_selection,
        "hourly_variables": list(request.hourly_variables),
    }

    mismatched_fields = [
        field_name
        for field_name, expected_value in expected_metadata.items()
        if response.get(field_name) != expected_value
    ]
    if mismatched_fields:
        raise WeatherCacheError(
            "Weather cache metadata does not match its request: "
            f"request_id={request.request_id}, "
            f"fields={','.join(sorted(mismatched_fields))}"
        )

    hourly = response.get("hourly")
    if not isinstance(hourly, Mapping):
        raise WeatherCacheError(
            "Weather cache entry has no hourly data: "
            f"request_id={request.request_id}"
        )

    timestamps = hourly.get("time")
    if not isinstance(timestamps, list) or not timestamps:
        raise WeatherCacheError(
            "Weather cache entry has no hourly timestamps: "
            f"request_id={request.request_id}"
        )
    if any(
        not isinstance(timestamp, str) or not timestamp.strip()
        for timestamp in timestamps
    ):
        raise WeatherCacheError(
            "Weather cache entry has invalid hourly timestamps: "
            f"request_id={request.request_id}"
        )

    expected_timestamps = expected_gmt_hourly_timestamps(request)
    if expected_timestamps is not None and tuple(timestamps) != expected_timestamps:
        raise WeatherCacheError(
            "Weather cache entry has an incomplete, duplicated, or unordered "
            "GMT hourly range: "
            f"expected={len(expected_timestamps)}, "
            f"actual={len(timestamps)}, "
            f"request_id={request.request_id}"
        )

    for variable in request.hourly_variables:
        values = hourly.get(variable)
        if not isinstance(values, list):
            raise WeatherCacheError(
                "Weather cache entry is missing requested variable "
                f"{variable!r}: request_id={request.request_id}"
            )
        if len(values) != len(timestamps):
            raise WeatherCacheError(
                "Weather cache hourly array length mismatch "
                f"for {variable!r}: request_id={request.request_id}"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ):
            raise WeatherCacheError(
                "Weather cache entry contains an invalid hourly value "
                f"for {variable!r}: request_id={request.request_id}"
            )


def _fsync_directory(directory: Path) -> None:
    try:
        directory_descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        logger.debug(
            "Directory fsync is unavailable for weather cache path: %s",
            directory,
        )
        return

    try:
        os.fsync(directory_descriptor)
    except OSError:
        logger.debug(
            "Directory fsync is unsupported for weather cache path: %s",
            directory,
        )
    finally:
        os.close(directory_descriptor)
