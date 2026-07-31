"""Persistent cache operations for raw Open-Meteo responses."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from crimenet.weather.open_meteo_client import (
    WeatherFetchError,
    WeatherRequest,
    validate_hourly_response,
)


def get_weather_cache_path(
    cache_directory: str | Path,
    request: WeatherRequest,
) -> Path:
    """Return the deterministic raw-output path for a weather request."""
    return (
        Path(cache_directory)
        / request.model
        / f"year={request.start_date.year}"
        / f"{request.request_id}.json"
    )


def is_valid_weather_cache(
    *,
    cache_path: Path,
    request: WeatherRequest,
) -> bool:
    """Return whether an existing cache file fully satisfies a request.

    A file is considered valid only when it:

    - exists as a regular file;
    - contains valid JSON;
    - matches the deterministic request identity;
    - contains the expected provider, model, cell, and date range;
    - contains complete hourly arrays for every requested variable.
    """
    if not cache_path.is_file():
        return False

    try:
        with cache_path.open(
            mode="r",
            encoding="utf-8",
        ) as file:
            payload = json.load(file)

        if not isinstance(payload, dict):
            return False

        if payload.get("request_id") != request.request_id:
            return False

        if payload.get("provider") != request.provider:
            return False

        if payload.get("model") != request.model:
            return False

        if (
            int(payload["weather_query_cell_id"])
            != request.weather_query_cell_id
        ):
            return False

        if (
            int(payload["h3_resolution"])
            != request.h3_resolution
        ):
            return False

        if payload.get("start_date") != (
            request.start_date.isoformat()
        ):
            return False

        if payload.get("end_date") != (
            request.end_date.isoformat()
        ):
            return False

        if payload.get("cell_selection") != (
            request.cell_selection
        ):
            return False

        cached_variables = tuple(
            str(variable)
            for variable in payload.get(
                "hourly_variables",
                [],
            )
        )

        if cached_variables != request.hourly_variables:
            return False

        validate_hourly_response(
            request=request,
            payload=payload,
        )

        return True

    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        WeatherFetchError,
    ):
        return False


def write_weather_cache(
    *,
    cache_path: Path,
    response: dict[str, Any],
) -> None:
    """Atomically persist a validated raw weather response."""
    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = cache_path.with_name(
        f".{cache_path.name}.{uuid4().hex}.tmp"
    )

    try:
        with temporary_path.open(
            mode="w",
            encoding="utf-8",
        ) as file:
            json.dump(
                response,
                file,
                ensure_ascii=False,
                separators=(",", ":"),
            )

            file.flush()
            os.fsync(file.fileno())

        os.replace(
            temporary_path,
            cache_path,
        )

    finally:
        temporary_path.unlink(
            missing_ok=True,
        )