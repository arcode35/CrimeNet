"""Execute Open-Meteo requests from a Spark weather manifest."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from pyspark.sql import DataFrame

from crimenet.weather.cache import (
    get_weather_cache_path,
    is_valid_weather_cache,
    write_weather_cache,
)
from crimenet.weather.open_meteo_client import (
    fetch_historical_weather,
)
from crimenet.weather.request_mapping import (
    weather_request_from_manifest_row,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class WeatherIngestionSummary:
    """Summary of one weather-manifest execution."""

    processed: int
    attempted: int
    downloaded: int
    cached: int
    failed: int
    failures: tuple[str, ...]


def fetch_weather_manifest(
    weather_request_manifest: DataFrame,
    *,
    cache_directory: str | Path,
    force: bool = False,
    maximum_requests: int | None = None,
    pause_seconds: float = 0.25,
) -> WeatherIngestionSummary:
    """Fetch and cache missing responses from a weather request manifest.

    Existing files are skipped only when they pass cache validation.

    Args:
        weather_request_manifest:
            Spark DataFrame containing one row per Open-Meteo request.

        cache_directory:
            Root directory for raw responses. This may be a local path or
            a Unity Catalog volume path such as:

            /Volumes/crimenet_dev/raw_files/landing/weather/open_meteo

        force:
            Redownload every examined request, even when a valid cached
            response already exists.

        maximum_requests:
            Maximum number of actual HTTP requests to attempt. Cached
            responses do not count toward this limit.

        pause_seconds:
            Delay after each attempted HTTP request.

    Returns:
        Counts and failure information for the execution.
    """
    if (
        maximum_requests is not None
        and maximum_requests <= 0
    ):
        raise ValueError(
            "maximum_requests must be positive"
        )

    if pause_seconds < 0:
        raise ValueError(
            "pause_seconds cannot be negative"
        )

    resolved_cache_directory = Path(
        cache_directory
    )

    processed_count = 0
    attempted_count = 0
    downloaded_count = 0
    cached_count = 0
    failed_count = 0
    failures: list[str] = []

    LOGGER.info(
        "Starting weather manifest ingestion: "
        "cache_directory=%s, force=%s, "
        "maximum_requests=%s, pause_seconds=%s",
        resolved_cache_directory,
        force,
        maximum_requests,
        pause_seconds,
    )

    rows = (
        weather_request_manifest
        .toLocalIterator()
    )

    LOGGER.info(
        "Spark iterator initialized; "
        "waiting for weather manifest rows"
    )

    for row in rows:
        request = (
            weather_request_from_manifest_row(
                row.asDict(recursive=True)
            )
        )

        cache_path = get_weather_cache_path(
            resolved_cache_directory,
            request,
        )

        if (
            not force
            and is_valid_weather_cache(
                cache_path=cache_path,
                request=request,
            )
        ):
            processed_count += 1
            cached_count += 1

            LOGGER.info(
                "Skipping valid cached weather request: "
                "request_id=%s, cell=%s, "
                "start=%s, end=%s, path=%s",
                request.request_id,
                request.weather_query_cell_id,
                request.start_date,
                request.end_date,
                cache_path,
            )

            continue

        if (
            maximum_requests is not None
            and attempted_count
            >= maximum_requests
        ):
            LOGGER.info(
                "Reached weather request limit: "
                "maximum_requests=%d",
                maximum_requests,
            )
            break

        processed_count += 1
        attempted_count += 1

        LOGGER.info(
            "Processing weather request: "
            "processed=%d, attempted=%d, "
            "request_id=%s, cell=%s, "
            "start=%s, end=%s",
            processed_count,
            attempted_count,
            request.request_id,
            request.weather_query_cell_id,
            request.start_date,
            request.end_date,
        )

        try:
            LOGGER.info(
                "Calling Open-Meteo: "
                "request_id=%s",
                request.request_id,
            )

            response = fetch_historical_weather(
                request
            )

            LOGGER.info(
                "Open-Meteo response received: "
                "request_id=%s",
                request.request_id,
            )

            write_weather_cache(
                cache_path=cache_path,
                response=response,
            )

            downloaded_count += 1

            LOGGER.info(
                "Cached weather request: "
                "request_id=%s, cell=%s, "
                "hours=%d, path=%s",
                request.request_id,
                request.weather_query_cell_id,
                len(response["hourly"]["time"]),
                cache_path,
            )

        except Exception as exc:
            failed_count += 1

            failure_message = (
                f"request_id={request.request_id}, "
                f"cell={request.weather_query_cell_id}, "
                f"start={request.start_date}, "
                f"end={request.end_date}, "
                f"error={type(exc).__name__}: {exc}"
            )

            failures.append(
                failure_message
            )

            LOGGER.exception(
                "Weather request failed: "
                "request_id=%s, cell=%s, "
                "start=%s, end=%s",
                request.request_id,
                request.weather_query_cell_id,
                request.start_date,
                request.end_date,
            )

        finally:
            if pause_seconds > 0:
                time.sleep(
                    pause_seconds
                )

    summary = WeatherIngestionSummary(
        processed=processed_count,
        attempted=attempted_count,
        downloaded=downloaded_count,
        cached=cached_count,
        failed=failed_count,
        failures=tuple(failures),
    )

    LOGGER.info(
        "Weather ingestion finished: "
        "processed=%d, attempted=%d, "
        "downloaded=%d, already_cached=%d, "
        "failed=%d",
        summary.processed,
        summary.attempted,
        summary.downloaded,
        summary.cached,
        summary.failed,
    )

    return summary