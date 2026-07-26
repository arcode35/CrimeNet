from __future__ import annotations

import logging
from pathlib import Path

from pyspark.sql import DataFrame

from crimenet.weather.cache import (
    get_weather_cache_path,
    write_weather_cache,
)
from crimenet.weather.open_meteo_client import (
    fetch_historical_weather,
)
from crimenet.weather.request_mapping import (
    weather_request_from_manifest_row,
)


logger = logging.getLogger(__name__)
def fetch_weather_manifest(
    weather_request_manifest: DataFrame,
    *,
    cache_directory: str | Path,
) -> None:
    cache_directory = Path(cache_directory)

    completed_count = 0
    cached_count = 0

    for row in (
        weather_request_manifest
        .toLocalIterator()
    ):
        request = (
            weather_request_from_manifest_row(
                row.asDict(recursive=True)
            )
        )

        cache_path = get_weather_cache_path(
            cache_directory,
            request,
        )

        if cache_path.exists():
            cached_count += 1

            logger.info(
                "Skipping cached weather request: "
                "request_id=%s, cell=%s, "
                "start=%s, end=%s",
                request.request_id,
                request.weather_query_cell_id,
                request.start_date,
                request.end_date,
            )
            continue

        try:
            response = fetch_historical_weather(
                request
            )

            write_weather_cache(
                cache_path=cache_path,
                response=response,
            )

            completed_count += 1

            hourly_count = len(
                response["hourly"]["time"]
            )

            logger.info(
                "Cached weather request: "
                "request_id=%s, cell=%s, "
                "hours=%d, path=%s",
                request.request_id,
                request.weather_query_cell_id,
                hourly_count,
                cache_path,
            )

        except Exception:
            logger.exception(
                "Weather request failed: "
                "request_id=%s, cell=%s, "
                "start=%s, end=%s",
                request.request_id,
                request.weather_query_cell_id,
                request.start_date,
                request.end_date,
            )
            raise

    logger.info(
        "Weather ingestion finished: "
        "downloaded=%d, already_cached=%d",
        completed_count,
        cached_count,
    )