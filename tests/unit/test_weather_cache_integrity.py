from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from crimenet.weather.cache import (
    WeatherCacheError,
    get_weather_cache_path,
    read_weather_cache,
    write_weather_cache,
)
from crimenet.weather.open_meteo_client import (
    WeatherRequest,
    build_weather_request_id,
)


def _weather_request() -> WeatherRequest:
    request_id = build_weather_request_id(
        provider="open_meteo",
        model="era5_land",
        weather_query_cell_id=987654321,
        start_date=date(2023, 1, 1),
        end_date=date(2023, 1, 1),
        hourly_variables=("temperature_2m",),
        timezone="GMT",
        cell_selection="nearest",
        h3_resolution=6,
    )
    return WeatherRequest(
        request_id=request_id,
        provider="open_meteo",
        model="era5_land",
        weather_query_cell_id=987654321,
        h3_resolution=6,
        latitude=29.7604,
        longitude=-95.3698,
        start_date=date(2023, 1, 1),
        end_date=date(2023, 1, 1),
        timezone="GMT",
        cell_selection="nearest",
        hourly_variables=("temperature_2m",),
    )


def _cached_response(
    request: WeatherRequest,
    *,
    temperature: float = 12.0,
) -> dict[str, object]:
    timestamps = [
        f"2023-01-01T{hour:02d}:00"
        for hour in range(24)
    ]
    return {
        "request_id": request.request_id,
        "provider": request.provider,
        "model": request.model,
        "weather_query_cell_id": request.weather_query_cell_id,
        "h3_resolution": request.h3_resolution,
        "query_latitude": request.latitude,
        "query_longitude": request.longitude,
        "grid_latitude": 29.75,
        "grid_longitude": -95.375,
        "grid_elevation": 15.0,
        "start_date": request.start_date.isoformat(),
        "end_date": request.end_date.isoformat(),
        "timezone": request.timezone,
        "utc_offset_seconds": 0,
        "cell_selection": request.cell_selection,
        "hourly_variables": list(request.hourly_variables),
        "hourly_units": {"temperature_2m": "°C"},
        "hourly": {
            "time": timestamps,
            "temperature_2m": [temperature] * len(timestamps),
        },
    }


def test_valid_cache_round_trip(tmp_path: Path) -> None:
    request = _weather_request()
    cache_path = get_weather_cache_path(
        tmp_path,
        request,
    )
    response = _cached_response(request)

    write_weather_cache(
        cache_path=cache_path,
        response=response,
    )

    assert read_weather_cache(
        cache_path=cache_path,
        request=request,
    ) == response


def test_missing_cache_is_a_miss(tmp_path: Path) -> None:
    request = _weather_request()

    assert (
        read_weather_cache(
            cache_path=get_weather_cache_path(
                tmp_path,
                request,
            ),
            request=request,
        )
        is None
    )


def test_partial_json_cache_is_not_a_hit(
    tmp_path: Path,
) -> None:
    request = _weather_request()
    cache_path = get_weather_cache_path(
        tmp_path,
        request,
    )
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(
        '{"request_id":',
        encoding="utf-8",
    )

    with pytest.raises(
        WeatherCacheError,
        match="unreadable",
    ):
        read_weather_cache(
            cache_path=cache_path,
            request=request,
        )


def test_cache_with_wrong_request_metadata_is_not_a_hit(
    tmp_path: Path,
) -> None:
    request = _weather_request()
    cache_path = get_weather_cache_path(
        tmp_path,
        request,
    )
    response = _cached_response(request)
    response["model"] = "era5"
    write_weather_cache(
        cache_path=cache_path,
        response=response,
    )

    with pytest.raises(
        WeatherCacheError,
        match="metadata does not match",
    ):
        read_weather_cache(
            cache_path=cache_path,
            request=request,
        )


def test_cache_with_truncated_hourly_array_is_not_a_hit(
    tmp_path: Path,
) -> None:
    request = _weather_request()
    cache_path = get_weather_cache_path(
        tmp_path,
        request,
    )
    response = _cached_response(request)
    hourly = response["hourly"]
    assert isinstance(hourly, dict)
    hourly["temperature_2m"] = [12.0] * 23
    write_weather_cache(
        cache_path=cache_path,
        response=response,
    )

    with pytest.raises(
        WeatherCacheError,
        match="array length mismatch",
    ):
        read_weather_cache(
            cache_path=cache_path,
            request=request,
        )


def test_cache_with_partial_hourly_range_is_not_a_hit(
    tmp_path: Path,
) -> None:
    request = _weather_request()
    cache_path = get_weather_cache_path(tmp_path, request)
    response = _cached_response(request)
    hourly = response["hourly"]
    assert isinstance(hourly, dict)
    timestamps = hourly["time"]
    assert isinstance(timestamps, list)
    hourly["time"] = timestamps[:-1]
    hourly["temperature_2m"] = [12.0] * 23
    write_weather_cache(cache_path=cache_path, response=response)

    with pytest.raises(WeatherCacheError, match="GMT hourly range"):
        read_weather_cache(cache_path=cache_path, request=request)


def test_interrupted_replace_preserves_previous_cache(
    tmp_path: Path,
) -> None:
    request = _weather_request()
    cache_path = get_weather_cache_path(
        tmp_path,
        request,
    )
    original_response = _cached_response(
        request,
        temperature=12.0,
    )
    write_weather_cache(
        cache_path=cache_path,
        response=original_response,
    )
    original_bytes = cache_path.read_bytes()

    with (
        patch(
            "crimenet.weather.cache.os.replace",
            side_effect=OSError("interrupted"),
        ),
        pytest.raises(OSError, match="interrupted"),
    ):
        write_weather_cache(
            cache_path=cache_path,
            response=_cached_response(
                request,
                temperature=99.0,
            ),
        )

    assert cache_path.read_bytes() == original_bytes
    assert list(cache_path.parent.glob(".*.tmp")) == []


def test_cache_path_rejects_inconsistent_request_id(
    tmp_path: Path,
) -> None:
    request = replace(
        _weather_request(),
        request_id="not-the-deterministic-id",
    )

    with pytest.raises(
        ValueError,
        match="deterministic request identity",
    ):
        get_weather_cache_path(
            tmp_path,
            request,
        )
