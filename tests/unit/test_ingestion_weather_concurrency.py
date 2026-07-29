from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest

from crimenet.weather.open_meteo_client import (
    OpenMeteoClientConfig,
    WeatherFetchError,
    WeatherRequest,
    build_weather_request_id,
)
from crimenet.weather.weather_ingestion import fetch_weather_manifest


class _Row:
    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values

    def asDict(self, *, recursive: bool) -> dict[str, Any]:
        assert recursive
        return dict(self.values)


class _Manifest:
    def __init__(self, rows: list[_Row]) -> None:
        self.rows = rows

    def toLocalIterator(self) -> object:
        return iter(self.rows)


def _manifest_values(cell: int) -> dict[str, Any]:
    values: dict[str, Any] = {
        "provider": "open_meteo",
        "model": "era5_land",
        "weather_query_cell_id": cell,
        "h3_resolution": 6,
        "query_latitude": 32.75,
        "query_longitude": -97.33,
        "start_date": date(2023, 1, 1),
        "end_date": date(2023, 1, 1),
        "timezone": "GMT",
        "cell_selection": "nearest",
        "hourly_variables": ["temperature_2m"],
        "crime_record_count": 1,
    }
    values["request_id"] = build_weather_request_id(
        provider=values["provider"],
        model=values["model"],
        weather_query_cell_id=cell,
        start_date=values["start_date"],
        end_date=values["end_date"],
        hourly_variables=values["hourly_variables"],
        timezone=values["timezone"],
        cell_selection=values["cell_selection"],
        h3_resolution=values["h3_resolution"],
    )
    return values


def _response(request: WeatherRequest) -> dict[str, Any]:
    timestamps = [
        f"{request.start_date.isoformat()}T{hour:02d}:00"
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
        "grid_latitude": request.latitude,
        "grid_longitude": request.longitude,
        "grid_elevation": 100.0,
        "start_date": request.start_date.isoformat(),
        "end_date": request.end_date.isoformat(),
        "timezone": request.timezone,
        "utc_offset_seconds": 0,
        "cell_selection": request.cell_selection,
        "hourly_variables": list(request.hourly_variables),
        "hourly_units": {"temperature_2m": "°C"},
        "hourly": {
            "time": timestamps,
            "temperature_2m": [10.0] * len(timestamps),
        },
    }


class _ConcurrentClient:
    def __init__(self, *, fail_cell: int | None = None) -> None:
        self.config = OpenMeteoClientConfig(max_concurrent_requests=2)
        self.fail_cell = fail_cell
        self.requests: list[int] = []

    def fetch_historical_weather(
        self,
        request: WeatherRequest,
    ) -> dict[str, Any]:
        self.requests.append(request.weather_query_cell_id)
        if request.weather_query_cell_id == self.fail_cell:
            raise WeatherFetchError(
                "simulated concurrent failure",
                request_id=request.request_id,
                category="http",
                retryable=True,
                status_code=503,
            )
        return _response(request)


def test_manifest_uses_bounded_concurrency_and_caches_every_result(
    tmp_path: Path,
) -> None:
    client = _ConcurrentClient()
    fetch_weather_manifest(
        _Manifest(
            [
                _Row(_manifest_values(111)),
                _Row(_manifest_values(222)),
                _Row(_manifest_values(333)),
            ]
        ),  # type: ignore[arg-type]
        cache_directory=tmp_path,
        client=client,  # type: ignore[arg-type]
    )
    assert sorted(client.requests) == [111, 222, 333]
    assert len(list(tmp_path.rglob("*.json"))) == 3


def test_concurrent_failure_is_propagated_after_executor_cleanup(
    tmp_path: Path,
) -> None:
    client = _ConcurrentClient(fail_cell=222)
    with pytest.raises(WeatherFetchError, match="concurrent failure"):
        fetch_weather_manifest(
            _Manifest(
                [
                    _Row(_manifest_values(111)),
                    _Row(_manifest_values(222)),
                    _Row(_manifest_values(333)),
                ]
            ),  # type: ignore[arg-type]
            cache_directory=tmp_path,
            client=client,  # type: ignore[arg-type]
        )
    assert 222 in client.requests
