from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest

from crimenet.weather.cache import (
    get_weather_cache_path,
    read_weather_cache,
    write_weather_cache,
)
from crimenet.weather.open_meteo_client import (
    WeatherFetchError,
    WeatherRequest,
    build_weather_request_id,
)
from crimenet.weather.request_mapping import (
    weather_request_from_manifest_row,
)
from crimenet.weather.weather_ingestion import (
    WeatherIngestionAuditEvent,
    fetch_weather_manifest,
)


def _manifest_values() -> dict[str, Any]:
    values: dict[str, Any] = {
        "provider": "open_meteo",
        "model": "era5_land",
        "weather_query_cell_id": 222333444,
        "h3_resolution": 6,
        "query_latitude": 32.75,
        "query_longitude": -97.33,
        "start_date": date(2022, 1, 1),
        "end_date": date(2022, 1, 1),
        "timezone": "GMT",
        "cell_selection": "nearest",
        "hourly_variables": ["temperature_2m"],
        "crime_record_count": 4,
    }
    values["request_id"] = build_weather_request_id(
        provider=values["provider"],
        model=values["model"],
        weather_query_cell_id=values[
            "weather_query_cell_id"
        ],
        start_date=values["start_date"],
        end_date=values["end_date"],
        hourly_variables=values["hourly_variables"],
        timezone=values["timezone"],
        cell_selection=values["cell_selection"],
        h3_resolution=values["h3_resolution"],
    )
    return values


def _cached_response(
    request: WeatherRequest,
) -> dict[str, Any]:
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
        "grid_elevation": 200.0,
        "start_date": request.start_date.isoformat(),
        "end_date": request.end_date.isoformat(),
        "timezone": request.timezone,
        "utc_offset_seconds": 0,
        "cell_selection": request.cell_selection,
        "hourly_variables": list(request.hourly_variables),
        "hourly_units": {"temperature_2m": "°C"},
        "hourly": {
            "time": timestamps,
            "temperature_2m": [3.5] * len(timestamps),
        },
    }


class _ManifestRow:
    def __init__(self, values: dict[str, Any]) -> None:
        self._values = values

    def asDict(
        self,
        *,
        recursive: bool,
    ) -> dict[str, Any]:
        assert recursive is True
        return dict(self._values)


class _Manifest:
    def __init__(self, *rows: _ManifestRow) -> None:
        self._rows = rows

    def toLocalIterator(self) -> object:
        return iter(self._rows)


class _StubClient:
    def __init__(
        self,
        *,
        response: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.requests: list[WeatherRequest] = []

    def fetch_historical_weather(
        self,
        request: WeatherRequest,
    ) -> dict[str, Any]:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


def test_existing_valid_cache_is_reused(
    tmp_path: Path,
) -> None:
    values = _manifest_values()
    request = weather_request_from_manifest_row(values)
    cache_path = get_weather_cache_path(
        tmp_path,
        request,
    )
    response = _cached_response(request)
    write_weather_cache(
        cache_path=cache_path,
        response=response,
    )
    client = _StubClient(response=response)

    fetch_weather_manifest(
        _Manifest(_ManifestRow(values)),
        cache_directory=tmp_path,
        client=client,
    )

    assert client.requests == []
    assert read_weather_cache(
        cache_path=cache_path,
        request=request,
    ) == response


def test_invalid_cache_is_audited_and_refetched(
    tmp_path: Path,
) -> None:
    values = _manifest_values()
    request = weather_request_from_manifest_row(values)
    cache_path = get_weather_cache_path(
        tmp_path,
        request,
    )
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("{partial", encoding="utf-8")
    response = _cached_response(request)
    client = _StubClient(response=response)
    audit_events: list[WeatherIngestionAuditEvent] = []

    fetch_weather_manifest(
        _Manifest(_ManifestRow(values)),
        cache_directory=tmp_path,
        client=client,
        audit_hook=audit_events.append,
    )

    assert client.requests == [request]
    assert [event.event_type for event in audit_events] == [
        "invalid_cache"
    ]
    assert read_weather_cache(
        cache_path=cache_path,
        request=request,
    ) == response


def test_failed_request_emits_structured_audit_event(
    tmp_path: Path,
) -> None:
    values = _manifest_values()
    failure = WeatherFetchError(
        "Open-Meteo returned HTTP 500",
        request_id=values["request_id"],
        category="http",
        retryable=True,
        status_code=500,
    )
    client = _StubClient(error=failure)
    audit_events: list[WeatherIngestionAuditEvent] = []

    with pytest.raises(
        WeatherFetchError,
        match="HTTP 500",
    ):
        fetch_weather_manifest(
            _Manifest(_ManifestRow(values)),
            cache_directory=tmp_path,
            client=client,
            audit_hook=audit_events.append,
        )

    assert len(audit_events) == 1
    event = audit_events[0]
    assert event.event_type == "request_failed"
    assert event.request_id == values["request_id"]
    assert event.error_type == "WeatherFetchError"
    assert event.error_category == "http"
    assert event.retryable is True
    assert event.status_code == 500
    assert event.occurred_at.tzinfo is not None


def test_manifest_rejects_tampered_request_identity() -> None:
    values = _manifest_values()
    values["request_id"] = "tampered"

    with pytest.raises(
        ValueError,
        match="deterministic identity",
    ):
        weather_request_from_manifest_row(values)
