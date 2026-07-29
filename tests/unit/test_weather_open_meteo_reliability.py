from __future__ import annotations

import json
import threading
from datetime import date
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import Mock, patch

import pytest
from requests import Response, Session
from requests.exceptions import (
    ConnectionError as RequestsConnectionError,
)
from requests.exceptions import Timeout

from crimenet.weather.open_meteo_client import (
    OpenMeteoClient,
    OpenMeteoClientConfig,
    WeatherFetchError,
    WeatherRequest,
    build_open_meteo_retry_policy,
    build_weather_request_id,
    fetch_historical_weather,
    is_retryable_http_status,
)


def _weather_request() -> WeatherRequest:
    request_values = {
        "provider": "open_meteo",
        "model": "era5_land",
        "weather_query_cell_id": 123456789,
        "h3_resolution": 6,
        "latitude": 32.7767,
        "longitude": -96.797,
        "start_date": date(2024, 1, 1),
        "end_date": date(2024, 1, 1),
        "timezone": "GMT",
        "cell_selection": "nearest",
        "hourly_variables": ("temperature_2m",),
    }
    request_id = build_weather_request_id(
        provider=request_values["provider"],
        model=request_values["model"],
        weather_query_cell_id=request_values[
            "weather_query_cell_id"
        ],
        start_date=request_values["start_date"],
        end_date=request_values["end_date"],
        hourly_variables=request_values["hourly_variables"],
        timezone=request_values["timezone"],
        cell_selection=request_values["cell_selection"],
        h3_resolution=request_values["h3_resolution"],
    )
    return WeatherRequest(
        request_id=request_id,
        **request_values,
    )


def _successful_payload() -> dict[str, object]:
    timestamps = [
        f"2024-01-01T{hour:02d}:00"
        for hour in range(24)
    ]
    return {
        "latitude": 32.75,
        "longitude": -96.75,
        "elevation": 130.0,
        "timezone": "GMT",
        "utc_offset_seconds": 0,
        "hourly_units": {"temperature_2m": "°C"},
        "hourly": {
            "time": timestamps,
            "temperature_2m": [8.5] * len(timestamps),
        },
    }


def _response(
    status_code: int,
    payload: object,
) -> Response:
    response = Response()
    response.status_code = status_code
    response.url = "https://archive-api.open-meteo.com/v1/archive"
    response.headers["Content-Type"] = "application/json"
    response._content = json.dumps(payload).encode("utf-8")
    return response


def test_request_id_is_stable_across_variable_order() -> None:
    common_values = {
        "provider": "open_meteo",
        "model": "era5_land",
        "weather_query_cell_id": 123,
        "start_date": date(2020, 1, 1),
        "end_date": date(2020, 12, 31),
        "timezone": "GMT",
        "cell_selection": "nearest",
        "h3_resolution": 6,
    }

    first = build_weather_request_id(
        **common_values,
        hourly_variables=("wind_speed_10m", "temperature_2m"),
    )
    second = build_weather_request_id(
        **common_values,
        hourly_variables=(" temperature_2m ", "wind_speed_10m"),
    )

    assert first == second
    assert len(first) == 64


def test_retry_policy_classifies_only_safe_get_failures() -> None:
    config = OpenMeteoClientConfig(
        max_retries=3,
        retryable_status_codes=(429, 500),
    )
    retry_policy = build_open_meteo_retry_policy(config)

    assert retry_policy.total == 3
    assert retry_policy.connect == 3
    assert retry_policy.read == 3
    assert retry_policy.status == 3
    assert retry_policy.respect_retry_after_header is True
    assert retry_policy.is_retry("GET", 429)
    assert retry_policy.is_retry("GET", 500)
    assert not retry_policy.is_retry("GET", 400)
    assert not retry_policy.is_retry("POST", 500)


def test_timeout_is_reported_as_retryable() -> None:
    session = Mock(spec=Session)
    session.get.side_effect = Timeout("sensitive URL must not be logged")

    with pytest.raises(WeatherFetchError) as error:
        fetch_historical_weather(
            _weather_request(),
            session=session,
        )

    assert error.value.category == "timeout"
    assert error.value.retryable is True
    assert "sensitive URL" not in str(error.value)
    session.close.assert_not_called()


def test_connection_failure_is_reported_as_retryable() -> None:
    session = Mock(spec=Session)
    session.get.side_effect = RequestsConnectionError(
        "sensitive URL must not be logged"
    )

    with pytest.raises(WeatherFetchError) as error:
        fetch_historical_weather(
            _weather_request(),
            session=session,
        )

    assert error.value.category == "connection"
    assert error.value.retryable is True
    assert "sensitive URL" not in str(error.value)


@pytest.mark.parametrize(
    ("status_code", "expected_retryable"),
    [
        (429, True),
        (500, True),
        (400, False),
    ],
)
def test_http_failure_classification(
    status_code: int,
    expected_retryable: bool,
) -> None:
    session = Mock(spec=Session)
    session.get.return_value = _response(
        status_code,
        {"reason": "test reason"},
    )

    with pytest.raises(WeatherFetchError) as error:
        fetch_historical_weather(
            _weather_request(),
            session=session,
        )

    assert error.value.category == "http"
    assert error.value.status_code == status_code
    assert error.value.retryable is expected_retryable
    assert is_retryable_http_status(status_code) is expected_retryable


def test_invalid_json_has_a_clear_error() -> None:
    response = Response()
    response.status_code = 200
    response._content = b"{invalid"
    session = Mock(spec=Session)
    session.get.return_value = response

    with pytest.raises(
        WeatherFetchError,
        match="invalid JSON",
    ) as error:
        fetch_historical_weather(
            _weather_request(),
            session=session,
        )

    assert error.value.category == "invalid_json"


def test_empty_response_is_rejected() -> None:
    response = Response()
    response.status_code = 200
    response._content = b""
    session = Mock(spec=Session)
    session.get.return_value = response

    with pytest.raises(
        WeatherFetchError,
        match="empty response",
    ) as error:
        fetch_historical_weather(
            _weather_request(),
            session=session,
        )

    assert error.value.category == "empty_response"


def test_empty_hourly_response_is_rejected() -> None:
    session = Mock(spec=Session)
    session.get.return_value = _response(
        200,
        {
            "timezone": "GMT",
            "hourly": {
                "time": [],
                "temperature_2m": [],
            },
        },
    )

    with pytest.raises(
        WeatherFetchError,
        match="no hourly timestamps",
    ):
        fetch_historical_weather(
            _weather_request(),
            session=session,
        )


def test_created_session_is_closed_and_timeout_is_configurable() -> None:
    session = Mock(spec=Session)
    session.get.return_value = _response(
        200,
        _successful_payload(),
    )
    config = OpenMeteoClientConfig(
        connect_timeout_seconds=2.5,
        read_timeout_seconds=9.0,
    )

    with patch(
        "crimenet.weather.open_meteo_client._build_open_meteo_session",
        return_value=session,
    ):
        result = fetch_historical_weather(
            _weather_request(),
            config=config,
        )

    assert result["hourly"]["temperature_2m"] == [8.5] * 24
    assert session.get.call_args.kwargs["timeout"] == (2.5, 9.0)
    session.close.assert_called_once_with()


def test_minimum_request_interval_is_enforced() -> None:
    session = Mock(spec=Session)
    session.get.return_value = _response(
        200,
        _successful_payload(),
    )
    clock_values = iter((0.0, 0.0, 0.25, 1.0))
    sleep = Mock()
    client = OpenMeteoClient(
        config=OpenMeteoClientConfig(
            minimum_request_interval_seconds=1.0,
        ),
        session=session,
        monotonic=lambda: next(clock_values),
        sleep=sleep,
    )

    client.fetch_historical_weather(_weather_request())
    client.fetch_historical_weather(_weather_request())

    sleep.assert_called_once_with(0.75)
    assert session.get.call_count == 2


class _RetryOnceHandler(BaseHTTPRequestHandler):
    attempts = 0

    def do_GET(self) -> None:
        type(self).attempts += 1
        if type(self).attempts == 1:
            status_code = 500
            payload: object = {"reason": "temporary"}
        else:
            status_code = 200
            payload = _successful_payload()

        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(
        self,
        format: str,
        *args: object,
    ) -> None:
        return


def test_retryable_http_error_succeeds_on_retry() -> None:
    _RetryOnceHandler.attempts = 0
    server = HTTPServer(
        ("127.0.0.1", 0),
        _RetryOnceHandler,
    )
    server_thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
    )
    server_thread.start()

    try:
        host, port = server.server_address
        config = OpenMeteoClientConfig(
            archive_url=f"http://{host}:{port}/archive",
            max_retries=1,
            retry_backoff_factor=0,
        )
        with OpenMeteoClient(config=config) as client:
            result = client.fetch_historical_weather(
                _weather_request()
            )
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)

    assert _RetryOnceHandler.attempts == 2
    assert result["hourly"]["temperature_2m"] == [8.5] * 24


def test_partial_gmt_hourly_range_is_rejected() -> None:
    payload = _successful_payload()
    hourly = payload["hourly"]
    assert isinstance(hourly, dict)
    hourly["time"] = ["2024-01-01T00:00"]
    hourly["temperature_2m"] = [8.5]
    session = Mock(spec=Session)
    session.get.return_value = _response(200, payload)

    with pytest.raises(
        WeatherFetchError,
        match="GMT hourly range",
    ) as error:
        fetch_historical_weather(
            _weather_request(),
            session=session,
        )

    assert error.value.category == "invalid_response"
    assert error.value.retryable is True
