"""open_meteo_client.py"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

import requests
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


def _build_open_meteo_session() -> Session:
    retry_policy = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        allowed_methods=frozenset({"GET"}),
        status_forcelist=(
            429,
            500,
            502,
            503,
            504,
        ),
        backoff_factor=1.0,
        respect_retry_after_header=True,
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry_policy,
        pool_connections=1,
        pool_maxsize=1,
    )

    session = Session()
    session.mount("https://", adapter)

    return session


OPEN_METEO_SESSION = _build_open_meteo_session()

OPEN_METEO_ARCHIVE_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
)

VALID_MODELS = frozenset(
    {
        "era5",
        "era5_land",
    }
)

VALID_CELL_SELECTIONS = frozenset(
    {
        "land",
        "sea",
        "nearest",
    }
)


@dataclass(frozen=True)
class WeatherRequest:
    request_id: str
    provider: str
    model: str

    weather_query_cell_id: int
    h3_resolution: int

    latitude: float
    longitude: float

    start_date: date
    end_date: date

    timezone: str
    cell_selection: str
    hourly_variables: tuple[str, ...]


class WeatherFetchError(RuntimeError):
    """Raised when historical weather retrieval fails."""

def fetch_historical_weather(
    request: WeatherRequest,
) -> dict[str, Any]:
    _validate_weather_request(request)

    params = {
        "latitude": request.latitude,
        "longitude": request.longitude,
        "start_date": request.start_date.isoformat(),
        "end_date": request.end_date.isoformat(),
        "hourly": ",".join(
            request.hourly_variables
        ),
        "models": request.model,
        "timezone": request.timezone,
        "cell_selection": request.cell_selection,
    }

    try:
        response = OPEN_METEO_SESSION.get(
            OPEN_METEO_ARCHIVE_URL,
            params=params,
            timeout=(30, 180),
        )
        response.raise_for_status()
    except requests.Timeout as exc:
        raise WeatherFetchError(
            "Open-Meteo request timed out: "
            f"request_id={request.request_id}"
        ) from exc
    except requests.HTTPError as exc:
        raise WeatherFetchError(
            _build_http_error_message(
                request_id=request.request_id,
                response=exc.response,
            )
        ) from exc
    except requests.RequestException as exc:
        raise WeatherFetchError(
            "Open-Meteo request failed: "
            f"request_id={request.request_id}, "
            f"error={exc}"
        ) from exc

    try:
        payload = response.json()
    except requests.JSONDecodeError as exc:
        raise WeatherFetchError(
            "Open-Meteo returned invalid JSON: "
            f"request_id={request.request_id}"
        ) from exc

    if payload.get("error") is True:
        raise WeatherFetchError(
            "Open-Meteo rejected the request: "
            f"request_id={request.request_id}, "
            f"reason={payload.get('reason')}"
        )

    validate_hourly_response(
        request=request,
        payload=payload,
    )

    return {
        "request_id": request.request_id,
        "provider": request.provider,
        "model": request.model,
        "weather_query_cell_id": (
            request.weather_query_cell_id
        ),
        "h3_resolution": request.h3_resolution,
        "query_latitude": request.latitude,
        "query_longitude": request.longitude,
        "grid_latitude": payload.get("latitude"),
        "grid_longitude": payload.get("longitude"),
        "grid_elevation": payload.get("elevation"),
        "start_date": request.start_date.isoformat(),
        "end_date": request.end_date.isoformat(),
        "timezone": payload.get(
            "timezone",
            request.timezone,
        ),
        "utc_offset_seconds": payload.get(
            "utc_offset_seconds"
        ),
        "cell_selection": request.cell_selection,
        "hourly_variables": list(
            request.hourly_variables
        ),
        "hourly_units": payload.get(
            "hourly_units",
            {},
        ),
        "hourly": payload["hourly"],
    }


def _validate_weather_request(
    request: WeatherRequest,
) -> None:
    if request.provider != "open_meteo":
        raise ValueError(
            "Unsupported weather provider: "
            f"{request.provider!r}"
        )

    if not request.request_id.strip():
        raise ValueError(
            "request_id must not be empty"
        )

    if request.model not in VALID_MODELS:
        supported = ", ".join(
            sorted(VALID_MODELS)
        )
        raise ValueError(
            f"Unsupported model {request.model!r}. "
            f"Supported models: {supported}"
        )

    if not -90.0 <= request.latitude <= 90.0:
        raise ValueError(
            "latitude must be between -90 and 90"
        )

    if not -180.0 <= request.longitude <= 180.0:
        raise ValueError(
            "longitude must be between "
            "-180 and 180"
        )

    if request.start_date > request.end_date:
        raise ValueError(
            "start_date must not be after end_date"
        )

    if request.cell_selection not in (
        VALID_CELL_SELECTIONS
    ):
        raise ValueError(
            "Unsupported cell_selection: "
            f"{request.cell_selection!r}"
        )

    if not request.timezone.strip():
        raise ValueError(
            "timezone must not be empty"
        )

    if not request.hourly_variables:
        raise ValueError(
            "At least one hourly variable "
            "is required"
        )

    normalized_variables = tuple(
        variable.strip()
        for variable in request.hourly_variables
    )

    if any(
        not variable
        for variable in normalized_variables
    ):
        raise ValueError(
            "hourly_variables cannot contain "
            "empty values"
        )

    if len(set(normalized_variables)) != len(
        normalized_variables
    ):
        raise ValueError(
            "hourly_variables contains duplicates"
        )


def validate_hourly_response(
    *,
    request: WeatherRequest,
    payload: Mapping[str, Any],
) -> None:
    hourly = payload.get("hourly")

    if not isinstance(hourly, dict):
        raise WeatherFetchError(
            "Open-Meteo returned no hourly data: "
            f"request_id={request.request_id}"
        )

    timestamps = hourly.get("time")

    if not isinstance(timestamps, list):
        raise WeatherFetchError(
            "Open-Meteo returned no hourly "
            "timestamps: "
            f"request_id={request.request_id}"
        )

    for variable in request.hourly_variables:
        values = hourly.get(variable)

        if not isinstance(values, list):
            raise WeatherFetchError(
                "Missing requested variable "
                f"{variable!r}: "
                f"request_id={request.request_id}"
            )

        if len(values) != len(timestamps):
            raise WeatherFetchError(
                "Hourly array length mismatch "
                f"for {variable!r}: "
                f"time={len(timestamps)}, "
                f"values={len(values)}, "
                f"request_id={request.request_id}"
            )


def _build_http_error_message(
    *,
    request_id: str,
    response: requests.Response,
) -> str:
    try:
        payload = response.json()
        reason = payload.get(
            "reason",
            response.text,
        )
    except requests.JSONDecodeError:
        reason = response.text

    return (
        "Open-Meteo returned "
        f"HTTP {response.status_code}: "
        f"request_id={request_id}, "
        f"reason={reason}"
    )