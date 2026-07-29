from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from datetime import time as datetime_time
from threading import BoundedSemaphore, Lock
from typing import Any, Self

from requests import Response, Session
from requests.adapters import HTTPAdapter
from requests.exceptions import (
    ConnectionError as RequestsConnectionError,
)
from requests.exceptions import (
    HTTPError,
    RequestException,
    Timeout,
)
from urllib3.util.retry import Retry

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_RETRYABLE_STATUS_CODES = (
    429,
    500,
    502,
    503,
    504,
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
class OpenMeteoClientConfig:
    """Transport and rate-limit settings for Open-Meteo requests."""

    archive_url: str = OPEN_METEO_ARCHIVE_URL
    connect_timeout_seconds: float = 30.0
    read_timeout_seconds: float = 180.0
    max_retries: int = 5
    retry_backoff_factor: float = 1.0
    retryable_status_codes: tuple[int, ...] = DEFAULT_RETRYABLE_STATUS_CODES
    max_concurrent_requests: int = 1
    minimum_request_interval_seconds: float = 0.0
    user_agent: str = "CrimeNet/0.2 Open-Meteo client"

    def __post_init__(self) -> None:
        if not self.archive_url.strip():
            raise ValueError("archive_url must not be empty")
        if self.connect_timeout_seconds <= 0:
            raise ValueError(
                "connect_timeout_seconds must be greater than zero"
            )
        if self.read_timeout_seconds <= 0:
            raise ValueError(
                "read_timeout_seconds must be greater than zero"
            )
        if self.max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if self.retry_backoff_factor < 0:
            raise ValueError(
                "retry_backoff_factor must not be negative"
            )
        if self.max_concurrent_requests <= 0:
            raise ValueError(
                "max_concurrent_requests must be greater than zero"
            )
        if self.minimum_request_interval_seconds < 0:
            raise ValueError(
                "minimum_request_interval_seconds must not be negative"
            )
        if not self.user_agent.strip():
            raise ValueError("user_agent must not be empty")
        if len(set(self.retryable_status_codes)) != len(
            self.retryable_status_codes
        ):
            raise ValueError(
                "retryable_status_codes must not contain duplicates"
            )
        if any(
            status_code < 400 or status_code > 599
            for status_code in self.retryable_status_codes
        ):
            raise ValueError(
                "retryable_status_codes must contain HTTP error statuses"
            )

    @property
    def timeout(self) -> tuple[float, float]:
        return (
            self.connect_timeout_seconds,
            self.read_timeout_seconds,
        )


DEFAULT_OPEN_METEO_CLIENT_CONFIG = OpenMeteoClientConfig()


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

    def __init__(
        self,
        message: str,
        *,
        request_id: str | None = None,
        category: str = "invalid_response",
        retryable: bool = False,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.request_id = request_id
        self.category = category
        self.retryable = retryable
        self.status_code = status_code


def normalize_hourly_variables(
    hourly_variables: Sequence[str],
) -> tuple[str, ...]:
    """Return the canonical variable order used in request identities."""

    if isinstance(hourly_variables, str):
        raise TypeError("hourly_variables must be a sequence of names")
    if any(
        not isinstance(variable, str)
        for variable in hourly_variables
    ):
        raise TypeError(
            "hourly_variables must contain only strings"
        )

    normalized_variables = tuple(
        sorted(
            {
                variable.strip()
                for variable in hourly_variables
                if variable.strip()
            }
        )
    )

    if not normalized_variables:
        raise ValueError("At least one hourly variable is required")

    return normalized_variables


def build_weather_request_id(
    *,
    provider: str,
    model: str,
    weather_query_cell_id: int,
    start_date: date,
    end_date: date,
    hourly_variables: Sequence[str],
    timezone: str,
    cell_selection: str,
    h3_resolution: int,
) -> str:
    """Build the stable SHA-256 identity used by the Spark manifest."""

    normalized_variables = normalize_hourly_variables(hourly_variables)
    identity = "|".join(
        (
            provider,
            model,
            str(weather_query_cell_id),
            start_date.isoformat(),
            end_date.isoformat(),
            ",".join(normalized_variables),
            timezone,
            cell_selection,
            str(h3_resolution),
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def expected_weather_request_id(request: WeatherRequest) -> str:
    return build_weather_request_id(
        provider=request.provider,
        model=request.model,
        weather_query_cell_id=request.weather_query_cell_id,
        start_date=request.start_date,
        end_date=request.end_date,
        hourly_variables=request.hourly_variables,
        timezone=request.timezone,
        cell_selection=request.cell_selection,
        h3_resolution=request.h3_resolution,
    )


def build_open_meteo_retry_policy(
    config: OpenMeteoClientConfig = DEFAULT_OPEN_METEO_CLIENT_CONFIG,
) -> Retry:
    """Create the urllib3 retry policy used by the requests adapter."""

    return Retry(
        total=config.max_retries,
        connect=config.max_retries,
        read=config.max_retries,
        status=config.max_retries,
        other=0,
        allowed_methods=frozenset({"GET"}),
        status_forcelist=config.retryable_status_codes,
        backoff_factor=config.retry_backoff_factor,
        respect_retry_after_header=True,
        raise_on_status=False,
    )


def is_retryable_http_status(
    status_code: int,
    config: OpenMeteoClientConfig = DEFAULT_OPEN_METEO_CLIENT_CONFIG,
) -> bool:
    return status_code in config.retryable_status_codes


def _build_open_meteo_session(
    config: OpenMeteoClientConfig = DEFAULT_OPEN_METEO_CLIENT_CONFIG,
) -> Session:
    adapter = HTTPAdapter(
        max_retries=build_open_meteo_retry_policy(config),
        pool_connections=config.max_concurrent_requests,
        pool_maxsize=config.max_concurrent_requests,
        pool_block=True,
    )

    session = Session()
    session.headers["User-Agent"] = config.user_agent
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


class OpenMeteoClient:
    """Lifecycle-managed, reusable Open-Meteo HTTP client."""

    def __init__(
        self,
        config: OpenMeteoClientConfig | None = None,
        *,
        session: Session | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config or DEFAULT_OPEN_METEO_CLIENT_CONFIG
        self._session = session or _build_open_meteo_session(self.config)
        self._owns_session = session is None
        self._monotonic = monotonic
        self._sleep = sleep
        self._request_slots = BoundedSemaphore(
            self.config.max_concurrent_requests
        )
        self._rate_limit_lock = Lock()
        self._last_request_started_at: float | None = None
        self._closed = False

    def __enter__(self) -> Self:
        if self._closed:
            raise RuntimeError("OpenMeteoClient is closed")
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        if self._owns_session:
            self._session.close()
        self._closed = True

    def fetch_historical_weather(
        self,
        request: WeatherRequest,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("OpenMeteoClient is closed")

        validate_weather_request(request)

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

        with self._request_slots:
            self._wait_for_rate_limit()
            response = self._get_response(
                request=request,
                params=params,
            )

        payload = _decode_response_payload(
            request=request,
            response=response,
        )

        if payload.get("error") is True:
            reason = _sanitize_reason(payload.get("reason"))
            raise WeatherFetchError(
                "Open-Meteo rejected the request: "
                f"request_id={request.request_id}, reason={reason}",
                request_id=request.request_id,
                category="provider_error",
                retryable=False,
            )

        _validate_hourly_response(
            request=request,
            payload=payload,
        )

        response_timezone = payload.get("timezone")
        if response_timezone != request.timezone:
            raise WeatherFetchError(
                "Open-Meteo returned an unexpected timezone: "
                f"request_id={request.request_id}",
                request_id=request.request_id,
                category="invalid_response",
                retryable=True,
            )

        return {
            "request_id": request.request_id,
            "provider": request.provider,
            "model": request.model,
            "weather_query_cell_id": request.weather_query_cell_id,
            "h3_resolution": request.h3_resolution,
            "query_latitude": request.latitude,
            "query_longitude": request.longitude,
            "grid_latitude": payload.get("latitude"),
            "grid_longitude": payload.get("longitude"),
            "grid_elevation": payload.get("elevation"),
            "start_date": request.start_date.isoformat(),
            "end_date": request.end_date.isoformat(),
            "timezone": response_timezone,
            "utc_offset_seconds": payload.get("utc_offset_seconds"),
            "cell_selection": request.cell_selection,
            "hourly_variables": list(request.hourly_variables),
            "hourly_units": payload.get("hourly_units", {}),
            "hourly": payload["hourly"],
        }

    def _wait_for_rate_limit(self) -> None:
        interval = self.config.minimum_request_interval_seconds
        if interval == 0:
            return

        with self._rate_limit_lock:
            now = self._monotonic()
            if self._last_request_started_at is not None:
                delay = self._last_request_started_at + interval - now
                if delay > 0:
                    self._sleep(delay)
            self._last_request_started_at = self._monotonic()

    def _get_response(
        self,
        *,
        request: WeatherRequest,
        params: dict[str, Any],
    ) -> Response:
        try:
            response = self._session.get(
                self.config.archive_url,
                params=params,
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            return response
        except Timeout as exc:
            raise WeatherFetchError(
                "Open-Meteo request timed out after configured retries: "
                f"request_id={request.request_id}",
                request_id=request.request_id,
                category="timeout",
                retryable=True,
            ) from exc
        except RequestsConnectionError as exc:
            raise WeatherFetchError(
                "Open-Meteo connection failed after configured retries: "
                f"request_id={request.request_id}",
                request_id=request.request_id,
                category="connection",
                retryable=True,
            ) from exc
        except HTTPError as exc:
            error_response = (
                exc.response
                if exc.response is not None
                else getattr(exc, "response", None)
            )
            status_code = (
                error_response.status_code
                if error_response is not None
                else None
            )
            raise WeatherFetchError(
                _build_http_error_message(
                    request_id=request.request_id,
                    response=error_response,
                ),
                request_id=request.request_id,
                category="http",
                retryable=(
                    status_code is not None
                    and is_retryable_http_status(
                        status_code,
                        self.config,
                    )
                ),
                status_code=status_code,
            ) from exc
        except RequestException as exc:
            raise WeatherFetchError(
                "Open-Meteo request failed: "
                f"request_id={request.request_id}, "
                f"error_type={type(exc).__name__}",
                request_id=request.request_id,
                category="request",
                retryable=False,
            ) from exc


def fetch_historical_weather(
    request: WeatherRequest,
    *,
    config: OpenMeteoClientConfig | None = None,
    session: Session | None = None,
) -> dict[str, Any]:
    """Fetch one request, closing only sessions created by this function."""

    with OpenMeteoClient(
        config=config,
        session=session,
    ) as client:
        return client.fetch_historical_weather(request)


def validate_weather_request(
    request: WeatherRequest,
) -> None:
    if not isinstance(request.request_id, str):
        raise TypeError("request_id must be a string")
    if not isinstance(request.provider, str):
        raise TypeError("provider must be a string")
    if not isinstance(request.model, str):
        raise TypeError("model must be a string")
    if (
        isinstance(request.weather_query_cell_id, bool)
        or not isinstance(request.weather_query_cell_id, int)
    ):
        raise TypeError("weather_query_cell_id must be an integer")
    if (
        isinstance(request.h3_resolution, bool)
        or not isinstance(request.h3_resolution, int)
    ):
        raise TypeError("h3_resolution must be an integer")
    if (
        isinstance(request.latitude, bool)
        or not isinstance(request.latitude, (int, float))
        or not math.isfinite(request.latitude)
    ):
        raise ValueError("latitude must be a finite number")
    if (
        isinstance(request.longitude, bool)
        or not isinstance(request.longitude, (int, float))
        or not math.isfinite(request.longitude)
    ):
        raise ValueError("longitude must be a finite number")
    if (
        not isinstance(request.start_date, date)
        or isinstance(request.start_date, datetime)
    ):
        raise TypeError("start_date must be a datetime.date")
    if (
        not isinstance(request.end_date, date)
        or isinstance(request.end_date, datetime)
    ):
        raise TypeError("end_date must be a datetime.date")
    if not isinstance(request.timezone, str):
        raise TypeError("timezone must be a string")
    if not isinstance(request.cell_selection, str):
        raise TypeError("cell_selection must be a string")
    if not isinstance(request.hourly_variables, tuple):
        raise TypeError("hourly_variables must be a tuple")

    if request.provider != "open_meteo":
        raise ValueError(
            "Unsupported weather provider: "
            f"{request.provider!r}"
        )

    if request.model not in VALID_MODELS:
        supported = ", ".join(sorted(VALID_MODELS))
        raise ValueError(
            f"Unsupported model {request.model!r}. "
            f"Supported models: {supported}"
        )

    if not 0 <= request.h3_resolution <= 15:
        raise ValueError("h3_resolution must be between 0 and 15")

    if not -90.0 <= request.latitude <= 90.0:
        raise ValueError("latitude must be between -90 and 90")

    if not -180.0 <= request.longitude <= 180.0:
        raise ValueError("longitude must be between -180 and 180")

    if request.start_date > request.end_date:
        raise ValueError("start_date must not be after end_date")

    if request.cell_selection not in VALID_CELL_SELECTIONS:
        raise ValueError(
            "Unsupported cell_selection: "
            f"{request.cell_selection!r}"
        )

    if not request.timezone.strip():
        raise ValueError("timezone must not be empty")

    normalized_variables = normalize_hourly_variables(
        request.hourly_variables
    )
    if request.hourly_variables != normalized_variables:
        raise ValueError(
            "hourly_variables must be stripped, unique, and sorted"
        )

    expected_request_id = expected_weather_request_id(request)
    if request.request_id != expected_request_id:
        raise ValueError(
            "request_id does not match the deterministic request identity: "
            f"expected={expected_request_id}, "
            f"actual={request.request_id}"
        )


def _validate_weather_request(
    request: WeatherRequest,
) -> None:
    """Backward-compatible private alias for older callers."""

    validate_weather_request(request)


def expected_gmt_hourly_timestamps(
    request: WeatherRequest,
) -> tuple[str, ...] | None:
    """Return the exact deployed hourly range, or None for non-GMT requests."""
    if request.timezone.strip().upper() != "GMT":
        return None

    current = datetime.combine(
        request.start_date,
        datetime_time.min,
    )
    stop = datetime.combine(
        request.end_date + timedelta(days=1),
        datetime_time.min,
    )
    timestamps: list[str] = []
    while current < stop:
        timestamps.append(current.strftime("%Y-%m-%dT%H:%M"))
        current += timedelta(hours=1)
    return tuple(timestamps)


def _decode_response_payload(
    *,
    request: WeatherRequest,
    response: Response,
) -> Mapping[str, Any]:
    content = getattr(response, "content", None)
    if content is None or content == b"" or content == "":
        raise WeatherFetchError(
            "Open-Meteo returned an empty response: "
            f"request_id={request.request_id}",
            request_id=request.request_id,
            category="empty_response",
            retryable=True,
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise WeatherFetchError(
            "Open-Meteo returned invalid JSON: "
            f"request_id={request.request_id}",
            request_id=request.request_id,
            category="invalid_json",
            retryable=True,
        ) from exc

    if not isinstance(payload, Mapping):
        raise WeatherFetchError(
            "Open-Meteo returned a non-object JSON response: "
            f"request_id={request.request_id}",
            request_id=request.request_id,
            category="invalid_json",
            retryable=True,
        )

    return payload


def _validate_hourly_response(
    *,
    request: WeatherRequest,
    payload: Mapping[str, Any],
) -> None:
    hourly = payload.get("hourly")

    if not isinstance(hourly, Mapping):
        raise WeatherFetchError(
            "Open-Meteo returned no hourly data: "
            f"request_id={request.request_id}",
            request_id=request.request_id,
            category="invalid_response",
            retryable=True,
        )

    timestamps = hourly.get("time")

    if not isinstance(timestamps, list) or not timestamps:
        raise WeatherFetchError(
            "Open-Meteo returned no hourly timestamps: "
            f"request_id={request.request_id}",
            request_id=request.request_id,
            category="invalid_response",
            retryable=True,
        )

    if any(
        not isinstance(timestamp, str) or not timestamp.strip()
        for timestamp in timestamps
    ):
        raise WeatherFetchError(
            "Open-Meteo returned invalid hourly timestamps: "
            f"request_id={request.request_id}",
            request_id=request.request_id,
            category="invalid_response",
            retryable=True,
        )

    expected_timestamps = expected_gmt_hourly_timestamps(request)
    if expected_timestamps is not None and tuple(timestamps) != expected_timestamps:
        raise WeatherFetchError(
            "Open-Meteo returned an incomplete, duplicated, or unordered "
            "GMT hourly range: "
            f"expected={len(expected_timestamps)}, "
            f"actual={len(timestamps)}, "
            f"request_id={request.request_id}",
            request_id=request.request_id,
            category="invalid_response",
            retryable=True,
        )

    for variable in request.hourly_variables:
        values = hourly.get(variable)

        if not isinstance(values, list):
            raise WeatherFetchError(
                "Missing requested variable "
                f"{variable!r}: "
                f"request_id={request.request_id}",
                request_id=request.request_id,
                category="invalid_response",
                retryable=True,
            )

        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ):
            raise WeatherFetchError(
                "Open-Meteo returned an invalid hourly value "
                f"for {variable!r}: "
                f"request_id={request.request_id}",
                request_id=request.request_id,
                category="invalid_response",
                retryable=True,
            )

        if len(values) != len(timestamps):
            raise WeatherFetchError(
                "Hourly array length mismatch "
                f"for {variable!r}: "
                f"time={len(timestamps)}, "
                f"values={len(values)}, "
                f"request_id={request.request_id}",
                request_id=request.request_id,
                category="invalid_response",
                retryable=True,
            )


def _build_http_error_message(
    *,
    request_id: str,
    response: Response | None,
) -> str:
    if response is None:
        return (
            "Open-Meteo returned an HTTP error without a response: "
            f"request_id={request_id}"
        )

    reason = "no structured error reason"
    try:
        payload = response.json()
        if isinstance(payload, Mapping):
            reason = _sanitize_reason(payload.get("reason"))
    except ValueError:
        pass

    return (
        "Open-Meteo returned "
        f"HTTP {response.status_code}: "
        f"request_id={request_id}, "
        f"reason={reason}"
    )


def _sanitize_reason(reason: object) -> str:
    if not isinstance(reason, (str, int, float, bool)):
        return "no structured error reason"
    return " ".join(str(reason).split())[:500]
