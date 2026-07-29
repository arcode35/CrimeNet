from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    wait,
)
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Literal

from pyspark.sql import DataFrame, Row

from crimenet.weather.cache import (
    WeatherCacheError,
    get_weather_cache_path,
    read_weather_cache,
    write_weather_cache,
)
from crimenet.weather.open_meteo_client import (
    OpenMeteoClient,
    OpenMeteoClientConfig,
    WeatherFetchError,
    WeatherRequest,
)
from crimenet.weather.request_mapping import (
    weather_request_from_manifest_row,
)

logger = logging.getLogger(__name__)

WeatherAuditEventType = Literal[
    "invalid_cache",
    "request_failed",
]


@dataclass(frozen=True)
class WeatherIngestionAuditEvent:
    """A persistence-ready event emitted for cache and request failures."""

    event_type: WeatherAuditEventType
    request_id: str
    provider: str
    model: str
    weather_query_cell_id: int
    start_date: str
    end_date: str
    cache_path: str
    error_type: str
    error_message: str
    occurred_at: datetime
    error_category: str | None = None
    retryable: bool | None = None
    status_code: int | None = None


WeatherAuditHook = Callable[[WeatherIngestionAuditEvent], None]
WeatherFetchOutcome = Literal["downloaded", "cached"]


def fetch_weather_manifest(
    weather_request_manifest: DataFrame,
    *,
    cache_directory: str | Path,
    client_config: OpenMeteoClientConfig | None = None,
    client: OpenMeteoClient | None = None,
    audit_hook: WeatherAuditHook | None = None,
) -> None:
    """Fetch a manifest, reusing only cache entries that fully validate."""

    if client is not None and client_config is not None:
        raise ValueError(
            "client_config cannot be supplied with an existing client"
        )

    cache_directory = Path(cache_directory)

    logger.info(
        "Starting weather manifest ingestion: cache_directory=%s",
        cache_directory,
    )

    rows = weather_request_manifest.toLocalIterator()

    logger.info(
        "Spark iterator initialized; waiting for manifest rows"
    )

    client_context = (
        nullcontext(client)
        if client is not None
        else OpenMeteoClient(config=client_config)
    )
    with client_context as active_client:
        _fetch_weather_rows(
            rows=rows,
            cache_directory=cache_directory,
            client=active_client,
            audit_hook=audit_hook,
        )


def _fetch_weather_rows(
    *,
    rows: Iterable[Row],
    cache_directory: Path,
    client: OpenMeteoClient,
    audit_hook: WeatherAuditHook | None,
) -> None:
    processed_count = 0
    outcomes: list[WeatherFetchOutcome] = []
    maximum_workers = (
        client.config.max_concurrent_requests
        if hasattr(client, "config")
        else 1
    )
    audit_lock = Lock()

    def synchronized_audit_hook(event: WeatherIngestionAuditEvent) -> None:
        if audit_hook is None:
            return
        with audit_lock:
            audit_hook(event)

    if maximum_workers == 1:
        for processed_count, row in enumerate(rows, start=1):
            outcomes.append(
                _fetch_weather_row(
                    row=row,
                    sequence_number=processed_count,
                    cache_directory=cache_directory,
                    client=client,
                    audit_hook=synchronized_audit_hook,
                )
            )
    else:
        row_iterator = iter(rows)
        executor = ThreadPoolExecutor(
            max_workers=maximum_workers,
            thread_name_prefix="crimenet-weather",
        )
        pending: set[Future[WeatherFetchOutcome]] = set()

        def submit_next() -> bool:
            nonlocal processed_count
            try:
                row = next(row_iterator)
            except StopIteration:
                return False
            processed_count += 1
            pending.add(
                executor.submit(
                    _fetch_weather_row,
                    row=row,
                    sequence_number=processed_count,
                    cache_directory=cache_directory,
                    client=client,
                    audit_hook=synchronized_audit_hook,
                )
            )
            return True

        try:
            for _ in range(maximum_workers):
                if not submit_next():
                    break
            while pending:
                completed, _ = wait(
                    pending,
                    return_when=FIRST_COMPLETED,
                )
                for future in completed:
                    pending.remove(future)
                    outcomes.append(future.result())
                    submit_next()
        except Exception:
            for future in pending:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    logger.info(
        "Weather ingestion finished: "
        "processed=%d, downloaded=%d, already_cached=%d",
        processed_count,
        outcomes.count("downloaded"),
        outcomes.count("cached"),
    )


def _fetch_weather_row(
    *,
    row: Row,
    sequence_number: int,
    cache_directory: Path,
    client: OpenMeteoClient,
    audit_hook: WeatherAuditHook | None,
) -> WeatherFetchOutcome:
    request = weather_request_from_manifest_row(
        row.asDict(recursive=True)
    )
    logger.info(
        "Processing weather request: "
        "number=%d, request_id=%s, cell=%s, "
        "start=%s, end=%s",
        sequence_number,
        request.request_id,
        request.weather_query_cell_id,
        request.start_date,
        request.end_date,
    )
    cache_path = get_weather_cache_path(
        cache_directory,
        request,
    )
    try:
        cached_response = read_weather_cache(
            cache_path=cache_path,
            request=request,
        )
    except WeatherCacheError as exc:
        logger.warning(
            "Ignoring invalid weather cache entry: "
            "request_id=%s, path=%s, error_type=%s",
            request.request_id,
            cache_path,
            type(exc).__name__,
        )
        _emit_audit_event(
            audit_hook=audit_hook,
            event=_build_audit_event(
                event_type="invalid_cache",
                request=request,
                cache_path=cache_path,
                error=exc,
            ),
        )
    else:
        if cached_response is not None:
            logger.info(
                "Reusing validated weather cache entry: "
                "request_id=%s, path=%s",
                request.request_id,
                cache_path,
            )
            return "cached"

    try:
        logger.info(
            "Calling Open-Meteo: request_id=%s",
            request.request_id,
        )
        response = client.fetch_historical_weather(request)
        write_weather_cache(
            cache_path=cache_path,
            response=response,
        )
        logger.info(
            "Cached weather request: "
            "request_id=%s, cell=%s, "
            "hours=%d, path=%s",
            request.request_id,
            request.weather_query_cell_id,
            len(response["hourly"]["time"]),
            cache_path,
        )
        return "downloaded"
    except Exception as exc:
        logger.exception(
            "Weather request failed: "
            "request_id=%s, cell=%s, "
            "start=%s, end=%s",
            request.request_id,
            request.weather_query_cell_id,
            request.start_date,
            request.end_date,
        )
        _emit_audit_event(
            audit_hook=audit_hook,
            event=_build_audit_event(
                event_type="request_failed",
                request=request,
                cache_path=cache_path,
                error=exc,
            ),
            original_error=exc,
        )
        raise


def _build_audit_event(
    *,
    event_type: WeatherAuditEventType,
    request: WeatherRequest,
    cache_path: Path,
    error: Exception,
) -> WeatherIngestionAuditEvent:
    retryable = None
    status_code = None
    error_category = None
    if isinstance(error, WeatherFetchError):
        error_category = error.category
        retryable = error.retryable
        status_code = error.status_code

    return WeatherIngestionAuditEvent(
        event_type=event_type,
        request_id=request.request_id,
        provider=request.provider,
        model=request.model,
        weather_query_cell_id=request.weather_query_cell_id,
        start_date=request.start_date.isoformat(),
        end_date=request.end_date.isoformat(),
        cache_path=str(cache_path),
        error_type=type(error).__name__,
        error_message=" ".join(str(error).split())[:1000],
        occurred_at=datetime.now(UTC),
        error_category=error_category,
        retryable=retryable,
        status_code=status_code,
    )


def _emit_audit_event(
    *,
    audit_hook: WeatherAuditHook | None,
    event: WeatherIngestionAuditEvent,
    original_error: Exception | None = None,
) -> None:
    if audit_hook is None:
        return

    try:
        audit_hook(event)
    except Exception as audit_error:
        logger.exception(
            "Weather audit hook failed: "
            "event_type=%s, request_id=%s, error_type=%s",
            event.event_type,
            event.request_id,
            type(audit_error).__name__,
        )
        if original_error is not None:
            original_error.add_note(
                "The weather audit hook also failed with "
                f"{type(audit_error).__name__}."
            )
