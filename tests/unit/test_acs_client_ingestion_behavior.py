from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import pytest
import requests

from crimenet.socioeconomic import acs_client, acs_ingestion
from crimenet.socioeconomic.acs_client import (
    ACS5_DATASET,
    CensusApiError,
    build_census_session,
    fetch_acs5_tracts,
)
from crimenet.socioeconomic.acs_ingestion import (
    get_acs5_tract_path,
    ingest_acs5_tract_vintages,
    is_valid_acs_cache,
    write_json_lines,
)


class _Response:
    def __init__(
        self,
        *,
        payload: object | None = None,
        status_code: int = 200,
        text: str = "",
        http_error: bool = False,
        json_error: bool = False,
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.text = text
        self.http_error = http_error
        self.json_error = json_error
        self.headers = {"Content-Type": "application/json"}

    def raise_for_status(self) -> None:
        if self.http_error:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self) -> object:
        if self.json_error:
            raise requests.exceptions.JSONDecodeError(
                "bad JSON",
                self.text,
                0,
            )
        return self.payload


class _Session:
    def __init__(
        self,
        response: _Response | None = None,
        *,
        request_error: Exception | None = None,
    ) -> None:
        self.response = response
        self.request_error = request_error
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    def get(self, url: str, **kwargs: Any) -> _Response:
        self.calls.append((url, kwargs))
        if self.request_error is not None:
            raise self.request_error
        assert self.response is not None
        return self.response

    def close(self) -> None:
        self.closed = True


def _success_payload() -> list[list[str]]:
    return [
        ["NAME", "B01003_001E", "state", "county", "tract"],
        ["Census Tract 1", "1234", "48", "113", "000100"],
        ["Census Tract 2", "5678", "48", "201", "000200"],
    ]


def test_census_session_has_bounded_get_retries() -> None:
    session = build_census_session()
    try:
        retry = session.get_adapter("https://").max_retries
        assert retry.total == 5
        assert retry.connect == 5
        assert retry.read == 5
        assert retry.status == 5
        assert retry.allowed_methods == frozenset({"GET"})
        assert 429 in retry.status_forcelist
        assert retry.respect_retry_after_header is True
    finally:
        session.close()


def test_fetch_acs_records_builds_keys_and_request_parameters() -> None:
    session = _Session(_Response(payload=_success_payload()))
    records = fetch_acs5_tracts(
        vintage=2023,
        state_fips="48",
        api_key="secret-key",
        session=session,  # type: ignore[arg-type]
    )

    assert len(records) == 2
    assert records[0]["geoid"] == "48113000100"
    assert records[0]["period_start_year"] == 2019
    assert records[0]["period_end_year"] == 2023
    assert records[0]["dataset"] == ACS5_DATASET
    assert records[0]["geography_type"] == "tract"
    assert records[0]["retrieved_at"].endswith("+00:00")
    assert session.closed is False

    url, kwargs = session.calls[0]
    assert url.endswith("/2023/acs/acs5")
    assert kwargs["params"]["for"] == "tract:*"
    assert kwargs["params"]["in"] == "state:48 county:*"
    assert kwargs["params"]["key"] == "secret-key"
    assert kwargs["timeout"] == (30, 180)


def test_fetch_acs_closes_an_internally_owned_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _Session(_Response(payload=_success_payload()))
    monkeypatch.setattr(acs_client, "build_census_session", lambda: session)
    fetch_acs5_tracts(vintage=2022)
    assert session.closed is True
    assert "key" not in session.calls[0][1]["params"]


@pytest.mark.parametrize(
    ("session", "message"),
    [
        (
            _Session(request_error=requests.ConnectionError("offline")),
            "request failed",
        ),
        (
            _Session(
                _Response(
                    status_code=503,
                    text="unavailable",
                    http_error=True,
                )
            ),
            "status=503",
        ),
        (
            _Session(
                _Response(
                    status_code=200,
                    text="<html>",
                    json_error=True,
                )
            ),
            "non-JSON",
        ),
        (
            _Session(_Response(payload={"error": "no data"})),
            "no tract records",
        ),
        (
            _Session(
                _Response(
                    payload=[
                        ["NAME", "state", "county", "tract"],
                        ["short", "48"],
                    ]
                )
            ),
            "unexpected number of values",
        ),
    ],
)
def test_fetch_acs_reports_external_failure_modes(
    session: _Session,
    message: str,
) -> None:
    with pytest.raises(CensusApiError, match=message):
        fetch_acs5_tracts(
            vintage=2023,
            session=session,  # type: ignore[arg-type]
        )


def test_census_error_does_not_echo_api_key() -> None:
    api_key = "do-not-log-this-key"
    session = _Session(
        _Response(
            payload={
                "error": f"invalid key: {api_key}",
            }
        )
    )

    with pytest.raises(CensusApiError) as error:
        fetch_acs5_tracts(
            vintage=2023,
            api_key=api_key,
            session=session,  # type: ignore[arg-type]
        )

    assert api_key not in str(error.value)


def test_census_request_traceback_suppresses_secret_bearing_cause() -> None:
    api_key = "do-not-log-this-key"
    session = _Session(
        request_error=requests.ConnectionError(
            f"https://api.census.gov/data?key={api_key}"
        )
    )

    with pytest.raises(CensusApiError) as error:
        fetch_acs5_tracts(
            vintage=2023,
            api_key=api_key,
            session=session,  # type: ignore[arg-type]
        )

    formatted = "".join(
        traceback.format_exception(
            type(error.value),
            error.value,
            error.value.__traceback__,
        )
    )
    assert error.value.__suppress_context__ is True
    assert api_key not in formatted


def test_json_lines_are_atomic_and_cache_is_partition_validated(
    tmp_path: Path,
) -> None:
    destination = get_acs5_tract_path(
        tmp_path,
        vintage=2023,
        state_fips="48",
    )
    records = [
        {
            "geoid": "48113000100",
            "acs_vintage": 2023,
            "state": "48",
            "name": "Tract 1",
        },
        {
            "geoid": "48201000200",
            "acs_vintage": 2023,
            "state": "48",
            "name": "Tract 2",
        },
    ]
    assert write_json_lines(iter(records), destination) == 2
    assert destination.is_file()
    assert list(destination.parent.glob(".*.tmp")) == []
    assert is_valid_acs_cache(
        destination,
        vintage=2023,
        state_fips="48",
    )
    assert not is_valid_acs_cache(
        destination,
        vintage=2022,
        state_fips="48",
    )
    assert not is_valid_acs_cache(
        destination,
        vintage=2023,
        state_fips="06",
    )
    assert not is_valid_acs_cache(
        destination,
        vintage=2023,
        state_fips="48",
        minimum_record_count=3,
    )


def test_acs_cache_rejects_duplicate_or_malformed_geoids(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "cache.jsonl"
    duplicate = {
        "geoid": "48113000100",
        "acs_vintage": 2023,
        "state": "48",
    }
    write_json_lines([duplicate, duplicate], destination)
    assert not is_valid_acs_cache(
        destination,
        vintage=2023,
        state_fips="48",
    )


@pytest.mark.parametrize(
    "content",
    [
        "",
        "{broken",
        "[]\n",
        '{"acs_vintage":"not-int","state":"48","geoid":"48113000100"}\n',
        '{"acs_vintage":2023,"state":"48"}\n',
    ],
)
def test_invalid_acs_cache_shapes_are_rejected(
    tmp_path: Path,
    content: str,
) -> None:
    path = tmp_path / "cache.jsonl"
    path.write_text(content, encoding="utf-8")
    assert not is_valid_acs_cache(
        path,
        vintage=2023,
        state_fips="48",
    )
    assert not is_valid_acs_cache(
        tmp_path / "missing.jsonl",
        vintage=2023,
        state_fips="48",
    )


def test_vintage_ingestion_skips_valid_cache_and_fetches_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing_path = get_acs5_tract_path(
        tmp_path,
        vintage=2022,
        state_fips="48",
    )
    write_json_lines(
        [
            {
                "geoid": "48113000100",
                "acs_vintage": 2022,
                "state": "48",
            }
        ],
        existing_path,
    )
    session = _Session()
    calls: list[tuple[int, str, str | None, object]] = []

    def fake_fetch(
        *,
        vintage: int,
        state_fips: str,
        api_key: str | None,
        session: object,
    ) -> list[dict[str, object]]:
        calls.append((vintage, state_fips, api_key, session))
        return [
            {
                "geoid": f"48{vintage:09d}"[-11:],
                "acs_vintage": vintage,
                "state": state_fips,
            }
        ]

    monkeypatch.setattr(
        acs_ingestion,
        "build_census_session",
        lambda: session,
    )
    monkeypatch.setattr(acs_ingestion, "fetch_acs5_tracts", fake_fetch)

    ingest_acs5_tract_vintages(
        landing_directory=tmp_path,
        start_vintage=2022,
        end_vintage=2023,
        state_fips="48",
        api_key=None,
    )
    assert [call[0] for call in calls] == [2023]
    assert calls[0][3] is session
    assert session.closed is True
    assert is_valid_acs_cache(
        get_acs5_tract_path(
            tmp_path,
            vintage=2023,
            state_fips="48",
        ),
        vintage=2023,
        state_fips="48",
    )


def test_vintage_ingestion_validates_range_and_closes_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="cannot be greater"):
        ingest_acs5_tract_vintages(
            landing_directory=tmp_path,
            start_vintage=2024,
            end_vintage=2023,
            state_fips="48",
            api_key=None,
        )

    session = _Session()
    monkeypatch.setattr(
        acs_ingestion,
        "build_census_session",
        lambda: session,
    )
    monkeypatch.setattr(
        acs_ingestion,
        "fetch_acs5_tracts",
        lambda **_kwargs: (_ for _ in ()).throw(CensusApiError("boom")),
    )
    with pytest.raises(CensusApiError, match="boom"):
        ingest_acs5_tract_vintages(
            landing_directory=tmp_path,
            start_vintage=2023,
            end_vintage=2023,
            state_fips="48",
            api_key=None,
            overwrite=True,
        )
    assert session.closed is True


def test_vintage_ingestion_rejects_semantically_partial_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _Session()
    monkeypatch.setattr(
        acs_ingestion,
        "build_census_session",
        lambda: session,
    )
    monkeypatch.setattr(
        acs_ingestion,
        "fetch_acs5_tracts",
        lambda **_kwargs: [
            {
                "geoid": "48113000100",
                "acs_vintage": 2023,
                "state": "48",
            }
        ],
    )

    with pytest.raises(CensusApiError, match="fewer tract records"):
        ingest_acs5_tract_vintages(
            landing_directory=tmp_path,
            start_vintage=2023,
            end_vintage=2023,
            state_fips="48",
            api_key=None,
            minimum_record_count=2,
        )

    assert session.closed is True
    assert not get_acs5_tract_path(
        tmp_path,
        vintage=2023,
        state_fips="48",
    ).exists()
