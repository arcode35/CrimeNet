"""Client for Census ACS 5-year tract data."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


CENSUS_API_BASE_URL = "https://api.census.gov/data"
ACS5_DATASET = "acs/acs5"

TEXAS_STATE_FIPS = "48"


# Estimate and margin-of-error variables are requested together.
ACS5_TRACT_VARIABLES = (
    # Population
    "B01003_001E",
    "B01003_001M",

    # Median age
    "B01002_001E",
    "B01002_001M",

    # Median household income
    "B19013_001E",
    "B19013_001M",

    # Poverty universe and population below poverty
    "B17001_001E",
    "B17001_001M",
    "B17001_002E",
    "B17001_002M",

    # Civilian labor force and unemployed population
    "B23025_003E",
    "B23025_003M",
    "B23025_005E",
    "B23025_005M",

    # Total, occupied, and vacant housing units
    "B25001_001E",
    "B25001_001M",
    "B25002_002E",
    "B25002_002M",
    "B25002_003E",
    "B25002_003M",

    # Occupied and renter-occupied housing units
    "B25003_001E",
    "B25003_001M",
    "B25003_003E",
    "B25003_003M",

    # Households and households without vehicles
    "B08201_001E",
    "B08201_001M",
    "B08201_002E",
    "B08201_002M",
)


class CensusApiError(RuntimeError):
    """Raised when the Census API returns an invalid response."""


def build_census_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1.0,
        status_forcelist=(
            429,
            500,
            502,
            503,
            504,
        ),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(
        max_retries=retry,
    )

    session = requests.Session()
    session.mount("https://", adapter)

    return session


def fetch_acs5_tracts(
    *,
    vintage: int,
    state_fips: str = TEXAS_STATE_FIPS,
    api_key: str,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Fetch all ACS 5-year census tracts in one state."""

    if not api_key:
        raise CensusApiError(
            "A Census API key is required. "
            "Set CENSUS_API_KEY before running the job."
        )

    active_session = (
        session
        if session is not None
        else build_census_session()
    )

    request_url = (
        f"{CENSUS_API_BASE_URL}/"
        f"{vintage}/{ACS5_DATASET}"
    )

    parameters = {
        "get": ",".join(
            (
                "NAME",
                *ACS5_TRACT_VARIABLES,
            )
        ),
        "for": "tract:*",
        "in": (
            f"state:{state_fips} "
            "county:*"
        ),
        "key": api_key,
    }

    response = active_session.get(
        request_url,
        params=parameters,
        headers={
            "Accept": "application/json",
        },
        timeout=(30, 180),
    )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise CensusApiError(
            "Census API request failed for "
            f"vintage={vintage}, "
            f"status={response.status_code}, "
            f"response={response.text[:500]!r}"
        ) from exc

    try:
        payload = response.json()
    except requests.exceptions.JSONDecodeError as exc:
        raise CensusApiError(
            "Census API returned a non-JSON response for "
            f"vintage={vintage}, "
            f"status={response.status_code}, "
            f"content_type="
            f"{response.headers.get('Content-Type')!r}, "
            f"response={response.text[:1000]!r}"
        ) from exc

    if (
        not isinstance(payload, list)
        or len(payload) < 2
    ):
        raise CensusApiError(
            "Census API returned no tract records "
            f"for vintage {vintage}: "
            f"{payload!r}"
        )

    header = payload[0]
    retrieved_at = datetime.now(UTC).isoformat()

    records: list[dict[str, Any]] = []

    for values in payload[1:]:
        if len(values) != len(header):
            raise CensusApiError(
                "Census API returned a row with "
                "an unexpected number of values."
            )

        record = dict(
            zip(
                header,
                values,
                strict=True,
            )
        )

        state = record["state"]
        county = record["county"]
        tract = record["tract"]

        record.update(
            {
                "geoid": (
                    f"{state}{county}{tract}"
                ),
                "acs_vintage": vintage,
                "period_start_year": vintage - 4,
                "period_end_year": vintage,
                "dataset": ACS5_DATASET,
                "geography_type": "tract",
                "retrieved_at": retrieved_at,
            }
        )

        records.append(record)

    return records