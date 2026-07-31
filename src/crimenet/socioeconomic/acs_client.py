"""Census API client and ACS 5-year tract source contract."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import requests
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


CENSUS_API_BASE_URL = "https://api.census.gov/data"
ACS5_DATASET = "acs/acs5"

# Leakage-safe mapping for crime years 2013–2025:
#
# crime 2013 -> ACS vintage 2012
# ...
# crime 2025 -> ACS vintage 2024
DEFAULT_START_VINTAGE = 2012
DEFAULT_END_VINTAGE = 2024


@dataclass(frozen=True)
class MetroGeography:
    """State and county coverage for one CrimeNet metro."""

    key: str
    state_fips: str
    county_fips: tuple[str, ...]


METRO_GEOGRAPHIES: dict[str, MetroGeography] = {
    "new_york": MetroGeography(
        key="new_york",
        state_fips="36",
        county_fips=(
            "005",  # Bronx County
            "047",  # Kings County
            "061",  # New York County
            "081",  # Queens County
            "085",  # Richmond County
        ),
    ),
    "chicago": MetroGeography(
        key="chicago",
        state_fips="17",
        county_fips=(
            "031",  # Cook County
            "043",  # DuPage County
        ),
    ),
    "san_francisco": MetroGeography(
        key="san_francisco",
        state_fips="06",
        county_fips=(
            "075",
        ),
    ),
    "seattle": MetroGeography(
        key="seattle",
        state_fips="53",
        county_fips=(
            "033",
        ),
    ),
    "baltimore": MetroGeography(
        key="baltimore",
        state_fips="24",
        county_fips=(
            "510",
        ),
    ),
    "washington_dc": MetroGeography(
        key="washington_dc",
        state_fips="11",
        county_fips=(
            "001",
        ),
    ),
}


# Estimate and margin-of-error variables are retained together.
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

    # Poverty universe and below-poverty population
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


def utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(UTC).isoformat()


def build_census_session() -> Session:
    """Create a retrying HTTP session for Census API requests."""
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        status=6,
        backoff_factor=1.0,
        status_forcelist=(
            429,
            500,
            502,
            503,
            504,
        ),
        allowed_methods=frozenset(
            {
                "GET",
            }
        ),
        respect_retry_after_header=True,
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=1,
        pool_maxsize=1,
    )

    session = Session()
    session.mount(
        "https://",
        adapter,
    )

    return session


def validate_fips(
    *,
    state_fips: str,
    county_fips: str,
) -> None:
    """Validate two-digit state and three-digit county codes."""
    if (
        len(state_fips) != 2
        or not state_fips.isdigit()
    ):
        raise ValueError(
            "state_fips must contain exactly two digits"
        )

    if (
        len(county_fips) != 3
        or not county_fips.isdigit()
    ):
        raise ValueError(
            "county_fips must contain exactly three digits"
        )


def fetch_acs5_tracts(
    *,
    vintage: int,
    state_fips: str,
    county_fips: str,
    api_key: str,
    session: Session,
) -> list[dict[str, Any]]:
    """Fetch all ACS 5-year tract records for one county."""
    validate_fips(
        state_fips=state_fips,
        county_fips=county_fips,
    )

    if not api_key.strip():
        raise CensusApiError(
            "The Census API key is empty"
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
            f"county:{county_fips}"
        ),
        "key": api_key,
    }

    try:
        response = session.get(
            request_url,
            params=parameters,
            headers={
                "Accept": "application/json",
                "User-Agent": (
                    "CrimeNet ACS tract ingestion/1.0"
                ),
            },
            timeout=(
                30,
                180,
            ),
        )

        response.raise_for_status()

    except requests.Timeout as exc:
        raise CensusApiError(
            "Census API request timed out: "
            f"vintage={vintage}, "
            f"state={state_fips}, "
            f"county={county_fips}"
        ) from exc

    except requests.RequestException as exc:
        raise CensusApiError(
            "Census API request failed: "
            f"vintage={vintage}, "
            f"state={state_fips}, "
            f"county={county_fips}, "
            f"error={exc}"
        ) from exc

    try:
        payload = response.json()

    except requests.exceptions.JSONDecodeError as exc:
        raise CensusApiError(
            "Census API returned non-JSON content: "
            f"vintage={vintage}, "
            f"state={state_fips}, "
            f"county={county_fips}, "
            f"status={response.status_code}, "
            f"content_type="
            f"{response.headers.get('Content-Type')!r}, "
            f"response={response.text[:1000]!r}"
        ) from exc

    if (
        not isinstance(payload, list)
        or len(payload) < 2
        or not isinstance(payload[0], list)
    ):
        raise CensusApiError(
            "Census API returned no tract records: "
            f"vintage={vintage}, "
            f"state={state_fips}, "
            f"county={county_fips}, "
            f"payload={payload!r}"
        )

    header = [
        str(column)
        for column in payload[0]
    ]

    required_columns = {
        "NAME",
        *ACS5_TRACT_VARIABLES,
        "state",
        "county",
        "tract",
    }

    missing_columns = (
        required_columns - set(header)
    )

    if missing_columns:
        raise CensusApiError(
            "Census response is missing columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    retrieved_at = utc_now()
    records: list[dict[str, Any]] = []

    for values in payload[1:]:
        if not isinstance(values, list):
            raise CensusApiError(
                "Census API returned a non-list row"
            )

        if len(values) != len(header):
            raise CensusApiError(
                "Census API returned a row with an "
                "unexpected number of values"
            )

        record = dict(
            zip(
                header,
                values,
                strict=True,
            )
        )

        returned_state = str(
            record["state"]
        )
        returned_county = str(
            record["county"]
        )
        returned_tract = str(
            record["tract"]
        )

        if returned_state != state_fips:
            raise CensusApiError(
                "Census API returned an unexpected state: "
                f"expected={state_fips}, "
                f"actual={returned_state}"
            )

        if returned_county != county_fips:
            raise CensusApiError(
                "Census API returned an unexpected county: "
                f"expected={county_fips}, "
                f"actual={returned_county}"
            )

        record.update(
            {
                "geoid": (
                    f"{returned_state}"
                    f"{returned_county}"
                    f"{returned_tract}"
                ),
                "acs_vintage": vintage,
                "period_start_year": vintage - 4,
                "period_end_year": vintage,
                "dataset": ACS5_DATASET,
                "geography_type": "tract",
                "retrieved_at": retrieved_at,
            }
        )

        records.append(
            record
        )

    return records