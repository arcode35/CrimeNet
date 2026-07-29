"""Stable keys and version semantics for solar-lighting features."""

from __future__ import annotations

from datetime import UTC, datetime

LIGHTING_QUERY_CELL_COLUMN = "lighting_query_cell_id"
SOLAR_TIMESTAMP_HOUR_COLUMN = "solar_timestamp_hour"
LIGHTING_DEFINITION_VERSION_COLUMN = "lighting_definition_version"

LIGHTING_DEFINITION_VERSION = "solar_elevation_twilight_v1"

LIGHTING_KEYS = (
    LIGHTING_QUERY_CELL_COLUMN,
    SOLAR_TIMESTAMP_HOUR_COLUMN,
    LIGHTING_DEFINITION_VERSION_COLUMN,
)

VALID_LIGHTING_CONDITIONS = (
    "daylight",
    "civil_twilight",
    "nautical_twilight",
    "astronomical_twilight",
    "night",
)


def normalize_solar_timestamp_hour(timestamp: datetime) -> datetime:
    """Return the UTC-hour key used by both lighting and Gold.

    Naive values are interpreted as UTC, matching the pipeline's required
    ``spark.sql.session.timeZone=UTC`` setting.
    """

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    else:
        timestamp = timestamp.astimezone(UTC)

    return timestamp.replace(minute=0, second=0, microsecond=0)


def lighting_join_key(
    *,
    query_cell_id: int,
    timestamp: datetime,
    definition_version: str = LIGHTING_DEFINITION_VERSION,
) -> tuple[int, datetime, str]:
    """Build the canonical versioned lighting key."""

    if not definition_version.strip():
        raise ValueError("Lighting definition version must not be blank.")

    return (
        query_cell_id,
        normalize_solar_timestamp_hour(timestamp),
        definition_version,
    )
