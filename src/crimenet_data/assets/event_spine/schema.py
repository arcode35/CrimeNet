"""Static contracts for the Gold crime event spine."""

from __future__ import annotations

EVENT_SPINE_SCHEMA_VERSION = "crime_event_spine_v1"

H3_RESOLUTION = 9
WEATHER_H3_RESOLUTION = 6

HISTORY_ROOT_SUFFIX = "national_feature_store/temporal/h3_r9/history"
EVENT_SPINE_ROOT_SUFFIX = "event_spine"

EVENT_SPINE_SUCCESS_MARKER = "_SUCCESS"
EVENT_SPINE_MANIFEST = "manifest.json"
EVENT_SPINE_LATEST_POINTER = "_latest.json"

PARTITION_COLUMNS = ["source_city", "occurrence_year"]

# The audited production history covers about 99.969% of modeled events.
MIN_HISTORY_COVERAGE_PCT = 99.90

# Nonexistent DST wall-clock times are expected to be a microscopic tail.
MAX_UNJOINABLE_EVENT_PCT = 0.10

REQUIRED_HISTORY_COLUMNS = frozenset(
    {
        "osm_h3_cell_id",
        "feature_available_at",
        "feature_version_id",
    }
)

COMPONENT_AVAILABILITY_COLUMNS = (
    "osm_available_at",
    "acs_release_date",
    "tiger_release_date",
)

TEMPORAL_INDEX_BASE_COLUMNS = (
    "osm_h3_cell_id",
    "feature_available_at",
    "feature_version_id",
)

HISTORY_KEY_COLUMNS = (
    "osm_h3_cell_id",
    "feature_available_at",
)

__all__ = [
    "COMPONENT_AVAILABILITY_COLUMNS",
    "EVENT_SPINE_LATEST_POINTER",
    "EVENT_SPINE_MANIFEST",
    "EVENT_SPINE_ROOT_SUFFIX",
    "EVENT_SPINE_SCHEMA_VERSION",
    "EVENT_SPINE_SUCCESS_MARKER",
    "H3_RESOLUTION",
    "HISTORY_KEY_COLUMNS",
    "HISTORY_ROOT_SUFFIX",
    "MAX_UNJOINABLE_EVENT_PCT",
    "MIN_HISTORY_COVERAGE_PCT",
    "PARTITION_COLUMNS",
    "REQUIRED_HISTORY_COLUMNS",
    "TEMPORAL_INDEX_BASE_COLUMNS",
    "WEATHER_H3_RESOLUTION",
]
