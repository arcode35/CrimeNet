"""Static contracts for the Gold crime event spine."""

from __future__ import annotations

EVENT_SPINE_SCHEMA_VERSION = "crime_event_spine_v2"

H3_RESOLUTION = 9
WEATHER_H3_RESOLUTION = 6

EVENT_SPINE_UNMATCHED_HISTORY_POLICY = "retain_event_with_null_features"

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
    "EVENT_SPINE_SCHEMA_VERSION",
    "EVENT_SPINE_UNMATCHED_HISTORY_POLICY",
    "H3_RESOLUTION",
    "HISTORY_KEY_COLUMNS",
    "MAX_UNJOINABLE_EVENT_PCT",
    "MIN_HISTORY_COVERAGE_PCT",
    "PARTITION_COLUMNS",
    "REQUIRED_HISTORY_COLUMNS",
    "TEMPORAL_INDEX_BASE_COLUMNS",
    "WEATHER_H3_RESOLUTION",
]
