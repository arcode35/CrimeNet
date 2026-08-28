"""Explicit leakage-safe feature contracts for transferable ML models."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable


ZERO_SHOT_HISTORY_PREFIXES = (
    "cell_crime_",
    "cell_violent_",
    "cell_property_",
    "city_crime_",
    "k1_crime_",
)

ZERO_SHOT_HISTORY_EXACT = frozenset({
    "has_crime_cell_28d",
    "has_crime_city_28d",
    "hours_since_last_crime_cell_capped_28d",
    "hours_since_last_crime_city_capped_28d",
    "cell_crime_24h_vs_28d_ratio",
    "cell_share_of_k1_crime_24h",
})

FORBIDDEN_PREDICTORS = frozenset(
    {
        "row_id", "model_row_id", "row_type", "event_indicator", "event_count",
        "is_observed_event", "integration_weight_cell_seconds",
        "integration_sample_id", "integration_sample_index", "crime_id",
        "feature_available_at", "feature_version_id", "snapshot_id", "split",
        "row_timestamp_utc", "model_timestamp_utc", "row_year", "osm_h3_cell_id",
        "weather_query_cell_id", "latitude", "longitude", "source_city",
    }
)

DEFAULT_TRANSFERABLE_NUMERIC = (
    "local_hour", "local_day_of_week", "local_hour_sin", "local_hour_cos",
    "local_day_of_week_sin", "local_day_of_week_cos", "socio_population",
    "socio_median_age", "socio_median_household_income", "socio_poverty_rate",
    "socio_unemployment_rate", "socio_vacancy_rate", "socio_renter_occupied_rate",
    "socio_no_vehicle_rate", "osm_poi_density_per_km2",
    "osm_nightlife_poi_density_per_km2", "osm_food_poi_density_per_km2",
    "osm_retail_poi_density_per_km2", "osm_transit_poi_density_per_km2",
    "osm_road_length_density_m_per_km2", "osm_major_road_density_m_per_km2",
    "osm_intersection_density_per_km2", "osm_dead_end_density_per_km2",
    "osm_building_density_per_km2", "osm_major_road_length_ratio",
    "osm_residential_road_length_ratio", "osm_service_road_length_ratio",
    "osm_one_way_road_length_ratio", "osm_tracked_poi_category_entropy",
    "osm_land_use_category_entropy", "osm_commercial_residential_mix_ratio",
    "weather_temperature_2m_c", "weather_relative_humidity_2m_pct",
    "weather_available", "solar_elevation_deg", "solar_azimuth_deg", "is_daylight",
)
LOCAL_HISTORY_ABLATION = (
    "cell_crime_count_6h", "cell_crime_count_24h", "cell_crime_count_7d",
    "cell_crime_count_28d", "cell_violent_count_6h", "cell_violent_count_24h",
    "cell_violent_count_7d", "cell_violent_count_28d",
    "cell_property_count_6h", "cell_property_count_24h",
    "cell_property_count_7d", "cell_property_count_28d", "k1_crime_count_6h",
    "k1_crime_count_24h", "k1_crime_count_7d", "k1_crime_count_28d",
    "has_crime_cell_28d", "hours_since_last_crime_cell_capped_28d",
    "cell_crime_24h_vs_28d_ratio", "cell_share_of_k1_crime_24h",
)
DEFAULT_TRANSFERABLE_CATEGORICAL = ("lighting_condition",)
CITY_HISTORY_ABLATION = (
    "city_crime_count_6h", "city_crime_count_24h", "city_crime_count_7d",
    "city_crime_count_28d", "has_crime_city_28d",
    "hours_since_last_crime_city_capped_28d",
)


@dataclass(frozen=True)
class FeatureContract:
    feature_set: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    contract_hash: str

    @property
    def all_features(self) -> tuple[str, ...]:
        return (*self.numeric, *self.categorical)


def _hash(feature_set: str, numeric: tuple[str, ...], categorical: tuple[str, ...]) -> str:
    payload = json.dumps(
        {"feature_set": feature_set, "numeric": numeric, "categorical": categorical},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def resolve_feature_contract(
    config: dict[str, Any], *, available_columns: Iterable[str]
) -> FeatureContract:
    feature_set = str(config.get("feature_set", "transferable_v2"))
    numeric = tuple(config.get("numeric", DEFAULT_TRANSFERABLE_NUMERIC))
    categorical = tuple(config.get("categorical", DEFAULT_TRANSFERABLE_CATEGORICAL))
    if config.get("include_local_history", False):
        numeric = (*numeric, *LOCAL_HISTORY_ABLATION)
    if config.get("include_city_history", False):
        numeric = (*numeric, *CITY_HISTORY_ABLATION)
    if config.get("include_solar_zenith", False):
        numeric = (*numeric, "solar_zenith_deg")
    combined = (*numeric, *categorical)
    duplicates = sorted({name for name in combined if combined.count(name) > 1})
    if duplicates:
        raise ValueError(f"Feature contract contains duplicates: {duplicates}")
    forbidden = sorted(
        name for name in combined
        if (
            name in FORBIDDEN_PREDICTORS
            or name.startswith("canonical_")
            or name.endswith("_id")
        )
    )
    if forbidden:
        raise ValueError(f"Forbidden predictors requested: {forbidden}")
    if config.get("zero_shot_geography", False):
        validate_zero_shot_feature_contract(numeric, categorical)
    missing = sorted(set(combined) - set(available_columns))
    if missing:
        raise ValueError(f"Configured features missing from model table: {missing}")
    return FeatureContract(
        feature_set=feature_set,
        numeric=numeric,
        categorical=categorical,
        contract_hash=_hash(feature_set, numeric, categorical),
    )


def validate_zero_shot_feature_contract(
    numeric: Iterable[str], categorical: Iterable[str]
) -> None:
    """Fail closed on predictors that encode target-geography crime history."""

    resolved = tuple(map(str, (*tuple(numeric), *tuple(categorical))))
    history = sorted(
        feature
        for feature in resolved
        if feature.startswith(ZERO_SHOT_HISTORY_PREFIXES)
        or feature in ZERO_SHOT_HISTORY_EXACT
    )
    direct = sorted(
        feature
        for feature in resolved
        if feature in FORBIDDEN_PREDICTORS
        or feature.startswith("canonical_")
        or feature.endswith("_id")
    )
    failures: list[str] = []
    if history:
        failures.append("crime-history predictors: " + ", ".join(history))
    if direct:
        failures.append("direct identity/target predictors: " + ", ".join(direct))
    if failures:
        raise ValueError("Zero-shot feature contract violation: " + " | ".join(failures))


__all__ = [
    "CITY_HISTORY_ABLATION", "DEFAULT_TRANSFERABLE_CATEGORICAL",
    "DEFAULT_TRANSFERABLE_NUMERIC", "FORBIDDEN_PREDICTORS", "FeatureContract",
    "LOCAL_HISTORY_ABLATION", "ZERO_SHOT_HISTORY_EXACT",
    "ZERO_SHOT_HISTORY_PREFIXES", "resolve_feature_contract",
    "validate_zero_shot_feature_contract",
]
