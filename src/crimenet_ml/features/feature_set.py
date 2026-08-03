"""Versioned feature lists ported from the CrimeNet Poisson implementation."""

from dataclasses import dataclass

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "mark": ("offense_mark",),
    "temporal": (
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "is_weekend",
        "is_late_night",
        "is_weekend_night",
    ),
    "history": (
        "cell_count_6h",
        "cell_count_24h",
        "cell_count_7d",
        "cell_count_28d",
        "hours_since_last_cell",
        "same_mark_cell_count_6h",
        "same_mark_cell_count_24h",
        "same_mark_cell_count_7d",
        "same_mark_cell_count_28d",
        "hours_since_last_same_mark_cell",
        "city_count_6h",
        "city_count_24h",
        "city_count_7d",
        "city_count_28d",
        "hours_since_last_city",
        "k1_count_6h",
        "k1_count_24h",
        "k1_count_7d",
        "k1_count_28d",
        "hours_since_last_k1",
        "same_mark_k1_count_6h",
        "same_mark_k1_count_24h",
        "same_mark_k1_count_7d",
        "same_mark_k1_count_28d",
        "hours_since_last_same_mark_k1",
        "crime_24h_vs_28d_ratio",
        "same_mark_24h_vs_28d_ratio",
        "cell_share_of_k1_crime_24h",
        "local_neighbor_crime_difference_24h",
        "local_neighbor_crime_difference_7d",
    ),
    "osm": (
        "log1p_poi_density",
        "log1p_nightlife_density",
        "log1p_retail_density",
        "log1p_transit_density",
        "log1p_road_density",
        "major_road_length_ratio",
        "intersection_density_per_km2",
        "dead_end_to_intersection_ratio",
        "log1p_building_density",
        "commercial_land_use_share",
        "green_space_share",
        "land_use_category_entropy",
    ),
    "socioeconomic": (
        "median_age",
        "log_median_household_income",
        "poverty_rate",
        "unemployment_rate",
        "vacancy_rate",
        "renter_occupied_rate",
        "no_vehicle_rate",
        "tract_population_density_per_km2",
        "acs_uncertainty_score",
    ),
    "neighborhood_context": (
        "poi_density_local_minus_k1",
        "nightlife_density_local_minus_k1",
        "road_density_local_minus_k1",
        "commercial_share_local_minus_k1",
        "poverty_rate_local_minus_k1",
    ),
    "weather": (
        "temperature_2m_c",
        "temperature_change_6h",
        "temperature_mean_24h",
        "temperature_range_24h",
        "temperature_monthly_hour_zscore",
    ),
    "lighting": ("solar_elevation_deg", "is_full_darkness"),
    "availability": (
        "has_osm_context",
        "has_socioeconomic_context",
        "weather_feature_available",
        "lighting_feature_available",
    ),
}


@dataclass(frozen=True)
class FeatureSet:
    name: str
    features: tuple[str, ...]
    categorical: tuple[str, ...] = ("offense_mark",)

    def __post_init__(self) -> None:
        if len(self.features) != len(set(self.features)):
            raise ValueError(f"Duplicate features in {self.name}")
        if not set(self.categorical).issubset(self.features):
            raise ValueError("Categorical columns must be model features")


HISTORY_V1 = FEATURE_GROUPS["mark"] + FEATURE_GROUPS["temporal"] + FEATURE_GROUPS["history"]
CORE_V1 = tuple(feature for group in FEATURE_GROUPS.values() for feature in group)
FEATURE_SETS = {
    "history_v1": FeatureSet("history_v1", HISTORY_V1),
    "core_v1": FeatureSet("core_v1", CORE_V1),
}

assert len(HISTORY_V1) == 40
assert len(CORE_V1) == 77


def get_feature_set(name: str) -> FeatureSet:
    try:
        return FEATURE_SETS[name]
    except KeyError as exc:
        choices = sorted(FEATURE_SETS)
        raise ValueError(f"Unknown feature set {name!r}; choose from {choices}") from exc
