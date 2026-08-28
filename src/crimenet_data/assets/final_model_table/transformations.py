"""Deterministic construction and validation of the canonical model table."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import polars as pl
import polars_h3 as plh3

from crimenet_data.assets.crime.sources import SOURCES
from crimenet_data.assets.integration.transforms import MODEL_SPLITS

FINAL_MODEL_TABLE_SCHEMA_VERSION = "final_model_table_v3"


HISTORY_MAX_SECONDS = 28 * 24 * 60 * 60
HISTORY_WINDOWS_SECONDS = {
    "6h": 6 * 60 * 60,
    "24h": 24 * 60 * 60,
    "7d": 7 * 24 * 60 * 60,
    "28d": 28 * 24 * 60 * 60,
}
CITY_TIMEZONES = {
    source_key: source.config.timezone for source_key, source in SOURCES.items()
}
SOURCE_IDS = {city: index for index, city in enumerate(CITY_TIMEZONES)}

SOCIOECONOMIC_FEATURE_COLUMNS = [
    "socio_population",
    "socio_median_age",
    "socio_median_household_income",
    "socio_poverty_rate",
    "socio_unemployment_rate",
    "socio_vacancy_rate",
    "socio_renter_occupied_rate",
    "socio_no_vehicle_rate",
]

STATIC_FEATURE_COLUMNS = [
    "osm_poi_density_per_km2",
    "osm_nightlife_poi_density_per_km2",
    "osm_food_poi_density_per_km2",
    "osm_retail_poi_density_per_km2",
    "osm_transit_poi_density_per_km2",
    "osm_road_length_density_m_per_km2",
    "osm_major_road_density_m_per_km2",
    "osm_intersection_density_per_km2",
    "osm_dead_end_density_per_km2",
    "osm_building_density_per_km2",
    "osm_major_road_length_ratio",
    "osm_residential_road_length_ratio",
    "osm_service_road_length_ratio",
    "osm_one_way_road_length_ratio",
    "osm_tracked_poi_category_entropy",
    "osm_land_use_category_entropy",
    "osm_commercial_residential_mix_ratio",
]

CONTEXT_FEATURE_COLUMNS = [
    "weather_temperature_2m_c",
    "weather_relative_humidity_2m_pct",
    *SOCIOECONOMIC_FEATURE_COLUMNS,
    *STATIC_FEATURE_COLUMNS,
]

LIGHTING_FEATURE_COLUMNS = [
    "solar_elevation_deg",
    "solar_zenith_deg",
    "solar_azimuth_deg",
    "lighting_condition",
    "is_daylight",
]

ENVIRONMENTAL_FEATURE_COLUMNS = [
    "weather_temperature_2m_c",
    "weather_relative_humidity_2m_pct",
    "weather_available",
    *LIGHTING_FEATURE_COLUMNS,
]

HISTORY_FEATURE_COLUMNS = [
    "cell_crime_count_6h",
    "cell_crime_count_24h",
    "cell_crime_count_7d",
    "cell_crime_count_28d",
    "cell_violent_count_6h",
    "cell_violent_count_24h",
    "cell_violent_count_7d",
    "cell_violent_count_28d",
    "cell_property_count_6h",
    "cell_property_count_24h",
    "cell_property_count_7d",
    "cell_property_count_28d",
    "city_crime_count_6h",
    "city_crime_count_24h",
    "city_crime_count_7d",
    "city_crime_count_28d",
    "k1_crime_count_6h",
    "k1_crime_count_24h",
    "k1_crime_count_7d",
    "k1_crime_count_28d",
    "has_crime_cell_28d",
    "hours_since_last_crime_cell_capped_28d",
    "has_crime_city_28d",
    "hours_since_last_crime_city_capped_28d",
    "cell_crime_24h_vs_28d_ratio",
    "cell_share_of_k1_crime_24h",
]

FINAL_COLUMNS = [
    "row_id",
    "model_row_id",
    "row_type",
    "event_indicator",
    "is_observed_event",
    "event_count",
    "integration_weight_cell_seconds",
    "source_city",
    "split",
    "row_year",
    "model_timestamp_utc",
    "row_timestamp_utc",
    "osm_h3_cell_id",
    "weather_query_cell_id",
    "latitude",
    "longitude",
    "integration_sample_id",
    "integration_sample_index",
    "crime_id",
    "canonical_family_code",
    "canonical_offense_family",
    "canonical_subtype_code",
    "canonical_offense_subtype",
    "feature_available_at",
    "feature_version_id",
    "local_hour",
    "local_day_of_week",
    "local_hour_sin",
    "local_hour_cos",
    "local_day_of_week_sin",
    "local_day_of_week_cos",
    *SOCIOECONOMIC_FEATURE_COLUMNS,
    *STATIC_FEATURE_COLUMNS,
    *ENVIRONMENTAL_FEATURE_COLUMNS,
    *HISTORY_FEATURE_COLUMNS,
]


class FinalModelContractError(RuntimeError):
    """A final-model-table input or output violates a hard contract."""



@dataclass(frozen=True)
class ModelSupportInterval:
    """One frozen source/split support interval copied from integration."""

    source_city: str
    split: str
    source_timezone: str
    start_utc: datetime
    end_utc: datetime
    coverage_basis: str
    coverage_reference: str


def _lazy(frame: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    return frame.lazy() if isinstance(frame, pl.DataFrame) else frame


def require_columns(
    frame: pl.DataFrame | pl.LazyFrame,
    columns: Sequence[str],
    *,
    name: str,
) -> None:
    missing = sorted(set(columns) - set(_lazy(frame).collect_schema().names()))
    if missing:
        raise FinalModelContractError(f"{name} missing columns: {missing}")


def validate_model_support_intervals(
    intervals: Sequence[ModelSupportInterval],
) -> None:
    if not intervals:
        raise FinalModelContractError("model support is empty")

    by_source: dict[str, list[ModelSupportInterval]] = {}
    for interval in intervals:
        if interval.split not in MODEL_SPLITS:
            raise FinalModelContractError(
                f"unknown model split in frozen support: {interval.split!r}"
            )
        if interval.start_utc.tzinfo is None or interval.end_utc.tzinfo is None:
            raise FinalModelContractError(
                "model-support interval timestamps must be timezone-aware"
            )
        if interval.start_utc >= interval.end_utc:
            raise FinalModelContractError(
                "model-support intervals must be positive and half-open"
            )
        by_source.setdefault(interval.source_city, []).append(interval)

    for source, source_intervals in by_source.items():
        ordered = sorted(source_intervals, key=lambda interval: interval.start_utc)
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if current.start_utc < previous.end_utc:
                raise FinalModelContractError(
                    f"{source}: frozen model-support intervals overlap: "
                    f"{previous.split} {previous.start_utc.isoformat()}.."
                    f"{previous.end_utc.isoformat()} versus "
                    f"{current.split} {current.start_utc.isoformat()}.."
                    f"{current.end_utc.isoformat()}"
                )


def model_support_expression(
    intervals: Sequence[ModelSupportInterval],
    *,
    timestamp_column: str = "model_timestamp_utc",
) -> pl.Expr:
    if not intervals:
        return pl.lit(False)
    timestamp = pl.col(timestamp_column)
    return pl.any_horizontal(
        *[
            (pl.col("source_city") == interval.source_city)
            & (timestamp >= pl.lit(interval.start_utc))
            & (timestamp < pl.lit(interval.end_utc))
            for interval in intervals
        ]
    )


def split_expression(
    intervals: Sequence[ModelSupportInterval],
    *,
    timestamp_column: str = "model_timestamp_utc",
) -> pl.Expr:
    """Assign the split recorded by frozen integration support."""
    result = pl.lit(None, dtype=pl.String)
    timestamp = pl.col(timestamp_column)
    for interval in reversed(list(intervals)):
        condition = (
            (pl.col("source_city") == interval.source_city)
            & (timestamp >= pl.lit(interval.start_utc))
            & (timestamp < pl.lit(interval.end_utc))
        )
        result = (
            pl.when(condition)
            .then(pl.lit(interval.split))
            .otherwise(result)
        )
    return result


def assign_model_split(
    rows: pl.DataFrame | pl.LazyFrame,
    *,
    support_intervals: Sequence[ModelSupportInterval],
) -> pl.LazyFrame:
    return _lazy(rows).with_columns(
        split_expression(
            support_intervals,
            timestamp_column="model_timestamp_utc",
        ).alias("split")
    )

def _optional_column(
    columns: set[str], name: str, dtype: pl.DataType
) -> pl.Expr:
    return pl.col(name).cast(dtype, strict=False) if name in columns else pl.lit(None, dtype=dtype)


def normalize_event_rows(
    events: pl.DataFrame | pl.LazyFrame,
    *,
    support_intervals: Sequence[ModelSupportInterval],
) -> pl.LazyFrame:
    """Normalize event targets while preserving Event Spine H3 feature state.

    The Event Spine has already selected its leakage-safe national H3/ACS/OSM
    feature version.  Do not re-resolve those features here: carrying them through
    verbatim is part of the final-table temporal-integrity contract.
    """
    events = _lazy(events)
    event_feature_columns = [
        *SOCIOECONOMIC_FEATURE_COLUMNS,
        *STATIC_FEATURE_COLUMNS,
    ]
    require_columns(
        events,
        [
            "crime_id",
            "source_city",
            "occurrence_timestamp_utc",
            "osm_h3_cell_id",
            "weather_query_cell_id",
            "canonical_family_code",
            "canonical_offense_family",
            "canonical_subtype_code",
            "canonical_offense_subtype",
            "feature_available_at",
            "feature_version_id",
            *event_feature_columns,
        ],
        name="event spine",
    )
    columns = set(events.collect_schema().names())
    result = (
        events.select(
            pl.concat_str(
                [pl.lit("event"), pl.col("source_city"), pl.col("crime_id")],
                separator="|",
            ).alias("row_id"),
            pl.lit("event").alias("row_type"),
            pl.lit(1, dtype=pl.Int8).alias("event_indicator"),
            pl.lit(1, dtype=pl.Int8).alias("event_count"),
            pl.lit(None, dtype=pl.Float64).alias(
                "integration_weight_cell_seconds"
            ),
            pl.col("source_city").cast(pl.String),
            pl.col("occurrence_timestamp_utc")
            .cast(pl.Datetime("us", time_zone="UTC"), strict=False)
            .alias("model_timestamp_utc"),
            pl.col("osm_h3_cell_id").cast(pl.Int64, strict=False),
            pl.col("weather_query_cell_id").cast(pl.Int64, strict=False),
            _optional_column(columns, "latitude", pl.Float64).alias("latitude"),
            _optional_column(columns, "longitude", pl.Float64).alias("longitude"),
            pl.lit(None, dtype=pl.String).alias("integration_sample_id"),
            pl.lit(None, dtype=pl.Int64).alias("integration_sample_index"),
            pl.col("crime_id").cast(pl.String),
            pl.col("canonical_family_code").cast(pl.String),
            pl.col("canonical_offense_family").cast(pl.String),
            pl.col("canonical_subtype_code").cast(pl.String),
            pl.col("canonical_offense_subtype").cast(pl.String),
            pl.col("feature_available_at")
            .cast(pl.Datetime("us", time_zone="UTC"), strict=False),
            pl.col("feature_version_id").cast(pl.String, strict=False),
            *event_feature_columns,
        )
        .with_columns(
            pl.col("row_id").alias("model_row_id"),
            pl.lit(True).alias("is_observed_event"),
            pl.col("model_timestamp_utc").alias("row_timestamp_utc"),
            pl.col("model_timestamp_utc").dt.year().cast(pl.Int32).alias("row_year"),
        )
    )
    return assign_model_split(
        result,
        support_intervals=support_intervals,
    ).filter(pl.col("split").is_not_null())

def normalize_integration_rows(
    integration: pl.DataFrame | pl.LazyFrame,
    *,
    support_intervals: Sequence[ModelSupportInterval],
) -> pl.LazyFrame:
    """Normalize split-aware integration samples and preserve MC weights."""
    integration = _lazy(integration)
    require_columns(
        integration,
        [
            "source_city",
            "split",
            "sample_index",
            "integration_timestamp_utc",
            "osm_h3_cell_id",
            "mc_weight_cell_hours",
        ],
        name="integration sampling",
    )
    columns = set(integration.collect_schema().names())
    if "integration_sample_id" in columns:
        sample_id = pl.col("integration_sample_id").cast(pl.String)
    else:
        sample_id = pl.concat_str(
            [
                pl.col("source_city"),
                pl.col("split"),
                pl.col("sample_index").cast(pl.String),
            ],
            separator="|",
        )
    if "integration_weight_cell_seconds" in columns:
        weight = pl.col("integration_weight_cell_seconds").cast(pl.Float64)
    else:
        weight = pl.col("mc_weight_cell_hours").cast(pl.Float64) * 3600.0

    result = (
        integration.select(
            pl.concat_str([pl.lit("integration"), sample_id], separator="|").alias(
                "row_id"
            ),
            pl.lit("integration").alias("row_type"),
            pl.lit(0, dtype=pl.Int8).alias("event_indicator"),
            pl.lit(0, dtype=pl.Int8).alias("event_count"),
            weight.alias("integration_weight_cell_seconds"),
            pl.col("source_city").cast(pl.String),
            pl.col("integration_timestamp_utc")
            .cast(pl.Datetime("us", time_zone="UTC"), strict=False)
            .alias("model_timestamp_utc"),
            pl.col("osm_h3_cell_id").cast(pl.Int64, strict=False),
            plh3.cell_to_parent("osm_h3_cell_id", 6)
            .cast(pl.Int64, strict=False)
            .alias("weather_query_cell_id"),
            _optional_column(columns, "latitude", pl.Float64).alias("latitude"),
            _optional_column(columns, "longitude", pl.Float64).alias("longitude"),
            sample_id.alias("integration_sample_id"),
            pl.col("sample_index").cast(pl.Int64).alias("integration_sample_index"),
            pl.lit(None, dtype=pl.String).alias("crime_id"),
            pl.lit(None, dtype=pl.String).alias("canonical_family_code"),
            pl.lit(None, dtype=pl.String).alias("canonical_offense_family"),
            pl.lit(None, dtype=pl.String).alias("canonical_subtype_code"),
            pl.lit(None, dtype=pl.String).alias("canonical_offense_subtype"),
            pl.col("split").cast(pl.String).alias("_upstream_split"),
        )
        .with_columns(
            pl.col("row_id").alias("model_row_id"),
            pl.lit(False).alias("is_observed_event"),
            pl.col("model_timestamp_utc").alias("row_timestamp_utc"),
            pl.col("model_timestamp_utc").dt.year().cast(pl.Int32).alias("row_year"),
        )
    )
    return assign_model_split(result, support_intervals=support_intervals)

# Public compatibility aliases retained for downstream imports.
prepare_observed_rows = normalize_event_rows
prepare_integration_rows = normalize_integration_rows


def build_query_rows(
    *,
    events: pl.DataFrame | pl.LazyFrame,
    integration: pl.DataFrame | pl.LazyFrame,
    support_intervals: Sequence[ModelSupportInterval],
) -> pl.LazyFrame:
    return pl.concat(
        [
            normalize_event_rows(events, support_intervals=support_intervals),
            normalize_integration_rows(
                integration,
                support_intervals=support_intervals,
            ),
        ],
        how="diagonal_relaxed",
    )

def validate_normalized_rows(
    rows: pl.DataFrame | pl.LazyFrame,
    *,
    expected_rows: int | None = None,
) -> dict[str, int]:
    """Fail closed on normalized identity, structural, and split contracts."""

    rows = _lazy(rows)
    if "_upstream_split" not in rows.collect_schema().names():
        rows = rows.with_columns(pl.lit(None, dtype=pl.String).alias("_upstream_split"))
    structural = [
        "row_id",
        "row_type",
        "source_city",
        "split",
        "model_timestamp_utc",
        "osm_h3_cell_id",
        "weather_query_cell_id",
    ]
    require_columns(
        rows,
        [*structural, "_upstream_split"],
        name="normalized model rows",
    )
    summary = rows.select(
        pl.len().alias("row_count"),
        pl.col("row_id").n_unique().alias("unique_row_ids"),
        pl.any_horizontal(*(pl.col(column).is_null() for column in structural))
        .sum()
        .alias("null_structural_rows"),
        pl.any_horizontal(
            ~plh3.is_valid_cell("osm_h3_cell_id").fill_null(False),
            (plh3.get_resolution("osm_h3_cell_id") != 9).fill_null(True),
            ~plh3.is_valid_cell("weather_query_cell_id").fill_null(False),
            (plh3.get_resolution("weather_query_cell_id") != 6).fill_null(True),
        )
        .sum()
        .alias("invalid_h3_rows"),
        (~pl.col("split").is_in(MODEL_SPLITS)).sum().alias("unknown_split_rows"),
        (
            (pl.col("row_type") == "integration")
            & (pl.col("split") != pl.col("_upstream_split"))
        )
        .sum()
        .alias("integration_split_mismatch_rows"),
    ).collect(engine="streaming").row(0, named=True)
    metrics = {key: int(value or 0) for key, value in summary.items()}
    metrics["duplicate_row_ids"] = (
        metrics["row_count"] - metrics.pop("unique_row_ids")
    )
    failures = {
        name: metrics[name]
        for name in (
            "null_structural_rows",
            "invalid_h3_rows",
            "unknown_split_rows",
            "integration_split_mismatch_rows",
            "duplicate_row_ids",
        )
        if metrics[name]
    }
    if expected_rows is not None and metrics["row_count"] != expected_rows:
        failures["row_count"] = metrics["row_count"]
    if failures:
        raise FinalModelContractError(
            f"normalized model-row validation failed: {failures}"
        )
    return metrics


def attach_environmental_features(
    rows: pl.DataFrame | pl.LazyFrame,
    environmental: pl.DataFrame | pl.LazyFrame,
) -> pl.LazyFrame:
    """Attach frozen H3-r6/hour environmental features with one hash join.

    Uniqueness is an upstream-store contract.  This stage intentionally avoids
    rescanning the environmental store just to re-prove it.
    """

    rows = _lazy(rows)
    environmental = _lazy(environmental)
    prepared = environmental.select(
        pl.col("h3_cell_id").cast(pl.Int64).alias("weather_query_cell_id"),
        pl.col("hour")
        .cast(pl.Datetime("us", time_zone="UTC"), strict=False)
        .alias("_environmental_hour"),
        *ENVIRONMENTAL_FEATURE_COLUMNS,
    )
    return (
        rows.with_columns(
            pl.col("model_timestamp_utc")
            .dt.truncate("1h")
            .alias("_environmental_hour")
        )
        .join(
            prepared,
            on=["weather_query_cell_id", "_environmental_hour"],
            how="left",
        )
        .with_columns(pl.col("weather_available").fill_null(False))
        .drop("_environmental_hour")
    )


def attach_temporal_features(
    rows: pl.DataFrame | pl.LazyFrame,
    history: pl.DataFrame | pl.LazyFrame,
) -> pl.LazyFrame:
    """Backward as-of national features without sorting the wide integration rows.

    The 155M-row integration table is given a temporary integer key.  Only the
    three lookup columns are sorted; the matched features are then joined back by
    the integer key.  This is deliberately RAM-hungry and CPU-efficient.
    """

    rows = _lazy(rows)
    temporary_rid = "_rid" not in rows.collect_schema().names()
    if temporary_rid:
        rows = (
            rows.with_row_index("_rid_local")
            .with_columns(pl.col("_rid_local").cast(pl.UInt64).alias("_rid"))
            .drop("_rid_local")
        )
    history = _lazy(history)
    feature_columns = [*SOCIOECONOMIC_FEATURE_COLUMNS, *STATIC_FEATURE_COLUMNS]

    lookup = (
        rows.select(
            "_rid",
            pl.col("osm_h3_cell_id").cast(pl.Int64),
            pl.col("model_timestamp_utc").cast(
                pl.Datetime("us", time_zone="UTC"), strict=False
            ),
        )
        .sort(["osm_h3_cell_id", "model_timestamp_utc"], multithreaded=True)
        .cache()
    )
    relevant_cells = lookup.select("osm_h3_cell_id").unique()
    right = (
        history.select(
            pl.col("osm_h3_cell_id").cast(pl.Int64),
            pl.col("feature_available_at").cast(
                pl.Datetime("us", time_zone="UTC"), strict=False
            ),
            pl.col("feature_version_id").cast(pl.String, strict=False),
            *feature_columns,
        )
        .join(relevant_cells, on="osm_h3_cell_id", how="semi")
        .sort(["osm_h3_cell_id", "feature_available_at"], multithreaded=True)
    )
    matched = (
        lookup.set_sorted(["osm_h3_cell_id", "model_timestamp_utc"])
        .join_asof(
            right.set_sorted(["osm_h3_cell_id", "feature_available_at"]),
            left_on="model_timestamp_utc",
            right_on="feature_available_at",
            by="osm_h3_cell_id",
            strategy="backward",
            allow_exact_matches=True,
            check_sortedness=False,
        )
        .select(
            "_rid",
            "feature_available_at",
            "feature_version_id",
            *feature_columns,
        )
    )
    result = rows.join(matched, on="_rid", how="left")
    return result.drop("_rid") if temporary_rid else result


def attach_static_features(
    rows: pl.DataFrame | pl.LazyFrame,
    history: pl.DataFrame | pl.LazyFrame,
) -> pl.LazyFrame:
    return attach_temporal_features(rows, history)


def prepare_lighting(lighting: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    lighting = _lazy(lighting)
    columns = set(lighting.collect_schema().names())
    cell = "h3_cell_id" if "h3_cell_id" in columns else "weather_query_cell_id"
    hour = "hour" if "hour" in columns else "solar_timestamp_hour"
    return lighting.select(
        pl.col(cell).cast(pl.Int64).alias("weather_query_cell_id"),
        pl.col(hour).alias("weather_timestamp"),
        *LIGHTING_FEATURE_COLUMNS,
    )


def join_lighting(
    rows: pl.DataFrame | pl.LazyFrame,
    lighting: pl.DataFrame | pl.LazyFrame,
) -> pl.LazyFrame:
    return _lazy(rows).join(
        prepare_lighting(lighting),
        on=["weather_query_cell_id", "weather_timestamp"],
        how="left",
    )


def prepare_history_events(events: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    events = _lazy(events)
    return events.select(
        pl.col("source_city").cast(pl.String),
        pl.col("osm_h3_cell_id").cast(pl.Int64),
        pl.col("occurrence_timestamp_utc").cast(
            pl.Datetime("us", time_zone="UTC"), strict=False
        ),
        pl.col("is_violent").fill_null(False),
        pl.col("is_property").fill_null(False),
    ).filter(
        pl.col("source_city").is_not_null()
        & pl.col("osm_h3_cell_id").is_not_null()
        & pl.col("occurrence_timestamp_utc").is_not_null()
    )


def _event_pulses(events: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate raw events once; all three history grains reuse this table."""

    return (
        events.with_columns(
            pl.col("source_city")
            .replace_strict(SOURCE_IDS, return_dtype=pl.UInt8)
            .alias("_source_id")
        )
        .select(
            "_source_id",
            "osm_h3_cell_id",
            "occurrence_timestamp_utc",
            "is_violent",
            "is_property",
        )
        .group_by(
            ["_source_id", "osm_h3_cell_id", "occurrence_timestamp_utc"]
        )
        .agg(
            pl.len().cast(pl.Int64).alias("_delta_all"),
            pl.col("is_violent").cast(pl.Int64).sum().alias("_delta_violent"),
            pl.col("is_property").cast(pl.Int64).sum().alias("_delta_property"),
        )
        .cache()
    )


def _prefix_from_pulses(pulses: pl.LazyFrame, *, grain: str) -> pl.LazyFrame:
    if grain == "cell":
        groups = ["_source_id", "osm_h3_cell_id"]
        grouped = pulses.select(
            *groups,
            "occurrence_timestamp_utc",
            "_delta_all",
            "_delta_violent",
            "_delta_property",
        )
        deltas = ["_delta_all", "_delta_violent", "_delta_property"]
    elif grain == "city":
        groups = ["_source_id"]
        grouped = pulses.group_by(
            ["_source_id", "occurrence_timestamp_utc"]
        ).agg(pl.col("_delta_all").sum().alias("_delta_all"))
        deltas = ["_delta_all"]
    elif grain == "k1":
        groups = ["_source_id", "osm_h3_cell_id"]
        # grid_disk is evaluated once per distinct event cell, not once per event.
        neighbors = (
            pulses.select(
                "_source_id",
                pl.col("osm_h3_cell_id").alias("_event_cell"),
            )
            .unique()
            .with_columns(
                plh3.grid_disk("_event_cell", 1).alias("osm_h3_cell_id")
            )
            .explode("osm_h3_cell_id", empty_as_null=False)
            .with_columns(pl.col("osm_h3_cell_id").cast(pl.Int64))
        )
        grouped = (
            pulses.select(
                "_source_id",
                "occurrence_timestamp_utc",
                pl.col("osm_h3_cell_id").alias("_event_cell"),
                "_delta_all",
            )
            .join(neighbors, on=["_source_id", "_event_cell"], how="inner")
            .group_by(
                ["_source_id", "osm_h3_cell_id", "occurrence_timestamp_utc"]
            )
            .agg(pl.col("_delta_all").sum().alias("_delta_all"))
        )
        deltas = ["_delta_all"]
    else:
        raise ValueError(f"unknown prefix grain: {grain}")

    ordered = grouped.sort(
        [*groups, "occurrence_timestamp_utc"], multithreaded=True
    )
    return (
        ordered.with_columns(
            pl.col(column)
            .cum_sum()
            .over(groups)
            .alias(column.replace("_delta_", "_cum_"))
            for column in deltas
        )
        .select(
            *groups,
            "occurrence_timestamp_utc",
            *[column.replace("_delta_", "_cum_") for column in deltas],
        )
        .cache()
    )


def _history_features_from_sorted_queries(
    query_base: pl.LazyFrame,
    prefix: pl.LazyFrame,
    *,
    by: list[str],
    cumulative_map: dict[str, str],
    output_prefix: str,
    recency: bool,
) -> pl.LazyFrame:
    """Five causal lookups with one left sort and one right sort per grain.

    Every lookup timestamp is a constant shift of model_timestamp_utc, so the
    ordering is identical.  We assert that ordering with set_sorted and chain the
    five as-of joins instead of sorting and hash-joining five separate results.
    """

    lookup_specs = [("end", 0), *HISTORY_WINDOWS_SECONDS.items()]
    lookups = query_base.with_columns(
        (
            pl.col("model_timestamp_utc") - pl.duration(seconds=seconds)
            if seconds
            else pl.col("model_timestamp_utc")
        ).alias(f"_lookup_{label}")
        for label, seconds in lookup_specs
    )

    result = lookups
    cumulative = list(cumulative_map)
    for label, _seconds in lookup_specs:
        lookup_column = f"_lookup_{label}"
        pulse_column = f"_pulse_{label}"
        right = prefix.select(
            *by,
            pl.col("occurrence_timestamp_utc").alias(pulse_column),
            *[
                pl.col(column).alias(f"{column}_{label}")
                for column in cumulative
            ],
        ).set_sorted([*by, pulse_column])
        result = result.set_sorted([*by, lookup_column]).join_asof(
            right,
            left_on=lookup_column,
            right_on=pulse_column,
            by=by,
            strategy="backward",
            allow_exact_matches=False,
            check_sortedness=False,
        )

    feature_exprs: list[pl.Expr] = []
    for window in HISTORY_WINDOWS_SECONDS:
        for source, target in cumulative_map.items():
            feature_exprs.append(
                (
                    pl.col(f"{source}_end").fill_null(0)
                    - pl.col(f"{source}_{window}").fill_null(0)
                )
                .clip(lower_bound=0)
                .cast(pl.Int64)
                .alias(f"{output_prefix}_{target}_{window}")
            )

    if recency:
        last = pl.col("_pulse_end")
        feature_exprs.extend(
            [
                (
                    last.is_not_null()
                    & (
                        last
                        >= pl.col("model_timestamp_utc")
                        - pl.duration(seconds=HISTORY_MAX_SECONDS)
                    )
                ).alias(f"has_crime_{output_prefix}_28d"),
                pl.when(last.is_not_null())
                .then(
                    (
                        (
                            pl.col("model_timestamp_utc") - last
                        ).dt.total_seconds()
                        / 3600.0
                    ).clip(upper_bound=HISTORY_MAX_SECONDS / 3600.0)
                )
                .otherwise(HISTORY_MAX_SECONDS / 3600.0)
                .alias(
                    f"hours_since_last_crime_{output_prefix}_capped_28d"
                ),
            ]
        )

    return result.select("_rid", *feature_exprs)


def _build_dynamic_history_features(
    *,
    row_keys: pl.LazyFrame,
    history_events: pl.DataFrame | pl.LazyFrame,
) -> pl.LazyFrame:
    events = prepare_history_events(history_events)
    pulses = _event_pulses(events)

    # Cell and k1 use identical query ordering, so this 155M-row sort is shared.
    cell_queries = (
        row_keys.select(
            "_rid", "_source_id", "osm_h3_cell_id", "model_timestamp_utc"
        )
        .sort(
            ["_source_id", "osm_h3_cell_id", "model_timestamp_utc"],
            multithreaded=True,
        )
        .cache()
    )
    city_queries = row_keys.select(
        "_rid", "_source_id", "model_timestamp_utc"
    ).sort(["_source_id", "model_timestamp_utc"], multithreaded=True)

    cell = _history_features_from_sorted_queries(
        cell_queries,
        _prefix_from_pulses(pulses, grain="cell"),
        by=["_source_id", "osm_h3_cell_id"],
        cumulative_map={
            "_cum_all": "crime_count",
            "_cum_violent": "violent_count",
            "_cum_property": "property_count",
        },
        output_prefix="cell",
        recency=True,
    )
    city = _history_features_from_sorted_queries(
        city_queries,
        _prefix_from_pulses(pulses, grain="city"),
        by=["_source_id"],
        cumulative_map={"_cum_all": "crime_count"},
        output_prefix="city",
        recency=True,
    )
    k1 = _history_features_from_sorted_queries(
        cell_queries,
        _prefix_from_pulses(pulses, grain="k1"),
        by=["_source_id", "osm_h3_cell_id"],
        cumulative_map={"_cum_all": "crime_count"},
        output_prefix="k1",
        recency=False,
    )

    return (
        cell.join(city, on="_rid", how="inner")
        .join(k1, on="_rid", how="inner")
        .with_columns(
            (
                (pl.col("cell_crime_count_24h").cast(pl.Float64) + 1.0)
                / (pl.col("cell_crime_count_28d").cast(pl.Float64) + 1.0)
            ).alias("cell_crime_24h_vs_28d_ratio"),
            (
                pl.col("cell_crime_count_24h").cast(pl.Float64)
                / pl.when(pl.col("k1_crime_count_24h") > 0)
                .then(pl.col("k1_crime_count_24h").cast(pl.Float64))
                .otherwise(1.0)
            ).alias("cell_share_of_k1_crime_24h"),
        )
    )


def attach_dynamic_history(
    *,
    rows: pl.DataFrame | pl.LazyFrame,
    history_events: pl.DataFrame | pl.LazyFrame,
) -> pl.LazyFrame:
    rows = _lazy(rows)
    temporary_rid = "_rid" not in rows.collect_schema().names()
    if temporary_rid:
        rows = rows.with_row_index("_rid")

    row_keys = (
        rows.select(
            "_rid", "source_city", "osm_h3_cell_id", "model_timestamp_utc"
        )
        .with_columns(
            pl.col("source_city")
            .replace_strict(SOURCE_IDS, return_dtype=pl.UInt8)
            .alias("_source_id")
        )
        .select("_rid", "_source_id", "osm_h3_cell_id", "model_timestamp_utc")
        .cache()
    )
    features = _build_dynamic_history_features(
        row_keys=row_keys,
        history_events=history_events,
    )
    result = rows.join(features, on="_rid", how="left")
    return result.drop("_rid") if temporary_rid else result


CALENDAR_FEATURE_COLUMNS = [
    "local_hour",
    "local_day_of_week",
    "local_hour_sin",
    "local_hour_cos",
    "local_day_of_week_sin",
    "local_day_of_week_cos",
]


def add_calendar_features(rows: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    """Compute each timezone once per row using a compact UInt8 city key."""

    rows = _lazy(rows)
    columns = set(rows.collect_schema().names())
    temporary_source_id = "_source_id" not in columns
    if temporary_source_id:
        rows = rows.with_columns(
            pl.col("source_city")
            .replace_strict(SOURCE_IDS, return_dtype=pl.UInt8)
            .alias("_source_id")
        )

    branches: list[pl.LazyFrame] = []
    for city, timezone in CITY_TIMEZONES.items():
        source_id = SOURCE_IDS[city]
        branch = (
            rows.filter(pl.col("_source_id") == source_id)
            .with_columns(
                pl.col("model_timestamp_utc")
                .dt.convert_time_zone(timezone)
                .alias("_local_timestamp")
            )
            .with_columns(
                pl.col("_local_timestamp").dt.hour().cast(pl.Int8).alias("local_hour"),
                (pl.col("_local_timestamp").dt.weekday() - 1)
                .cast(pl.Int8)
                .alias("local_day_of_week"),
            )
            .drop("_local_timestamp")
        )
        branches.append(branch)

    result = pl.concat(branches, how="vertical").with_columns(
        (pl.col("local_hour") * (2.0 * math.pi / 24.0)).sin().alias("local_hour_sin"),
        (pl.col("local_hour") * (2.0 * math.pi / 24.0)).cos().alias("local_hour_cos"),
        (pl.col("local_day_of_week") * (2.0 * math.pi / 7.0))
        .sin()
        .alias("local_day_of_week_sin"),
        (pl.col("local_day_of_week") * (2.0 * math.pi / 7.0))
        .cos()
        .alias("local_day_of_week_cos"),
    )
    return result.drop("_source_id") if temporary_source_id else result


def validate_model_support(
    rows: pl.DataFrame | pl.LazyFrame,
    *,
    intervals: Sequence[ModelSupportInterval],
) -> dict[str, int]:
    rows = _lazy(rows)
    outside = int(
        rows.select(
            (~model_support_expression(intervals)).sum().alias("outside")
        )
        .collect(engine="streaming")
        .item()
    )
    if outside:
        raise FinalModelContractError(
            f"model rows outside frozen integration/model support: {outside}"
        )
    return {"rows_outside_frozen_model_support": outside}

def validate_final_model_table(
    rows: pl.DataFrame | pl.LazyFrame,
    *,
    support_intervals: Sequence[ModelSupportInterval],
    expected_rows: int | None = None,
) -> dict[str, object]:
    rows = _lazy(rows)
    require_columns(rows, FINAL_COLUMNS, name="final model table")
    structural = [
        "row_id",
        "row_type",
        "source_city",
        "split",
        "model_timestamp_utc",
        "osm_h3_cell_id",
        "weather_query_cell_id",
    ]
    expected_split = split_expression(
        support_intervals,
        timestamp_column="model_timestamp_utc",
    )
    metrics = rows.select(
        pl.len().alias("row_count"),
        pl.col("row_id").n_unique().alias("unique_row_ids"),
        pl.any_horizontal(*(pl.col(column).is_null() for column in structural))
        .sum()
        .alias("null_structural_rows"),
        (~pl.col("split").is_in(MODEL_SPLITS)).sum().alias("unknown_split_rows"),
        (pl.col("split") != expected_split).sum().alias("split_mismatch_rows"),
        (
            (pl.col("row_type") == "integration")
            & pl.col("_upstream_split").is_not_null()
            & (pl.col("split") != pl.col("_upstream_split"))
        )
        .sum()
        .alias("integration_split_mismatch_rows"),
        (
            (pl.col("row_type") == "event")
            & ((pl.col("event_indicator") != 1) | (pl.col("event_count") != 1))
        )
        .sum()
        .alias("invalid_event_semantics_rows"),
        (
            (pl.col("row_type") == "integration")
            & ((pl.col("event_indicator") != 0) | (pl.col("event_count") != 0))
        )
        .sum()
        .alias("invalid_integration_semantics_rows"),
        (
            (pl.col("row_type") == "event")
            & pl.col("integration_weight_cell_seconds").is_not_null()
        )
        .sum()
        .alias("event_weight_rows"),
        (
            (pl.col("row_type") == "integration")
            & (
                pl.col("integration_weight_cell_seconds").is_null()
                | ~pl.col("integration_weight_cell_seconds").is_finite()
                | (pl.col("integration_weight_cell_seconds") <= 0)
            )
        )
        .sum()
        .alias("invalid_integration_weight_rows"),
        (~pl.col("_environmental_row_exists")).sum().alias(
            "structural_environmental_missing_rows"
        ),
        (
            pl.col("feature_available_at").is_not_null()
            & (pl.col("feature_available_at") > pl.col("model_timestamp_utc"))
        )
        .sum()
        .alias("future_feature_rows"),
        (~pl.col("weather_available")).sum().alias("weather_unavailable_rows"),
        pl.any_horizontal(
            *(pl.col(column).is_null() for column in LIGHTING_FEATURE_COLUMNS)
        ).sum().alias("lighting_missing_rows"),
        (
            (pl.col("row_type") == "event")
            & pl.col("feature_available_at").is_null()
        ).sum().alias("event_h3_feature_missing_rows"),
        (
            (pl.col("row_type") == "integration")
            & pl.col("feature_available_at").is_null()
        ).sum().alias("integration_h3_feature_missing_rows"),
    ).collect(engine="streaming").row(0, named=True)
    metrics = {key: int(value or 0) for key, value in metrics.items()}
    if expected_rows is not None and metrics["row_count"] != expected_rows:
        raise FinalModelContractError(
            "final model-table cardinality changed: "
            f"expected={expected_rows}, actual={metrics['row_count']}"
        )
    duplicate_rows = metrics["row_count"] - metrics.pop("unique_row_ids")
    metrics["duplicate_row_ids"] = duplicate_rows
    fatal_names = {
        "null_structural_rows",
        "unknown_split_rows",
        "split_mismatch_rows",
        "integration_split_mismatch_rows",
        "invalid_event_semantics_rows",
        "invalid_integration_semantics_rows",
        "event_weight_rows",
        "invalid_integration_weight_rows",
        "structural_environmental_missing_rows",
        "lighting_missing_rows",
        "future_feature_rows",
        "duplicate_row_ids",
    }
    failures = {name: metrics[name] for name in fatal_names if metrics[name]}
    if metrics["row_count"] == 0:
        failures["row_count"] = 0
    if failures:
        raise FinalModelContractError(
            f"final model-table validation failed: {failures}"
        )
    return metrics


# Kept as a named validator for callers that validate a complete enriched slice.
validate_final_partition = validate_final_model_table

def audit_final_model_table(
    rows: pl.DataFrame | pl.LazyFrame,
) -> list[dict[str, object]]:
    rows = _lazy(rows)
    null_features = [
        "weather_temperature_2m_c",
        "weather_relative_humidity_2m_pct",
        *SOCIOECONOMIC_FEATURE_COLUMNS,
        *STATIC_FEATURE_COLUMNS,
    ]
    return (
        rows.group_by("source_city", "split", "row_type")
        .agg(
            pl.len().alias("row_count"),
            pl.col("osm_h3_cell_id").n_unique().alias("unique_h3_cells"),
            pl.col("model_timestamp_utc").min().alias("min_timestamp_utc"),
            pl.col("model_timestamp_utc").max().alias("max_timestamp_utc"),
            (pl.col("row_type") == "event").sum().alias("event_count"),
            (pl.col("row_type") == "integration").sum().alias(
                "integration_count"
            ),
            pl.col("weather_available").sum().alias("weather_available_count"),
            (~pl.col("weather_available")).sum().alias("weather_unavailable_count"),
            pl.col("feature_available_at").count().alias("national_h3_matched_count"),
            pl.col("feature_available_at").null_count().alias(
                "national_h3_unmatched_count"
            ),
            pl.col("integration_weight_cell_seconds").count().alias(
                "integration_weight_count"
            ),
            pl.col("integration_weight_cell_seconds").min().alias(
                "integration_weight_min"
            ),
            pl.col("integration_weight_cell_seconds").max().alias(
                "integration_weight_max"
            ),
            pl.col("integration_weight_cell_seconds").sum().alias(
                "integration_weight_sum"
            ),
            *[
                pl.col(column).null_count().alias(f"null__{column}")
                for column in null_features
            ],
        )
        .with_columns(
            (
                100.0 * pl.col("weather_available_count") / pl.col("row_count")
            ).alias("weather_available_pct"),
            (
                100.0 * pl.col("national_h3_matched_count") / pl.col("row_count")
            ).alias("national_h3_matched_pct"),
            *[
                (100.0 * pl.col(f"null__{column}") / pl.col("row_count")).alias(
                    f"null_pct__{column}"
                )
                for column in null_features
            ],
        )
        .sort("source_city", "split", "row_type")
        .collect(engine="streaming")
        .to_dicts()
    )


def finalize_model_table(rows: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    rows = _lazy(rows)
    require_columns(rows, FINAL_COLUMNS, name="final model table")
    return rows.select(FINAL_COLUMNS)


def build_final_model_table(
    *,
    events: pl.DataFrame | pl.LazyFrame,
    integration: pl.DataFrame | pl.LazyFrame,
    environmental: pl.DataFrame | pl.LazyFrame,
    temporal_history: pl.DataFrame | pl.LazyFrame,
    support_intervals: Sequence[ModelSupportInterval],
) -> tuple[pl.LazyFrame, dict[str, object]]:
    """Build once; validation/auditing belongs on the persisted snapshot.

    This plan is tuned for a high-core, high-RAM host:
      * integration national features sort only a thin integer-key lookup frame;
      * one cached integer row key is shared by context/history enrichment;
      * cell+k1 share one query-row sort;
      * history uses five chained as-of probes per grain without repeated sorts;
      * H3 k1 adjacency is computed once per distinct event cell;
      * no collect() occurs in the expensive build graph.
    """

    validate_model_support_intervals(support_intervals)

    # Give each side a compact globally unique temporary UInt64 key before any
    # branching.  The top bit separates integration rows from event rows.
    event_rows = (
        normalize_event_rows(
            events,
            support_intervals=support_intervals,
        )
        .with_row_index("_rid_local")
        .with_columns(pl.col("_rid_local").cast(pl.UInt64).alias("_rid"))
        .drop("_rid_local")
    )
    integration_base = (
        normalize_integration_rows(
            integration,
            support_intervals=support_intervals,
        )
        .with_row_index("_rid_local")
        .with_columns(
            (
                pl.col("_rid_local").cast(pl.UInt64)
                + pl.lit(1 << 63, dtype=pl.UInt64)
            ).alias("_rid")
        )
        .drop("_rid_local")
    )
    integration_rows = attach_temporal_features(
        integration_base,
        temporal_history,
    )

    rows = pl.concat(
        [event_rows, integration_rows],
        how="diagonal_relaxed",
    )

    # Cache only the narrow hot set.  On a large-RAM machine this avoids repeatedly
    # rescanning/recomputing the wide event+integration branch while keeping the
    # full final table lazy/streamable.
    row_keys = (
        rows.select(
            "_rid",
            "source_city",
            "model_timestamp_utc",
            "osm_h3_cell_id",
            "weather_query_cell_id",
        )
        .with_columns(
            pl.col("source_city")
            .replace_strict(SOURCE_IDS, return_dtype=pl.UInt8)
            .alias("_source_id")
        )
        .select(
            "_rid",
            "_source_id",
            "model_timestamp_utc",
            "osm_h3_cell_id",
            "weather_query_cell_id",
        )
        .cache()
    )

    environmental_features = attach_environmental_features(
        row_keys.select(
            "_rid", "weather_query_cell_id", "model_timestamp_utc"
        ),
        environmental,
    ).select("_rid", *ENVIRONMENTAL_FEATURE_COLUMNS)

    calendar_features = add_calendar_features(
        row_keys.select("_rid", "_source_id", "model_timestamp_utc")
    ).select("_rid", *CALENDAR_FEATURE_COLUMNS)

    history_features = _build_dynamic_history_features(
        row_keys=row_keys,
        history_events=events,
    )

    feature_bundle = environmental_features.join(
        calendar_features, on="_rid", how="inner"
    ).join(history_features, on="_rid", how="inner")

    final = (
        rows.join(feature_bundle, on="_rid", how="left")
        .drop("_rid")
    )
    return finalize_model_table(final), {
        "build_execution_policy": "single_expensive_sink_then_persisted_audit",
        "history_query_sort_policy": "shared_cell_k1_plus_city",
        "history_asof_policy": "five_chained_probes_per_grain_no_resort",
        "history_join_key": "temporary_integer_row_index",
        "k1_adjacency_policy": "once_per_distinct_event_cell",
        "calendar_policy": "one_timezone_conversion_per_row_via_city_branches",
        "integration_h3_enrichment_policy": "thin_backward_asof_national_temporal_history",
        "environmental_enrichment_policy": "exact_h3_r6_utc_hour_left_join",
    }


__all__ = [
    "CALENDAR_FEATURE_COLUMNS",
    "CONTEXT_FEATURE_COLUMNS",
    "ENVIRONMENTAL_FEATURE_COLUMNS",
    "FINAL_COLUMNS",
    "FINAL_MODEL_TABLE_SCHEMA_VERSION",
    "FinalModelContractError",
    "HISTORY_FEATURE_COLUMNS",
    "LIGHTING_FEATURE_COLUMNS",
    "MODEL_SPLITS",
    "ModelSupportInterval",
    "SOCIOECONOMIC_FEATURE_COLUMNS",
    "SOURCE_IDS",
    "STATIC_FEATURE_COLUMNS",
    "add_calendar_features",
    "assign_model_split",
    "attach_dynamic_history",
    "attach_environmental_features",
    "attach_static_features",
    "attach_temporal_features",
    "audit_final_model_table",
    "build_final_model_table",
    "build_query_rows",
    "finalize_model_table",
    "join_lighting",
    "model_support_expression",
    "normalize_event_rows",
    "normalize_integration_rows",
    "prepare_history_events",
    "prepare_integration_rows",
    "prepare_lighting",
    "prepare_observed_rows",
    "require_columns",
    "split_expression",
    "validate_final_model_table",
    "validate_final_partition",
    "validate_model_support",
    "validate_model_support_intervals",
    "validate_normalized_rows",
]
