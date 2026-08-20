from __future__ import annotations

from datetime import UTC, datetime
import math

import polars as pl
import polars_h3 as plh3


# =============================================================================
# Temporal contract
# =============================================================================


TRAIN_END_UTC = datetime(
    2024,
    1,
    1,
    tzinfo=UTC,
)

VALIDATION_END_UTC = datetime(
    2025,
    1,
    1,
    tzinfo=UTC,
)

HISTORY_MAX_SECONDS = (
    28 * 24 * 60 * 60
)

HISTORY_WINDOWS_SECONDS = {
    "6h":
        6 * 60 * 60,

    "24h":
        24 * 60 * 60,

    "7d":
        7 * 24 * 60 * 60,

    "28d":
        28 * 24 * 60 * 60,
}


CITY_TIMEZONES = {
    "baltimore":
        "America/New_York",

    "chicago":
        "America/Chicago",

    "dallas":
        "America/Chicago",

    "fort_worth":
        "America/Chicago",

    "new_york":
        "America/New_York",

    "san_francisco":
        "America/Los_Angeles",

    "seattle":
        "America/Los_Angeles",

    "washington_dc":
        "America/New_York",
}


# =============================================================================
# Model feature contract
# =============================================================================


CONTEXT_FEATURE_COLUMNS = [
    # Weather: intentionally only the fields
    # proven to contain useful data.
    "weather_temperature_2m_c",
    "weather_relative_humidity_2m_pct",

    # Socioeconomic.
    "socio_population",
    "socio_median_age",
    "socio_median_household_income",
    "socio_poverty_rate",
    "socio_unemployment_rate",
    "socio_vacancy_rate",
    "socio_renter_occupied_rate",
    "socio_no_vehicle_rate",

    # OSM: derived structural features only.
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


LIGHTING_FEATURE_COLUMNS = [
    "solar_elevation_deg",
    "lighting_condition",
    "is_daylight",
]


HISTORY_FEATURE_COLUMNS = [
    # Exact H3 cell.
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

    # City-wide.
    "city_crime_count_6h",
    "city_crime_count_24h",
    "city_crime_count_7d",
    "city_crime_count_28d",

    # H3 k=1 neighborhood.
    "k1_crime_count_6h",
    "k1_crime_count_24h",
    "k1_crime_count_7d",
    "k1_crime_count_28d",

    # Recency.
    "has_crime_cell_28d",
    "hours_since_last_crime_cell_capped_28d",

    "has_crime_city_28d",
    "hours_since_last_crime_city_capped_28d",

    # Relative state.
    "cell_crime_24h_vs_28d_ratio",
    "cell_share_of_k1_crime_24h",
]


FINAL_COLUMNS = [
    # Loss/control.
    "model_row_id",
    "row_type",
    "is_observed_event",
    "event_count",
    "integration_weight_cell_seconds",

    # Domain.
    "source_city",
    "split",
    "row_year",
    "row_timestamp_utc",

    # Spatial keys.
    "osm_h3_cell_id",
    "weather_query_cell_id",

    # Observed-event marks.
    # NULL for Monte Carlo rows; these are labels/control,
    # not general predictors.
    "canonical_family_code",
    "canonical_offense_family",
    "canonical_subtype_code",
    "canonical_offense_subtype",

    # Calendar.
    "local_hour",
    "local_day_of_week",
    "local_hour_sin",
    "local_hour_cos",
    "local_day_of_week_sin",
    "local_day_of_week_cos",

    *CONTEXT_FEATURE_COLUMNS,
    *LIGHTING_FEATURE_COLUMNS,
    *HISTORY_FEATURE_COLUMNS,
]


# =============================================================================
# Helpers
# =============================================================================


def require_columns(
    frame: pl.LazyFrame,
    columns: list[str],
    *,
    name: str,
) -> None:
    schema = (
        frame.collect_schema()
    )

    missing = [
        column
        for column in columns
        if column not in schema
    ]

    if missing:
        raise ValueError(
            f"{name} missing columns: "
            f"{missing}"
        )


def split_expression(
    timestamp_column: str,
) -> pl.Expr:
    timestamp = pl.col(
        timestamp_column
    )

    return (
        pl.when(
            timestamp
            < pl.lit(TRAIN_END_UTC)
        )
        .then(
            pl.lit("train")
        )

        .when(
            timestamp
            < pl.lit(
                VALIDATION_END_UTC
            )
        )
        .then(
            pl.lit("validation")
        )

        .otherwise(
            pl.lit("test")
        )
    )



# =============================================================================
# Query rows
# =============================================================================


def prepare_observed_rows(
    events: pl.LazyFrame,
) -> pl.LazyFrame:
    require_columns(
        events,
        [
            "crime_id",
            "source_city",
            "source_timezone",
            "occurrence_timestamp_utc",
            "weather_timestamp",
            "osm_h3_cell_id",
            "weather_query_cell_id",

            "canonical_family_code",
            "canonical_offense_family",
            "canonical_subtype_code",
            "canonical_offense_subtype",

            "include_in_model",
            "is_criminal_event",

            *CONTEXT_FEATURE_COLUMNS,
        ],
        name="event_spine",
    )

    return (
        events

        .filter(
            pl.col(
                "include_in_model"
            )
            .fill_null(False)

            &

            pl.col(
                "is_criminal_event"
            )
            .fill_null(False)

            &

            pl.col(
                "occurrence_timestamp_utc"
            )
            .is_not_null()

            &

            pl.col(
                "osm_h3_cell_id"
            )
            .is_not_null()

            &

            pl.col(
                "weather_query_cell_id"
            )
            .is_not_null()
        )

        .with_columns(
            pl.concat_str(
                [
                    pl.lit("observed"),
                    pl.col("source_city"),
                    pl.col("crime_id"),
                ],
                separator="|",
            )
            .alias(
                "model_row_id"
            ),

            pl.lit(
                "observed_event"
            )
            .alias(
                "row_type"
            ),

            pl.lit(
                True
            )
            .alias(
                "is_observed_event"
            ),

            pl.lit(
                1,
                dtype=pl.Int8,
            )
            .alias(
                "event_count"
            ),

            # Observed log-intensity term has no
            # quadrature/integration weight.
            pl.lit(
                0.0,
                dtype=pl.Float64,
            )
            .alias(
                "integration_weight_cell_seconds"
            ),

            pl.col(
                "occurrence_timestamp_utc"
            )
            .cast(
                pl.Datetime(
                    "us",
                    time_zone="UTC",
                )
            )
            .alias(
                "row_timestamp_utc"
            ),
        )

        .with_columns(
            split_expression(
                "row_timestamp_utc"
            )
            .alias("split"),

            pl.col(
                "row_timestamp_utc"
            )
            .dt.year()
            .cast(pl.Int32)
            .alias("row_year"),
        )

        .select(
            "model_row_id",
            "row_type",
            "is_observed_event",
            "event_count",
            "integration_weight_cell_seconds",

            "source_city",
            "source_timezone",
            "split",
            "row_year",
            "row_timestamp_utc",

            "osm_h3_cell_id",
            "weather_query_cell_id",
            "weather_timestamp",

            "canonical_family_code",
            "canonical_offense_family",
            "canonical_subtype_code",
            "canonical_offense_subtype",

            *CONTEXT_FEATURE_COLUMNS,
        )
    )


def prepare_integration_rows(
    integration: pl.LazyFrame,
) -> pl.LazyFrame:
    require_columns(
        integration,
        [
            "integration_sample_id",
            "source_city",
            "source_timezone",
            "sample_timestamp_utc",
            "weather_timestamp",
            "osm_h3_cell_id",
            "weather_query_cell_id",
            "integration_weight_cell_seconds",

            *CONTEXT_FEATURE_COLUMNS,
        ],
        name="integration_context",
    )

    return (
        integration

        .with_columns(
            pl.concat_str(
                [
                    pl.lit("integration"),
                    pl.col(
                        "integration_sample_id"
                    ),
                ],
                separator="|",
            )
            .alias(
                "model_row_id"
            ),

            pl.lit(
                "integration_sample"
            )
            .alias(
                "row_type"
            ),

            pl.lit(False)
            .alias(
                "is_observed_event"
            ),

            pl.lit(
                0,
                dtype=pl.Int8,
            )
            .alias(
                "event_count"
            ),

            pl.col(
                "sample_timestamp_utc"
            )
            .cast(
                pl.Datetime(
                    "us",
                    time_zone="UTC",
                )
            )
            .alias(
                "row_timestamp_utc"
            ),

            # No observed mark exists for a
            # Monte Carlo integration row.
            pl.lit(
                None,
                dtype=pl.String,
            )
            .alias(
                "canonical_family_code"
            ),

            pl.lit(
                None,
                dtype=pl.String,
            )
            .alias(
                "canonical_offense_family"
            ),

            pl.lit(
                None,
                dtype=pl.String,
            )
            .alias(
                "canonical_subtype_code"
            ),

            pl.lit(
                None,
                dtype=pl.String,
            )
            .alias(
                "canonical_offense_subtype"
            ),
        )

        .with_columns(
            split_expression(
                "row_timestamp_utc"
            )
            .alias("split"),

            pl.col(
                "row_timestamp_utc"
            )
            .dt.year()
            .cast(pl.Int32)
            .alias("row_year"),
        )

        .select(
            "model_row_id",
            "row_type",
            "is_observed_event",
            "event_count",
            "integration_weight_cell_seconds",

            "source_city",
            "source_timezone",
            "split",
            "row_year",
            "row_timestamp_utc",

            "osm_h3_cell_id",
            "weather_query_cell_id",
            "weather_timestamp",

            "canonical_family_code",
            "canonical_offense_family",
            "canonical_subtype_code",
            "canonical_offense_subtype",

            *CONTEXT_FEATURE_COLUMNS,
        )
    )


def build_query_rows(
    *,
    events: pl.LazyFrame,
    integration: pl.LazyFrame,
) -> pl.LazyFrame:
    return pl.concat(
        [
            prepare_observed_rows(
                events
            ),

            prepare_integration_rows(
                integration
            ),
        ],
        how="vertical_relaxed",
    )


# =============================================================================
# Lighting
# =============================================================================


def prepare_lighting(
    lighting: pl.LazyFrame,
) -> pl.LazyFrame:
    return (
        lighting
        .select(
            pl.col(
                "weather_query_cell_id"
            )
            .cast(pl.Int64),

            pl.col(
                "solar_timestamp_hour"
            )
            .alias(
                "weather_timestamp"
            ),

            "solar_elevation_deg",
            "lighting_condition",
            "is_daylight",
        )
        .with_columns(
            pl.lit(True)
            .alias(
                "_lighting_matched"
            )
        )
    )


def join_lighting(
    rows: pl.LazyFrame,
    lighting: pl.LazyFrame,
) -> pl.LazyFrame:
    return rows.join(
        prepare_lighting(
            lighting
        ),
        on=[
            "weather_query_cell_id",
            "weather_timestamp",
        ],
        how="left",
        validate="m:1",
    )


# =============================================================================
# History event source
# =============================================================================

def prepare_history_events(
    events: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Historical state is defined by event occurrence time.

    An event may contribute to a query at time t iff:

        occurrence_timestamp_utc < t

    Exact equality is forbidden later by the strict as-of lookup.
    """

    return (
        events
        .filter(
            pl.col("include_in_model")
            .fill_null(False)

            & pl.col("is_criminal_event")
            .fill_null(False)

            & pl.col("osm_h3_cell_id")
            .is_not_null()

            & pl.col("occurrence_timestamp_utc")
            .is_not_null()
        )
        .select(
            "crime_id",
            "source_city",

            pl.col("osm_h3_cell_id")
            .cast(pl.Int64),

            pl.col("occurrence_timestamp_utc")
            .cast(
                pl.Datetime(
                    "us",
                    time_zone="UTC",
                )
            ),

            pl.col("is_violent")
            .fill_null(False),

            pl.col("is_property")
            .fill_null(False),
        )
    )

# =============================================================================
# Prefix-state tables
# =============================================================================


def build_cell_prefix(
    events: pl.LazyFrame,
) -> pl.LazyFrame:
    groups = [
        "source_city",
        "osm_h3_cell_id",
    ]

    return (
        events

        .group_by(
            [
                *groups,
                "occurrence_timestamp_utc",
            ]
        )

        .agg(
            pl.len()
            .cast(pl.Int64)
            .alias("_delta_all"),

            pl.col("is_violent")
            .cast(pl.Int64)
            .sum()
            .alias("_delta_violent"),

            pl.col("is_property")
            .cast(pl.Int64)
            .sum()
            .alias("_delta_property"),
        )

        .sort(
            [
                *groups,
                "occurrence_timestamp_utc",
            ]
        )

        .with_columns(
            pl.col("_delta_all")
            .cum_sum()
            .over(groups)
            .alias("_cum_all"),

            pl.col("_delta_violent")
            .cum_sum()
            .over(groups)
            .alias("_cum_violent"),

            pl.col("_delta_property")
            .cum_sum()
            .over(groups)
            .alias("_cum_property"),
        )
    )


def build_city_prefix(
    events: pl.LazyFrame,
) -> pl.LazyFrame:
    groups = [
        "source_city",
    ]

    return (
        events

        .group_by(
            [
                *groups,
                "occurrence_timestamp_utc",
            ]
        )

        .agg(
            pl.len()
            .cast(pl.Int64)
            .alias("_delta_all")
        )

        .sort(
            [
                *groups,
                "occurrence_timestamp_utc",
            ]
        )

        .with_columns(
            pl.col("_delta_all")
            .cum_sum()
            .over(groups)
            .alias("_cum_all")
        )
    )


def build_k1_prefix(
    events: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Each reported crime contributes to its own H3-9
    cell and all H3-9 cells at grid distance 1.
    """

    groups = [
        "source_city",
        "osm_h3_cell_id",
    ]

    return (
        events

        .select(
            "source_city",
            "occurrence_timestamp_utc",
            "osm_h3_cell_id",
        )

        .with_columns(
            plh3.grid_disk(
                "osm_h3_cell_id",
                1,
            )
            .alias(
                "_target_h3_cells"
            )
        )

        .explode(
            "_target_h3_cells"
        )

        .select(
            "source_city",
            "occurrence_timestamp_utc",

            pl.col(
                "_target_h3_cells"
            )
            .cast(pl.Int64)
            .alias(
                "osm_h3_cell_id"
            ),
        )

        .group_by(
            [
                *groups,
                "occurrence_timestamp_utc",
            ]
        )

        .agg(
            pl.len()
            .cast(pl.Int64)
            .alias("_delta_all")
        )

        .sort(
            [
                *groups,
                "occurrence_timestamp_utc",
            ]
        )

        .with_columns(
            pl.col("_delta_all")
            .cum_sum()
            .over(groups)
            .alias("_cum_all")
        )
    )


# =============================================================================
# STRICT as-of lookup
# =============================================================================


def prefix_lookup(
    *,
    rows: pl.LazyFrame,
    prefix: pl.LazyFrame,
    by: list[str],
    query_time: pl.Expr,
    cumulative_columns: list[str],
    suffix: str,
    include_pulse_time: bool = False,
) -> pl.LazyFrame:
    """
    STRICT backward occurrence-time lookup:

        historical.occurrence_timestamp_utc < query_timestamp

    NOT <=.

    This prevents:
    - self-history leakage
    - simultaneous-event leakage
    - future-event leakage
    """
    left = (
        rows
        .select(
            "model_row_id",
            *by,

            query_time.alias(
                "_lookup_time"
            ),
        )
        .sort(
            [
                *by,
                "_lookup_time",
            ]
        )
    )

    right = (
        prefix
        .select(
            *by,

            pl.col("occurrence_timestamp_utc")
            .alias("_pulse_time"),

            *cumulative_columns,
        )
        .sort(
            [
                *by,
                "_pulse_time",
            ]
        )
    )

    joined = left.join_asof(
        right,

        left_on="_lookup_time",
        right_on="_pulse_time",

        by=by,

        strategy="backward",

        # CRITICAL:
        # exact timestamp equality is forbidden.
        allow_exact_matches=False,

        check_sortedness=False,

        coalesce=False,
    )

    selected = [
        pl.col("model_row_id"),
    ]

    if include_pulse_time:
        selected.append(
            pl.col(
                "_pulse_time"
            )
            .alias(f"_last_occurrence_at_utc{suffix}")
        )

    selected.extend(
        [
            pl.col(column)
            .fill_null(0)
            .cast(pl.Int64)
            .alias(
                f"{column}{suffix}"
            )
            for column
            in cumulative_columns
        ]
    )

    return joined.select(
        selected
    )


# =============================================================================
# Generic rolling state
# =============================================================================


def attach_prefix_history(
    *,
    rows: pl.LazyFrame,
    prefix: pl.LazyFrame,
    by: list[str],
    cumulative_map: dict[str, str],
    output_prefix: str,
    add_recency: bool,
) -> pl.LazyFrame:
    cumulative_columns = list(
        cumulative_map
    )

    end = prefix_lookup(
        rows=rows,
        prefix=prefix,
        by=by,

        query_time=
            pl.col(
                "row_timestamp_utc"
            ),

        cumulative_columns=
            cumulative_columns,

        suffix="_end",

        include_pulse_time=
            add_recency,
    )

    result = rows.join(
        end,
        on="model_row_id",
        how="left",
        validate="1:1",
    )

    for (
        window_name,
        seconds,
    ) in (
        HISTORY_WINDOWS_SECONDS
        .items()
    ):
        lower = prefix_lookup(
            rows=rows,
            prefix=prefix,
            by=by,

            query_time=(
                pl.col(
                    "row_timestamp_utc"
                )
                -
                pl.duration(
                    seconds=seconds
                )
            ),

            cumulative_columns=
                cumulative_columns,

            suffix=
                f"_lower_{window_name}",

            include_pulse_time=False,
        )

        result = result.join(
            lower,
            on="model_row_id",
            how="left",
            validate="1:1",
        )

        expressions = []

        for (
            cumulative_column,
            output_base,
        ) in cumulative_map.items():
            expressions.append(
                (
                    pl.col(
                        f"{cumulative_column}"
                        "_end"
                    )
                    .fill_null(0)

                    -

                    pl.col(
                        f"{cumulative_column}"
                        f"_lower_{window_name}"
                    )
                    .fill_null(0)
                )
                .clip(
                    lower_bound=0
                )
                .cast(pl.Int64)
                .alias(
                    f"{output_prefix}_"
                    f"{output_base}_"
                    f"{window_name}"
                )
            )

        result = (
            result
            .with_columns(
                expressions
            )
        )

    if add_recency:
        last_column = (
            "_last_occurrence_at_utc"
            "_end"
        )

        result = (
            result

            .with_columns(
                (
                    pl.col(last_column).is_not_null()
                    &
                    (
                        pl.col(last_column)
                        >=
                        (
                            pl.col("row_timestamp_utc")
                            - pl.duration(
                                seconds=HISTORY_MAX_SECONDS
                            )
                        )
                    )
                )
                .alias(
                    f"has_{output_prefix}_28d"
                ),

                pl.when(
                    pl.col(
                        last_column
                    )
                    .is_not_null()
                )
                .then(
                    (
                        (
                            pl.col(
                                "row_timestamp_utc"
                            )
                            -
                            pl.col(
                                last_column
                            )
                        )
                        .dt.total_seconds()
                        .cast(pl.Float64)

                        / 3600.0
                    )
                    .clip(
                        upper_bound=
                            HISTORY_MAX_SECONDS
                            / 3600.0
                    )
                )
                .otherwise(
                    HISTORY_MAX_SECONDS
                    / 3600.0
                )
                .alias(
                    "hours_since_last_"
                    f"{output_prefix}"
                    "_capped_28d"
                ),
            )
        )

    temporary_columns = [
        column
        for column
        in result.collect_schema()
        .names()
        if (
            column.startswith(
                "_cum_"
            )
            or
            column.startswith(
                "_delta_"
            )
        )
    ]

    return result.drop(
        temporary_columns
    )


# =============================================================================
# Complete history graph
# =============================================================================


def attach_dynamic_history(
    *,
    rows: pl.LazyFrame,
    history_events: pl.LazyFrame,
) -> pl.LazyFrame:
    cell_prefix = (
        build_cell_prefix(
            history_events
        )
    )

    city_prefix = (
        build_city_prefix(
            history_events
        )
    )

    k1_prefix = (
        build_k1_prefix(
            history_events
        )
    )

    # ------------------------------------------------------------------
    # Same H3-9 cell
    # ------------------------------------------------------------------

    result = attach_prefix_history(
        rows=rows,

        prefix=cell_prefix,

        by=[
            "source_city",
            "osm_h3_cell_id",
        ],
        cumulative_map={
            "_cum_all":
                "crime_count",

            "_cum_violent":
                "violent_count",

            "_cum_property":
                "property_count",
        },

        output_prefix="cell",

        add_recency=True,
    )

    result = result.rename(
        {
            "has_cell_28d":
                "has_crime_cell_28d",

            "hours_since_last_cell_capped_28d":
                "hours_since_last_crime_cell_capped_28d",
        }
    )

    # ------------------------------------------------------------------
    # City-wide
    # ------------------------------------------------------------------

    result = attach_prefix_history(
        rows=result,

        prefix=city_prefix,

        by=[
            "source_city",
        ],

        cumulative_map={
            "_cum_all":
                "crime_count",
        },

        output_prefix="city",

        add_recency=True,
    )

    result = result.rename(
        {
            "has_city_28d":
                "has_crime_city_28d",

            "hours_since_last_city_capped_28d":
                "hours_since_last_crime_city_capped_28d",
        }
    )

    # ------------------------------------------------------------------
    # k=1 neighborhood
    # ------------------------------------------------------------------

    result = attach_prefix_history(
        rows=result,

        prefix=k1_prefix,

        by=[
            "source_city",
            "osm_h3_cell_id",
        ],

        cumulative_map={
            "_cum_all":
                "crime_count",
        },

        output_prefix="k1",

        add_recency=False,
    )

    # ------------------------------------------------------------------
    # Ratios
    # ------------------------------------------------------------------

    return result.with_columns(
        (
            (
                pl.col(
                    "cell_crime_count_24h"
                )
                .cast(pl.Float64)
                + 1.0
            )
            /
            (
                pl.col(
                    "cell_crime_count_28d"
                )
                .cast(pl.Float64)
                + 1.0
            )
        )
        .alias(
            "cell_crime_24h_vs_28d_ratio"
        ),

        (
            pl.col(
                "cell_crime_count_24h"
            )
            .cast(pl.Float64)

            /

            pl.when(
                pl.col(
                    "k1_crime_count_24h"
                )
                > 0
            )
            .then(
                pl.col(
                    "k1_crime_count_24h"
                )
                .cast(pl.Float64)
            )
            .otherwise(
                pl.lit(1.0)
            )
        )
        .alias(
            "cell_share_of_k1_crime_24h"
        ),
    )


# =============================================================================
# Calendar features
# =============================================================================


def local_hour_expression() -> pl.Expr:
    result = pl.lit(
        None,
        dtype=pl.Int8,
    )

    for city, timezone in reversed(
        list(
            CITY_TIMEZONES.items()
        )
    ):
        result = (
            pl.when(
                pl.col(
                    "source_city"
                )
                == city
            )
            .then(
                pl.col(
                    "row_timestamp_utc"
                )
                .dt.convert_time_zone(
                    timezone
                )
                .dt.hour()
            )
            .otherwise(result)
        )

    return result.alias(
        "local_hour"
    )


def local_weekday_expression() -> pl.Expr:
    result = pl.lit(
        None,
        dtype=pl.Int8,
    )

    for city, timezone in reversed(
        list(
            CITY_TIMEZONES.items()
        )
    ):
        result = (
            pl.when(
                pl.col(
                    "source_city"
                )
                == city
            )
            .then(
                (
                    pl.col(
                        "row_timestamp_utc"
                    )
                    .dt.convert_time_zone(
                        timezone
                    )
                    .dt.weekday()
                    - 1
                )
                .cast(pl.Int8)
            )
            .otherwise(result)
        )

    return result.alias(
        "local_day_of_week"
    )


def add_calendar_features(
    rows: pl.LazyFrame,
) -> pl.LazyFrame:
    rows = rows.with_columns(
        local_hour_expression(),
        local_weekday_expression(),
    )

    return rows.with_columns(
        (
            pl.col(
                "local_hour"
            )
            .cast(pl.Float64)
            * (
                2.0
                * math.pi
                / 24.0
            )
        )
        .sin()
        .alias(
            "local_hour_sin"
        ),

        (
            pl.col(
                "local_hour"
            )
            .cast(pl.Float64)
            * (
                2.0
                * math.pi
                / 24.0
            )
        )
        .cos()
        .alias(
            "local_hour_cos"
        ),

        (
            pl.col(
                "local_day_of_week"
            )
            .cast(pl.Float64)
            * (
                2.0
                * math.pi
                / 7.0
            )
        )
        .sin()
        .alias(
            "local_day_of_week_sin"
        ),

        (
            pl.col(
                "local_day_of_week"
            )
            .cast(pl.Float64)
            * (
                2.0
                * math.pi
                / 7.0
            )
        )
        .cos()
        .alias(
            "local_day_of_week_cos"
        ),
    )


# =============================================================================
# Validation
# =============================================================================


def validate_final_partition(
    rows: pl.LazyFrame,
    *,
    expected_rows: int,
) -> dict[str, object]:
    history_audit_columns = [
        column
        for column
        in rows.collect_schema().names()
        if column.startswith(
            "_last_occurrence_at_utc"
        )
    ]

    leak_expressions = []

    for column in (
        history_audit_columns
    ):
        leak_expressions.append(
            (
                pl.col(column)
                .is_not_null()

                &

                (
                    pl.col(column)
                    >=
                    pl.col(
                        "row_timestamp_utc"
                    )
                )
            )
            .sum()
            .cast(pl.Int64)
            .alias(
                f"leak__{column}"
            )
        )

    negative_history = [
        column
        for column
        in rows.collect_schema().names()
        if (
            "_count_"
            in column
            and (
                column.startswith("cell_")
                or column.startswith("city_")
                or column.startswith("k1_")
            )
        )
    ]

    metrics = (
        rows
        .select(
            pl.len()
            .cast(pl.Int64)
            .alias("rows"),

            pl.col(
                "model_row_id"
            )
            .n_unique()
            .cast(pl.Int64)
            .alias(
                "unique_row_ids"
            ),

            pl.col(
                "_lighting_matched"
            )
            .fill_null(False)
            .not_()
            .sum()
            .cast(pl.Int64)
            .alias(
                "missing_lighting"
            ),

            (
                pl.col("split")
                !=
                split_expression(
                    "row_timestamp_utc"
                )
            )
            .sum()
            .cast(pl.Int64)
            .alias(
                "split_mismatches"
            ),

            (
                pl.col(
                    "is_observed_event"
                )
                &
                (
                    pl.col(
                        "event_count"
                    )
                    != 1
                )
            )
            .sum()
            .cast(pl.Int64)
            .alias(
                "bad_observed_event_count"
            ),

            (
                ~pl.col(
                    "is_observed_event"
                )
                &
                (
                    pl.col(
                        "event_count"
                    )
                    != 0
                )
            )
            .sum()
            .cast(pl.Int64)
            .alias(
                "bad_integration_event_count"
            ),

            (
                pl.col(
                    "is_observed_event"
                )
                &
                (
                    pl.col(
                        "integration_weight_cell_seconds"
                    )
                    != 0.0
                )
            )
            .sum()
            .cast(pl.Int64)
            .alias(
                "bad_observed_weight"
            ),

            (
                ~pl.col(
                    "is_observed_event"
                )
                &
                (
                    pl.col(
                        "integration_weight_cell_seconds"
                    )
                    <= 0.0
                )
            )
            .sum()
            .cast(pl.Int64)
            .alias(
                "bad_integration_weight"
            ),

            *[
                (
                    pl.col(column)
                    < 0
                )
                .sum()
                .cast(pl.Int64)
                .alias(
                    f"negative__{column}"
                )
                for column
                in negative_history
            ],

            *leak_expressions,
        )

        .collect()

        .row(
            0,
            named=True,
        )
    )

    if (
        metrics["rows"]
        != expected_rows
    ):
        raise ValueError(
            "Final partition cardinality changed: "
            f"expected={expected_rows:,}, "
            f"actual={metrics['rows']:,}"
        )

    fatal = {
        key: value
        for key, value
        in metrics.items()
        if (
            key != "rows"
            and
            key != "unique_row_ids"
            and
            value != 0
        )
    }

    if (
        metrics["unique_row_ids"]
        != metrics["rows"]
    ):
        fatal[
            "duplicate_model_row_ids"
        ] = (
            metrics["rows"]
            -
            metrics[
                "unique_row_ids"
            ]
        )

    if fatal:
        raise ValueError(
            "Final model-table validation "
            f"failed: {fatal}"
        )

    return metrics


# =============================================================================
# Final projection
# =============================================================================


def finalize_model_table(
    rows: pl.LazyFrame,
) -> pl.LazyFrame:
    require_columns(
        rows,
        FINAL_COLUMNS,
        name="final model table",
    )

    # Explicit allow-list.
    #
    # This is much safer than trying to enumerate every
    # provenance/debug field that should be removed.
    return rows.select(
        FINAL_COLUMNS
    )