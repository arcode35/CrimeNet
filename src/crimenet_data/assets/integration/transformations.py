import polars as pl
import polars_h3 as plh3
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# =============================================================================
# Sampling contract
# =============================================================================

H3_RESOLUTION = 9
WEATHER_H3_RESOLUTION = 6

DEFAULT_INTEGRATION_POOL_SIZE = 5

SAMPLING_VERSION = (
    "integration_mc_v3"
)
# Explicit deterministic PRNG.
#
# We intentionally do not use pl.Expr.hash() because Polars documents that
# hash results are only stable within a given Polars version.
PRNG_MODULUS = 2_147_483_647
PRNG_MULTIPLIER = 48_271
PRNG_SEED = 17_081_726


# =============================================================================
# Helpers
# =============================================================================



def build_observation_windows_from_local_ranges(
    *,
    observation_ranges: dict[
        str,
        list[tuple[str, str]],
    ],
    city_timezones: dict[str, str],
) -> pl.LazyFrame:
    """
    Convert authoritative source-observation ranges into year-split UTC
    windows.

    Input timestamps are LOCAL wall-clock timestamps.

    Example:

        {
            "dallas": [
                (
                    "2014-01-01T00:00:00",
                    "2026-07-25T00:00:00",
                )
            ]
        }

    The end is EXCLUSIVE.

    A range crossing calendar years is automatically split at each local
    Jan-1 boundary because annual OSM support changes there.
    """

    rows: list[dict[str, object]] = []

    for city, ranges in observation_ranges.items():
        if city not in city_timezones:
            raise ValueError(
                f"No timezone configured for city={city!r}"
            )

        tz = ZoneInfo(
            city_timezones[city]
        )

        for range_index, (
            start_text,
            end_text,
        ) in enumerate(ranges):
            start_local = datetime.fromisoformat(
                start_text
            )
            end_local = datetime.fromisoformat(
                end_text
            )

            if (
                start_local.tzinfo is not None
                or end_local.tzinfo is not None
            ):
                raise ValueError(
                    "Observation-range timestamps must "
                    "be naive LOCAL wall-clock times. "
                    f"city={city!r}"
                )

            start_local = start_local.replace(
                tzinfo=tz
            )
            end_local = end_local.replace(
                tzinfo=tz
            )

            if end_local <= start_local:
                raise ValueError(
                    "Observation window must have "
                    "positive duration: "
                    f"{city}: "
                    f"{start_text} -> {end_text}"
                )

            cursor = start_local
            segment_index = 0

            while cursor < end_local:
                next_year = datetime(
                    cursor.year + 1,
                    1,
                    1,
                    tzinfo=tz,
                )

                segment_end = min(
                    end_local,
                    next_year,
                )

                rows.append(
                    {
                        "observation_window_id": (
                            f"{city}"
                            f"|{range_index}"
                            f"|{segment_index}"
                            f"|{cursor.year}"
                        ),
                        "source_city": city,
                        "source_timezone":
                            city_timezones[city],
                        "support_year":
                            cursor.year,
                        "support_valid_from":
                            cursor.astimezone(
                                timezone.utc
                            ),
                        "support_valid_to":
                            segment_end.astimezone(
                                timezone.utc
                            ),
                    }
                )

                cursor = segment_end
                segment_index += 1

    if not rows:
        raise ValueError(
            "No observation windows were configured."
        )

    return pl.DataFrame(
        rows,
        schema={
            "observation_window_id":
                pl.String,
            "source_city":
                pl.String,
            "source_timezone":
                pl.String,
            "support_year":
                pl.Int32,
            "support_valid_from":
                pl.Datetime(
                    "us",
                    "UTC",
                ),
            "support_valid_to":
                pl.Datetime(
                    "us",
                    "UTC",
                ),
        },
    ).lazy()

def split_from_year(
    year_column: str,
) -> pl.Expr:
    return (
        pl.when(
            pl.col(year_column) <= 2023
        )
        .then(pl.lit("train"))
        .when(
            pl.col(year_column) == 2024
        )
        .then(pl.lit("validation"))
        .otherwise(pl.lit("test"))
    )


def city_code_expr(
    city_codes: dict[str, int],
) -> pl.Expr:
    """
    Deterministic integer code used only for pseudo-random seed mixing.
    """

    expression = pl.lit(
        None,
        dtype=pl.Int64,
    )

    for city, code in reversed(
        list(city_codes.items())
    ):
        expression = (
            pl.when(
                pl.col("source_city") == city
            )
            .then(
                pl.lit(
                    code,
                    dtype=pl.Int64,
                )
            )
            .otherwise(expression)
        )

    return expression


# =============================================================================
# 1. Select the events that actually define the modeled process
# =============================================================================


def select_modeled_events(
    event_spine: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Only events represented by the model should affect integration intervals.

    Rows excluded from modeling must not create point-process state changes.
    """

    return (
        event_spine
        .filter(
            pl.col("include_in_model")
            .fill_null(False)
            & pl.col("is_criminal_event")
            .fill_null(False)
            & pl.col(
                "occurrence_timestamp_utc"
            ).is_not_null()
        )
        .select(
            "crime_id",
            "source_city",
            "source_timezone",
            "occurrence_timestamp_utc",
            "occurrence_year",
            "osm_h3_cell_id",
        )
    )


# =============================================================================
# 2. Build city/year spatial support from canonical OSM H3-9
# =============================================================================
def prepare_spatial_support(
    spatial_domain_mask: pl.LazyFrame,
) -> pl.LazyFrame:

    support = (
        spatial_domain_mask
        .filter(
            pl.col(
                "inside_observation_domain"
            )
        )
        .select(
            "source_city",

            pl.col(
                "support_year"
            )
            .cast(pl.Int32),

            pl.col(
                "osm_h3_cell_id"
            )
            .cast(pl.Int64),
        )
        .sort(
            [
                "source_city",
                "support_year",
                "osm_h3_cell_id",
            ]
        )
    )

    return support.with_columns(
        (
            pl.col(
                "osm_h3_cell_id"
            )
            .rank(
                method="ordinal"
            )
            .over(
                [
                    "source_city",
                    "support_year",
                ]
            )
            .cast(pl.Int64)
            - 1
        )
        .alias(
            "support_cell_index"
        ),

        pl.len()
        .over(
            [
                "source_city",
                "support_year",
            ]
        )
        .cast(pl.Int64)
        .alias(
            "supported_cell_count"
        ),
    )
def build_support_counts(
    spatial_support: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    One row per city/year containing the size of the spatial integration
    domain.
    """

    return (
        spatial_support
        .select(
            "source_city",
            "support_year",
            "supported_cell_count",
        )
        .unique()
    )


# =============================================================================
# 3. Construct continuous-time inter-event intervals
# =============================================================================

def build_support_intervals(
    modeled_events: pl.LazyFrame,
    observation_windows: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Build complete temporal integration support.

    For each authoritative observation window:

        observation_start
              ↓
            event
              ↓
            event
              ↓
            ...
              ↓
        observation_end

    Every adjacent pair of boundaries becomes exactly one integration
    interval.

    This correctly handles:

        - prefix before first event
        - suffix after last event
        - exactly one event
        - zero events
        - simultaneous events
        - partial years
        - year boundaries

    observation_valid_to is exclusive.
    """

    windows = observation_windows.select(
        "observation_window_id",
        "source_city",
        "source_timezone",
        "support_year",
        "support_valid_from",
        "support_valid_to",
    )

    # =====================================================================
    # Observation-start sentinels
    # =====================================================================

    starts = windows.select(
        "observation_window_id",
        "source_city",
        "source_timezone",
        "support_year",
        "support_valid_from",
        "support_valid_to",

        pl.col(
            "support_valid_from"
        )
        .alias(
            "_boundary_timestamp"
        ),

        pl.lit(
            0,
            dtype=pl.Int64,
        )
        .alias(
            "_events_at_boundary"
        ),

        pl.lit(True)
        .alias(
            "_is_observation_start"
        ),

        pl.lit(False)
        .alias(
            "_is_observation_end"
        ),
    )

    # =====================================================================
    # Observation-end sentinels
    # =====================================================================

    ends = windows.select(
        "observation_window_id",
        "source_city",
        "source_timezone",
        "support_year",
        "support_valid_from",
        "support_valid_to",

        pl.col(
            "support_valid_to"
        )
        .alias(
            "_boundary_timestamp"
        ),

        pl.lit(
            0,
            dtype=pl.Int64,
        )
        .alias(
            "_events_at_boundary"
        ),

        pl.lit(False)
        .alias(
            "_is_observation_start"
        ),

        pl.lit(True)
        .alias(
            "_is_observation_end"
        ),
    )

    # =====================================================================
    # Map actual events into observation windows
    # =====================================================================

    events = (
        modeled_events
        .select(
            "crime_id",
            "source_city",

            pl.col(
                "occurrence_year"
            )
            .cast(pl.Int32)
            .alias(
                "support_year"
            ),

            pl.col(
                "occurrence_timestamp_utc"
            )
            .alias(
                "_boundary_timestamp"
            ),
        )
        .join(
            windows,
            on=[
                "source_city",
                "support_year",
            ],
            how="inner",
        )
        .filter(
            (
                pl.col(
                    "_boundary_timestamp"
                )
                >=
                pl.col(
                    "support_valid_from"
                )
            )
            &
            (
                pl.col(
                    "_boundary_timestamp"
                )
                <
                pl.col(
                    "support_valid_to"
                )
            )
        )
        .group_by(
            [
                "observation_window_id",
                "source_city",
                "source_timezone",
                "support_year",
                "support_valid_from",
                "support_valid_to",
                "_boundary_timestamp",
            ]
        )
        .agg(
            pl.len()
            .cast(pl.Int64)
            .alias(
                "_events_at_boundary"
            )
        )
        .with_columns(
            pl.lit(False)
            .alias(
                "_is_observation_start"
            ),

            pl.lit(False)
            .alias(
                "_is_observation_end"
            ),
        )
    )

    # =====================================================================
    # Start + events + end
    #
    # Grouping is important because an event may occur exactly at
    # observation_start. In that case the sentinel and event boundary
    # become one boundary and we preserve event multiplicity.
    # =====================================================================

    boundaries = (
        pl.concat(
            [
                starts,
                events,
                ends,
            ],
            how="vertical_relaxed",
        )
        .group_by(
            [
                "observation_window_id",
                "source_city",
                "source_timezone",
                "support_year",
                "support_valid_from",
                "support_valid_to",
                "_boundary_timestamp",
            ]
        )
        .agg(
            pl.col(
                "_events_at_boundary"
            )
            .sum()
            .alias(
                "_events_at_boundary"
            ),

            pl.col(
                "_is_observation_start"
            )
            .max()
            .alias(
                "_is_observation_start"
            ),

            pl.col(
                "_is_observation_end"
            )
            .max()
            .alias(
                "_is_observation_end"
            ),
        )
        .sort(
            [
                "observation_window_id",
                "_boundary_timestamp",
            ]
        )
    )

    # =====================================================================
    # Convert adjacent boundaries into intervals
    # =====================================================================

    intervals = (
        boundaries
        .with_columns(
            pl.col(
                "_boundary_timestamp"
            )
            .shift(-1)
            .over(
                "observation_window_id"
            )
            .alias(
                "_next_boundary_timestamp"
            )
        )
        .filter(
            pl.col(
                "_next_boundary_timestamp"
            )
            .is_not_null()
        )
        .filter(
            pl.col(
                "_next_boundary_timestamp"
            )
            >
            pl.col(
                "_boundary_timestamp"
            )
        )
        .with_columns(
            pl.col(
                "_boundary_timestamp"
            )
            .alias(
                "interval_valid_from"
            ),

            pl.col(
                "_next_boundary_timestamp"
            )
            .alias(
                "interval_valid_to"
            ),

            pl.col(
                "_events_at_boundary"
            )
            .alias(
                "events_at_left_boundary"
            ),

            (
                pl.when(
                    pl.col(
                        "_is_observation_start"
                    )
                    & (
                        pl.col(
                            "_events_at_boundary"
                        )
                        > 0
                    )
                )
                .then(
                    pl.lit(
                        "observation_start_event"
                    )
                )
                .when(
                    pl.col(
                        "_is_observation_start"
                    )
                )
                .then(
                    pl.lit(
                        "observation_start"
                    )
                )
                .when(
                    pl.col(
                        "_events_at_boundary"
                    )
                    > 0
                )
                .then(
                    pl.lit("event")
                )
                .otherwise(
                    pl.lit("boundary")
                )
            )
            .alias(
                "left_boundary_type"
            ),
        )
    )

    # =====================================================================
    # Duration + split + stable ID
    # =====================================================================

    intervals = (
        intervals
        .with_columns(
            (
                pl.col(
                    "interval_valid_to"
                )
                -
                pl.col(
                    "interval_valid_from"
                )
            )
            .dt.total_microseconds()
            .cast(pl.Int64)
            .alias(
                "support_duration_microseconds"
            )
        )
        .filter(
            # We sample strictly INSIDE the interval.
            # Therefore we need at least one interior microsecond.
            pl.col(
                "support_duration_microseconds"
            )
            >= 2
        )
        .with_columns(
            (
                pl.col(
                    "support_duration_microseconds"
                )
                .cast(pl.Float64)
                / 1_000_000.0
            )
            .alias(
                "support_duration_seconds"
            ),

            split_from_year(
                "support_year"
            )
            .alias(
                "split"
            ),

            pl.concat_str(
                [
                    pl.col(
                        "observation_window_id"
                    ),
                    pl.col(
                        "interval_valid_from"
                    )
                    .dt.epoch("us")
                    .cast(pl.String),
                    pl.col(
                        "interval_valid_to"
                    )
                    .dt.epoch("us")
                    .cast(pl.String),
                ],
                separator="|",
            )
            .alias(
                "support_interval_id"
            ),
        )
        .select(
            "support_interval_id",
            "observation_window_id",

            "source_city",
            "source_timezone",
            "support_year",
            "split",

            pl.col(
                "interval_valid_from"
            )
            .alias(
                "support_valid_from"
            ),

            pl.col(
                "interval_valid_to"
            )
            .alias(
                "support_valid_to"
            ),

            "support_duration_microseconds",
            "support_duration_seconds",

            "left_boundary_type",
            "events_at_left_boundary",
        )
    )

    return intervals


# =============================================================================
# 4. Attach spatial measure
# =============================================================================


def attach_spatial_measure(
    intervals: pl.LazyFrame,
    spatial_support: pl.LazyFrame,
) -> pl.LazyFrame:
    support_counts = (
        build_support_counts(
            spatial_support
        )
    )

    return intervals.join(
        support_counts,
        on=[
            "source_city",
            "support_year",
        ],
        how="left",
        validate="m:1",
    )


# =============================================================================
# 5. Expand K samples per support interval
# =============================================================================


def expand_sample_indices(
    intervals: pl.LazyFrame,
    *,
    samples_per_interval: int,
) -> pl.LazyFrame:
    if samples_per_interval <= 0:
        raise ValueError(
            "samples_per_interval "
            "must be positive"
        )

    return (
        intervals
        .with_columns(
            pl.int_ranges(
                0,
                samples_per_interval,
                dtype=pl.Int32,
            )
            .alias(
                "integration_sample_index"
            )
        )
        .explode(
            "integration_sample_index"
        )
        .with_columns(
            pl.lit(
                samples_per_interval
            )
            .cast(pl.Int32)
            .alias(
                "integration_pool_size"
            )
        )
    )


# =============================================================================
# 6. Deterministic uniform numbers
# =============================================================================
def add_deterministic_uniforms(
    samples: pl.LazyFrame,
    *,
    city_codes: dict[str, int],
) -> pl.LazyFrame:
    """
    Produce two deterministic U(0,1)-like values:

        _u_time
        _u_cell

    using explicit integer modular arithmetic.
    """

    samples = samples.with_columns(
        city_code_expr(
            city_codes
        )
        .alias("_city_code"),

        pl.col(
            "support_valid_from"
        )
        .dt.epoch("us")
        .mod(PRNG_MODULUS)
        .cast(pl.Int64)
        .alias("_start_mod"),

        pl.col(
            "support_valid_to"
        )
        .dt.epoch("us")
        .mod(PRNG_MODULUS)
        .cast(pl.Int64)
        .alias("_end_mod"),

        (
            pl.col(
                "integration_sample_index"
            )
            .cast(pl.Int64)
            + 1
        )
        .alias("_sample_number"),
    )

    # -----------------------------------------------------------------
    # Initial deterministic state
    # -----------------------------------------------------------------

    samples = samples.with_columns(
        (
            (
                pl.col("_start_mod")
                * 1_000_003
            )
            + (
                pl.col("_end_mod")
                * 1_000_033
            )
            + (
                pl.col("_city_code")
                * 1_009
            )
            + (
                pl.col("_sample_number")
                * 104_729
            )
            + PRNG_SEED
        )
        .mod(PRNG_MODULUS)
        .alias("_state_0")
    )

    # -----------------------------------------------------------------
    # First PRNG step: time
    # -----------------------------------------------------------------

    samples = samples.with_columns(
        (
            (
                pl.col("_state_0")
                * PRNG_MULTIPLIER
            )
            + 1
        )
        .mod(PRNG_MODULUS)
        .alias("_state_time")
    )

    # -----------------------------------------------------------------
    # Second PRNG step: cell
    # -----------------------------------------------------------------

    samples = samples.with_columns(
        (
            (
                pl.col("_state_time")
                * PRNG_MULTIPLIER
            )
            + 1
        )
        .mod(PRNG_MODULUS)
        .alias("_state_cell")
    )

    # -----------------------------------------------------------------
    # Convert integer states to values in approximately (0, 1)
    # -----------------------------------------------------------------

    return samples.with_columns(
        (
            (
                pl.col("_state_time")
                .cast(pl.Float64)
                + 0.5
            )
            / float(PRNG_MODULUS)
        )
        .alias("_u_time"),

        (
            (
                pl.col("_state_cell")
                .cast(pl.Float64)
                + 0.5
            )
            / float(PRNG_MODULUS)
        )
        .alias("_u_cell"),
    )


# =============================================================================
# 7. Sample a continuous time and discrete H3 cell index
# =============================================================================


def sample_space_time(
    samples: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Time:
        uniformly inside the open inter-event interval.

    Space:
        uniformly from every supported H3-9 cell for that city/year.
    """

    samples = samples.with_columns(
        (
            (
                pl.col("_u_time")
                * (
                    pl.col(
                        "support_duration_microseconds"
                    )
                    - 1
                )
                .cast(pl.Float64)
            )
            .floor()
            .cast(pl.Int64)
            + 1
        )
        .alias("_sample_offset_us"),

        (
            pl.col("_u_cell")
            * pl.col(
                "supported_cell_count"
            )
            .cast(pl.Float64)
        )
        .floor()
        .cast(pl.Int64)
        .alias(
            "support_cell_index"
        ),
    )

    return samples.with_columns(
        (
            pl.col(
                "support_valid_from"
            )
            + pl.duration(
                microseconds=pl.col(
                    "_sample_offset_us"
                )
            )
        )
        .alias(
            "sample_timestamp_utc"
        )
    )


# =============================================================================
# 8. Assign actual H3 cell
# =============================================================================


def assign_sampled_cells(
    samples: pl.LazyFrame,
    spatial_support: pl.LazyFrame,
) -> pl.LazyFrame:
    cell_lookup = (
        spatial_support
        .select(
            "source_city",
            "support_year",
            "support_cell_index",
            "osm_h3_cell_id",
        )
    )

    return (
        samples
        .join(
            cell_lookup,
            on=[
                "source_city",
                "support_year",
                "support_cell_index",
            ],
            how="left",
            validate="m:1",
        )
        .with_columns(
            plh3.cell_to_parent(
                "osm_h3_cell_id",
                WEATHER_H3_RESOLUTION,
            )
            .cast(pl.Int64)
            .alias(
                "weather_query_cell_id"
            )
        )
    )


# =============================================================================
# 9. Monte Carlo measure / integration weights
# =============================================================================


def add_integration_weights(
    samples: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    For uniform sampling:

        domain measure
            = duration_seconds × supported_cell_count

        weight per sample
            = domain_measure / K

    Unit:
        cell-seconds
    """

    return samples.with_columns(
        (
            1.0
            / pl.col(
                "supported_cell_count"
            )
            .cast(pl.Float64)
        )
        .alias(
            "cell_sampling_probability"
        ),

        (
            1.0
            / pl.col(
                "support_duration_seconds"
            )
        )
        .alias(
            "time_sampling_density_per_second"
        ),

        (
            1.0
            / (
                pl.col(
                    "support_duration_seconds"
                )
                * pl.col(
                    "supported_cell_count"
                )
                .cast(pl.Float64)
            )
        )
        .alias(
            "joint_sampling_density_per_cell_second"
        ),

        (
            pl.col(
                "support_duration_seconds"
            )
            * pl.col(
                "supported_cell_count"
            )
            .cast(pl.Float64)
        )
        .alias(
            "integration_domain_cell_seconds"
        ),

        (
            pl.col(
                "support_duration_seconds"
            )
            * pl.col(
                "supported_cell_count"
            )
            .cast(pl.Float64)
            / pl.col(
                "integration_pool_size"
            )
            .cast(pl.Float64)
        )
        .alias(
            "integration_weight_cell_seconds"
        ),
    )


# =============================================================================
# 10. Final projection
# =============================================================================


def finalize_integration_samples(
    samples: pl.LazyFrame,
) -> pl.LazyFrame:
    return (
        samples
        .with_columns(
            pl.lit(
                SAMPLING_VERSION
            )
            .alias(
                "sampling_version"
            ),

            pl.col(
                "sample_timestamp_utc"
            )
            .alias(
                "history_cutoff_timestamp_utc"
            ),

            pl.concat_str(
                [
                    pl.col(
                        "support_interval_id"
                    ),
                    pl.col(
                        "integration_sample_index"
                    ).cast(pl.String),
                    pl.lit(
                        SAMPLING_VERSION
                    ),
                ],
                separator="|",
            )
            .alias(
                "integration_sample_id"
            ),
        )
        .select(
            # Identity / lineage
            "integration_sample_id",
            "sampling_version",

            # Domain
            "source_city",
            "source_timezone",
            "support_year",
            "split",

            # Inter-event support interval
            "observation_window_id",
            "support_interval_id",
            "support_valid_from",
            "support_valid_to",
            "support_duration_seconds",
            "left_boundary_type",
            "events_at_left_boundary",
            # Monte Carlo sample
            "integration_sample_index",
            "integration_pool_size",
            "sample_timestamp_utc",
            "history_cutoff_timestamp_utc",

            # Spatial sample
            "osm_h3_cell_id",
            "weather_query_cell_id",
            "support_cell_index",
            "supported_cell_count",

            # Sampling distribution
            "cell_sampling_probability",
            "time_sampling_density_per_second",
            "joint_sampling_density_per_cell_second",

            # Measure represented by the sample
            "integration_domain_cell_seconds",
            "integration_weight_cell_seconds",
            
        )
    )


# =============================================================================
# Complete integration transformation graph
# =============================================================================

def build_integration_samples(
    *,
    modeled_events: pl.LazyFrame,
    spatial_support: pl.LazyFrame,
    observation_windows: pl.LazyFrame,
    city_codes: dict[str, int],
    samples_per_interval: int = DEFAULT_INTEGRATION_POOL_SIZE,
) -> tuple[
    pl.LazyFrame,
    pl.LazyFrame,
    pl.LazyFrame,
]:
    """
    Build Monte Carlo integration samples over the explicitly defined
    statistical observation domain.

    Important:
        modeled_events must already be filtered to the final temporal and
        spatial observation domain. spatial_support must be the exact support
        used to perform that spatial event partition.

    Returns:
        samples
        intervals
        spatial_support
    """

    # -------------------------------------------------------------------------
    # Complete temporal support
    #
    # Uses:
    #   observation start
    #       -> events
    #       -> observation end
    #
    # rather than inferring the domain from consecutive events alone.
    # -------------------------------------------------------------------------

    intervals = (
        build_support_intervals(
            modeled_events,
            observation_windows,
        )
    )

    # -------------------------------------------------------------------------
    # Attach number of valid H3 cells for each city/year.
    # -------------------------------------------------------------------------

    intervals = (
        attach_spatial_measure(
            intervals,
            spatial_support,
        )
    )

    # -------------------------------------------------------------------------
    # Expand each interval into K Monte Carlo samples.
    # -------------------------------------------------------------------------

    samples = (
        expand_sample_indices(
            intervals,
            samples_per_interval=
                samples_per_interval,
        )
    )

    # -------------------------------------------------------------------------
    # Deterministic pseudo-random U(0, 1) values.
    # -------------------------------------------------------------------------

    samples = (
        add_deterministic_uniforms(
            samples,
            city_codes=city_codes,
        )
    )

    # -------------------------------------------------------------------------
    # Sample:
    #   - timestamp inside interval
    #   - spatial support index
    # -------------------------------------------------------------------------

    samples = (
        sample_space_time(
            samples
        )
    )

    # -------------------------------------------------------------------------
    # Resolve sampled support index to actual H3-9 cell.
    # -------------------------------------------------------------------------

    samples = (
        assign_sampled_cells(
            samples,
            spatial_support,
        )
    )

    # -------------------------------------------------------------------------
    # Monte Carlo measure.
    #
    # weight =
    #     interval_duration_seconds
    #     × supported_cell_count
    #     / K
    # -------------------------------------------------------------------------

    samples = (
        add_integration_weights(
            samples
        )
    )

    # -------------------------------------------------------------------------
    # Stable IDs + final schema.
    # -------------------------------------------------------------------------

    samples = (
        finalize_integration_samples(
            samples
        )
    )

    return (
        samples,
        intervals,
        spatial_support,
    )
