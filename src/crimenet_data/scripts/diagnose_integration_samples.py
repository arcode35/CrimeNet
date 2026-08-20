import polars as pl
import polars_h3 as plh3


INTEGRATION_ROOT = (
    "gs://crimenet/gold/integration_samples"
)

H3_RESOLUTION = 9
WEATHER_H3_RESOLUTION = 6

REL_TOL = 1e-12
ABS_TOL = 1e-9


# =============================================================================
# Helpers
# =============================================================================


def scan_integration() -> pl.LazyFrame:
    credentials = (
        pl.CredentialProviderGCP()
    )

    return pl.scan_delta(
        INTEGRATION_ROOT,
        credential_provider=credentials,
    )


def print_section(
    title: str,
) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_check(
    name: str,
    value: int,
) -> None:
    status = (
        "PASS"
        if value == 0
        else "FAIL"
    )

    print(
        f"{status:4}  "
        f"{name:<50} "
        f"{value:,}"
    )


# =============================================================================
# Schema
# =============================================================================


EXPECTED_COLUMNS = [
    "integration_sample_id",
    "sampling_version",

    "source_city",
    "source_timezone",
    "support_year",
    "split",

    "observation_window_id",
    "support_interval_id",
    "support_valid_from",
    "support_valid_to",
    "support_duration_seconds",

    "left_boundary_type",
    "events_at_left_boundary",

    "integration_sample_index",
    "integration_pool_size",
    "sample_timestamp_utc",
    "history_cutoff_timestamp_utc",

    "osm_h3_cell_id",
    "weather_query_cell_id",
    "support_cell_index",
    "supported_cell_count",

    "cell_sampling_probability",
    "time_sampling_density_per_second",
    "joint_sampling_density_per_cell_second",

    "integration_domain_cell_seconds",
    "integration_weight_cell_seconds",
]


def validate_schema(
    samples: pl.LazyFrame,
) -> None:
    schema = (
        samples.collect_schema()
    )

    actual = schema.names()

    missing = sorted(
        set(EXPECTED_COLUMNS)
        - set(actual)
    )

    unexpected = sorted(
        set(actual)
        - set(EXPECTED_COLUMNS)
    )

    print_section(
        "SCHEMA"
    )

    print(
        f"columns: {len(actual)}"
    )

    print(
        f"missing: {missing}"
    )

    print(
        f"unexpected: {unexpected}"
    )

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


# =============================================================================
# Global structural statistics
# =============================================================================


def global_statistics(
    samples: pl.LazyFrame,
) -> dict:
    return (
        samples
        .select(
            pl.len()
            .cast(pl.Int64)
            .alias("rows"),

            pl.col(
                "integration_sample_id"
            )
            .n_unique()
            .cast(pl.Int64)
            .alias(
                "unique_sample_ids"
            ),

            pl.col(
                "support_interval_id"
            )
            .n_unique()
            .cast(pl.Int64)
            .alias(
                "unique_intervals"
            ),

            pl.col(
                "observation_window_id"
            )
            .n_unique()
            .cast(pl.Int64)
            .alias(
                "observation_windows"
            ),

            pl.col(
                "source_city"
            )
            .n_unique()
            .cast(pl.Int64)
            .alias("cities"),

            pl.col(
                "sampling_version"
            )
            .n_unique()
            .cast(pl.Int64)
            .alias(
                "sampling_versions"
            ),

            pl.col(
                "integration_pool_size"
            )
            .min()
            .alias("min_k"),

            pl.col(
                "integration_pool_size"
            )
            .max()
            .alias("max_k"),

            pl.col(
                "sample_timestamp_utc"
            )
            .min()
            .alias("first_sample"),

            pl.col(
                "sample_timestamp_utc"
            )
            .max()
            .alias("last_sample"),

            pl.col(
                "integration_weight_cell_seconds"
            )
            .sum()
            .alias(
                "total_weight"
            ),
        )
        .collect()
        .row(
            0,
            named=True,
        )
    )


# =============================================================================
# Nulls
# =============================================================================


def null_diagnostics(
    samples: pl.LazyFrame,
) -> pl.DataFrame:
    schema = (
        samples.collect_schema()
    )

    columns = schema.names()

    result = (
        samples
        .select(
            [
                pl.col(column)
                .null_count()
                .alias(column)
                for column
                in columns
            ]
        )
        .collect()
    )

    rows = []

    for column in columns:
        count = int(
            result[column][0]
        )

        if count:
            rows.append(
                {
                    "column": column,
                    "null_rows": count,
                }
            )

    return pl.DataFrame(
        rows,
        schema={
            "column":
                pl.String,
            "null_rows":
                pl.Int64,
        },
    )


# =============================================================================
# Row-level invariants
# =============================================================================


def row_invariants(
    samples: pl.LazyFrame,
) -> dict:
    duration_from_timestamps = (
        (
            pl.col(
                "support_valid_to"
            )
            -
            pl.col(
                "support_valid_from"
            )
        )
        .dt.total_microseconds()
        .cast(pl.Float64)
        / 1_000_000.0
    )

    expected_cell_probability = (
        1.0
        /
        pl.col(
            "supported_cell_count"
        )
        .cast(pl.Float64)
    )

    expected_time_density = (
        1.0
        /
        pl.col(
            "support_duration_seconds"
        )
    )

    expected_joint_density = (
        1.0
        /
        (
            pl.col(
                "support_duration_seconds"
            )
            *
            pl.col(
                "supported_cell_count"
            )
            .cast(pl.Float64)
        )
    )

    expected_domain = (
        pl.col(
            "support_duration_seconds"
        )
        *
        pl.col(
            "supported_cell_count"
        )
        .cast(pl.Float64)
    )

    expected_weight = (
        expected_domain
        /
        pl.col(
            "integration_pool_size"
        )
        .cast(pl.Float64)
    )

    expected_weather_cell = (
        plh3.cell_to_parent(
            "osm_h3_cell_id",
            WEATHER_H3_RESOLUTION,
        )
        .cast(pl.Int64)
    )

    bad_split = (
        (
            (
                pl.col("support_year")
                <= 2023
            )
            &
            (
                pl.col("split")
                != "train"
            )
        )
        |
        (
            (
                pl.col("support_year")
                == 2024
            )
            &
            (
                pl.col("split")
                != "validation"
            )
        )
        |
        (
            (
                pl.col("support_year")
                >= 2025
            )
            &
            (
                pl.col("split")
                != "test"
            )
        )
    )

    return (
        samples
        .select(
            (
                pl.col(
                    "support_valid_to"
                )
                <=
                pl.col(
                    "support_valid_from"
                )
            )
            .sum()
            .alias(
                "nonpositive_intervals"
            ),

            (
                pl.col(
                    "sample_timestamp_utc"
                )
                <=
                pl.col(
                    "support_valid_from"
                )
            )
            .sum()
            .alias(
                "samples_at_or_before_start"
            ),

            (
                pl.col(
                    "sample_timestamp_utc"
                )
                >=
                pl.col(
                    "support_valid_to"
                )
            )
            .sum()
            .alias(
                "samples_at_or_after_end"
            ),

            (
                pl.col(
                    "history_cutoff_timestamp_utc"
                )
                !=
                pl.col(
                    "sample_timestamp_utc"
                )
            )
            .sum()
            .alias(
                "history_cutoff_mismatch"
            ),

            (
                pl.col(
                    "supported_cell_count"
                )
                <= 0
            )
            .sum()
            .alias(
                "nonpositive_cell_count"
            ),

            (
                pl.col(
                    "support_cell_index"
                )
                < 0
            )
            .sum()
            .alias(
                "negative_cell_index"
            ),

            (
                pl.col(
                    "support_cell_index"
                )
                >=
                pl.col(
                    "supported_cell_count"
                )
            )
            .sum()
            .alias(
                "cell_index_out_of_bounds"
            ),

            (
                pl.col(
                    "integration_pool_size"
                )
                <= 0
            )
            .sum()
            .alias(
                "nonpositive_pool_size"
            ),

            (
                pl.col(
                    "integration_sample_index"
                )
                < 0
            )
            .sum()
            .alias(
                "negative_sample_index"
            ),

            (
                pl.col(
                    "integration_sample_index"
                )
                >=
                pl.col(
                    "integration_pool_size"
                )
            )
            .sum()
            .alias(
                "sample_index_out_of_bounds"
            ),

            (
                pl.col(
                    "integration_weight_cell_seconds"
                )
                <= 0
            )
            .sum()
            .alias(
                "nonpositive_weight"
            ),

            (
                ~pl.col(
                    "support_duration_seconds"
                )
                .is_close(
                    duration_from_timestamps,
                    rel_tol=
                        REL_TOL,
                    abs_tol=
                        ABS_TOL,
                )
            )
            .sum()
            .alias(
                "duration_formula_mismatch"
            ),

            (
                ~pl.col(
                    "cell_sampling_probability"
                )
                .is_close(
                    expected_cell_probability,
                    rel_tol=
                        REL_TOL,
                    abs_tol=
                        ABS_TOL,
                )
            )
            .sum()
            .alias(
                "cell_probability_mismatch"
            ),

            (
                ~pl.col(
                    "time_sampling_density_per_second"
                )
                .is_close(
                    expected_time_density,
                    rel_tol=
                        REL_TOL,
                    abs_tol=
                        ABS_TOL,
                )
            )
            .sum()
            .alias(
                "time_density_mismatch"
            ),

            (
                ~pl.col(
                    "joint_sampling_density_per_cell_second"
                )
                .is_close(
                    expected_joint_density,
                    rel_tol=
                        REL_TOL,
                    abs_tol=
                        ABS_TOL,
                )
            )
            .sum()
            .alias(
                "joint_density_mismatch"
            ),

            (
                ~pl.col(
                    "integration_domain_cell_seconds"
                )
                .is_close(
                    expected_domain,
                    rel_tol=
                        REL_TOL,
                    abs_tol=
                        ABS_TOL,
                )
            )
            .sum()
            .alias(
                "domain_measure_mismatch"
            ),

            (
                ~pl.col(
                    "integration_weight_cell_seconds"
                )
                .is_close(
                    expected_weight,
                    rel_tol=
                        REL_TOL,
                    abs_tol=
                        ABS_TOL,
                )
            )
            .sum()
            .alias(
                "weight_formula_mismatch"
            ),

            (
                plh3.get_resolution(
                    "osm_h3_cell_id"
                )
                != H3_RESOLUTION
            )
            .sum()
            .alias(
                "wrong_osm_h3_resolution"
            ),

            (
                expected_weather_cell
                !=
                pl.col(
                    "weather_query_cell_id"
                )
            )
            .sum()
            .alias(
                "wrong_weather_parent"
            ),

            bad_split
            .sum()
            .alias(
                "split_mismatch"
            ),

            (
                (
                    pl.col(
                        "events_at_left_boundary"
                    )
                    > 0
                )
                &
                ~pl.col(
                    "left_boundary_type"
                )
                .is_in(
                    [
                        "event",
                        "observation_start_event",
                    ]
                )
            )
            .sum()
            .alias(
                "event_boundary_semantic_error"
            ),

            (
                (
                    pl.col(
                        "events_at_left_boundary"
                    )
                    == 0
                )
                &
                pl.col(
                    "left_boundary_type"
                )
                .is_in(
                    [
                        "event",
                        "observation_start_event",
                    ]
                )
            )
            .sum()
            .alias(
                "zero_event_boundary_semantic_error"
            ),
        )
        .collect()
        .row(
            0,
            named=True,
        )
    )


# =============================================================================
# Interval-level invariants
# =============================================================================


def interval_invariants(
    samples: pl.LazyFrame,
) -> dict:
    per_interval = (
        samples
        .group_by(
            "support_interval_id"
        )
        .agg(
            pl.len()
            .cast(pl.Int64)
            .alias(
                "sample_rows"
            ),

            pl.col(
                "integration_sample_index"
            )
            .n_unique()
            .cast(pl.Int64)
            .alias(
                "unique_sample_indices"
            ),

            pl.col(
                "integration_pool_size"
            )
            .n_unique()
            .alias(
                "pool_size_values"
            ),

            pl.col(
                "integration_pool_size"
            )
            .first()
            .cast(pl.Int64)
            .alias("k"),

            pl.col(
                "supported_cell_count"
            )
            .n_unique()
            .alias(
                "cell_count_values"
            ),

            pl.col(
                "support_duration_seconds"
            )
            .n_unique()
            .alias(
                "duration_values"
            ),

            pl.col(
                "integration_domain_cell_seconds"
            )
            .n_unique()
            .alias(
                "domain_values"
            ),

            pl.col(
                "integration_weight_cell_seconds"
            )
            .sum()
            .alias(
                "summed_weights"
            ),

            pl.col(
                "integration_domain_cell_seconds"
            )
            .first()
            .alias(
                "domain_measure"
            ),

            pl.col(
                "integration_sample_index"
            )
            .min()
            .alias(
                "min_sample_index"
            ),

            pl.col(
                "integration_sample_index"
            )
            .max()
            .alias(
                "max_sample_index"
            ),
        )
    )

    return (
        per_interval
        .select(
            pl.len()
            .alias("intervals"),

            (
                pl.col("sample_rows")
                !=
                pl.col("k")
            )
            .sum()
            .alias(
                "wrong_samples_per_interval"
            ),

            (
                pl.col(
                    "unique_sample_indices"
                )
                !=
                pl.col("k")
            )
            .sum()
            .alias(
                "duplicate_or_missing_sample_indices"
            ),

            (
                pl.col(
                    "pool_size_values"
                )
                != 1
            )
            .sum()
            .alias(
                "inconsistent_pool_size"
            ),

            (
                pl.col(
                    "cell_count_values"
                )
                != 1
            )
            .sum()
            .alias(
                "inconsistent_supported_cell_count"
            ),

            (
                pl.col(
                    "duration_values"
                )
                != 1
            )
            .sum()
            .alias(
                "inconsistent_interval_duration"
            ),

            (
                pl.col(
                    "domain_values"
                )
                != 1
            )
            .sum()
            .alias(
                "inconsistent_domain_measure"
            ),

            (
                pl.col(
                    "min_sample_index"
                )
                != 0
            )
            .sum()
            .alias(
                "bad_min_sample_index"
            ),

            (
                pl.col(
                    "max_sample_index"
                )
                !=
                (
                    pl.col("k")
                    - 1
                )
            )
            .sum()
            .alias(
                "bad_max_sample_index"
            ),

            (
                ~pl.col(
                    "summed_weights"
                )
                .is_close(
                    pl.col(
                        "domain_measure"
                    ),
                    rel_tol=
                        REL_TOL,
                    abs_tol=
                        ABS_TOL,
                )
            )
            .sum()
            .alias(
                "weight_conservation_failures"
            ),
        )
        .collect()
        .row(
            0,
            named=True,
        )
    )


# =============================================================================
# Temporal tiling
# =============================================================================


def temporal_tiling_diagnostics(
    samples: pl.LazyFrame,
) -> tuple[
    dict,
    pl.DataFrame,
]:
    intervals = (
        samples
        .select(
            "support_interval_id",
            "observation_window_id",
            "source_city",
            "support_year",
            "split",
            "support_valid_from",
            "support_valid_to",
            "support_duration_seconds",
            "events_at_left_boundary",
            "left_boundary_type",
            "supported_cell_count",
        )
        .unique(
            subset=[
                "support_interval_id"
            ]
        )
    )

    ordered = (
        intervals
        .sort(
            [
                "observation_window_id",
                "support_valid_from",
            ]
        )
        .with_columns(
            pl.col(
                "support_valid_to"
            )
            .shift(1)
            .over(
                "observation_window_id"
            )
            .alias(
                "_previous_end"
            )
        )
    )

    tiling = (
        ordered
        .select(
            (
                pl.col(
                    "_previous_end"
                )
                .is_not_null()
                &
                (
                    pl.col(
                        "support_valid_from"
                    )
                    >
                    pl.col(
                        "_previous_end"
                    )
                )
            )
            .sum()
            .alias(
                "internal_temporal_gaps"
            ),

            (
                pl.col(
                    "_previous_end"
                )
                .is_not_null()
                &
                (
                    pl.col(
                        "support_valid_from"
                    )
                    <
                    pl.col(
                        "_previous_end"
                    )
                )
            )
            .sum()
            .alias(
                "temporal_overlaps"
            ),
        )
        .collect()
        .row(
            0,
            named=True,
        )
    )

    windows = (
        intervals
        .group_by(
            [
                "observation_window_id",
                "source_city",
                "support_year",
                "split",
            ]
        )
        .agg(
            pl.len()
            .alias(
                "intervals"
            ),

            pl.col(
                "support_valid_from"
            )
            .min()
            .alias(
                "window_start"
            ),

            pl.col(
                "support_valid_to"
            )
            .max()
            .alias(
                "window_end"
            ),

            pl.col(
                "support_duration_seconds"
            )
            .sum()
            .alias(
                "summed_interval_seconds"
            ),

            pl.col(
                "events_at_left_boundary"
            )
            .sum()
            .alias(
                "events_in_domain"
            ),

            pl.col(
                "support_duration_seconds"
            )
            .max()
            .alias(
                "max_interval_seconds"
            ),

            pl.col(
                "support_duration_seconds"
            )
            .mean()
            .alias(
                "mean_interval_seconds"
            ),

            pl.col(
                "supported_cell_count"
            )
            .first()
            .alias(
                "supported_cells"
            ),
        )
        .with_columns(
            (
                (
                    pl.col(
                        "window_end"
                    )
                    -
                    pl.col(
                        "window_start"
                    )
                )
                .dt.total_microseconds()
                .cast(pl.Float64)
                / 1_000_000.0
            )
            .alias(
                "window_span_seconds"
            )
        )
        .with_columns(
            (
                pl.col(
                    "summed_interval_seconds"
                )
                -
                pl.col(
                    "window_span_seconds"
                )
            )
            .alias(
                "tiling_error_seconds"
            )
        )
        .sort(
            [
                "source_city",
                "support_year",
            ]
        )
        .collect()
    )

    return (
        tiling,
        windows,
    )


# =============================================================================
# Sampling distribution
# =============================================================================


def sampling_distribution(
    samples: pl.LazyFrame,
) -> tuple[
    dict,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
]:
    analyzed = (
        samples
        .with_columns(
            (
                (
                    pl.col(
                        "sample_timestamp_utc"
                    )
                    -
                    pl.col(
                        "support_valid_from"
                    )
                )
                .dt.total_microseconds()
                .cast(pl.Float64)
                /
                (
                    (
                        pl.col(
                            "support_valid_to"
                        )
                        -
                        pl.col(
                            "support_valid_from"
                        )
                    )
                    .dt.total_microseconds()
                    .cast(pl.Float64)
                )
            )
            .alias(
                "_time_fraction"
            ),

            (
                (
                    pl.col(
                        "support_cell_index"
                    )
                    .cast(pl.Float64)
                    + 0.5
                )
                /
                pl.col(
                    "supported_cell_count"
                )
                .cast(pl.Float64)
            )
            .alias(
                "_cell_fraction"
            ),
        )
    )

    summary = (
        analyzed
        .select(
            pl.col(
                "_time_fraction"
            )
            .min()
            .alias(
                "time_min"
            ),

            pl.col(
                "_time_fraction"
            )
            .max()
            .alias(
                "time_max"
            ),

            pl.col(
                "_time_fraction"
            )
            .mean()
            .alias(
                "time_mean"
            ),

            pl.col(
                "_time_fraction"
            )
            .std()
            .alias(
                "time_std"
            ),

            pl.col(
                "_cell_fraction"
            )
            .min()
            .alias(
                "cell_min"
            ),

            pl.col(
                "_cell_fraction"
            )
            .max()
            .alias(
                "cell_max"
            ),

            pl.col(
                "_cell_fraction"
            )
            .mean()
            .alias(
                "cell_mean"
            ),

            pl.col(
                "_cell_fraction"
            )
            .std()
            .alias(
                "cell_std"
            ),
        )
        .collect()
        .row(
            0,
            named=True,
        )
    )

    time_deciles = (
        analyzed
        .with_columns(
            (
                pl.col(
                    "_time_fraction"
                )
                * 10
            )
            .floor()
            .cast(pl.Int8)
            .clip(
                0,
                9,
            )
            .alias(
                "decile"
            )
        )
        .group_by(
            "decile"
        )
        .len()
        .sort(
            "decile"
        )
        .with_columns(
            (
                pl.col("len")
                /
                pl.col("len")
                .sum()
            )
            .alias(
                "proportion"
            )
        )
        .collect()
    )

    cell_deciles = (
        analyzed
        .with_columns(
            (
                pl.col(
                    "_cell_fraction"
                )
                * 10
            )
            .floor()
            .cast(pl.Int8)
            .clip(
                0,
                9,
            )
            .alias(
                "decile"
            )
        )
        .group_by(
            "decile"
        )
        .len()
        .sort(
            "decile"
        )
        .with_columns(
            (
                pl.col("len")
                /
                pl.col("len")
                .sum()
            )
            .alias(
                "proportion"
            )
        )
        .collect()
    )

    by_sample_index = (
        analyzed
        .group_by(
            "integration_sample_index"
        )
        .agg(
            pl.len()
            .alias("rows"),

            pl.col(
                "_time_fraction"
            )
            .mean()
            .alias(
                "mean_time_fraction"
            ),

            pl.col(
                "_cell_fraction"
            )
            .mean()
            .alias(
                "mean_cell_fraction"
            ),
        )
        .sort(
            "integration_sample_index"
        )
        .collect()
    )

    return (
        summary,
        time_deciles,
        cell_deciles,
        by_sample_index,
    )


# =============================================================================
# City/year diagnostics
# =============================================================================


def city_year_summary(
    samples: pl.LazyFrame,
) -> pl.DataFrame:
    intervals = (
        samples
        .select(
            "support_interval_id",
            "source_city",
            "support_year",
            "split",
            "support_duration_seconds",
            "events_at_left_boundary",
            "supported_cell_count",
        )
        .unique(
            subset=[
                "support_interval_id"
            ]
        )
    )

    return (
        intervals
        .group_by(
            [
                "source_city",
                "support_year",
                "split",
            ]
        )
        .agg(
            pl.len()
            .alias(
                "intervals"
            ),

            pl.col(
                "events_at_left_boundary"
            )
            .sum()
            .alias(
                "modeled_events"
            ),

            pl.col(
                "support_duration_seconds"
            )
            .sum()
            .alias(
                "support_seconds"
            ),

            pl.col(
                "support_duration_seconds"
            )
            .mean()
            .alias(
                "mean_interval_seconds"
            ),

            pl.col(
                "support_duration_seconds"
            )
            .max()
            .alias(
                "max_interval_seconds"
            ),

            pl.col(
                "supported_cell_count"
            )
            .first()
            .alias(
                "supported_cells"
            ),
        )
        .sort(
            [
                "source_city",
                "support_year",
            ]
        )
        .collect()
    )


# =============================================================================
# Boundary diagnostics
# =============================================================================


def boundary_summary(
    samples: pl.LazyFrame,
) -> pl.DataFrame:
    return (
        samples
        .select(
            "support_interval_id",
            "left_boundary_type",
            "events_at_left_boundary",
        )
        .unique(
            subset=[
                "support_interval_id"
            ]
        )
        .group_by(
            "left_boundary_type"
        )
        .agg(
            pl.len()
            .alias(
                "intervals"
            ),

            pl.col(
                "events_at_left_boundary"
            )
            .sum()
            .alias(
                "events"
            ),
        )
        .sort(
            "left_boundary_type"
        )
        .collect()
    )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    samples = (
        scan_integration()
    )

    # -------------------------------------------------------------------------
    # Schema
    # -------------------------------------------------------------------------

    validate_schema(
        samples
    )

    # -------------------------------------------------------------------------
    # Global statistics
    # -------------------------------------------------------------------------

    print_section(
        "GLOBAL STATISTICS"
    )

    global_stats = (
        global_statistics(
            samples
        )
    )

    for key, value in (
        global_stats.items()
    ):
        print(
            f"{key}: {value}"
        )

    # -------------------------------------------------------------------------
    # Nulls
    # -------------------------------------------------------------------------

    print_section(
        "NULL DIAGNOSTICS"
    )

    nulls = (
        null_diagnostics(
            samples
        )
    )

    if nulls.height == 0:
        print(
            "PASS  no null values"
        )
    else:
        print(
            nulls
        )

    # -------------------------------------------------------------------------
    # Global ID uniqueness
    # -------------------------------------------------------------------------

    print_section(
        "GLOBAL CARDINALITY"
    )

    duplicate_ids = (
        global_stats["rows"]
        -
        global_stats[
            "unique_sample_ids"
        ]
    )

    print_check(
        "duplicate integration_sample_id",
        duplicate_ids,
    )

    # -------------------------------------------------------------------------
    # Row invariants
    # -------------------------------------------------------------------------

    print_section(
        "ROW-LEVEL INVARIANTS"
    )

    row_checks = (
        row_invariants(
            samples
        )
    )

    for name, value in (
        row_checks.items()
    ):
        print_check(
            name,
            int(value),
        )

    # -------------------------------------------------------------------------
    # Interval invariants
    # -------------------------------------------------------------------------

    print_section(
        "INTERVAL-LEVEL INVARIANTS"
    )

    interval_checks = (
        interval_invariants(
            samples
        )
    )

    interval_count = int(
        interval_checks.pop(
            "intervals"
        )
    )

    print(
        f"intervals: {interval_count:,}"
    )

    for name, value in (
        interval_checks.items()
    ):
        print_check(
            name,
            int(value),
        )

    # -------------------------------------------------------------------------
    # Temporal tiling
    # -------------------------------------------------------------------------

    print_section(
        "TEMPORAL TILING"
    )

    (
        tiling,
        window_summary,
    ) = (
        temporal_tiling_diagnostics(
            samples
        )
    )

    for name, value in (
        tiling.items()
    ):
        print_check(
            name,
            int(value),
        )

    max_tiling_error = (
        window_summary
        .select(
            pl.col(
                "tiling_error_seconds"
            )
            .abs()
            .max()
        )
        .item()
    )

    print(
        "max absolute observation-window "
        f"tiling error: {max_tiling_error}"
    )

    print()
    print(
        window_summary
    )

    # -------------------------------------------------------------------------
    # Boundary semantics
    # -------------------------------------------------------------------------

    print_section(
        "BOUNDARY SUMMARY"
    )

    print(
        boundary_summary(
            samples
        )
    )

    # -------------------------------------------------------------------------
    # Monte Carlo sampling quality
    # -------------------------------------------------------------------------

    print_section(
        "SAMPLING DISTRIBUTION"
    )

    (
        distribution,
        time_deciles,
        cell_deciles,
        by_sample_index,
    ) = sampling_distribution(
        samples
    )

    print(
        "Expected for a uniform distribution:"
    )
    print(
        "mean ≈ 0.5"
    )
    print(
        "std  ≈ 0.288675"
    )

    print()
    print(
        distribution
    )

    print()
    print(
        "TIME DECILES"
    )
    print(
        time_deciles
    )

    print()
    print(
        "CELL DECILES"
    )
    print(
        cell_deciles
    )

    print()
    print(
        "BY SAMPLE INDEX"
    )
    print(
        by_sample_index
    )

    # -------------------------------------------------------------------------
    # City/year statistical domain
    # -------------------------------------------------------------------------

    print_section(
        "CITY / YEAR SUMMARY"
    )

    city_year = (
        city_year_summary(
            samples
        )
    )

    print(
        city_year
    )

    city_year.write_csv(
        "integration_city_year_diagnostics.csv"
    )

    window_summary.write_csv(
        "integration_window_diagnostics.csv"
    )

    time_deciles.write_csv(
        "integration_time_deciles.csv"
    )

    cell_deciles.write_csv(
        "integration_cell_deciles.csv"
    )

    print()
    print(
        "Diagnostic CSVs written."
    )


if __name__ == "__main__":
    main()