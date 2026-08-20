import dagster as dg
import polars as pl
import polars_h3 as plh3

from crimenet_data.assets.event_spine import (
    event_spine,
)
from crimenet_data.assets.osm.silver import (
    osm_h3_silver_assets,
)
SPATIAL_DOMAIN_MASK_ROOT = (
    "gs://crimenet/silver/spatial_support/"
    "jurisdiction_h3_mask"
)
from .transformations import (
    DEFAULT_INTEGRATION_POOL_SIZE,
    H3_RESOLUTION,
    SAMPLING_VERSION,
    build_integration_samples,
    build_observation_windows_from_local_ranges,
    prepare_spatial_support,
    select_modeled_events,
)

OBSERVATION_RANGES_LOCAL = {
    "baltimore": [
        (
            "2022-01-01T00:00:00",
            "2026-01-01T00:00:00",
        ),
    ],

    "chicago": [
        (
            "2014-01-01T00:00:00",
            "2026-01-01T00:00:00",
        ),
    ],

    "dallas": [
        (
            "2017-01-01T00:00:00",
            "2026-07-24T00:00:00",
        ),
    ],

    "fort_worth": [
        (
            "2016-01-01T00:00:00",
            "2026-07-20T00:00:00",
        ),
    ],

    "new_york": [
        (
            "2014-01-01T00:00:00",
            "2026-01-01T00:00:00",
        ),
    ],

    "san_francisco": [
        (
            "2018-01-01T00:00:00",
            "2026-01-01T00:00:00",
        ),
    ],

    "seattle": [
        (
            "2014-01-01T00:00:00",
            "2026-01-01T00:00:00",
        ),
    ],

    "washington_dc": [
        (
            "2019-01-01T00:00:00",
            "2026-01-01T00:00:00",
        ),
    ],
}

# =============================================================================
# Storage
# =============================================================================


EVENT_SPINE_ROOT = (
    "gs://crimenet/gold/event_spine"
)

OSM_ROOT = (
    "gs://crimenet/silver/osm_h3_features"
)

INTEGRATION_STAGING_ROOT = (
    "gs://crimenet/gold_staging/integration_samples"
)

INTEGRATION_ROOT = (
    "gs://crimenet/gold/integration_samples"
)


# =============================================================================
# City metadata
# =============================================================================


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


# These codes exist only to seed the deterministic sampler.
CITY_CODES = {
    city: index
    for index, city in enumerate(
        sorted(CITY_TIMEZONES),
        start=1,
    )
}


# =============================================================================
# I/O
# =============================================================================


def scan_delta(
    path: str,
    *,
    credentials: pl.CredentialProviderGCP,
) -> pl.LazyFrame:
    return pl.scan_delta(
        path,
        credential_provider=credentials,
    )


def write_delta(
    frame: pl.LazyFrame,
    *,
    path: str,
    credentials: pl.CredentialProviderGCP,
) -> None:
    frame.sink_delta(
        path,
        mode="overwrite",
        credential_provider=credentials,
        delta_write_options={
            "partition_by": [
                "split",
                "source_city",
                "support_year",
            ],
            "schema_mode":
                "overwrite",
        },
    )


# =============================================================================
# OSM support validation
# =============================================================================
def validate_events_inside_spatial_support(
    *,
    modeled_events: pl.LazyFrame,
    spatial_support: pl.LazyFrame,
) -> None:
    event_cells = (
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
                "osm_h3_cell_id"
            )
            .cast(pl.Int64),
        )
    )

    support_keys = (
        spatial_support
        .select(
            "source_city",
            "support_year",
            "osm_h3_cell_id",
        )
        .with_columns(
            pl.lit(True).alias("_inside_support")
        )
    )

    invalid = (
        event_cells
        .join(
            support_keys,
            on=[
                "source_city",
                "support_year",
                "osm_h3_cell_id",
            ],
            how="left",
            validate="m:1",
        )
        .filter(
            pl.col(
                "_inside_support"
            )
            .is_null()
        )
    )

    summary = (
        invalid
        .group_by(
            [
                "source_city",
                "support_year",
            ]
        )
        .agg(
            pl.len()
            .alias(
                "outside_events"
            )
        )
        .sort(
            "outside_events",
            descending=True,
        )
        .collect()
    )

    invalid_count = (
        summary[
            "outside_events"
        ]
        .sum()
        if summary.height
        else 0
    )

    if invalid_count:
        raise ValueError(
            "Modeled observed events fall outside "
            "the spatial integration support: "
            f"{invalid_count:,}\n"
            f"{summary}"
        )
def validate_city_metadata(
    modeled_events: pl.LazyFrame,
    *,
    city_timezones: dict[str, str],
) -> None:
    actual_cities = set(
        modeled_events
        .select(
            "source_city"
        )
        .unique()
        .collect()
        .get_column(
            "source_city"
        )
        .to_list()
    )

    configured_cities = set(
        city_timezones
    )

    missing = (
        actual_cities
        - configured_cities
    )

    if missing:
        raise ValueError(
            "Modeled events contain cities with "
            "no configured timezone: "
            f"{sorted(missing)}"
        )
def partition_events_by_observation_domain(
    modeled_events: pl.LazyFrame,
    observation_windows: pl.LazyFrame,
) -> tuple[
    pl.LazyFrame,
    pl.LazyFrame,
]:
    """
    Partition otherwise model-eligible events into:

        in_domain
        excluded

    An event may match:
        0 windows -> deliberately outside modeled observation domain
        1 window  -> valid modeled event
        >1 window -> invalid overlapping observation windows
    """

    windows = observation_windows.select(
        "observation_window_id",
        "source_city",
        "support_year",
        "support_valid_from",
        "support_valid_to",
    )

    candidates = (
        modeled_events
        .with_columns(
            pl.col(
                "occurrence_year"
            )
            .cast(pl.Int32)
            .alias(
                "support_year"
            )
        )
        .join(
            windows,
            on=[
                "source_city",
                "support_year",
            ],
            how="left",
        )
        .with_columns(
            (
                (
                    pl.col(
                        "occurrence_timestamp_utc"
                    )
                    >=
                    pl.col(
                        "support_valid_from"
                    )
                )
                &
                (
                    pl.col(
                        "occurrence_timestamp_utc"
                    )
                    <
                    pl.col(
                        "support_valid_to"
                    )
                )
            )
            .fill_null(False)
            .alias(
                "_inside_observation_window"
            )
        )
    )

    membership = (
        candidates
        .group_by(
            "crime_id"
        )
        .agg(
            pl.col(
                "_inside_observation_window"
            )
            .sum()
            .cast(pl.Int32)
            .alias(
                "_observation_window_matches"
            ),

            pl.col(
                "observation_window_id"
            )
            .filter(
                pl.col(
                    "_inside_observation_window"
                )
            )
            .first()
            .alias(
                "observation_window_id"
            ),
        )
    )

    annotated = (
        modeled_events
        .join(
            membership,
            on="crime_id",
            how="left",
            validate="1:1",
        )
        .with_columns(
            pl.col("_observation_window_matches").fill_null(0)
        )
    )

    overlapping = (
        annotated
        .filter(
            pl.col(
                "_observation_window_matches"
            )
            > 1
        )
        .select(
            pl.len()
            .alias("rows")
        )
        .collect()
        .item()
    )

    if overlapping:
        raise ValueError(
            "Observation windows overlap: "
            f"{overlapping:,} modeled events "
            "belong to multiple windows."
        )

    in_domain = (
        annotated
        .filter(
            pl.col(
                "_observation_window_matches"
            )
            == 1
        )
        .drop(
            "_observation_window_matches"
        )
    )

    excluded = (
        annotated
        .filter(
            pl.col(
                "_observation_window_matches"
            )
            == 0
        )
        .drop(
            "_observation_window_matches"
        )
    )

    return (
        in_domain,
        excluded,
    )
def validate_osm_support_source(
    osm: pl.LazyFrame,
) -> dict[str, int]:
    source_stats = (
        osm
        .select(
            pl.len()
            .cast(pl.Int64)
            .alias("rows"),

            (
                pl.col(
                    "osm_h3_resolution"
                )
                != H3_RESOLUTION
            )
            .sum()
            .cast(pl.Int64)
            .alias(
                "wrong_resolution_rows"
            ),
        )
        .collect()
        .row(
            0,
            named=True,
        )
    )

    key_stats = (
        osm
        .group_by(
            [
                "source_city",
                "snapshot_year",
                "osm_h3_cell_id",
            ]
        )
        .len()
        .select(
            pl.len()
            .cast(pl.Int64)
            .alias("unique_keys"),

            (
                pl.col("len") > 1
            )
            .sum()
            .cast(pl.Int64)
            .alias(
                "duplicate_keys"
            ),
        )
        .collect()
        .row(
            0,
            named=True,
        )
    )

    if (
        source_stats[
            "wrong_resolution_rows"
        ]
        != 0
    ):
        raise ValueError(
            "OSM spatial support contains "
            "non-H3-9 rows."
        )

    if (
        key_stats["duplicate_keys"]
        != 0
    ):
        raise ValueError(
            "OSM spatial support contains "
            f"{key_stats['duplicate_keys']:,} "
            "duplicate city/year/cell keys."
        )

    if (
        source_stats["rows"]
        != key_stats["unique_keys"]
    ):
        raise ValueError(
            "OSM support cardinality invariant failed: "
            f"rows={source_stats['rows']:,}, "
            f"unique_keys={key_stats['unique_keys']:,}"
        )

    return {
        "osm_support_rows":
            int(
                source_stats["rows"]
            ),

        "osm_support_unique_keys":
            int(
                key_stats["unique_keys"]
            ),
    }


# =============================================================================
# Interval validation before sample expansion
# =============================================================================


def validate_intervals(
    intervals: pl.LazyFrame,
) -> dict[str, int | float]:
    metrics = (
        intervals
        .select(
            pl.len()
            .cast(pl.Int64)
            .alias(
                "support_intervals"
            ),

            pl.col(
                "supported_cell_count"
            )
            .is_null()
            .sum()
            .cast(pl.Int64)
            .alias(
                "missing_spatial_support"
            ),

            (
                pl.col(
                    "supported_cell_count"
                )
                <= 0
            )
            .fill_null(False)
            .sum()
            .cast(pl.Int64)
            .alias(
                "nonpositive_cell_counts"
            ),

            (
                pl.col(
                    "support_duration_seconds"
                )
                <= 0
            )
            .sum()
            .cast(pl.Int64)
            .alias(
                "nonpositive_durations"
            ),

            pl.col(
                "support_duration_seconds"
            )
            .sum()
            .alias(
                "total_support_seconds"
            ),
        )
        .collect()
        .row(
            0,
            named=True,
        )
    )

    if (
        metrics[
            "missing_spatial_support"
        ]
        != 0
    ):
        raise ValueError(
            "Integration intervals exist without "
            "matching OSM city/year spatial support: "
            f"{metrics['missing_spatial_support']:,}"
        )

    if (
        metrics[
            "nonpositive_cell_counts"
        ]
        != 0
    ):
        raise ValueError(
            "Integration support contains "
            "non-positive cell counts."
        )

    if (
        metrics[
            "nonpositive_durations"
        ]
        != 0
    ):
        raise ValueError(
            "Integration support contains "
            "non-positive durations."
        )

    return metrics


# =============================================================================
# Materialized output validation
# =============================================================================


def validate_materialized_samples(
    samples: pl.LazyFrame,
    *,
    expected_rows: int,
    samples_per_interval: int,
) -> dict[str, int | float]:
    expected_weight = (
        pl.col(
            "support_duration_seconds"
        )
        * pl.col(
            "supported_cell_count"
        )
        .cast(pl.Float64)
        / samples_per_interval
    )

    metrics = (
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
                "osm_h3_cell_id"
            )
            .is_null()
            .sum()
            .cast(pl.Int64)
            .alias(
                "missing_sample_cells"
            ),

            (
                pl.col(
                    "sample_timestamp_utc"
                )
                <= pl.col(
                    "support_valid_from"
                )
            )
            .sum()
            .cast(pl.Int64)
            .alias(
                "samples_at_or_before_start"
            ),

            (
                pl.col(
                    "sample_timestamp_utc"
                )
                >= pl.col(
                    "support_valid_to"
                )
            )
            .sum()
            .cast(pl.Int64)
            .alias(
                "samples_at_or_after_end"
            ),

            (
                pl.col(
                    "integration_weight_cell_seconds"
                )
                <= 0
            )
            .sum()
            .cast(pl.Int64)
            .alias(
                "nonpositive_weights"
            ),
            (
                ~pl.col(
                    "integration_weight_cell_seconds"
                ).is_close(
                    expected_weight,
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                )
            )
            .sum()
            .cast(pl.Int64)
            .alias(
                "incorrect_weights"
            ),
            (
                pl.col(
                    "integration_sample_index"
                )
                < 0
            )
            .sum()
            .cast(pl.Int64)
            .alias(
                "negative_sample_indices"
            ),

            (
                pl.col(
                    "integration_sample_index"
                )
                >= samples_per_interval
            )
            .sum()
            .cast(pl.Int64)
            .alias(
                "sample_index_overflow"
            ),

            (
                plh3.get_resolution(
                    "osm_h3_cell_id"
                )
                != H3_RESOLUTION
            )
            .sum()
            .cast(pl.Int64)
            .alias(
                "wrong_h3_resolution"
            ),

            (
                pl.col(
                    "sampling_version"
                )
                != SAMPLING_VERSION
            )
            .sum()
            .cast(pl.Int64)
            .alias(
                "wrong_sampling_version"
            ),

            pl.col(
                "integration_weight_cell_seconds"
            )
            .sum()
            .alias(
                "total_integration_weight_cell_seconds"
            ),
        )
        .collect()
        .row(
            0,
            named=True,
        )
    )

    if metrics["rows"] != expected_rows:
        raise ValueError(
            "Integration sample cardinality changed: "
            f"expected={expected_rows:,}, "
            f"actual={metrics['rows']:,}"
        )

    if (
        metrics["unique_sample_ids"]
        != metrics["rows"]
    ):
        raise ValueError(
            "integration_sample_id uniqueness violated: "
            f"rows={metrics['rows']:,}, "
            f"unique={metrics['unique_sample_ids']:,}"
        )

    fatal_fields = [
        "missing_sample_cells",
        "samples_at_or_before_start",
        "samples_at_or_after_end",
        "nonpositive_weights",
        "incorrect_weights",
        "negative_sample_indices",
        "sample_index_overflow",
        "wrong_h3_resolution",
        "wrong_sampling_version",
    ]

    failures = {
        field:
            int(metrics[field])
        for field in fatal_fields
        if metrics[field] != 0
    }

    if failures:
        raise ValueError(
            "Integration validation failed: "
            f"{failures}"
        )

    return metrics

def validate_spatial_domain_mask(
    *,
    osm: pl.LazyFrame,
    spatial_domain_mask: pl.LazyFrame,
) -> dict[str, int]:
    osm_keys = (
        osm
        .filter(
            pl.col("osm_h3_resolution")
            == H3_RESOLUTION
        )
        .select(
            "source_city",

            pl.col("snapshot_year")
            .cast(pl.Int32)
            .alias("support_year"),

            plh3.str_to_int(
                "osm_h3_cell_id"
            )
            .cast(pl.Int64)
            .alias("osm_h3_cell_id"),
        )
    )

    mask_keys = (
        spatial_domain_mask
        .select(
            "source_city",

            pl.col("support_year")
            .cast(pl.Int32),

            pl.col("osm_h3_cell_id")
            .cast(pl.Int64),

            "inside_observation_domain",
        )
        .with_columns(
            pl.lit(True)
            .alias("_mask_matched")
        )
    )

    metrics = (
        osm_keys
        .join(
            mask_keys,
            on=[
                "source_city",
                "support_year",
                "osm_h3_cell_id",
            ],
            how="left",
            validate="1:1",
        )
        .select(
            pl.len()
            .alias("osm_cells"),

            pl.col("_mask_matched")
            .is_null()
            .sum()
            .alias("missing_mask_cells"),

            pl.col("inside_observation_domain")
            .fill_null(False)
            .sum()
            .alias("inside_cells"),
        )
        .collect()
        .row(0, named=True)
    )

    if metrics["missing_mask_cells"]:
        raise ValueError(
            "Spatial-domain mask is incomplete: "
            f"{metrics['missing_mask_cells']:,} "
            "OSM support cells have no jurisdiction decision."
        )

    return metrics
# =============================================================================
# Asset
# =============================================================================
def partition_events_by_spatial_support(
    modeled_events: pl.LazyFrame,
    spatial_support: pl.LazyFrame,
) -> tuple[
    pl.LazyFrame,
    pl.LazyFrame,
]:
    """
    Partition temporally valid modeled events into:

        in_domain:
            event H3 belongs to the exact spatial support used
            by the compensator.

        excluded:
            event H3 lies outside that model spatial support.

    This guarantees that observed-event log-intensity terms and
    compensator integration use the same spatial domain.
    """

    support_keys = (
        spatial_support
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
        .with_columns(
            pl.lit(True)
            .alias(
                "_inside_spatial_support"
            )
        )
    )

    annotated = (
        modeled_events
        .with_columns(
            pl.col(
                "occurrence_year"
            )
            .cast(pl.Int32)
            .alias(
                "support_year"
            ),

            pl.col(
                "osm_h3_cell_id"
            )
            .cast(pl.Int64),
        )
        .join(
            support_keys,
            on=[
                "source_city",
                "support_year",
                "osm_h3_cell_id",
            ],
            how="left",
            validate="m:1",
        )
        .with_columns(
            pl.col(
                "_inside_spatial_support"
            )
            .fill_null(False)
        )
    )

    in_domain = (
        annotated
        .filter(
            pl.col(
                "_inside_spatial_support"
            )
        )
        .drop(
            [
                "_inside_spatial_support",
                "support_year",
            ]
        )
    )

    excluded = (
        annotated
        .filter(
            ~pl.col(
                "_inside_spatial_support"
            )
        )
        .drop(
            [
                "_inside_spatial_support",
                "support_year",
            ]
        )
    )

    return (
        in_domain,
        excluded,
    )
def diagnose_event_spatial_failures(
    *,
    modeled_events: pl.LazyFrame,
    osm: pl.LazyFrame,
    spatial_domain_mask: pl.LazyFrame,
) -> pl.DataFrame:

    events = (
        modeled_events
        .select(
            "crime_id",
            "source_city",

            pl.col("occurrence_year")
            .cast(pl.Int32)
            .alias("support_year"),

            pl.col("osm_h3_cell_id")
            .cast(pl.Int64),
        )
    )

    osm_keys = (
        osm
        .filter(
            pl.col("osm_h3_resolution")
            == H3_RESOLUTION
        )
        .select(
            "source_city",

            pl.col("snapshot_year")
            .cast(pl.Int32)
            .alias("support_year"),

            plh3.str_to_int(
                "osm_h3_cell_id"
            )
            .cast(pl.Int64)
            .alias("osm_h3_cell_id"),
        )
        .with_columns(
            pl.lit(True)
            .alias("_osm_present")
        )
    )

    mask = (
        spatial_domain_mask
        .select(
            "source_city",

            pl.col("support_year")
            .cast(pl.Int32),

            pl.col("osm_h3_cell_id")
            .cast(pl.Int64),

            "inside_observation_domain",
        )
        .with_columns(
            pl.lit(True)
            .alias("_mask_present")
        )
    )

    classified = (
        events
        .join(
            osm_keys,
            on=[
                "source_city",
                "support_year",
                "osm_h3_cell_id",
            ],
            how="left",
            validate="m:1",
        )
        .join(
            mask,
            on=[
                "source_city",
                "support_year",
                "osm_h3_cell_id",
            ],
            how="left",
            validate="m:1",
        )
        .with_columns(
            pl.when(
                pl.col("_osm_present").is_null()
            )
            .then(
                pl.lit("missing_from_osm")
            )
            .when(
                pl.col("_mask_present").is_null()
            )
            .then(
                pl.lit("missing_from_mask")
            )
            .when(
                ~pl.col(
                    "inside_observation_domain"
                ).fill_null(False)
            )
            .then(
                pl.lit("outside_jurisdiction")
            )
            .otherwise(
                pl.lit("inside_support")
            )
            .alias("failure_reason")
        )
    )

    return (
        classified
        .filter(
            pl.col("failure_reason")
            != "inside_support"
        )
        .group_by(
            [
                "failure_reason",
                "source_city",
                "support_year",
            ]
        )
        .agg(
            pl.len()
            .alias("events")
        )
        .sort(
            "events",
            descending=True,
        )
        .collect()
    )
@dg.asset(
    name="integration_samples",
    group_name="integration",
    compute_kind="polars",

    # event_spine is read from external Delta storage rather than passed
    # through memory, so declare the dependency explicitly.
    deps=[
        event_spine,
        *osm_h3_silver_assets,
    ],
)
def integration_samples(
    context: dg.AssetExecutionContext,
) -> dg.MaterializeResult:
    credentials = pl.CredentialProviderGCP()
    k = DEFAULT_INTEGRATION_POOL_SIZE

    # -------------------------------------------------------------------------
    # Scan + validate source support.
    # -------------------------------------------------------------------------
    events = scan_delta(
        EVENT_SPINE_ROOT,
        credentials=credentials,
    )
    osm = scan_delta(
        OSM_ROOT,
        credentials=credentials,
    )

    osm_metrics = validate_osm_support_source(osm)
    context.log.info(
        "OSM integration support validated: "
        f"{osm_metrics['osm_support_rows']:,} unique H3-9 city/year cells"
    )

    # -------------------------------------------------------------------------
    # Candidate modeled events.
    # -------------------------------------------------------------------------
    candidate_modeled_events = select_modeled_events(events)
    validate_city_metadata(
        candidate_modeled_events,
        city_timezones=CITY_TIMEZONES,
    )

    observation_windows = build_observation_windows_from_local_ranges(
        observation_ranges=OBSERVATION_RANGES_LOCAL,
        city_timezones=CITY_TIMEZONES,
    )

    # -------------------------------------------------------------------------
    # Temporal partition.
    # -------------------------------------------------------------------------
    (
        temporally_valid_events,
        temporal_excluded_events,
    ) = partition_events_by_observation_domain(
        candidate_modeled_events,
        observation_windows,
    )

    candidate_count = (
        candidate_modeled_events.select(pl.len()).collect().item()
    )
    temporally_valid_count = (
        temporally_valid_events.select(pl.len()).collect().item()
    )
    temporal_excluded_count = (
        temporal_excluded_events.select(pl.len()).collect().item()
    )

    if candidate_count != temporally_valid_count + temporal_excluded_count:
        raise ValueError(
            "Temporal partition cardinality failed: "
            f"candidate={candidate_count:,}, "
            f"temporally_valid={temporally_valid_count:,}, "
            f"temporal_excluded={temporal_excluded_count:,}"
        )

    context.log.info(
        "Temporal observation-domain partition validated: "
        f"candidate={candidate_count:,}, "
        f"in_domain={temporally_valid_count:,}, "
        f"excluded={temporal_excluded_count:,}"
    )

    if temporal_excluded_count:
        temporal_summary = (
            temporal_excluded_events
            .group_by(["source_city", "occurrence_year"])
            .agg(pl.len().alias("excluded_events"))
            .sort(["source_city", "occurrence_year"])
            .collect()
        )
        context.log.info(
            "Events intentionally excluded from the temporal observation domain:\n"
            f"{temporal_summary}"
        )

    # -------------------------------------------------------------------------
    # Exact model spatial support.
    #
    # The mask is defined over OSM-feature-covered H3-9 cells and retains only
    # cells admitted by the jurisdiction rule. This exact support is reused for
    # both event membership and compensator integration.
    # -------------------------------------------------------------------------
    spatial_domain_mask = scan_delta(
        SPATIAL_DOMAIN_MASK_ROOT,
        credentials=credentials,
    )
    mask_metrics = validate_spatial_domain_mask(
        osm=osm,
        spatial_domain_mask=spatial_domain_mask,
    )
    spatial_support = prepare_spatial_support(spatial_domain_mask)

    context.log.info(
        "Spatial-domain mask validated: "
        f"osm_cells={mask_metrics['osm_cells']:,}, "
        f"inside_cells={mask_metrics['inside_cells']:,}"
    )

    # -------------------------------------------------------------------------
    # Spatial partition.
    # -------------------------------------------------------------------------
    (
        modeled_events,
        spatial_excluded_events,
    ) = partition_events_by_spatial_support(
        temporally_valid_events,
        spatial_support,
    )

    modeled_count = modeled_events.select(pl.len()).collect().item()
    spatial_excluded_count = (
        spatial_excluded_events.select(pl.len()).collect().item()
    )

    if temporally_valid_count != modeled_count + spatial_excluded_count:
        raise ValueError(
            "Spatial partition cardinality failed: "
            f"temporally_valid={temporally_valid_count:,}, "
            f"modeled={modeled_count:,}, "
            f"spatial_excluded={spatial_excluded_count:,}"
        )

    if (
        candidate_count
        != temporal_excluded_count + spatial_excluded_count + modeled_count
    ):
        raise ValueError(
            "Final observation-domain partition failed: "
            f"candidate={candidate_count:,}, "
            f"temporal_excluded={temporal_excluded_count:,}, "
            f"spatial_excluded={spatial_excluded_count:,}, "
            f"modeled={modeled_count:,}"
        )

    if spatial_excluded_count:
        spatial_summary = (
            spatial_excluded_events
            .group_by(["source_city", "occurrence_year"])
            .agg(pl.len().alias("excluded_events"))
            .sort(["source_city", "occurrence_year"])
            .collect()
        )
        context.log.info(
            "Events intentionally excluded from the spatial observation domain: "
            f"{spatial_excluded_count:,}\n{spatial_summary}"
        )

    context.log.info(
        "Final modeled observation domain: "
        f"candidate={candidate_count:,}, "
        f"temporal_excluded={temporal_excluded_count:,}, "
        f"spatial_excluded={spatial_excluded_count:,}, "
        f"modeled={modeled_count:,}"
    )

    # Hard invariant: every event retained for the likelihood must be in the
    # exact spatial support used by the compensator.
    validate_events_inside_spatial_support(
        modeled_events=modeled_events,
        spatial_support=spatial_support,
    )

    # -------------------------------------------------------------------------
    # Build integration samples from the same domain-filtered event set and
    # the same spatial-support LazyFrame.
    # -------------------------------------------------------------------------
    samples, intervals, _spatial_support = build_integration_samples(
        modeled_events=modeled_events,
        spatial_support=spatial_support,
        observation_windows=observation_windows,
        city_codes=CITY_CODES,
        samples_per_interval=k,
    )

    interval_metrics = validate_intervals(intervals)
    interval_count = int(interval_metrics["support_intervals"])
    expected_sample_rows = interval_count * k

    context.log.info(
        "Integration support validated: "
        f"{interval_count:,} intervals, "
        f"K={k}, expected_samples={expected_sample_rows:,}"
    )

    # -------------------------------------------------------------------------
    # Materialize staging, validate persisted output, then publish.
    # -------------------------------------------------------------------------
    context.log.info("Writing integration sample staging table")
    write_delta(
        samples,
        path=INTEGRATION_STAGING_ROOT,
        credentials=credentials,
    )

    staged = scan_delta(
        INTEGRATION_STAGING_ROOT,
        credentials=credentials,
    )
    sample_metrics = validate_materialized_samples(
        staged,
        expected_rows=expected_sample_rows,
        samples_per_interval=k,
    )

    context.log.info(
        "Integration samples passed validation: "
        f"rows={sample_metrics['rows']:,}, "
        f"unique_ids={sample_metrics['unique_sample_ids']:,}"
    )

    context.log.info("Publishing integration samples")
    write_delta(
        staged,
        path=INTEGRATION_ROOT,
        credentials=credentials,
    )

    return dg.MaterializeResult(
        metadata={
            "sampling_version": SAMPLING_VERSION,
            "samples_per_interval": k,
            "candidate_modeled_events": candidate_count,
            "modeled_events": modeled_count,
            "temporal_excluded_events": temporal_excluded_count,
            "spatial_excluded_events": spatial_excluded_count,
            "excluded_events": (
                temporal_excluded_count + spatial_excluded_count
            ),
            "support_intervals": interval_count,
            "integration_sample_rows": sample_metrics["rows"],
            "osm_support_cells": osm_metrics["osm_support_rows"],
            "total_support_seconds": interval_metrics["total_support_seconds"],
            "total_integration_weight_cell_seconds": sample_metrics[
                "total_integration_weight_cell_seconds"
            ],
            "output": dg.MetadataValue.text(INTEGRATION_ROOT),
        }
    )
