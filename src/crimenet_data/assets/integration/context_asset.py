import duckdb
import dagster as dg
import polars as pl
import polars_h3 as plh3


from crimenet_data.assets.integration import (
    integration_samples,
)
from crimenet_data.assets.osm.silver import (
    osm_h3_silver_assets,
)
from crimenet_data.assets.weather.silver import (
    weather_silver_assets,
)
from crimenet_data.assets.socioeconomic.silver import (
    socioeconomic_silver_assets,
)
from crimenet_data.assets.tract_resources.silver import (
    tract_resource_silver_assets,
)

from .enrichment import (
    build_integration_base_context,
    build_integration_context,
)


# =============================================================================
# Storage
# =============================================================================


INTEGRATION_ROOT = (
    "gs://crimenet/gold/integration_samples"
)

INTEGRATION_BASE_STAGING_ROOT = (
    "gs://crimenet/gold_staging/integration_context_base"
)

INTEGRATION_CONTEXT_STAGING_ROOT = (
    "gs://crimenet/gold_staging/integration_context"
)

INTEGRATION_CONTEXT_ROOT = (
    "gs://crimenet/gold/integration_context"
)

INTEGRATION_TRACT_MAPPING_ROOT = (
    "gs://crimenet/silver/tract_resources/"
    "integration_h3_tract_mapping"
)


ACS_CALENDAR_ROOT = (
    "gs://crimenet/raw_files/landing/"
    "tract_resources/acs_vintage_calendar"
)

TRACT_BOUNDARIES_POLARS_ROOT = (
    "gs://crimenet/raw_files/landing/"
    "tract_resources/census_tract_boundaries_polars_v2"
)

SOCIOECONOMIC_ROOT = (
    "gs://crimenet/silver/socioeconomic/tract"
)

OSM_ROOT = (
    "gs://crimenet/silver/osm_h3_features"
)

LAND_WEATHER_ROOT = (
    "gs://crimenet/silver/weather/land_hourly"
)

COASTAL_WEATHER_ROOT = (
    "gs://crimenet/silver/weather/coastal_hourly"
)


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


LEAK_FLAGS = [
    "_sample_local_year_mismatch",
    "_leak_acs_future_release",
    "_leak_future_acs_vintage",
    "_leak_future_tiger_year",
    "_leak_future_osm_snapshot",
    "_leak_future_weather",
    "_weather_hour_alignment_violation",
]


# =============================================================================
# I/O
# =============================================================================

CITY_STATE_FIPS = {
    "baltimore": "24",
    "chicago": "17",
    "dallas": "48",
    "fort_worth": "48",
    "new_york": "36",
    "san_francisco": "06",
    "seattle": "53",
    "washington_dc": "11",
}
def scan_delta(
    path: str,
    *,
    credentials: pl.CredentialProviderGCP,
) -> pl.LazyFrame:
    return pl.scan_delta(
        path,
        credential_provider=
            credentials,
    )


import random
import time

def write_delta(
    frame: pl.LazyFrame,
    *,
    path: str,
    credentials: pl.CredentialProviderGCP,
    context: dg.AssetExecutionContext,
    max_attempts: int = 3,
) -> None:
    storage_options = {
        # Retry individual GCS HTTP/object-store operations.
        "max_retries": "20",

        # Total period during which retries may continue.
        "retry_timeout": "600s",

        # Overall HTTP request timeout.
        # Keep this >= retry_timeout.
        "timeout": "600s",

        # Fail relatively quickly when a connection
        # cannot even be established.
        "connect_timeout": "30s",
    }

    for attempt in range(
        1,
        max_attempts + 1,
    ):
        try:
            frame.sink_delta(
                path,
                mode="overwrite",

                credential_provider=credentials,

                storage_options=
                    storage_options,

                delta_write_options={
                    "partition_by": [
                        "split",
                        "source_city",
                        "support_year",
                    ],

                    "schema_mode":
                        "overwrite",

                    "target_file_size":
                        64 * 1024 * 1024,
                },
            )

            return

        except Exception as exc:
            if attempt == max_attempts:
                context.log.error(
                    "Delta write failed after "
                    f"{max_attempts} attempts: "
                    f"{path}"
                )
                raise

            delay = min(
                60.0,
                5.0
                * (
                    2
                    ** (
                        attempt - 1
                    )
                ),
            )

            delay += random.uniform(
                0,
                2,
            )

            context.log.warning(
                f"Delta write failed: {path}. "
                f"Attempt "
                f"{attempt}/{max_attempts}. "
                f"Retrying entire write in "
                f"{delay:.1f}s. "
                f"Error: {exc!r}"
            )

            time.sleep(
                delay
            )
def build_h3_tract_mapping(
    *,
    base: pl.LazyFrame,
    boundaries: pl.LazyFrame,
) -> pl.DataFrame:
    """
    Construct:

        source_city
        × tiger_line_year
        × H3-9

            -> tract_geoid

    The spatial join is executed independently for each
    city/TIGER-vintage pair so DuckDB can use its optimized
    SPATIAL_JOIN operator.

    IMPORTANT:
        The JOIN itself contains ONLY the spatial predicate.
        Year/state restrictions are pushed into the input relations.
    """

    # =========================================================================
    # Unique spatial states requiring tract assignment.
    # =========================================================================

    support_points = (
        base
        .select(
            "source_city",
            "tiger_line_year",
            "osm_h3_cell_id",
            "sample_latitude",
            "sample_longitude",
        )
        .filter(
            pl.col(
                "tiger_line_year"
            )
            .is_not_null()
        )
        .unique(
            subset=[
                "source_city",
                "tiger_line_year",
                "osm_h3_cell_id",
            ]
        )
        .collect()
    )

    if support_points.height == 0:
        raise ValueError(
            "No integration H3 support points "
            "available for tract mapping."
        )

    # =========================================================================
    # Relevant tract boundaries only.
    # =========================================================================

    tiger_years = (
        support_points
        .get_column(
            "tiger_line_year"
        )
        .unique()
        .to_list()
    )

    tract_boundaries = (
        boundaries
        .filter(
            pl.col(
                "boundary_vintage"
            )
            .is_in(
                tiger_years
            )
        )
        .select(
            pl.col(
                "geoid"
            )
            .alias(
                "tract_geoid"
            ),

            pl.col(
                "boundary_vintage"
            )
            .cast(pl.Int32),

            pl.col(
                "tract_geometry_wkb"
            )
            .cast(pl.Binary),
        )
        .with_columns(
            pl.col(
                "tract_geoid"
            )
            .str.slice(
                0,
                2,
            )
            .alias(
                "state_fips"
            )
        )
        .collect()
    )

    # =========================================================================
    # DuckDB setup
    # =========================================================================

    con = duckdb.connect(
        database=":memory:"
    )

    try:
        con.execute(
            "INSTALL spatial;"
        )

        con.execute(
            "LOAD spatial;"
        )

        # Avoid unnecessary ordering guarantees during this large operation.
        con.execute(
            "SET preserve_insertion_order = false;"
        )

        # ---------------------------------------------------------------------
        # Register points.
        # ---------------------------------------------------------------------

        con.register(
            "support_points_raw",
            support_points.to_arrow(),
        )

        con.execute(
            """
            CREATE TEMP TABLE support_points AS
            SELECT
                source_city,
                tiger_line_year,
                osm_h3_cell_id,
                sample_latitude,
                sample_longitude,

                ST_Point(
                    sample_longitude,
                    sample_latitude
                ) AS geometry

            FROM support_points_raw
            """
        )

        # ---------------------------------------------------------------------
        # Register tract polygons.
        # ---------------------------------------------------------------------

        con.register(
            "tract_boundaries_raw",
            tract_boundaries.to_arrow(),
        )

        con.execute(
            """
            CREATE TEMP TABLE tract_boundaries AS
            SELECT
                tract_geoid,
                boundary_vintage,
                state_fips,

                ST_GeomFromWKB(
                    tract_geometry_wkb
                ) AS geometry

            FROM tract_boundaries_raw

            WHERE
                tract_geometry_wkb IS NOT NULL
            """
        )

        # =====================================================================
        # Run one optimized spatial join per city × TIGER vintage.
        # =====================================================================

        combinations = (
            support_points
            .select(
                "source_city",
                "tiger_line_year",
            )
            .unique()
            .sort(
                [
                    "source_city",
                    "tiger_line_year",
                ]
            )
            .rows()
        )

        results: list[pl.DataFrame] = []

        for (
            source_city,
            tiger_line_year,
        ) in combinations:
            state_fips = (
                CITY_STATE_FIPS[
                    source_city
                ]
            )

            # -------------------------------------------------------------
            # Filtering occurs inside the subqueries.
            #
            # Therefore the JOIN itself has ONE condition:
            #
            #     ST_ContainsProperly(...)
            #
            # allowing DuckDB to use SPATIAL_JOIN.
            # -------------------------------------------------------------

            result_arrow = (
                con.execute(
                    """
                    SELECT
                        p.source_city,
                        p.tiger_line_year,
                        p.osm_h3_cell_id,
                        p.sample_latitude,
                        p.sample_longitude,
                        b.tract_geoid

                    FROM (
                        SELECT *
                        FROM support_points
                        WHERE
                            source_city = ?
                            AND tiger_line_year = ?
                    ) AS p

                    LEFT JOIN (
                        SELECT *
                        FROM tract_boundaries
                        WHERE
                            boundary_vintage = ?
                            AND state_fips = ?
                    ) AS b

                    ON ST_ContainsProperly(
                        b.geometry,
                        p.geometry
                    )
                    """,
                    [
                        source_city,
                        tiger_line_year,
                        tiger_line_year,
                        state_fips,
                    ],
                )
                .fetch_arrow_table()
            )

            result = (
                pl.from_arrow(
                    result_arrow
                )
            )

            print(
                "Mapped "
                f"{source_city} "
                f"TIGER {tiger_line_year}: "
                f"{result.height:,} H3 states"
            )

            results.append(
                result
            )

    finally:
        con.close()

    # =========================================================================
    # Combine mappings
    # =========================================================================

    mapping = (
        pl.concat(
            results,
            how="vertical",
        )
        .with_columns(
            pl.col(
                "tiger_line_year"
            )
            .cast(pl.Int32),

            pl.col(
                "tract_geoid"
            )
            .is_not_null()
            .alias(
                "_tract_mapping_matched"
            ),
        )
    )

    # =========================================================================
    # Cardinality invariant
    # =========================================================================

    expected_keys = (
        support_points.height
    )

    actual_keys = (
        mapping.height
    )

    if (
        actual_keys
        != expected_keys
    ):
        raise ValueError(
            "H3 -> tract mapping changed cardinality: "
            f"expected={expected_keys:,}, "
            f"actual={actual_keys:,}"
        )

    # =========================================================================
    # No H3 state may belong to >1 tract.
    # =========================================================================

    duplicate_count = (
        mapping
        .group_by(
            [
                "source_city",
                "tiger_line_year",
                "osm_h3_cell_id",
            ]
        )
        .len()
        .filter(
            pl.col("len") > 1
        )
        .height
    )

    if duplicate_count:
        raise ValueError(
            "H3 -> tract mapping is not m:1: "
            f"{duplicate_count:,} support keys "
            "matched multiple census tracts."
        )

    return mapping
def validate_base_context(
    base: pl.LazyFrame,
    *,
    expected_rows: int,
) -> dict[str, int | float]:
    metrics = (
        base
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

            (
                ~pl.col(
                    "_acs_calendar_matched"
                )
                .fill_null(False)
            )
            .sum()
            .cast(pl.Int64)
            .alias(
                "missing_acs"
            ),

            (
                ~pl.col(
                    "_osm_matched"
                )
                .fill_null(False)
            )
            .sum()
            .cast(pl.Int64)
            .alias(
                "missing_osm"
            ),

            pl.col(
                "_sample_local_year_mismatch"
            )
            .fill_null(False)
            .sum()
            .cast(pl.Int64)
            .alias(
                "local_year_mismatch"
            ),

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

    if (
        metrics["rows"]
        != expected_rows
    ):
        raise ValueError(
            "Base context row count changed: "
            f"expected={expected_rows:,}, "
            f"actual={metrics['rows']:,}"
        )

    if (
        metrics["unique_sample_ids"]
        != metrics["rows"]
    ):
        raise ValueError(
            "Base context sample IDs are not unique: "
            f"rows={metrics['rows']:,}, "
            f"unique_ids="
            f"{metrics['unique_sample_ids']:,}"
        )

    if metrics["missing_acs"]:
        raise ValueError(
            "Missing ACS calendar context for "
            f"{metrics['missing_acs']:,} "
            "integration samples."
        )

    if metrics["missing_osm"]:
        raise ValueError(
            "Missing OSM context for "
            f"{metrics['missing_osm']:,} "
            "integration samples."
        )

    if metrics["local_year_mismatch"]:
        raise ValueError(
            "Integration samples have support_year "
            "different from sample local year: "
            f"{metrics['local_year_mismatch']:,}"
        )

    return metrics
def validate_final_context(
    context_frame: pl.LazyFrame,
    *,
    expected_rows: int,
    expected_weight: float,
) -> dict[str, int | float]:
    metrics = (
        context_frame
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
                "_tract_mapping_matched"
            )
            .fill_null(False)
            .sum()
            .cast(pl.Int64)
            .alias(
                "tract_matches"
            ),

            pl.col(
                "_socioeconomic_matched"
            )
            .fill_null(False)
            .sum()
            .cast(pl.Int64)
            .alias(
                "socio_matches"
            ),

            pl.col(
                "weather_temperature_2m_c"
            )
            .is_not_null()
            .sum()
            .cast(pl.Int64)
            .alias(
                "weather_matches"
            ),

            *[
                pl.col(flag)
                .fill_null(False)
                .sum()
                .cast(pl.Int64)
                .alias(flag)
                for flag in LEAK_FLAGS
            ],

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

    rows = int(
        metrics["rows"]
    )

    if rows != expected_rows:
        raise ValueError(
            "Integration context cardinality changed: "
            f"expected={expected_rows:,}, "
            f"actual={rows:,}"
        )

    if (
        metrics["unique_sample_ids"]
        != rows
    ):
        raise ValueError(
            "Integration context sample IDs "
            "are not unique."
        )

    for flag in LEAK_FLAGS:
        if metrics[flag]:
            raise ValueError(
                f"{flag}="
                f"{metrics[flag]:,}"
            )

    tract_rate = (
        metrics["tract_matches"]
        / rows
    )

    socio_rate = (
        metrics["socio_matches"]
        / rows
    )

    weather_rate = (
        metrics["weather_matches"]
        / rows
    )

    if tract_rate < 0.99:
        raise ValueError(
            "Tract coverage too low: "
            f"{tract_rate:.6%}"
        )

    if socio_rate < 0.99:
        raise ValueError(
            "Socioeconomic coverage too low: "
            f"{socio_rate:.6%}"
        )

    if weather_rate < 0.999:
        raise ValueError(
            "Weather coverage too low: "
            f"{weather_rate:.6%}"
        )

    relative_weight_error = (
        abs(
            metrics["total_weight"]
            - expected_weight
        )
        /
        abs(expected_weight)
    )

    if (
        relative_weight_error
        > 1e-12
    ):
        raise ValueError(
            "Integration measure changed "
            "during enrichment: "
            f"relative_error="
            f"{relative_weight_error:.3e}"
        )

    return {
        **metrics,

        "tract_match_rate":
            tract_rate,

        "socio_match_rate":
            socio_rate,

        "weather_match_rate":
            weather_rate,
    }
import time


PARTITION_COLUMNS = [
    "split",
    "source_city",
    "support_year",
]


def write_context_partitions(
    *,
    base: pl.LazyFrame,
    tract_mapping: pl.LazyFrame,
    socioeconomic: pl.LazyFrame,
    path: str,
    credentials: pl.CredentialProviderGCP,
    context: dg.AssetExecutionContext,
) -> None:
    partitions = (
        base
        .select(
            *PARTITION_COLUMNS
        )
        .unique()
        .sort(
            PARTITION_COLUMNS
        )
        .collect()
    )

    total = partitions.height

    for index, row in enumerate(
        partitions.iter_rows(
            named=True
        ),
        start=1,
    ):
        split = row["split"]
        city = row["source_city"]
        year = int(
            row["support_year"]
        )

        context.log.info(
            f"[{index}/{total}] "
            f"Enriching/writing "
            f"{split} / {city} / {year}"
        )

        # -------------------------------------------------------------
        # Filter BEFORE performing socioeconomic/tract enrichment.
        #
        # This prevents us from rebuilding the entire 17.4M-row
        # enrichment graph for every partition.
        # -------------------------------------------------------------

        base_partition = (
            base
            .filter(
                (pl.col("split") == split)
                &
                (
                    pl.col("source_city")
                    == city
                )
                &
                (
                    pl.col("support_year")
                    == year
                )
            )
        )

        enriched_partition = (
            build_integration_context(
                base=
                    base_partition,
                tract_mapping=
                    tract_mapping,
                socioeconomic=
                    socioeconomic,
            )
        )

        predicate = (
            f"split = '{split}' "
            f"AND source_city = '{city}' "
            f"AND support_year = {year}"
        )

        # -------------------------------------------------------------
        # Retry at the PARTITION level.
        #
        # delta-rs already retries individual HTTP requests. This gives
        # us another recovery boundary around the complete Delta write.
        # -------------------------------------------------------------

        max_attempts = 4

        for attempt in range(
            1,
            max_attempts + 1,
        ):
            try:
                enriched_partition.sink_delta(
                    path,
                    mode="overwrite",
                    credential_provider=
                        credentials,
                    delta_write_options={
                        "partition_by":
                            PARTITION_COLUMNS,

                        "predicate":
                            predicate,

                        "target_file_size":
                            64 * 1024 * 1024,
                    },
                )

                context.log.info(
                    f"[{index}/{total}] "
                    f"Completed "
                    f"{split} / {city} / {year}"
                )

                break

            except Exception:
                if attempt == max_attempts:
                    raise

                delay_seconds = (
                    5 * attempt
                )

                context.log.warning(
                    f"Write failed for "
                    f"{split} / {city} / {year}; "
                    f"retrying partition "
                    f"({attempt}/{max_attempts})"
                )

                time.sleep(
                    delay_seconds
                )
@dg.asset(
    name="integration_context",
    group_name="integration",
    compute_kind="polars_duckdb",
    deps=[
        integration_samples,
        *osm_h3_silver_assets,
        *weather_silver_assets,
        *socioeconomic_silver_assets,
        *tract_resource_silver_assets,
    ],
)

def integration_context(
    context: dg.AssetExecutionContext,
) -> dg.MaterializeResult:
    credentials = (
        pl.CredentialProviderGCP()
    )

    # =====================================================================
    # Scan inputs
    # =====================================================================

    samples = scan_delta(
        INTEGRATION_ROOT,
        credentials=credentials,
    )

    acs_calendar = scan_delta(
        ACS_CALENDAR_ROOT,
        credentials=credentials,
    )

    tract_boundaries = scan_delta(
        TRACT_BOUNDARIES_POLARS_ROOT,
        credentials=credentials,
    )

    socioeconomic = scan_delta(
        SOCIOECONOMIC_ROOT,
        credentials=credentials,
    )

    osm = scan_delta(
        OSM_ROOT,
        credentials=credentials,
    )

    land_weather = scan_delta(
        LAND_WEATHER_ROOT,
        credentials=credentials,
    )

    coastal_weather = scan_delta(
        COASTAL_WEATHER_ROOT,
        credentials=credentials,
    )

    # =====================================================================
    # Establish source cardinality + MC measure.
    # =====================================================================

    source_metrics = (
        samples
        .select(
            pl.len()
            .cast(pl.Int64)
            .alias("rows"),

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

    source_rows = int(
        source_metrics["rows"]
    )

    source_weight = float(
        source_metrics[
            "total_weight"
        ]
    )

    context.log.info(
        "Integration context source: "
        f"{source_rows:,} samples"
    )

    # =====================================================================
    # Build exogenous context that does not require tract lookup.
    # =====================================================================

    base = (
        build_integration_base_context(
            samples=samples,
            acs_calendar=
                acs_calendar,
            osm=osm,
            land_weather=
                land_weather,
            coastal_weather=
                coastal_weather,
            city_timezones=
                CITY_TIMEZONES,
        )
    )

    context.log.info(
        "Writing integration base-context staging"
    )

    write_delta(
        base,
        path=INTEGRATION_BASE_STAGING_ROOT,
        credentials=credentials,
        context=context,
    )
    persisted_base = scan_delta(
        INTEGRATION_BASE_STAGING_ROOT,
        credentials=credentials,
    )

    base_metrics = (
        validate_base_context(
            persisted_base,
            expected_rows=
                source_rows,
        )
    )

    context.log.info(
        "Base context validated: "
        f"rows={base_metrics['rows']:,}"
    )

    # =====================================================================
    # Build reusable H3-9 center -> TIGER tract mapping.
    # =====================================================================

    context.log.info(
        "Building H3-9 -> census tract mapping"
    )

    tract_mapping_df = (
        build_h3_tract_mapping(
            base=
                persisted_base,
            boundaries=
                tract_boundaries,
        )
    )

    mapping_rows = (
        tract_mapping_df.height
    )

    mapping_matches = int(
        tract_mapping_df
        .get_column(
            "_tract_mapping_matched"
        )
        .sum()
    )

    mapping_rate = (
        mapping_matches
        / mapping_rows
        if mapping_rows
        else 0.0
    )

    context.log.info(
        "H3 tract mapping: "
        f"{mapping_matches:,}/"
        f"{mapping_rows:,} "
        f"({mapping_rate:.6%})"
    )

    tract_mapping_df.lazy().sink_delta(
        INTEGRATION_TRACT_MAPPING_ROOT,
        mode="overwrite",
        credential_provider=
            credentials,
        delta_write_options={
            "schema_mode":
                "overwrite",
        },
    )

    tract_mapping = scan_delta(
        INTEGRATION_TRACT_MAPPING_ROOT,
        credentials=credentials,
    )

    # =====================================================================
    # Add tract + socioeconomic context.
    # =====================================================================
    write_context_partitions(
        base=
            persisted_base,

        tract_mapping=
            tract_mapping,

        socioeconomic=
            socioeconomic,

        path=
            INTEGRATION_CONTEXT_STAGING_ROOT,

        credentials=
            credentials,

        context=
            context,
    )

    staged = scan_delta(
        INTEGRATION_CONTEXT_STAGING_ROOT,
        credentials=credentials,
    )

    # =====================================================================
    # Hard validation
    # =====================================================================

    metrics = (
        validate_final_context(
            staged,
            expected_rows=
                source_rows,
            expected_weight=
                source_weight,
        )
    )

    context.log.info(
        "Integration context validated: "
        f"tract={metrics['tract_match_rate']:.6%}, "
        f"socio={metrics['socio_match_rate']:.6%}, "
        f"weather={metrics['weather_match_rate']:.6%}"
    )

    # =====================================================================
    # Publish
    # =====================================================================

    context.log.info(
        "Publishing integration context"
    )

    write_delta(
        staged,
        path=INTEGRATION_CONTEXT_ROOT,
        credentials=credentials,
        context=context,
    )

    return dg.MaterializeResult(
        metadata={
            "rows":
                source_rows,

            "columns":
                len(
                    staged.collect_schema()
                ),

            "tract_mapping_keys":
                mapping_rows,

            "tract_mapping_rate":
                mapping_rate,

            "tract_match_rate":
                metrics[
                    "tract_match_rate"
                ],

            "socioeconomic_match_rate":
                metrics[
                    "socio_match_rate"
                ],

            "weather_match_rate":
                metrics[
                    "weather_match_rate"
                ],

            "integration_measure_cell_seconds":
                metrics[
                    "total_weight"
                ],

            "output":
                dg.MetadataValue.text(
                    INTEGRATION_CONTEXT_ROOT
                ),
        }
    )