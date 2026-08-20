from datetime import UTC, datetime, timedelta
import random
import time

import dagster as dg
from dagster import (
    AssetExecutionContext,
    MaterializeResult,
)

import polars as pl

from crimenet_data.assets.event_spine import (
    event_spine,
)
from crimenet_data.assets.integration.context_asset import (
    integration_context,
)
from crimenet_data.assets.lighting import (
    solar_lighting_conditions,
)

from .transformations import (
    HISTORY_MAX_SECONDS,
    FINAL_COLUMNS,
    attach_dynamic_history,
    build_query_rows,
    finalize_model_table,
    join_lighting,
    prepare_history_events,
    validate_final_partition,
)


# =============================================================================
# Storage
# =============================================================================


EVENT_SPINE_ROOT = (
    "gs://crimenet/gold/event_spine"
)

INTEGRATION_CONTEXT_ROOT = (
    "gs://crimenet/gold/"
    "integration_context"
)

LIGHTING_ROOT = (
    "gs://crimenet/silver/"
    "solar_lighting_conditions"
)

FINAL_STAGING_ROOT = (
    "gs://crimenet/gold_staging_/"
    "model_table_nyc_timestamp_fix"
)

FINAL_ROOT = (
    "gs://crimenet/gold/"
    "model_table"
)


PARTITION_COLUMNS = [
    "split",
    "source_city",
    "row_year",
]


# =============================================================================
# I/O
# =============================================================================


def scan_delta(
    path: str,
    *,
    credentials:
        pl.CredentialProviderGCP,
) -> pl.LazyFrame:
    return pl.scan_delta(
        path,
        credential_provider=
            credentials,
    )


def sink_with_retry(
    frame: pl.LazyFrame,
    *,
    path: str,
    credentials:
        pl.CredentialProviderGCP,
    context:
        AssetExecutionContext,
    predicate: str | None = None,
    first_write: bool = False,
    max_attempts: int = 3,
) -> None:
    for attempt in range(
        1,
        max_attempts + 1,
    ):
        try:
            options = {
                "target_file_size":
                    64
                    * 1024
                    * 1024,
            }

            if first_write:
                options.update(
                    {
                        "partition_by":
                            PARTITION_COLUMNS,

                        "schema_mode":
                            "overwrite",
                    }
                )

            elif predicate:
                options[
                    "predicate"
                ] = predicate

            frame.sink_delta(
                path,
                mode="overwrite",

                credential_provider=
                    credentials,

                delta_write_options=
                    options,
            )

            return

        except Exception as exc:
            if (
                attempt
                == max_attempts
            ):
                context.log.error(
                    "Delta write failed "
                    f"after {max_attempts} "
                    f"attempts: {path}"
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

            delay += (
                random.uniform(
                    0,
                    2,
                )
            )

            context.log.warning(
                f"Delta write failed: "
                f"{path}. "
                f"Attempt "
                f"{attempt}/"
                f"{max_attempts}. "
                f"Retrying in "
                f"{delay:.1f}s. "
                f"Error: {exc!r}"
            )

            time.sleep(
                delay
            )

# =============================================================================
# Asset
# =============================================================================

def get_staging_state(
    *,
    credentials: pl.CredentialProviderGCP,
    context: AssetExecutionContext,
) -> tuple[
    bool,
    set[tuple[str, str, int]],
]:
    try:
        staged = scan_delta(
            FINAL_STAGING_ROOT,
            credentials=credentials,
        )

        completed_df = (
            staged
            .select(
                "split",
                "source_city",
                "row_year",
            )
            .unique()
            .collect()
        )

        completed = {
            (
                row["split"],
                row["source_city"],
                int(row["row_year"]),
            )
            for row in completed_df.iter_rows(
                named=True
            )
        }

        context.log.info(
            "Existing staging table found: "
            f"{len(completed)} completed partitions"
        )

        return True, completed

    except Exception as exc:
        context.log.info(
            "No usable staging table found; "
            "starting from scratch. "
            f"Reason: {exc!r}"
        )

        return False, set()

@dg.asset(
    name="final_model_table",
    group_name="model",
    compute_kind="polars",

    deps=[
        event_spine,
        integration_context,
        solar_lighting_conditions,
    ],
)
def final_model_table(
    context:
        AssetExecutionContext,
) -> MaterializeResult:
    credentials = (
        pl.CredentialProviderGCP()
    )

    # -------------------------------------------------------------------------
    # Inputs
    # -------------------------------------------------------------------------

    events = scan_delta(
        EVENT_SPINE_ROOT,
        credentials=credentials,
    )

    integration = scan_delta(
        INTEGRATION_CONTEXT_ROOT,
        credentials=credentials,
    )

    lighting = scan_delta(
        LIGHTING_ROOT,
        credentials=credentials,
    )
    # -------------------------------------------------------------------------
    # Common observed + Monte Carlo query universe.
    # -------------------------------------------------------------------------

    all_queries = (
        build_query_rows(
            events=events,
            integration=integration,
        )
    )

    query_rows = int(
        all_queries
        .select(
            pl.len()
        )
        .collect()
        .item()
    )

    # -------------------------------------------------------------------------
    # Partition manifest.
    # -------------------------------------------------------------------------

    manifest = (
        all_queries
        .select(
            *PARTITION_COLUMNS
        )
        .unique()
        .sort(
            PARTITION_COLUMNS
        )
        .collect()
    )

    partition_count = (
        manifest.height
    )

    context.log.info(
        "Building final model table: "
        f"{query_rows:,} rows across "
        f"{partition_count} partitions"
    )

    history = (
        prepare_history_events(
            events
        )
    )

    staging_exists, completed_partitions = (
        get_staging_state(
            credentials=credentials,
            context=context,
        )
    )

    first_write = not staging_exists
    # -------------------------------------------------------------------------
    # Process independently by split/city/year.
    # -------------------------------------------------------------------------

    for (
        index,
        partition,
    ) in enumerate(
        manifest.iter_rows(
            named=True
        ),
        start=1,
    ):
        split = (
            partition["split"]
        )

        city = (
            partition[
                "source_city"
            ]
        )

        year = int(
            partition[
                "row_year"
            ]
        )
        partition_key = (
            split,
            city,
            year,
        )

        if partition_key in completed_partitions:
            context.log.info(
                f"[{index}/{partition_count}] "
                f"Skipping completed "
                f"{split} / {city} / {year}"
            )
            continue
        context.log.info(
            f"[{index}/{partition_count}] "
            f"Building "
            f"{split} / "
            f"{city} / "
            f"{year}"
        )

        year_start = datetime(
            year,
            1,
            1,
            tzinfo=UTC,
        )

        year_end = datetime(
            year + 1,
            1,
            1,
            tzinfo=UTC,
        )

        history_start = (
            year_start
            -
            timedelta(
                seconds=
                    HISTORY_MAX_SECONDS
            )
        )

        # -------------------------------------------------------------
        # Query partition.
        # -------------------------------------------------------------

        query_partition = (
            all_queries
            .filter(
                (
                    pl.col(
                        "source_city"
                    )
                    == city
                )
                &
                (
                    pl.col(
                        "row_year"
                    )
                    == year
                )
                &
                (
                    pl.col(
                        "split"
                    )
                    == split
                )
            )
        )

        expected_rows = int(
            query_partition
            .select(
                pl.len()
            )
            .collect()
            .item()
        )

        if expected_rows == 0:
            continue


        history_partition = (
            history
            .filter(
                (pl.col("source_city") == city)
                &
                (
                    pl.col("occurrence_timestamp_utc")
                    >= pl.lit(history_start)
                )
                &
                (
                    pl.col("occurrence_timestamp_utc")
                    < pl.lit(year_end)
                )
            )
        )

        # -------------------------------------------------------------
        # Lighting.
        # -------------------------------------------------------------

        lighting_partition = (
            lighting
            .filter(
                pl.col(
                    "solar_year"
                )
                == year
            )
        )

        enriched = join_lighting(
            query_partition,
            lighting_partition,
        )

        # -------------------------------------------------------------
        # Occurrence-causal crime state
        # -------------------------------------------------------------

        enriched = (
            attach_dynamic_history(
                rows=enriched,
                history_events=
                    history_partition,
            )
        )

        # -------------------------------------------------------------
        # Calendar.
        # -------------------------------------------------------------

        from .transformations import (
            add_calendar_features,
        )

        enriched = (
            add_calendar_features(
                enriched
            )
        )

        # -------------------------------------------------------------
        # Hard leakage validation BEFORE dropping audit columns.
        # -------------------------------------------------------------

        metrics = (
            validate_final_partition(
                enriched,
                expected_rows=
                    expected_rows,
            )
        )

        context.log.info(
            f"[{index}/{partition_count}] "
            f"Validated "
            f"{split} / "
            f"{city} / "
            f"{year}: "
            f"{metrics['rows']:,} rows"
        )

        final_partition = (
            finalize_model_table(
                enriched
            )
        )

        predicate = (
            f"split = '{split}' "
            f"AND "
            f"source_city = '{city}' "
            f"AND "
            f"row_year = {year}"
        )

        sink_with_retry(
            final_partition,

            path=
                FINAL_STAGING_ROOT,

            credentials=
                credentials,

            context=
                context,

            predicate=
                predicate,

            first_write=
                first_write,
        )

        first_write = False

    # -------------------------------------------------------------------------
    # Global validation.
    # -------------------------------------------------------------------------

    staged = scan_delta(
        FINAL_STAGING_ROOT,
        credentials=credentials,
    )

    staged_stats = (
        staged
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
                "unique_rows"
            ),

            pl.col(
                "is_observed_event"
            )
            .sum()
            .cast(pl.Int64)
            .alias(
                "observed_rows"
            ),

            (
                ~pl.col(
                    "is_observed_event"
                )
            )
            .sum()
            .cast(pl.Int64)
            .alias(
                "integration_rows"
            ),

            pl.col(
                "integration_weight_cell_seconds"
            )
            .sum()
            .alias(
                "total_integration_weight"
            ),
        )
        .collect()
        .row(
            0,
            named=True,
        )
    )

    if (
        staged_stats["rows"]
        != query_rows
    ):
        raise ValueError(
            "Final staged row count "
            "does not match query universe: "
            f"staged="
            f"{staged_stats['rows']:,}, "
            f"queries={query_rows:,}"
        )

    if (
        staged_stats["unique_rows"]
        != staged_stats["rows"]
    ):
        raise ValueError(
            "Final model row IDs "
            "are not unique."
        )

    # -------------------------------------------------------------------------
    # Verify exact chronological split boundaries.
    # -------------------------------------------------------------------------

    split_checks = (
        staged
        .select(
            (
                (
                    pl.col("split")
                    == "train"
                )
                &
                (
                    pl.col(
                        "row_timestamp_utc"
                    )
                    >=
                    pl.lit(
                        datetime(
                            2024,
                            1,
                            1,
                            tzinfo=UTC,
                        )
                    )
                )
            )
            .sum()
            .alias(
                "train_future_rows"
            ),

            (
                (
                    pl.col("split")
                    == "validation"
                )
                &
                (
                    (
                        pl.col(
                            "row_timestamp_utc"
                        )
                        <
                        pl.lit(
                            datetime(
                                2024,
                                1,
                                1,
                                tzinfo=UTC,
                            )
                        )
                    )
                    |
                    (
                        pl.col(
                            "row_timestamp_utc"
                        )
                        >=
                        pl.lit(
                            datetime(
                                2025,
                                1,
                                1,
                                tzinfo=UTC,
                            )
                        )
                    )
                )
            )
            .sum()
            .alias(
                "validation_boundary_rows"
            ),

            (
                (
                    pl.col("split")
                    == "test"
                )
                &
                (
                    pl.col(
                        "row_timestamp_utc"
                    )
                    <
                    pl.lit(
                        datetime(
                            2025,
                            1,
                            1,
                            tzinfo=UTC,
                        )
                    )
                )
            )
            .sum()
            .alias(
                "test_past_rows"
            ),
        )
        .collect()
        .row(
            0,
            named=True,
        )
    )

    if any(
        split_checks.values()
    ):
        raise ValueError(
            "Strict chronological split "
            f"validation failed: "
            f"{split_checks}"
        )

    # -------------------------------------------------------------------------
    # Publish final curated table.
    #
    # Use the same retry strategy that just successfully
    # published integration_context.
    # -------------------------------------------------------------------------

    context.log.info(
        "Publishing final model table"
    )

    # Whole-table overwrite is now considerably narrower
    # than the 238-column context table.
    for attempt in range(
        1,
        4,
    ):
        try:
            staged.sink_delta(
                FINAL_ROOT,

                mode="overwrite",

                credential_provider=
                    credentials,

                delta_write_options={
                    "partition_by":
                        PARTITION_COLUMNS,

                    "schema_mode":
                        "overwrite",

                    "target_file_size":
                        64
                        * 1024
                        * 1024,
                },
            )

            break

        except Exception:
            if attempt == 3:
                raise

            delay = (
                5.0
                * (
                    2
                    ** (
                        attempt - 1
                    )
                )
                +
                random.uniform(
                    0,
                    2,
                )
            )

            context.log.warning(
                "Final publish failed. "
                f"Attempt {attempt}/3. "
                f"Retrying in "
                f"{delay:.1f}s."
            )

            time.sleep(
                delay
            )

    context.log.info(
        "Final model table published: "
        f"rows={staged_stats['rows']:,}, "
        f"observed="
        f"{staged_stats['observed_rows']:,}, "
        f"integration="
        f"{staged_stats['integration_rows']:,}, "
        f"columns={len(FINAL_COLUMNS)}"
    )

    return MaterializeResult(
        metadata={
            "rows": staged_stats["rows"],
            "observed_rows": staged_stats["observed_rows"],
            "integration_rows": staged_stats["integration_rows"],
            "columns": len(FINAL_COLUMNS),
            "history_source_mode": "strict_occurrence",
            "output": dg.MetadataValue.text(FINAL_ROOT),
        }
    )

