from datetime import UTC, datetime

import dagster as dg
import polars as pl
import pvlib

from crimenet_data.assets.event_spine import (
    event_spine,
)
from crimenet_data.assets.integration.asset import (
    integration_samples,
)
from crimenet_data.observability.context import (
    log_context,
)
from crimenet_data.observability.logger import (
    get_logger,
)

from .transformations import (
    LIGHTING_DEFINITION_VERSION,
    build_required_lighting_keys,
    compute_lighting_conditions,
    validate_lighting_results,
    validate_required_lighting_keys,
)


log = get_logger(__name__)


# =============================================================================
# Storage
# =============================================================================


EVENT_SPINE_ROOT = (
    "gs://crimenet/gold/event_spine"
)

INTEGRATION_ROOT = (
    "gs://crimenet/gold/integration_samples"
)

LIGHTING_KEYS_ROOT = (
    "gs://crimenet/gold_staging/"
    "lighting_required_keys"
)

LIGHTING_ROOT = (
    "gs://crimenet/silver/"
    "solar_lighting_conditions"
)


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


def write_required_keys(
    frame: pl.LazyFrame,
    *,
    credentials:
        pl.CredentialProviderGCP,
) -> None:
    frame.sink_delta(
        LIGHTING_KEYS_ROOT,
        mode="overwrite",
        credential_provider=credentials,
        delta_write_options={
            "partition_by": [
                "solar_year",
            ],
            "schema_mode":
                "overwrite",
        },
    )


def write_lighting(
    frame: pl.LazyFrame,
    *,
    credentials:
        pl.CredentialProviderGCP,
) -> None:
    frame.sink_delta(
        LIGHTING_ROOT,
        mode="overwrite",
        credential_provider=credentials,
        delta_write_options={
            "partition_by": [
                "solar_year",
            ],
            "schema_mode":
                "overwrite",
        },
    )


# =============================================================================
# Required lighting keys
# =============================================================================


@dg.asset(
    name="lighting_required_keys",
    group_name="lighting",
    compute_kind="polars",
    deps=[
        event_spine,
        integration_samples,
    ],
)
def lighting_required_keys(
    context: dg.AssetExecutionContext,
) -> dg.MaterializeResult:
    with log_context(
        run_id=context.run_id,
        asset_key=
            context.asset_key.to_user_string(),
    ):
        credentials = (
            pl.CredentialProviderGCP()
        )

        events = scan_delta(
            EVENT_SPINE_ROOT,
            credentials=credentials,
        )

        integration = scan_delta(
            INTEGRATION_ROOT,
            credentials=credentials,
        )

        required_keys = (
            build_required_lighting_keys(
                events=events,
                integration=integration,
            )
        )

        log.info(
            "lighting_key_generation_started",
            target_uri=
                LIGHTING_KEYS_ROOT,
        )

        write_required_keys(
            required_keys,
            credentials=credentials,
        )

        published = scan_delta(
            LIGHTING_KEYS_ROOT,
            credentials=credentials,
        )

        metrics = (
            validate_required_lighting_keys(
                published
            )
        )

        log.info(
            "lighting_key_generation_completed",
            rows=
                metrics["rows"],
            cells=
                metrics["cells"],
            target_uri=
                LIGHTING_KEYS_ROOT,
        )

        return dg.MaterializeResult(
            metadata={
                "rows":
                    metrics["rows"],

                "h3_cells":
                    metrics["cells"],

                "min_timestamp":
                    str(
                        metrics[
                            "min_timestamp"
                        ]
                    ),

                "max_timestamp":
                    str(
                        metrics[
                            "max_timestamp"
                        ]
                    ),

                "target_uri":
                    LIGHTING_KEYS_ROOT,
            }
        )


# =============================================================================
# Solar lighting
# =============================================================================


@dg.asset(
    name="solar_lighting_conditions",
    group_name="lighting",
    compute_kind="pvlib",
    deps=[
        lighting_required_keys,
    ],
)
def solar_lighting_conditions(
    context: dg.AssetExecutionContext,
) -> dg.MaterializeResult:
    with log_context(
        run_id=context.run_id,
        asset_key=
            context.asset_key.to_user_string(),
    ):
        credentials = (
            pl.CredentialProviderGCP()
        )

        required_keys = scan_delta(
            LIGHTING_KEYS_ROOT,
            credentials=credentials,
        )

        expected_rows = int(
            required_keys
            .select(
                pl.len()
            )
            .collect()
            .item()
        )

        log.info(
            "solar_lighting_started",
            required_cell_hours=
                expected_rows,
            pvlib_version=
                pvlib.__version__,
            definition_version=
                LIGHTING_DEFINITION_VERSION,
        )

        computed_at = datetime.now(
            UTC
        )

        lighting = (
            compute_lighting_conditions(
                required_keys
            )
            .with_columns(
                pl.lit(
                    computed_at
                )
                .alias(
                    "computed_at_utc"
                )
            )
        )

        write_lighting(
            lighting,
            credentials=credentials,
        )

        published = scan_delta(
            LIGHTING_ROOT,
            credentials=credentials,
        )

        metrics = (
            validate_lighting_results(
                published,
                expected_rows=
                    expected_rows,
            )
        )

        condition_counts = (
            published
            .group_by(
                "lighting_condition"
            )
            .agg(
                pl.len()
                .alias("rows")
            )
            .sort(
                "lighting_condition"
            )
            .collect()
        )

        log.info(
            "lighting_condition_distribution",
            distribution=
                condition_counts.to_dicts(),
        )

        log.info(
            "solar_lighting_completed",
            rows=
                metrics["rows"],
            target_uri=
                LIGHTING_ROOT,
        )

        return dg.MaterializeResult(
            metadata={
                "rows":
                    metrics["rows"],

                "unique_keys":
                    metrics[
                        "unique_keys"
                    ],

                "pvlib_version":
                    pvlib.__version__,

                "lighting_definition_version":
                    LIGHTING_DEFINITION_VERSION,

                "source_uri":
                    LIGHTING_KEYS_ROOT,

                "target_uri":
                    LIGHTING_ROOT,
            }
        )


lighting_assets = [
    lighting_required_keys,
    solar_lighting_conditions,
]