from collections.abc import Mapping
from contextlib import ExitStack

import dagster as dg
import polars as pl
from dagster import AssetExecutionContext

from crimenet_data.assets.crime.canonical import (
    CANONICAL_MAPPING_VERSION,
    apply_canonical_crosswalk,
    cleanse_canonical_source,
    project_canonical_schema,
    validate_canonical_crosswalk,
)
from crimenet_data.assets.crime.normalization import (
    normalization_requires_duckdb,
    normalize_source,
)
from crimenet_data.assets.crime.sources import (
    SILVER_SOURCE_KEYS,
    AdapterContext,
    get_source,
)
from crimenet_data.observability.context import log_context
from crimenet_data.observability.logger import get_logger
from crimenet_data.resources.crime_lake import CrimeLakeResources
from crimenet_data.resources.duckdb import DuckDBResource

log = get_logger(__name__)
CROSSWALK_ASSET_KEY = dg.AssetKey(["reference", "canonical_crime_crosswalk"])
KNOWN_STALE_MONTGOMERY_BRONZE_ROWS = 1_515_753


def deduplicate_source(lf: pl.LazyFrame, source_key: str) -> pl.LazyFrame:
    """Apply the source contract's record identity before crosswalk expansion."""

    keys = get_source(source_key).config.deduplication_keys
    if not keys:
        return lf
    missing = set(keys) - set(lf.collect_schema().names())
    if missing:
        raise KeyError(
            f"Cannot deduplicate {source_key!r}; keys are missing: {sorted(missing)}"
        )
    return lf.unique(subset=list(keys), keep="last", maintain_order=False)


def build_silver(
    bronze_lf: pl.LazyFrame,
    crosswalk_lf: pl.LazyFrame,
    *,
    source_key: str,
    adapter_context: AdapterContext,
) -> pl.LazyFrame:
    """Build one source's canonical rows without persisting an intermediate table."""

    source = get_source(source_key)
    normalized = normalize_source(
        bronze_lf,
        source_key,
        connection=adapter_context.duckdb,
    )
    deduplicated = deduplicate_source(normalized, source_key)
    adapted = source.adapt_to_silver(deduplicated, adapter_context)
    cleansed = cleanse_canonical_source(adapted, source_key)
    mapped = apply_canonical_crosswalk(cleansed, crosswalk_lf, source_key)
    return project_canonical_schema(mapped, source_key)


def build_unified_silver(
    bronze_frames: Mapping[str, pl.LazyFrame],
    crosswalk_lf: pl.LazyFrame,
    *,
    adapter_contexts: Mapping[str, AdapterContext] | None = None,
) -> pl.LazyFrame:
    """Union approved source frames under the exact canonical contract."""

    if not bronze_frames:
        raise ValueError("At least one Silver source frame is required")
    unsupported = set(bronze_frames) - set(SILVER_SOURCE_KEYS)
    if unsupported:
        raise ValueError(f"Sources are not Silver-enabled: {sorted(unsupported)}")
    contexts = adapter_contexts or {}
    frames = [
        build_silver(
            bronze_frames[source_key],
            crosswalk_lf,
            source_key=source_key,
            adapter_context=contexts.get(source_key, AdapterContext()),
        )
        for source_key in SILVER_SOURCE_KEYS
        if source_key in bronze_frames
    ]
    return pl.concat(frames, how="vertical")


def silver_mapping_summary(lf: pl.LazyFrame, source_key: str) -> pl.LazyFrame:
    """Produce one audit row without materializing source incidents."""

    mapped = pl.col("canonical_mapping_found").fill_null(False)
    included = pl.col("include_in_model").fill_null(False)
    review = pl.col("review_required").fill_null(False)
    taxonomy_is_present = pl.any_horizontal(
        *(
            pl.col(key).is_not_null()
            for key in get_source(source_key).config.crosswalk_keys
        )
    )
    return lf.select(
        pl.lit(source_key).alias("source_city"),
        pl.len().alias("output_rows"),
        mapped.sum().alias("mapped_rows"),
        (~mapped).sum().alias("unmapped_rows"),
        ((~mapped) & taxonomy_is_present)
        .sum()
        .alias("unexpected_unmapped_rows"),
        included.sum().alias("include_in_model_rows"),
        (pl.col("mapping_action") == "drop")
        .fill_null(False)
        .sum()
        .alias("drop_rows"),
        (pl.col("mapping_action") == "exclude_non_criminal")
        .fill_null(False)
        .sum()
        .alias("excluded_rows"),
        review.sum().alias("review_required_rows"),
    )


def validate_montgomery_bronze_row_count(row_count: int) -> None:
    """Block the one known corrupt physical-line Montgomery snapshot."""

    if row_count == KNOWN_STALE_MONTGOMERY_BRONZE_ROWS:
        raise RuntimeError(
            "Montgomery Bronze is the known stale 1,515,753-row snapshot. "
            "Rematerialize Bronze with logical CSV record parsing before Silver."
        )


@dg.asset(
    name="silver_crime_offenses",
    group_name="silver_crime",
    deps=[
        CROSSWALK_ASSET_KEY,
        *[f"bronze_{source_key}" for source_key in SILVER_SOURCE_KEYS],
    ],
    pool="crime_silver_offenses_writer",
)
def silver_crime_offenses(
    context: AssetExecutionContext,
    crime_lake: CrimeLakeResources,
    duckdb_resource: DuckDBResource,
) -> dg.MaterializeResult:
    with log_context(
        run_id=context.run_id,
        asset_key=context.asset_key.to_user_string(),
    ):
        crosswalk = validate_canonical_crosswalk(crime_lake.resolve_crosswalk())
        crosswalk_lf = crosswalk.lazy()

        bronze_frames: dict[str, pl.LazyFrame] = {}
        snapshot_uris: dict[str, str] = {}
        for source_key in SILVER_SOURCE_KEYS:
            snapshot_uri = crime_lake.resolve_current_bronze_snapshot(source_key)
            snapshot_uris[source_key] = snapshot_uri
            bronze_frames[source_key] = crime_lake.scan_bronze_snapshot(
                source_key,
                snapshot_uri=snapshot_uri,
            )

        montgomery_rows = (
            bronze_frames["montgomery_county_md"]
            .select(pl.len().alias("rows"))
            .collect()
            .item()
        )
        validate_montgomery_bronze_row_count(montgomery_rows)

        source_frames: dict[str, pl.LazyFrame] = {}
        with ExitStack() as stack:
            for source_key in SILVER_SOURCE_KEYS:
                source = get_source(source_key)
                log.info(
                    "silver_source_processing_started",
                    source_city=source_key,
                    bronze_snapshot_uri=snapshot_uris[source_key],
                    crosswalk_keys=list(source.config.crosswalk_keys),
                    mapping_version=CANONICAL_MAPPING_VERSION,
                )
                adapter_context = AdapterContext()
                if normalization_requires_duckdb(source_key):
                    connection = stack.enter_context(
                        duckdb_resource.get_connection()
                    )
                    adapter_context = AdapterContext(duckdb=connection)
                source_frames[source_key] = build_silver(
                    bronze_frames[source_key],
                    crosswalk_lf,
                    source_key=source_key,
                    adapter_context=adapter_context,
                )

            summaries = pl.collect_all(
                [
                    silver_mapping_summary(source_frames[source_key], source_key)
                    for source_key in SILVER_SOURCE_KEYS
                ]
            )
            summary_rows = [summary.row(0, named=True) for summary in summaries]
            for summary in summary_rows:
                source_key = str(summary["source_city"])
                log.info(
                    "silver_source_mapping_summary",
                    bronze_snapshot_uri=snapshot_uris[source_key],
                    crosswalk_keys=list(
                        get_source(source_key).config.crosswalk_keys
                    ),
                    mapping_version=CANONICAL_MAPPING_VERSION,
                    **summary,
                )
                if summary["unmapped_rows"]:
                    log.error(
                        "silver_source_unmapped_rows",
                        bronze_snapshot_uri=snapshot_uris[source_key],
                        crosswalk_keys=list(
                            get_source(source_key).config.crosswalk_keys
                        ),
                        mapping_version=CANONICAL_MAPPING_VERSION,
                        **summary,
                    )
                log.info(
                    "silver_source_processing_completed",
                    bronze_snapshot_uri=snapshot_uris[source_key],
                    mapping_version=CANONICAL_MAPPING_VERSION,
                    **summary,
                )

            unexpected = [
                summary
                for summary in summary_rows
                if summary["unexpected_unmapped_rows"]
            ]
            if unexpected:
                raise RuntimeError(
                    "Silver overwrite blocked by populated taxonomy values missing "
                    f"from canonical crosswalk v1.4: {unexpected}"
                )

            silver_lf = pl.concat(
                [source_frames[source_key] for source_key in SILVER_SOURCE_KEYS],
                how="vertical",
            )
            target_uri = crime_lake.silver_crime_offenses_uri
            log.info(
                "silver_unified_write_started",
                target_uri=target_uri,
                source_count=len(SILVER_SOURCE_KEYS),
                mapping_version=CANONICAL_MAPPING_VERSION,
                partitioning_columns=["source_city", "occurrence_year"],
            )
            crime_lake.write_delta_table(
                lf=silver_lf,
                target_uri=target_uri,
                partitioning_columns=["source_city", "occurrence_year"],
            )
            log.info(
                "silver_unified_write_completed",
                target_uri=target_uri,
                source_count=len(SILVER_SOURCE_KEYS),
                mapping_version=CANONICAL_MAPPING_VERSION,
            )

        return dg.MaterializeResult(
            metadata={
                "target_uri": target_uri,
                "sources_processed": len(SILVER_SOURCE_KEYS),
                "mapping_version": CANONICAL_MAPPING_VERSION,
                "partitioning_columns": ["source_city", "occurrence_year"],
                "montgomery_bronze_rows": montgomery_rows,
                "event_grain": "source offense/event row",
            }
        )


crime_silver_assets = [silver_crime_offenses]
