import os
from collections.abc import Mapping
from contextlib import ExitStack
from datetime import UTC, datetime
from uuid import uuid4

import dagster as dg
import polars as pl
from dagster import AssetExecutionContext

from crimenet_data.assets.crime.canonical import (
    CANONICAL_CRIME_SCHEMA,
    CANONICAL_MAPPING_VERSION,
    apply_canonical_crosswalk,
    cleanse_canonical_source,
    project_canonical_schema,
    validate_canonical_crosswalk,
)
from crimenet_data.assets.crime.common.source_bounds import (
    SOURCE_COORDINATE_BOUNDS,
    SOURCE_COORDINATE_BOUNDS_VALID_COLUMN,
    apply_source_coordinate_bounds,
    globally_valid_coordinate_expr,
    source_coordinate_bounds_summary,
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
SILVER_SCHEMA_VERSION = "crime_silver_v1"
DUPLICATE_SEMANTIC_COLUMNS = (
    "occurrence_timestamp",
    "report_timestamp",
    "source_offense_code",
    "source_offense_category",
    "source_offense_description",
    "source_auxiliary",
    "source_severity",
    "latitude",
    "longitude",
    "location_label",
    "location_type",
    "police_district",
    "local_area",
)


def silver_duplicate_id_examples(
    lf: pl.LazyFrame,
    *,
    limit: int = 25,
) -> pl.LazyFrame:
    duplicate_ids = (
        lf.group_by("crime_id")
        .agg(pl.len().alias("_rows"))
        .filter(pl.col("_rows") > 1)
        .select("crime_id")
    )

    return (
        lf.join(
            duplicate_ids,
            on="crime_id",
            how="inner",
        )
        .select(
            [
                "source_city",
                "crime_id",
                "source_record_id",
                "occurrence_timestamp",
                "report_timestamp",
                "source_offense_code",
                "source_offense_category",
                "source_offense_description",
                "source_auxiliary",
                "source_severity",
                "latitude",
                "longitude",
                "source_file_uri",
            ]
        )
        .sort(
            [
                "crime_id",
                "occurrence_timestamp",
            ]
        )
        .head(limit)
    )


def silver_duplicate_id_summary(
    lf: pl.LazyFrame,
    source_key: str,
) -> pl.LazyFrame:
    duplicate_groups = (
        lf.group_by(
            [
                "crime_id",
                "source_record_id",
            ]
        )
        .agg(
            pl.len().alias("row_count"),
            pl.struct(list(DUPLICATE_SEMANTIC_COLUMNS))
            .n_unique()
            .alias("semantic_variants"),
        )
        .filter(pl.col("row_count") > 1)
    )

    return duplicate_groups.select(
        pl.lit(source_key).alias("source_city"),
        pl.len().alias("duplicate_ids"),
        (pl.col("row_count").sum() - pl.len()).alias("duplicate_surplus_rows"),
        ((pl.col("semantic_variants") == 1).sum()).alias("exact_duplicate_ids"),
        (
            pl.when(pl.col("semantic_variants") == 1)
            .then(pl.col("row_count") - 1)
            .otherwise(0)
            .sum()
        ).alias("exact_duplicate_surplus_rows"),
        ((pl.col("semantic_variants") > 1).sum()).alias("identity_collision_ids"),
        (
            pl.when(pl.col("semantic_variants") > 1)
            .then(pl.col("row_count") - 1)
            .otherwise(0)
            .sum()
        ).alias("identity_collision_surplus_rows"),
    )


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

    adapted = adapt_silver_source(
        bronze_lf,
        source_key=source_key,
        adapter_context=adapter_context,
    )
    return map_silver_source(adapted, crosswalk_lf, source_key=source_key)


def adapt_silver_source(
    bronze_lf: pl.LazyFrame,
    *,
    source_key: str,
    adapter_context: AdapterContext,
) -> pl.LazyFrame:
    """Normalize, deduplicate, and adapt a Bronze source before quality filtering."""

    source = get_source(source_key)
    normalized = normalize_source(
        bronze_lf,
        source_key,
        connection=adapter_context.duckdb,
    )
    deduplicated = deduplicate_source(normalized, source_key)
    return source.adapt_to_silver(deduplicated, adapter_context)


def map_silver_source(
    adapted: pl.LazyFrame,
    crosswalk_lf: pl.LazyFrame,
    *,
    source_key: str,
) -> pl.LazyFrame:
    """Apply generic validity filtering, v1.4 mapping, and canonical projection."""

    cleansed = cleanse_canonical_source(adapted, source_key)
    mapped = apply_canonical_crosswalk(cleansed, crosswalk_lf, source_key)
    if source_key in SOURCE_COORDINATE_BOUNDS:
        mapped = apply_source_coordinate_bounds(mapped, source_key)
    else:
        # Registered archival/non-modeled adapters remain testable, but cannot
        # become model eligible without first joining SILVER_SOURCE_KEYS and the
        # exact bounds registry (which is validated at registry import time).
        mapped = mapped.with_columns(
            pl.lit(None, dtype=pl.Boolean).alias(SOURCE_COORDINATE_BOUNDS_VALID_COLUMN),
            pl.lit(False).alias("include_in_model"),
        )
    return project_canonical_schema(mapped, source_key)


def zero_zero_coordinate_count(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.select(
        ((pl.col("latitude") == 0.0) & (pl.col("longitude") == 0.0))
        .fill_null(False)
        .sum()
        .alias("zero_zero_coordinate_rows")
    )


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
    bounds_valid = pl.col(SOURCE_COORDINATE_BOUNDS_VALID_COLUMN).fill_null(False)
    globally_valid_coordinates = globally_valid_coordinate_expr()
    review = pl.col("review_required").fill_null(False)
    taxonomy_is_present = pl.any_horizontal(
        *(
            pl.col(key)
            .cast(pl.String, strict=False)
            .fill_null("")
            .str.strip_chars()
            .ne("")
            for key in get_source(source_key).config.crosswalk_keys
        )
    )
    return lf.select(
        pl.lit(source_key).alias("source_city"),
        pl.len().alias("output_rows"),
        mapped.sum().alias("mapped_rows"),
        (~mapped).sum().alias("unmapped_rows"),
        ((~mapped) & taxonomy_is_present).sum().alias("unexpected_unmapped_rows"),
        included.sum().alias("include_in_model_rows"),
        (globally_valid_coordinates & bounds_valid)
        .sum()
        .alias("inside_source_bounds_rows"),
        (globally_valid_coordinates & ~bounds_valid)
        .sum()
        .alias("outside_source_bounds_rows"),
        (pl.col("mapping_action") == "drop").fill_null(False).sum().alias("drop_rows"),
        (pl.col("mapping_action") == "exclude_non_criminal")
        .fill_null(False)
        .sum()
        .alias("excluded_rows"),
        review.sum().alias("review_required_rows"),
    )


def silver_unmapped_key_counts(
    lf: pl.LazyFrame,
    source_key: str,
    *,
    limit: int = 25,
) -> pl.LazyFrame:
    """Return the most frequent populated source keys missing from the crosswalk."""

    keys = list(get_source(source_key).config.crosswalk_keys)
    taxonomy_is_present = pl.any_horizontal(
        *(
            pl.col(key)
            .cast(pl.String, strict=False)
            .fill_null("")
            .str.strip_chars()
            .ne("")
            for key in keys
        )
    )
    return (
        lf.filter(
            ~pl.col("canonical_mapping_found").fill_null(False) & taxonomy_is_present
        )
        .group_by(keys)
        .agg(pl.len().alias("row_count"))
        .sort("row_count", descending=True)
        .head(limit)
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
        snapshot_id = str(uuid4())
        created_at_utc = datetime.now(UTC)
        crosswalk = validate_canonical_crosswalk(crime_lake.resolve_crosswalk())
        crosswalk_lf = crosswalk.lazy()
        crosswalk_sha256 = crime_lake.canonical_crosswalk_sha256()

        bronze_frames: dict[str, pl.LazyFrame] = {}
        snapshot_uris: dict[str, str] = {}
        for source_key in SILVER_SOURCE_KEYS:
            snapshot_uri = crime_lake.resolve_current_bronze_snapshot(source_key)
            snapshot_uris[source_key] = snapshot_uri
            bronze_frames[source_key] = crime_lake.scan_bronze_snapshot(
                source_key,
                snapshot_uri=snapshot_uri,
            )

        bronze_counts = pl.collect_all(
            [
                bronze_frames[source_key].select(pl.len().alias("rows"))
                for source_key in SILVER_SOURCE_KEYS
            ]
        )
        input_rows = {
            source_key: int(count.item())
            for source_key, count in zip(
                SILVER_SOURCE_KEYS,
                bronze_counts,
                strict=True,
            )
        }
        montgomery_rows = input_rows["montgomery_county_md"]
        validate_montgomery_bronze_row_count(montgomery_rows)

        adapted_frames: dict[str, pl.LazyFrame] = {}
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
                    connection = stack.enter_context(duckdb_resource.get_connection())
                    adapter_context = AdapterContext(duckdb=connection)
                adapted_frames[source_key] = adapt_silver_source(
                    bronze_frames[source_key],
                    source_key=source_key,
                    adapter_context=adapter_context,
                )

            zero_zero_counts = pl.collect_all(
                [
                    zero_zero_coordinate_count(adapted_frames[source_key])
                    for source_key in SILVER_SOURCE_KEYS
                ]
            )
            coordinate_bounds_summaries = pl.collect_all(
                [
                    source_coordinate_bounds_summary(
                        adapted_frames[source_key], source_key
                    )
                    for source_key in SILVER_SOURCE_KEYS
                ]
            )
            coordinate_bounds_rows = [
                summary.row(0, named=True) for summary in coordinate_bounds_summaries
            ]
            for source_key, count, bounds_summary in zip(
                SILVER_SOURCE_KEYS,
                zero_zero_counts,
                coordinate_bounds_rows,
                strict=True,
            ):
                log.info(
                    "silver_source_zero_zero_coordinates",
                    source_city=source_key,
                    bronze_snapshot_uri=snapshot_uris[source_key],
                    zero_zero_coordinate_rows=int(count.item()),
                )
                log.info(
                    "silver_source_coordinate_bounds_summary",
                    bronze_snapshot_uri=snapshot_uris[source_key],
                    **bounds_summary,
                )
                source_frames[source_key] = map_silver_source(
                    adapted_frames[source_key],
                    crosswalk_lf,
                    source_key=source_key,
                )

            bounds_total_rows = sum(
                int(row["input_rows"]) for row in coordinate_bounds_rows
            )
            bounds_inside_rows = sum(
                int(row["inside_source_bounds_rows"]) for row in coordinate_bounds_rows
            )
            bounds_outside_rows = sum(
                int(row["outside_source_bounds_rows"]) for row in coordinate_bounds_rows
            )
            log.info(
                "silver_coordinate_bounds_summary",
                total_rows=bounds_total_rows,
                inside_source_bounds_rows=bounds_inside_rows,
                outside_source_bounds_rows=bounds_outside_rows,
                outside_source_bounds_pct=(
                    100.0 * bounds_outside_rows / bounds_total_rows
                    if bounds_total_rows
                    else 0.0
                ),
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
                summary["input_rows"] = input_rows[source_key]
                log.info(
                    "silver_source_mapping_summary",
                    bronze_snapshot_uri=snapshot_uris[source_key],
                    crosswalk_keys=list(get_source(source_key).config.crosswalk_keys),
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
                unexpected_sources = {
                    str(summary["source_city"]) for summary in unexpected
                }
                unmapped_key_frames = pl.collect_all(
                    [
                        silver_unmapped_key_counts(
                            source_frames[source_key], source_key
                        )
                        for source_key in SILVER_SOURCE_KEYS
                        if source_key in unexpected_sources
                    ]
                )
                for source_key, unmapped_keys in zip(
                    (
                        source_key
                        for source_key in SILVER_SOURCE_KEYS
                        if source_key in unexpected_sources
                    ),
                    unmapped_key_frames,
                    strict=True,
                ):
                    log.error(
                        "silver_source_top_unmapped_keys",
                        source_city=source_key,
                        bronze_snapshot_uri=snapshot_uris[source_key],
                        crosswalk_keys=list(
                            get_source(source_key).config.crosswalk_keys
                        ),
                        top_unmapped_keys=unmapped_keys.to_dicts(),
                        mapping_version=CANONICAL_MAPPING_VERSION,
                    )
                raise RuntimeError(
                    "Silver publication blocked by populated taxonomy values missing "
                    f"from canonical crosswalk v1.4: {unexpected}"
                )

            review_required = [
                summary for summary in summary_rows if summary["review_required_rows"]
            ]
            if review_required:
                raise RuntimeError(
                    "Silver publication blocked by review-required mappings: "
                    f"{review_required}"
                )
            duplicate_summaries = pl.collect_all(
                [
                    silver_duplicate_id_summary(
                        source_frames[source_key],
                        source_key,
                    )
                    for source_key in SILVER_SOURCE_KEYS
                ]
            )

            duplicate_summary_rows = [
                summary.row(0, named=True) for summary in duplicate_summaries
            ]

            duplicate_sources: list[str] = []
            LASD_COMPARE_COLUMNS = [
                "occurrence_timestamp",
                "report_timestamp",
                "source_offense_code",
                "source_offense_category",
                "source_offense_description",
                "source_auxiliary",
                "latitude",
                "longitude",
                "location_label",
                "police_district",
                "local_area",
                "source_file_uri",
            ]

            lasd = source_frames["los_angeles_county_sheriff"]

            collision_ids = (
                lasd.group_by("crime_id")
                .agg(
                    pl.len().alias("rows"),
                    pl.struct(
                        [c for c in LASD_COMPARE_COLUMNS if c != "source_file_uri"]
                    )
                    .n_unique()
                    .alias("semantic_variants"),
                )
                .filter((pl.col("rows") > 1) & (pl.col("semantic_variants") > 1))
                .select("crime_id")
            )

            lasd_collision_examples = (
                lasd.join(
                    collision_ids,
                    on="crime_id",
                    how="inner",
                )
                .select(["crime_id", "source_record_id"] + LASD_COMPARE_COLUMNS)
                .sort(["crime_id", "source_file_uri"])
                .head(50)
                .collect(engine="streaming")
            )

            log.error(
                "lasd_identity_collision_details",
                rows=lasd_collision_examples.to_dicts(),
            )
            for summary in duplicate_summary_rows:
                if summary["duplicate_ids"]:
                    source_key = str(summary["source_city"])
                    duplicate_sources.append(source_key)
                    log.error(
                        "silver_source_duplicate_identity_summary",
                        **summary,
                    )

            for source_key in duplicate_sources:
                duplicate_examples = silver_duplicate_id_examples(
                    source_frames[source_key],
                    limit=25,
                ).collect(engine="streaming")

                log.error(
                    "silver_source_duplicate_identity_examples",
                    source_city=source_key,
                    duplicate_examples=duplicate_examples.to_dicts(),
                )

            silver_lf = pl.concat(
                [source_frames[source_key] for source_key in SILVER_SOURCE_KEYS],
                how="vertical",
            )
            actual_schema = silver_lf.collect_schema()
            if actual_schema != CANONICAL_CRIME_SCHEMA:
                raise RuntimeError(
                    "Silver publication blocked by canonical schema mismatch: "
                    f"expected={CANONICAL_CRIME_SCHEMA}, actual={actual_schema}"
                )
            snapshot_uri = crime_lake.silver_snapshot_uri(snapshot_id)
            log.info(
                "silver_snapshot_write_started",
                snapshot_id=snapshot_id,
                snapshot_uri=snapshot_uri,
                source_count=len(SILVER_SOURCE_KEYS),
                mapping_version=CANONICAL_MAPPING_VERSION,
                partitioning_columns=["source_city", "occurrence_year"],
            )
            manifest = crime_lake.publish_silver_snapshot(
                silver_lf,
                snapshot_id=snapshot_id,
                created_at_utc=created_at_utc,
                mapping_version=CANONICAL_MAPPING_VERSION,
                schema_version=SILVER_SCHEMA_VERSION,
                crosswalk_sha256=crosswalk_sha256,
                source_snapshots=snapshot_uris,
                per_source=summary_rows,
                git_commit_sha=(
                    os.environ.get("GIT_COMMIT_SHA") or os.environ.get("GITHUB_SHA")
                ),
            )
            log.info(
                "silver_snapshot_published",
                snapshot_id=snapshot_id,
                snapshot_uri=snapshot_uri,
                row_count=manifest["row_count"],
                parquet_file_count=manifest["parquet_file_count"],
                source_count=len(SILVER_SOURCE_KEYS),
                mapping_version=CANONICAL_MAPPING_VERSION,
            )

        return dg.MaterializeResult(
            metadata={
                "snapshot_id": snapshot_id,
                "snapshot_uri": snapshot_uri,
                "row_count": manifest["row_count"],
                "include_in_model_rows": manifest["include_in_model_rows"],
                "sources_processed": len(SILVER_SOURCE_KEYS),
                "mapping_version": CANONICAL_MAPPING_VERSION,
                "partitioning_columns": ["source_city", "occurrence_year"],
                "montgomery_bronze_rows": montgomery_rows,
                "event_grain": "source offense/event row",
            }
        )


crime_silver_assets = [silver_crime_offenses]
