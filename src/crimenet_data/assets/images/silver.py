from pathlib import Path
import tempfile

import dagster as dg
from google.cloud import storage
import polars as pl

from .transformations import (
    ImageryPreprocessSettings,
    build_temporal_index,
    preprocess_item_manifest,
)


BRONZE_IMAGERY_ITEMS_URI = (
    "gs://crimenet/bronze/imagery/optimized/manifests/imagery_items.parquet"
)
SILVER_H3_CANDIDATES_URI = (
    "gs://crimenet/silver/imagery/h3_candidates/part-00000.parquet"
)
SILVER_H3_TEMPORAL_INDEX_URI = (
    "gs://crimenet/silver/imagery/h3_temporal_index/part-00000.parquet"
)


class ImageryPreprocessConfig(dg.Config):
    workers: int = 8
    target_h3_resolution: int = 9
    naip_context_margin_m: float = 32.0
    sentinel_context_margin_m: float = 600.0
    sentinel_max_local_bad_fraction: float = 0.20
    min_single_source_coverage_fraction: float = 0.995


# -----------------------------------------------------------------------------
# GCS parquet helpers
# -----------------------------------------------------------------------------


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got {uri!r}")
    bucket, _, blob = uri[5:].partition("/")
    if not bucket or not blob:
        raise ValueError(f"Invalid GCS object URI: {uri!r}")
    return bucket, blob


def _read_parquet(uri: str) -> pl.DataFrame:
    return pl.read_parquet(
        uri,
        credential_provider=pl.CredentialProviderGCP(),
    )


def _write_parquet(uri: str, df: pl.DataFrame) -> None:
    bucket_name, blob_name = _parse_gs_uri(uri)
    client = storage.Client()

    with tempfile.TemporaryDirectory(prefix="crimenet_imagery_silver_") as tmpdir:
        local_path = Path(tmpdir) / "part-00000.parquet"
        df.write_parquet(
            local_path,
            compression="zstd",
            compression_level=6,
            statistics=True,
        )
        client.bucket(bucket_name).blob(blob_name).upload_from_filename(
            str(local_path),
            content_type="application/octet-stream",
        )


def _count_source(df: pl.DataFrame, source: str) -> int:
    if df.is_empty() or "source" not in df.columns:
        return 0
    return df.filter(pl.col("source") == source).height


# -----------------------------------------------------------------------------
# Assets
# -----------------------------------------------------------------------------


@dg.asset(
    group_name="imagery",
    compute_kind="rasterio/polars",
    description=(
        "Preprocess optimized NAIP and Sentinel-2 COGs into one row per "
        "(source item, H3-9 cell), including exact raster windows and local "
        "Sentinel SCL cloud/shadow quality metrics. No per-H3 image files are "
        "materialized."
    ),
)
def silver_imagery_h3_candidates(
    context: dg.AssetExecutionContext,
    config: ImageryPreprocessConfig,
) -> dg.MaterializeResult:
    context.log.info(f"Reading imagery manifest: {BRONZE_IMAGERY_ITEMS_URI}")
    manifest = _read_parquet(BRONZE_IMAGERY_ITEMS_URI)

    failed_ingestion = manifest.filter(pl.col("status") == "error")
    if failed_ingestion.height:
        context.log.warning(
            f"Bronze imagery manifest still contains {failed_ingestion.height:,} "
            "failed source items. They will not be preprocessed."
        )

    settings = ImageryPreprocessSettings(
        target_h3_resolution=config.target_h3_resolution,
        workers=config.workers,
        naip_context_margin_m=config.naip_context_margin_m,
        sentinel_context_margin_m=config.sentinel_context_margin_m,
        sentinel_max_local_bad_fraction=config.sentinel_max_local_bad_fraction,
        min_single_source_coverage_fraction=(
            config.min_single_source_coverage_fraction
        ),
    )

    candidates = preprocess_item_manifest(
        manifest,
        settings,
        log=context.log.info,
    )

    if candidates.is_empty():
        raise dg.Failure("Imagery preprocessing produced zero H3 candidates.")

    context.log.info(f"Writing H3 candidate table: {SILVER_H3_CANDIDATES_URI}")
    _write_parquet(SILVER_H3_CANDIDATES_URI, candidates)

    errors = candidates.filter(pl.col("error").is_not_null()).height
    sentinel = candidates.filter(pl.col("source") == "sentinel2")
    sentinel_usable = sentinel.filter(pl.col("is_usable")).height
    requires_mosaic = candidates.filter(pl.col("requires_mosaic") == True).height

    unique_h3 = candidates.select(pl.col("h3_cell").n_unique()).item()

    return dg.MaterializeResult(
        metadata={
            "output_uri": dg.MetadataValue.path(SILVER_H3_CANDIDATES_URI),
            "candidate_rows": candidates.height,
            "unique_h3_cells": int(unique_h3),
            "naip_candidates": _count_source(candidates, "naip"),
            "sentinel_candidates": sentinel.height,
            "sentinel_usable_candidates": sentinel_usable,
            "requires_mosaic_rows": requires_mosaic,
            "preprocess_error_rows": errors,
        }
    )


@dg.asset(
    deps=[silver_imagery_h3_candidates],
    group_name="imagery",
    compute_kind="polars",
    description=(
        "Build a leakage-safe temporal image index per H3. NAIP selects the "
        "best-overlap source per acquisition; Sentinel selects the locally "
        "clearest usable source per H3/month. valid_from_utc is always the "
        "actual capture timestamp, enabling backward as-of joins."
    ),
)
def silver_imagery_h3_temporal_index(
    context: dg.AssetExecutionContext,
) -> dg.MaterializeResult:
    context.log.info(f"Reading H3 candidates: {SILVER_H3_CANDIDATES_URI}")
    candidates = _read_parquet(SILVER_H3_CANDIDATES_URI)

    temporal = build_temporal_index(candidates)
    if temporal.is_empty():
        raise dg.Failure("Temporal imagery index is empty.")

    # Hard leakage invariant: an image cannot become valid before it existed.
    future_validity = temporal.filter(
        pl.col("valid_from_utc") != pl.col("capture_timestamp_utc")
    )
    if future_validity.height:
        raise dg.Failure(
            f"Temporal leakage invariant violated for {future_validity.height:,} rows: "
            "valid_from_utc must equal capture_timestamp_utc."
        )

    # Each selected Sentinel row must be the unique local winner for H3/month.
    sentinel_dupes = (
        temporal
        .filter(pl.col("source") == "sentinel2")
        .group_by(["h3_cell", "capture_period"])
        .len()
        .filter(pl.col("len") > 1)
    )
    if sentinel_dupes.height:
        raise dg.Failure(
            f"Found {sentinel_dupes.height:,} duplicated Sentinel H3/month selections."
        )

    context.log.info(
        f"Writing temporal imagery index: {SILVER_H3_TEMPORAL_INDEX_URI}"
    )
    _write_parquet(SILVER_H3_TEMPORAL_INDEX_URI, temporal)

    return dg.MaterializeResult(
        metadata={
            "output_uri": dg.MetadataValue.path(SILVER_H3_TEMPORAL_INDEX_URI),
            "rows": temporal.height,
            "unique_h3_cells": int(
                temporal.select(pl.col("h3_cell").n_unique()).item()
            ),
            "naip_rows": _count_source(temporal, "naip"),
            "sentinel_rows": _count_source(temporal, "sentinel2"),
            "mosaic_required_rows": temporal.filter(
                pl.col("requires_mosaic") == True
            ).height,
        }
    )


@dg.asset_check(asset=silver_imagery_h3_temporal_index)
def imagery_temporal_index_integrity_check(
    context: dg.AssetCheckExecutionContext,
) -> dg.AssetCheckResult:
    temporal = _read_parquet(SILVER_H3_TEMPORAL_INDEX_URI)

    null_keys = temporal.filter(
        pl.any_horizontal(
            pl.col("h3_cell").is_null(),
            pl.col("capture_timestamp_utc").is_null(),
            pl.col("gcs_uri").is_null(),
            pl.col("valid_from_utc").is_null(),
        )
    ).height

    wrong_resolution = temporal.filter(pl.col("h3_resolution") != 9).height

    bad_intervals = temporal.filter(
        pl.col("valid_to_utc").is_not_null()
        & (pl.col("valid_to_utc") <= pl.col("valid_from_utc"))
    ).height

    passed = null_keys == 0 and wrong_resolution == 0 and bad_intervals == 0

    if not passed:
        context.log.error(
            f"null_keys={null_keys:,}, wrong_resolution={wrong_resolution:,}, "
            f"bad_intervals={bad_intervals:,}"
        )

    return dg.AssetCheckResult(
        passed=passed,
        metadata={
            "null_key_rows": null_keys,
            "wrong_h3_resolution_rows": wrong_resolution,
            "bad_validity_intervals": bad_intervals,
        },
    )