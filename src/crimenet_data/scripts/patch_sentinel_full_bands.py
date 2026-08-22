#!/usr/bin/env python3
"""Patch CrimeNet's existing 7-band Sentinel-2 stacks with the six missing bands.

Existing stack layout is intentionally preserved for backward compatibility:

    1  B02
    2  B03
    3  B04
    4  B08
    5  B11
    6  B12
    7  SCL

The patch appends:

    8  B05
    9  B06
    10 B07
    11 B8A
    12 B01
    13 B09

This means the already-materialized silver imagery tables remain valid:
- gcs_uri does not change
- raster width/height/CRS/transform do not change
- H3 window offsets do not change
- SCL remains band 7

For OlmoEarth, read the 12 spectral channels in this order:

    [1, 2, 3, 4, 8, 9, 10, 11, 5, 6, 12, 13]

which corresponds to:

    B02 B03 B04 B08 B05 B06 B07 B8A B11 B12 B01 B09

The script is resumable. A successfully patched GCS object receives custom metadata
``crimenet_band_layout_version=sentinel2_full12_scl_v1``. Reruns skip those objects.
Uploads replace the original object only after a complete local patched raster has been
built, and use a GCS generation precondition to avoid clobbering concurrent changes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
from typing import Iterable

from google.cloud import storage
import planetary_computer as pc
import polars as pl
import pystac
import rasterio
from rasterio.enums import Resampling
from rasterio.shutil import copy as rio_copy
from rasterio.vrt import WarpedVRT
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"

DEFAULT_MANIFEST_URI = (
    "gs://crimenet/bronze/imagery/optimized/manifests/imagery_items.parquet"
)
DEFAULT_REPORT_URI = (
    "gs://crimenet/bronze/imagery/optimized/manifests/"
    "sentinel_full_band_patch.parquet"
)

PATCH_VERSION = "sentinel2_full12_scl_v1"
PATCH_METADATA_KEY = "crimenet_band_layout_version"

# Keep the first seven positions unchanged so the already-built silver tables and
# current SCL=7 preprocessing remain compatible.
EXISTING_LAYOUT = {
    1: "B02",
    2: "B03",
    3: "B04",
    4: "B08",
    5: "B11",
    6: "B12",
    7: "SCL",
}

APPENDED_LAYOUT = {
    8: "B05",
    9: "B06",
    10: "B07",
    11: "B8A",
    12: "B01",
    13: "B09",
}

FINAL_LAYOUT = {**EXISTING_LAYOUT, **APPENDED_LAYOUT}
MISSING_ASSET_KEYS = ["B05", "B06", "B07", "B8A", "B01", "B09"]

# Exact channel order expected by OlmoEarth's Sentinel-2 input.
OLMOEARTH_RASTER_BAND_INDEXES = [1, 2, 3, 4, 8, 9, 10, 11, 5, 6, 12, 13]
SCL_RASTER_BAND_INDEX = 7

_thread_local = threading.local()


@dataclass
class PatchResult:
    item_id: str
    gcs_uri: str
    status: str
    duration_seconds: float
    bytes_before: int | None = None
    bytes_after: int | None = None
    error: str | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Append B05/B06/B07/B8A/B01/B09 to CrimeNet's existing "
            "7-band Sentinel-2 stacks while preserving all current raster geometry."
        )
    )
    p.add_argument("--manifest-uri", default=DEFAULT_MANIFEST_URI)
    p.add_argument("--report-uri", default=DEFAULT_REPORT_URI)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="0 means all Sentinel items")
    p.add_argument(
        "--cities",
        nargs="*",
        default=None,
        help="Optional source_cities filter, e.g. --cities dallas fort_worth",
    )
    p.add_argument(
        "--item-id",
        action="append",
        default=None,
        help="Patch only this exact Sentinel item ID; may be repeated.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--work-dir",
        default=None,
        help="Local scratch directory. Defaults to the system temp directory.",
    )
    p.add_argument(
        "--no-make-cog",
        action="store_true",
        help=(
            "Skip the final GDAL COG CreateCopy pass. The output remains a tiled, "
            "compressed GeoTIFF, but COG conversion is recommended for production."
        ),
    )
    p.add_argument(
        "--zstd-level",
        type=int,
        default=6,
        help="ZSTD compression level for the intermediate tiled GeoTIFF.",
    )
    return p.parse_args()


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got {uri!r}")
    bucket, _, blob = uri[5:].partition("/")
    if not bucket or not blob:
        raise ValueError(f"Invalid GCS URI: {uri!r}")
    return bucket, blob


def _gcs_client() -> storage.Client:
    client = getattr(_thread_local, "gcs_client", None)
    if client is None:
        client = storage.Client()
        _thread_local.gcs_client = client
    return client


def _http_session() -> requests.Session:
    session = getattr(_thread_local, "http_session", None)
    if session is not None:
        return session

    retry = Retry(
        total=6,
        connect=6,
        read=6,
        status=6,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    _thread_local.http_session = session
    return session


def _download_gcs(uri: str, local_path: Path) -> tuple[int, int, dict[str, str]]:
    bucket_name, blob_name = _parse_gs_uri(uri)
    blob = _gcs_client().bucket(bucket_name).blob(blob_name)
    blob.reload()
    generation = int(blob.generation)
    size = int(blob.size or 0)
    metadata = dict(blob.metadata or {})
    blob.download_to_filename(str(local_path))
    return generation, size, metadata


def _upload_gcs_replace(
    uri: str,
    local_path: Path,
    *,
    expected_generation: int,
    metadata: dict[str, str],
) -> int:
    bucket_name, blob_name = _parse_gs_uri(uri)
    blob = _gcs_client().bucket(bucket_name).blob(blob_name)
    blob.metadata = metadata
    blob.upload_from_filename(
        str(local_path),
        content_type="image/tiff",
        if_generation_match=expected_generation,
        timeout=600,
    )
    blob.reload()
    return int(blob.size or local_path.stat().st_size)


def _upload_file(uri: str, local_path: Path, content_type: str) -> None:
    bucket_name, blob_name = _parse_gs_uri(uri)
    blob = storage.Client().bucket(bucket_name).blob(blob_name)
    blob.upload_from_filename(str(local_path), content_type=content_type)


def _read_manifest(uri: str) -> pl.DataFrame:
    with tempfile.TemporaryDirectory(prefix="crimenet_manifest_") as tmp:
        local = Path(tmp) / "imagery_items.parquet"
        bucket_name, blob_name = _parse_gs_uri(uri)
        storage.Client().bucket(bucket_name).blob(blob_name).download_to_filename(str(local))
        return pl.read_parquet(local)


def _manifest_records(
    df: pl.DataFrame,
    *,
    cities: set[str] | None,
    item_ids: set[str] | None,
    limit: int,
) -> list[dict]:
    required = {"collection", "item_id", "gcs_uri", "status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    filtered = (
        df.filter(pl.col("collection") == COLLECTION)
        .filter(pl.col("status").is_in(["uploaded", "exists"]))
        .filter(pl.col("gcs_uri").is_not_null())
        .unique(subset=["item_id"], keep="first")
        .sort("item_id")
    )

    records = filtered.to_dicts()

    if cities:
        records = [
            r
            for r in records
            if cities.intersection(set(r.get("source_cities") or []))
        ]

    if item_ids:
        records = [r for r in records if r["item_id"] in item_ids]

    if limit > 0:
        records = records[:limit]

    return records


def _fetch_signed_item(item_id: str) -> pystac.Item:
    url = f"{STAC_API}/collections/{COLLECTION}/items/{item_id}"
    response = _http_session().get(url, timeout=60)
    response.raise_for_status()
    raw_item = pystac.Item.from_dict(response.json())
    return pc.sign(raw_item)


def _validate_missing_assets(item: pystac.Item) -> dict[str, str]:
    missing = [key for key in MISSING_ASSET_KEYS if key not in item.assets]
    if missing:
        raise KeyError(f"STAC item {item.id} is missing assets: {missing}")
    return {key: item.assets[key].href for key in MISSING_ASSET_KEYS}


def _descriptions_match(src: rasterio.io.DatasetReader) -> bool:
    if src.count != 13:
        return False
    descriptions = list(src.descriptions)
    for idx, expected in FINAL_LAYOUT.items():
        actual = descriptions[idx - 1]
        if actual is not None and actual.upper() != expected.upper():
            return False
    return True


def _copy_existing_bands(
    src: rasterio.io.DatasetReader,
    dst: rasterio.io.DatasetWriter,
) -> None:
    # Geometry is identical, so source and destination pixel windows line up exactly.
    for _, window in dst.block_windows(1):
        for band_idx in range(1, 8):
            arr = src.read(band_idx, window=window)
            dst.write(arr, band_idx, window=window)


def _append_remote_band(
    *,
    href: str,
    dst: rasterio.io.DatasetWriter,
    dst_band_index: int,
    dst_crs,
    dst_transform,
    dst_width: int,
    dst_height: int,
    dst_dtype: str,
    dst_nodata,
) -> None:
    # Spectral bands are continuous reflectance. Bilinear is appropriate when the
    # 20 m / 60 m source bands are aligned to the existing 10 m stack grid.
    with rasterio.open(href) as src_band:
        src_nodata = src_band.nodata if src_band.nodata is not None else 0
        with WarpedVRT(
            src_band,
            crs=dst_crs,
            transform=dst_transform,
            width=dst_width,
            height=dst_height,
            resampling=Resampling.bilinear,
            src_nodata=src_nodata,
            nodata=dst_nodata,
            dtype=dst_dtype,
        ) as vrt:
            for _, window in dst.block_windows(dst_band_index):
                arr = vrt.read(1, window=window, out_dtype=dst_dtype)
                dst.write(arr, dst_band_index, window=window)


def _build_patched_tiff(
    *,
    existing_path: Path,
    output_gtiff: Path,
    signed_assets: dict[str, str],
    zstd_level: int,
) -> None:
    with rasterio.open(existing_path) as src:
        if src.count == 13 and _descriptions_match(src):
            shutil.copyfile(existing_path, output_gtiff)
            return
        if src.count != 7:
            raise ValueError(
                f"Expected existing Sentinel stack to have 7 bands or already-patched 13; "
                f"got {src.count} in {existing_path}"
            )
        if src.crs is None:
            raise ValueError(f"Existing Sentinel stack has no CRS: {existing_path}")

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            count=13,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            compress="ZSTD",
            zstd_level=zstd_level,
            BIGTIFF="IF_SAFER",
            interleave="band",
        )
        # All current Sentinel bands are stored in a common integer dataset dtype.
        dst_dtype = src.dtypes[0]
        dst_nodata = src.nodata if src.nodata is not None else 0
        profile.update(dtype=dst_dtype, nodata=dst_nodata)

        with rasterio.open(output_gtiff, "w", **profile) as dst:
            _copy_existing_bands(src, dst)

            for dst_band_index, asset_key in APPENDED_LAYOUT.items():
                _append_remote_band(
                    href=signed_assets[asset_key],
                    dst=dst,
                    dst_band_index=dst_band_index,
                    dst_crs=src.crs,
                    dst_transform=src.transform,
                    dst_width=src.width,
                    dst_height=src.height,
                    dst_dtype=dst_dtype,
                    dst_nodata=dst_nodata,
                )

            # Copy dataset tags, then add explicit layout metadata.
            dst.update_tags(**src.tags())
            dst.update_tags(
                CRIMENET_BAND_LAYOUT_VERSION=PATCH_VERSION,
                CRIMENET_BAND_LAYOUT=json.dumps(FINAL_LAYOUT, sort_keys=True),
                CRIMENET_OLMOEARTH_BAND_INDEXES=",".join(
                    str(x) for x in OLMOEARTH_RASTER_BAND_INDEXES
                ),
                CRIMENET_SCL_BAND_INDEX=str(SCL_RASTER_BAND_INDEX),
            )

            for idx, name in FINAL_LAYOUT.items():
                dst.set_band_description(idx, name)

            # Preserve any tags that existed on the original seven bands.
            for idx in range(1, 8):
                tags = src.tags(idx)
                if tags:
                    dst.update_tags(idx, **tags)


def _make_cog(src_path: Path, dst_path: Path) -> None:
    # No overview pyramid is needed for CrimeNet's native-resolution window reads.
    # The COG driver still reorganizes blocks for efficient range access.
    rio_copy(
        str(src_path),
        str(dst_path),
        driver="COG",
        compress="ZSTD",
        blocksize=512,
        overview_resampling="nearest",
        BIGTIFF="IF_SAFER",
    )


def _check_gcs_metadata(uri: str) -> tuple[bool, int | None, dict[str, str]]:
    bucket_name, blob_name = _parse_gs_uri(uri)
    blob = _gcs_client().bucket(bucket_name).blob(blob_name)
    blob.reload()
    metadata = dict(blob.metadata or {})
    return (
        metadata.get(PATCH_METADATA_KEY) == PATCH_VERSION,
        int(blob.size or 0),
        metadata,
    )


def patch_one(record: dict, args: argparse.Namespace) -> PatchResult:
    started = time.monotonic()
    item_id = record["item_id"]
    gcs_uri = record["gcs_uri"]

    try:
        already_patched, existing_size, existing_metadata = _check_gcs_metadata(gcs_uri)
        if already_patched:
            return PatchResult(
                item_id=item_id,
                gcs_uri=gcs_uri,
                status="exists",
                duration_seconds=time.monotonic() - started,
                bytes_before=existing_size,
                bytes_after=existing_size,
            )

        if args.dry_run:
            return PatchResult(
                item_id=item_id,
                gcs_uri=gcs_uri,
                status="would_patch",
                duration_seconds=time.monotonic() - started,
                bytes_before=existing_size,
            )

        work_root = Path(args.work_dir) if args.work_dir else None
        with tempfile.TemporaryDirectory(
            prefix=f"sentinel_patch_{item_id[:24]}_",
            dir=str(work_root) if work_root else None,
        ) as tmp:
            tmpdir = Path(tmp)
            existing_path = tmpdir / "existing_7band.tif"
            patched_gtiff = tmpdir / "patched_13band_gtiff.tif"
            final_path = tmpdir / "patched_13band.tif"

            generation, bytes_before, metadata = _download_gcs(gcs_uri, existing_path)

            # Handle a 13-band object that predates/omitted our GCS metadata marker.
            with rasterio.open(existing_path) as src_check:
                if src_check.count == 13 and _descriptions_match(src_check):
                    metadata[PATCH_METADATA_KEY] = PATCH_VERSION
                    metadata["crimenet_band_layout"] = json.dumps(
                        FINAL_LAYOUT, sort_keys=True
                    )
                    # Re-upload only to add metadata because the GCS object itself is already correct.
                    bytes_after = _upload_gcs_replace(
                        gcs_uri,
                        existing_path,
                        expected_generation=generation,
                        metadata=metadata,
                    )
                    return PatchResult(
                        item_id=item_id,
                        gcs_uri=gcs_uri,
                        status="metadata_fixed",
                        duration_seconds=time.monotonic() - started,
                        bytes_before=bytes_before,
                        bytes_after=bytes_after,
                    )
                if src_check.count != 7:
                    raise ValueError(
                        f"Unexpected current stack band count {src_check.count}; expected 7 or 13"
                    )

            item = _fetch_signed_item(item_id)
            signed_assets = _validate_missing_assets(item)

            with rasterio.Env(
                GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                GDAL_HTTP_MULTIRANGE="YES",
                GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
                GDAL_HTTP_MAX_RETRY="5",
                GDAL_HTTP_RETRY_DELAY="2",
                GDAL_HTTP_VERSION="2",
                CPL_VSIL_CURL_CHUNK_SIZE="1048576",
            ):
                _build_patched_tiff(
                    existing_path=existing_path,
                    output_gtiff=patched_gtiff,
                    signed_assets=signed_assets,
                    zstd_level=args.zstd_level,
                )

                if args.no_make_cog:
                    shutil.move(patched_gtiff, final_path)
                else:
                    _make_cog(patched_gtiff, final_path)

            # Validate the exact geometry invariant before replacing the GCS object.
            with rasterio.open(existing_path) as before, rasterio.open(final_path) as after:
                if after.count != 13:
                    raise ValueError(f"Patched output has {after.count} bands; expected 13")
                if (
                    before.width != after.width
                    or before.height != after.height
                    or before.crs != after.crs
                    or before.transform != after.transform
                ):
                    raise ValueError(
                        "Patched raster geometry changed. Refusing to overwrite because "
                        "the existing silver H3 windows would become invalid."
                    )
                if not _descriptions_match(after):
                    raise ValueError("Patched output band descriptions do not match expected layout")

            metadata = dict(metadata)
            metadata[PATCH_METADATA_KEY] = PATCH_VERSION
            metadata["crimenet_band_layout"] = json.dumps(FINAL_LAYOUT, sort_keys=True)
            metadata["crimenet_olmoearth_band_indexes"] = ",".join(
                str(x) for x in OLMOEARTH_RASTER_BAND_INDEXES
            )
            metadata["crimenet_scl_band_index"] = str(SCL_RASTER_BAND_INDEX)

            bytes_after = _upload_gcs_replace(
                gcs_uri,
                final_path,
                expected_generation=generation,
                metadata=metadata,
            )

        return PatchResult(
            item_id=item_id,
            gcs_uri=gcs_uri,
            status="patched",
            duration_seconds=time.monotonic() - started,
            bytes_before=bytes_before,
            bytes_after=bytes_after,
        )

    except Exception as exc:
        return PatchResult(
            item_id=item_id,
            gcs_uri=gcs_uri,
            status="error",
            duration_seconds=time.monotonic() - started,
            error=repr(exc),
        )


def _write_report(results: Iterable[PatchResult], uri: str) -> None:
    rows = [asdict(r) for r in results]
    if not rows:
        return
    df = pl.DataFrame(rows).sort(["status", "item_id"])
    with tempfile.TemporaryDirectory(prefix="sentinel_patch_report_") as tmp:
        local = Path(tmp) / "sentinel_full_band_patch.parquet"
        df.write_parquet(local, compression="zstd", statistics=True)
        _upload_file(uri, local, "application/octet-stream")


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    log = logging.getLogger("sentinel-band-patch")

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    manifest = _read_manifest(args.manifest_uri)
    records = _manifest_records(
        manifest,
        cities=set(args.cities) if args.cities else None,
        item_ids=set(args.item_id) if args.item_id else None,
        limit=args.limit,
    )

    log.info("Sentinel source items selected: %s", f"{len(records):,}")
    log.info("Workers: %d", args.workers)
    log.info("Output band layout: %s", FINAL_LAYOUT)
    log.info("OlmoEarth raster band indexes: %s", OLMOEARTH_RASTER_BAND_INDEXES)
    log.info("SCL remains band: %d", SCL_RASTER_BAND_INDEX)

    if args.dry_run:
        log.info("Dry-run mode: no GCS objects will be modified.")

    results: list[PatchResult] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(patch_one, record, args): record for record in records}

        for i, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            before_mib = (result.bytes_before or 0) / (1024 * 1024)
            after_mib = (result.bytes_after or 0) / (1024 * 1024)

            if result.status == "error":
                log.error(
                    "[%s/%s] %s | %s | %.1fs | %s",
                    f"{i:,}",
                    f"{len(records):,}",
                    result.status,
                    result.item_id,
                    result.duration_seconds,
                    result.error,
                )
            else:
                log.info(
                    "[%s/%s] %s | %s | %.1f -> %.1f MiB | %.1fs",
                    f"{i:,}",
                    f"{len(records):,}",
                    result.status,
                    result.item_id,
                    before_mib,
                    after_mib,
                    result.duration_seconds,
                )

    if not args.dry_run:
        _write_report(results, args.report_uri)
        log.info("Patch report: %s", args.report_uri)

    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    log.info("Final status counts: %s", counts)

    errors = counts.get("error", 0)
    if errors:
        log.error(
            "%d item(s) failed. Re-run the same command; completed items will be skipped.",
            errors,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())