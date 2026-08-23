#!/usr/bin/env python3
"""Aggressively patch CrimeNet's selected 7-band Sentinel-2 stacks to 13 bands in place.

This is a one-time preprocessing pass for OlmoEarth. It preserves the exact raster
geometry and the first seven bands, so the already-built silver H3 window offsets
remain valid and no silver table needs to be rebuilt.

Existing layout (unchanged):
    1 B02, 2 B03, 3 B04, 4 B08, 5 B11, 6 B12, 7 SCL
Appended layout:
    8 B05, 9 B06, 10 B07, 11 B8A, 12 B01, 13 B09

OlmoEarth spectral order is therefore read with raster indexes:
    [1, 2, 3, 4, 8, 9, 10, 11, 5, 6, 12, 13]

Performance design:
- patch multiple source scenes concurrently (default 8)
- inside every scene, range-read all six missing Planetary Computer bands in parallel
- overlap those six remote reads with the local copy of the existing seven bands
- write a tiled ZSTD COG directly, one output block at a time
- upload only after the entire local result validates
- replace the GCS object with an if-generation-match precondition
- mark completed objects in GCS metadata so reruns are resumable

The first seven bands and raster geometry never change. Only bands 8..13 are appended.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import logging
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any

from google.cloud import storage
import planetary_computer as pc
import polars as pl
import pystac
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"

DEFAULT_TEMPORAL_INDEX_URI = (
    "gs://crimenet/silver/imagery/h3_temporal_index/part-00000.parquet"
)
DEFAULT_REPORT_URI = (
    "gs://crimenet/silver/imagery/sentinel13_patch_report.parquet"
)

PATCH_VERSION = "sentinel2_full12_scl_v2_parallel"
PATCH_METADATA_KEY = "crimenet_band_layout_version"

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
    width: int | None = None
    height: int | None = None
    error: str | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Patch selected CrimeNet Sentinel stacks to local 13-band OlmoEarth inputs."
    )
    p.add_argument("--temporal-index-uri", default=DEFAULT_TEMPORAL_INDEX_URI)
    p.add_argument("--report-uri", default=DEFAULT_REPORT_URI)
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent source scenes. Each scene also uses --band-workers remote readers.",
    )
    p.add_argument(
        "--band-workers",
        type=int,
        default=6,
        help="Parallel missing-band reads per source scene. Six is the natural maximum.",
    )
    p.add_argument("--limit", type=int, default=0, help="0 = all selected Sentinel scenes")
    p.add_argument(
        "--item-id",
        action="append",
        default=None,
        help="Patch only this exact item ID; may be repeated.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--work-dir",
        default=None,
        help="Local scratch directory. Defaults to the system temp directory.",
    )
    p.add_argument(
        "--block-size",
        type=int,
        default=512,
        help="COG block size. 512 is a good range-read compromise.",
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
        total=8,
        connect=8,
        read=8,
        status=8,
        backoff_factor=0.75,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=64, pool_maxsize=64)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    _thread_local.http_session = session
    return session


def _read_selected_scenes(uri: str, item_ids: set[str] | None, limit: int) -> list[dict]:
    df = pl.read_parquet(
        uri,
        columns=[
            "source",
            "item_id",
            "gcs_uri",
            "selected_in_period",
            "is_usable",
            "error",
        ],
        credential_provider=pl.CredentialProviderGCP(),
    )
    scenes = (
        df.filter(pl.col("source") == "sentinel2")
        .filter(pl.col("selected_in_period") == True)
        .filter(pl.col("is_usable") == True)
        .filter(pl.col("error").is_null())
        .select("item_id", "gcs_uri")
        .unique(subset=["item_id"], keep="first")
        .sort("item_id")
    )
    if item_ids:
        scenes = scenes.filter(pl.col("item_id").is_in(sorted(item_ids)))
    if limit > 0:
        scenes = scenes.head(limit)
    return scenes.to_dicts()


def _fetch_signed_assets(item_id: str) -> dict[str, str]:
    url = f"{STAC_API}/collections/{COLLECTION}/items/{item_id}"
    response = _http_session().get(url, timeout=60)
    response.raise_for_status()
    item = pc.sign(pystac.Item.from_dict(response.json()))
    missing = [key for key in MISSING_ASSET_KEYS if key not in item.assets]
    if missing:
        raise KeyError(f"Sentinel item {item_id} missing assets {missing}")
    return {key: item.assets[key].href for key in MISSING_ASSET_KEYS}


def _descriptions_match(src: rasterio.io.DatasetReader) -> bool:
    if src.count != 13:
        return False
    for idx, expected in FINAL_LAYOUT.items():
        actual = src.descriptions[idx - 1]
        if actual is not None and actual.upper() != expected.upper():
            return False
    return True


def _blob_state(uri: str) -> tuple[Any, int, int, dict[str, str]]:
    bucket_name, blob_name = _parse_gs_uri(uri)
    blob = _gcs_client().bucket(bucket_name).blob(blob_name)
    blob.reload()
    return blob, int(blob.generation), int(blob.size or 0), dict(blob.metadata or {})


def _download_blob(blob, local_path: Path) -> None:
    blob.download_to_filename(str(local_path), timeout=900)


def _mark_metadata_only(blob, generation: int, metadata: dict[str, str]) -> None:
    blob.metadata = metadata
    blob.patch(if_generation_match=generation, timeout=120)


def _upload_replace(
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
        timeout=1800,
    )
    blob.reload()
    return int(blob.size or local_path.stat().st_size)


def _raster_env() -> rasterio.Env:
    opts = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "GDAL_HTTP_MULTIRANGE": "YES",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "GDAL_HTTP_MAX_RETRY": "8",
        "GDAL_HTTP_RETRY_DELAY": "1",
        "GDAL_HTTP_VERSION": "2",
        "CPL_VSIL_CURL_CHUNK_SIZE": "2097152",
        "VSI_CACHE": "TRUE",
        "VSI_CACHE_SIZE": "134217728",
    }
    if os.getenv("CPL_MACHINE_IS_GCE", "").upper() == "YES":
        opts["CPL_MACHINE_IS_GCE"] = "YES"
    return rasterio.Env(**opts)


def _build_patched_cog(
    *,
    existing_path: Path,
    output_path: Path,
    signed_assets: dict[str, str],
    band_workers: int,
    block_size: int,
) -> tuple[int, int]:
    """Build one 13-band COG while reading all six missing bands concurrently."""
    remote_sources: dict[str, rasterio.io.DatasetReader] = {}
    remote_vrts: dict[str, WarpedVRT] = {}

    try:
        with rasterio.open(existing_path) as src:
            if src.count != 7:
                raise ValueError(f"Expected 7-band source, got {src.count}: {existing_path}")
            if src.crs is None:
                raise ValueError(f"Source has no CRS: {existing_path}")

            dst_dtype = src.dtypes[0]
            dst_nodata = src.nodata if src.nodata is not None else 0

            for key, href in signed_assets.items():
                remote = rasterio.open(href, sharing=False)
                src_nodata = remote.nodata if remote.nodata is not None else 0
                vrt = WarpedVRT(
                    remote,
                    crs=src.crs,
                    transform=src.transform,
                    width=src.width,
                    height=src.height,
                    resampling=Resampling.bilinear,
                    src_nodata=src_nodata,
                    nodata=dst_nodata,
                    dtype=dst_dtype,
                )
                remote_sources[key] = remote
                remote_vrts[key] = vrt

            profile = {
                "driver": "COG",
                "width": src.width,
                "height": src.height,
                "count": 13,
                "dtype": dst_dtype,
                "crs": src.crs,
                "transform": src.transform,
                "nodata": dst_nodata,
                "compress": "ZSTD",
                "blocksize": block_size,
                "BIGTIFF": "IF_SAFER",
            }

            with rasterio.open(output_path, "w", **profile) as dst:
                with ThreadPoolExecutor(
                    max_workers=min(max(1, band_workers), len(MISSING_ASSET_KEYS)),
                    thread_name_prefix="sentinel-missing-band",
                ) as band_pool:
                    for _, window in dst.block_windows(1):
                        # Start all six remote reads first.
                        futures = {
                            key: band_pool.submit(
                                remote_vrts[key].read,
                                1,
                                window=window,
                                out_dtype=dst_dtype,
                            )
                            for key in MISSING_ASSET_KEYS
                        }

                        # Copy the seven already-local bands while remote requests are in flight.
                        local = src.read(list(range(1, 8)), window=window)
                        dst.write(local, indexes=list(range(1, 8)), window=window)

                        # Only the main scene thread writes to the destination dataset.
                        for dst_idx, key in APPENDED_LAYOUT.items():
                            dst.write(futures[key].result(), dst_idx, window=window)

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
                for idx in range(1, 8):
                    tags = src.tags(idx)
                    if tags:
                        dst.update_tags(idx, **tags)

            return src.width, src.height
    finally:
        for vrt in remote_vrts.values():
            try:
                vrt.close()
            except Exception:
                pass
        for remote in remote_sources.values():
            try:
                remote.close()
            except Exception:
                pass


def _patched_metadata(metadata: dict[str, str]) -> dict[str, str]:
    out = dict(metadata)
    out[PATCH_METADATA_KEY] = PATCH_VERSION
    out["crimenet_band_layout"] = json.dumps(FINAL_LAYOUT, sort_keys=True)
    out["crimenet_olmoearth_band_indexes"] = ",".join(
        str(x) for x in OLMOEARTH_RASTER_BAND_INDEXES
    )
    out["crimenet_scl_band_index"] = str(SCL_RASTER_BAND_INDEX)
    return out


def patch_one(record: dict, args: argparse.Namespace) -> PatchResult:
    started = time.monotonic()
    item_id = record["item_id"]
    gcs_uri = record["gcs_uri"]

    try:
        blob, generation, bytes_before, metadata = _blob_state(gcs_uri)
        if metadata.get(PATCH_METADATA_KEY) == PATCH_VERSION:
            return PatchResult(
                item_id=item_id,
                gcs_uri=gcs_uri,
                status="exists",
                duration_seconds=time.monotonic() - started,
                bytes_before=bytes_before,
                bytes_after=bytes_before,
            )

        if args.dry_run:
            return PatchResult(
                item_id=item_id,
                gcs_uri=gcs_uri,
                status="would_patch",
                duration_seconds=time.monotonic() - started,
                bytes_before=bytes_before,
            )

        work_root = Path(args.work_dir) if args.work_dir else None
        with tempfile.TemporaryDirectory(
            prefix=f"sentinel13_{item_id[:24]}_",
            dir=str(work_root) if work_root else None,
        ) as tmp:
            tmpdir = Path(tmp)
            existing_path = tmpdir / "existing.tif"
            output_path = tmpdir / "patched13.tif"
            _download_blob(blob, existing_path)

            with rasterio.open(existing_path) as before:
                width, height = before.width, before.height
                if before.count == 13 and _descriptions_match(before):
                    _mark_metadata_only(
                        blob,
                        generation,
                        _patched_metadata(metadata),
                    )
                    return PatchResult(
                        item_id=item_id,
                        gcs_uri=gcs_uri,
                        status="metadata_fixed",
                        duration_seconds=time.monotonic() - started,
                        bytes_before=bytes_before,
                        bytes_after=bytes_before,
                        width=width,
                        height=height,
                    )
                if before.count != 7:
                    raise ValueError(
                        f"Unexpected source band count {before.count}; expected 7 or 13"
                    )

            signed_assets = _fetch_signed_assets(item_id)
            with _raster_env():
                width, height = _build_patched_cog(
                    existing_path=existing_path,
                    output_path=output_path,
                    signed_assets=signed_assets,
                    band_workers=args.band_workers,
                    block_size=args.block_size,
                )

            # Validate the invariant that makes existing silver H3 windows reusable.
            with rasterio.open(existing_path) as before, rasterio.open(output_path) as after:
                if after.count != 13:
                    raise ValueError(f"Patched output has {after.count} bands, expected 13")
                if (
                    before.width != after.width
                    or before.height != after.height
                    or before.crs != after.crs
                    or before.transform != after.transform
                ):
                    raise ValueError(
                        "Patched raster geometry changed; refusing to replace source because "
                        "existing silver H3 windows would become invalid."
                    )
                if not _descriptions_match(after):
                    raise ValueError("Patched band descriptions do not match expected layout")

            bytes_after = _upload_replace(
                gcs_uri,
                output_path,
                expected_generation=generation,
                metadata=_patched_metadata(metadata),
            )

        return PatchResult(
            item_id=item_id,
            gcs_uri=gcs_uri,
            status="patched",
            duration_seconds=time.monotonic() - started,
            bytes_before=bytes_before,
            bytes_after=bytes_after,
            width=width,
            height=height,
        )
    except Exception as exc:
        return PatchResult(
            item_id=item_id,
            gcs_uri=gcs_uri,
            status="error",
            duration_seconds=time.monotonic() - started,
            error=repr(exc),
        )


def _write_report(results: list[PatchResult], uri: str) -> None:
    if not results:
        return
    df = pl.DataFrame([asdict(r) for r in results], infer_schema_length=None).sort(
        ["status", "item_id"]
    )
    with tempfile.TemporaryDirectory(prefix="sentinel13_report_") as tmp:
        local = Path(tmp) / "report.parquet"
        df.write_parquet(local, compression="zstd", compression_level=3, statistics=True)
        bucket_name, blob_name = _parse_gs_uri(uri)
        storage.Client().bucket(bucket_name).blob(blob_name).upload_from_filename(
            str(local), content_type="application/octet-stream"
        )


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    log = logging.getLogger("sentinel13-max-patch")

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.band_workers < 1:
        raise ValueError("--band-workers must be >= 1")
    if args.block_size < 128:
        raise ValueError("--block-size must be >= 128")

    records = _read_selected_scenes(
        args.temporal_index_uri,
        set(args.item_id) if args.item_id else None,
        args.limit,
    )

    log.info("Selected Sentinel source scenes: %s", f"{len(records):,}")
    log.info(
        "Concurrency: scenes=%d x missing-bands=%d => up to ~%d remote reads",
        args.workers,
        min(args.band_workers, 6),
        args.workers * min(args.band_workers, 6),
    )
    log.info("Patch is in-place; bands 1..7 and raster geometry are preserved exactly.")
    if args.dry_run:
        log.info("Dry-run: no GCS objects will be modified.")

    results: list[PatchResult] = []
    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="sentinel-scene") as pool:
        futures = {pool.submit(patch_one, rec, args): rec for rec in records}
        for i, fut in enumerate(as_completed(futures), start=1):
            result = fut.result()
            results.append(result)
            before = (result.bytes_before or 0) / (1024 * 1024)
            after = (result.bytes_after or 0) / (1024 * 1024)
            if result.status == "error":
                log.error(
                    "[%s/%s] %s | %s | %.1fs | %s",
                    f"{i:,}", f"{len(records):,}", result.status,
                    result.item_id, result.duration_seconds, result.error,
                )
            else:
                log.info(
                    "[%s/%s] %s | %s | %.1f -> %.1f MiB | %.1fs",
                    f"{i:,}", f"{len(records):,}", result.status,
                    result.item_id, before, after, result.duration_seconds,
                )

    if not args.dry_run:
        _write_report(results, args.report_uri)
        log.info("Report: %s", args.report_uri)

    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    log.info("Final status counts: %s", counts)

    errors = counts.get("error", 0)
    if errors:
        log.error("%d scene(s) failed. Rerun the same command; completed scenes are skipped.", errors)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
