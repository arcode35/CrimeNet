#!/usr/bin/env python3
"""
CrimeNet Sentinel-2 -> OlmoEarth foundation embeddings, Vast-optimized.

This is a standalone, disk-first replacement for the original GCP/Dagster
`gold_patched13_async_v5.py` execution engine.  The scientific recipe is kept
compatible with foundation_v1 while the I/O and scheduling are redesigned for
machines with many CPU cores, hundreds of GB of RAM, fast NVMe, and 1-N GPUs.

Pipeline
--------
1. stage
   - rclone the entire B2 Sentinel tree to local NVMe first.
   - optionally stage existing foundation_v1 Sentinel parquet metadata/shards.
2. inventory
   - parse B2 selection manifests + local TIFF hierarchy into compact Parquet.
   - reject incomplete 13-file scenes.
3. candidates
   - reproduce the old Silver H3-r9 + 600 m context / SCL quality calculation.
4. select
   - reproduce the exact best-Sentinel-per-H3/month ordering.
   - anti-join existing Gold H3/months by default.
   - build a temporal frame plan and validity patch sidecar.
5. context (optional but default when existing Gold is supplied)
   - reacquire only old selected scenes needed for <=400-day temporal context
     from Microsoft Planetary Computer.
6. frames
   - CPU-heavy raster stage: resample the 12 spectral bands + SCL into the exact
     128x128 target crop and write a temporary, compressed Parquet frame cache.
     Raw GeoTIFF bytes themselves are intentionally NOT packed into Parquet.
7. embed
   - one OlmoEarth encoder replica per visible GPU.
   - fixed H3 partitions are distributed across GPUs.
   - independent dynamic batch size per temporal length T=1..12.
   - bf16 inference, pinned host memory, non-blocking H2D copies.
8. finalize / publish
   - produce flat, canonical Parquet shards with the original foundation_v1
     output schema, plus recipe/success/validity manifests.
   - publish to a B2 staging prefix with rclone and verify sizes.

Important
---------
* Existing foundation_v1 embeddings are never recomputed.
* Existing H3/month rows remain the canonical historical selections.  New B2
  candidates for months already present in Gold are skipped rather than silently
  replacing history and changing every later temporal sequence.
* When historical context hydration is enabled, old selected item_ids are
  reacquired from Planetary Computer only to reconstruct the raw temporal frames
  needed by NEW embeddings.  Their embeddings are not regenerated.
* Remote B2 raw imagery is NEVER deleted automatically.  Delete it only after
  the published extension has been audited.

Dependencies
------------
Python:
  pip install -U numpy polars pyarrow rasterio h3 pyproj requests \
      pystac-client planetary-computer torch olmoearth-pretrain-minimal
System:
  rclone

Environment for rclone/B2 is whatever your configured rclone remote requires.
Planetary Computer does not require a key for normal public access; an optional
PC_SDK_SUBSCRIPTION_KEY is honored by planetary-computer if configured.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import dataclasses
import datetime as dt
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence
from urllib.parse import urlsplit

import h3
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import Window, bounds as window_bounds, from_bounds


# -----------------------------------------------------------------------------
# Scientific contract: foundation_v1 Sentinel / OlmoEarth
# -----------------------------------------------------------------------------

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
SENTINEL_COLLECTION = "sentinel-2-l2a"
OLMOEARTH_MODEL_ID = "OLMOEARTH_V1_2_BASE"
ENCODER_NAME = "olmoearth-v1.2-base"

SPECTRAL_BANDS = (
    "B02", "B03", "B04", "B08",
    "B05", "B06", "B07", "B8A",
    "B11", "B12", "B01", "B09",
)
REQUIRED_ASSETS = (
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B11", "B12", "SCL",
)
ASSET_GSD_M = {
    "B01": 60,
    "B02": 10,
    "B03": 10,
    "B04": 10,
    "B05": 20,
    "B06": 20,
    "B07": 20,
    "B08": 10,
    "B8A": 20,
    "B09": 60,
    "B11": 20,
    "B12": 20,
    "SCL": 20,
}

TARGET_H3_RESOLUTION = 9
SENTINEL_CONTEXT_MARGIN_M = 600.0
SENTINEL_MAX_LOCAL_BAD_FRACTION = 0.20
MIN_SINGLE_SOURCE_COVERAGE_FRACTION = 0.995
SENTINEL_IMAGE_SIZE = 128
SENTINEL_MAX_TIMESTEPS = 12
SENTINEL_LOOKBACK_DAYS = 400
OLMO_INPUT_RES_M = 10
OLMO_PATCH_SIZE = 8

# Sentinel-2 L2A SCL classes.  Same BAD_CLASSES as the old Silver/Gold path.
SCL_NO_DATA = 0
SCL_SATURATED_OR_DEFECTIVE = 1
SCL_DARK_AREA = 2
SCL_CLOUD_SHADOW = 3
SCL_VEGETATION = 4
SCL_NOT_VEGETATED = 5
SCL_WATER = 6
SCL_UNCLASSIFIED = 7
SCL_CLOUD_MEDIUM_PROBABILITY = 8
SCL_CLOUD_HIGH_PROBABILITY = 9
SCL_THIN_CIRRUS = 10
SCL_SNOW_OR_ICE = 11
CLOUD_CLASSES = {8, 9, 10}
SHADOW_CLASSES = {3}
INVALID_CLASSES = {0, 1}
SNOW_CLASSES = {11}
BAD_CLASSES = CLOUD_CLASSES | SHADOW_CLASSES | INVALID_CLASSES | SNOW_CLASSES

# Exact output columns used by foundation_v1 Sentinel parquet shards.
OUTPUT_COLUMNS = [
    "source",
    "h3_cell",
    "h3_resolution",
    "capture_period",
    "capture_timestamp_utc",
    "valid_from_utc",
    "valid_to_utc",
    "item_id",
    "gcs_uri",  # legacy field name; extension stores a source URI/prefix here.
    "coverage_fraction",
    "requires_mosaic",
    "local_bad_fraction",
    "local_clear_fraction",
    "s2_processing_baseline",
    "temporal_sequence_length",
    "sentinel_input_mode",
    "embedding",
    "embedding_dim",
    "encoder_name",
    "encoder_version",
]

# Columns needed from existing Gold.  Excludes the large embedding vector.
EXISTING_META_COLUMNS = [
    "source",
    "h3_cell",
    "h3_resolution",
    "capture_period",
    "capture_timestamp_utc",
    "valid_from_utc",
    "valid_to_utc",
    "item_id",
    "gcs_uri",
    "coverage_fraction",
    "requires_mosaic",
    "local_bad_fraction",
    "local_clear_fraction",
    "s2_processing_baseline",
]

# Binary frame cache: 12 x 128 x 128 uint16 + 128 x 128 uint8 SCL.
SPECTRAL_DN_SHAPE = (SENTINEL_IMAGE_SIZE, SENTINEL_IMAGE_SIZE, len(SPECTRAL_BANDS))
SCL_SHAPE = (SENTINEL_IMAGE_SIZE, SENTINEL_IMAGE_SIZE)
SPECTRAL_DN_BYTES = int(np.prod(SPECTRAL_DN_SHAPE)) * np.dtype(np.uint16).itemsize
SCL_BYTES = int(np.prod(SCL_SHAPE)) * np.dtype(np.uint8).itemsize


# -----------------------------------------------------------------------------
# Logging / utilities
# -----------------------------------------------------------------------------

_PRINT_LOCK = threading.Lock()
_TLS = threading.local()


def log(msg: str) -> None:
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with _PRINT_LOCK:
        print(f"[{now}] {msg}", flush=True)


def die(msg: str) -> "NoReturn":
    raise SystemExit(msg)


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso_z(x: dt.datetime) -> str:
    if x.tzinfo is None:
        x = x.replace(tzinfo=dt.timezone.utc)
    return x.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def parse_datetime(value: str | dt.datetime) -> dt.datetime:
    if isinstance(value, dt.datetime):
        x = value
    else:
        x = dt.datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    if x.tzinfo is None:
        x = x.replace(tzinfo=dt.timezone.utc)
    return x.astimezone(dt.timezone.utc)


def safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def stable_bucket(h3_cell: str, bucket_count: int) -> int:
    # Avoid Python hash randomization; this remains stable across hosts/runs.
    digest = hashlib.blake2b(h3_cell.encode("ascii"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % bucket_count


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def run(cmd: Sequence[str], *, env: dict[str, str] | None = None) -> None:
    log("RUN " + " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), check=True, env=env)


def run_capture(cmd: Sequence[str]) -> str:
    log("RUN " + " ".join(map(str, cmd)))
    p = subprocess.run(list(map(str, cmd)), check=True, text=True, capture_output=True)
    return p.stdout


def rclone_exists() -> None:
    if shutil.which("rclone") is None:
        die("rclone is required but was not found on PATH")


def raise_nofile_limit(target: int = 131072) -> None:
    """Raise the per-process FD limit for aggressive Rasterio scene caches."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        desired = min(max(soft, target), hard)
        if desired > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (desired, hard))
        new_soft, new_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        log(f"RLIMIT_NOFILE soft={new_soft:,} hard={new_hard:,}")
    except Exception as exc:
        log(f"WARN could not raise RLIMIT_NOFILE: {exc}")


def parse_item_capture_time(item_id: str) -> dt.datetime:
    m = re.search(r"_MSIL2A_(\d{8}T\d{6})_", item_id)
    if not m:
        raise ValueError(f"Cannot parse capture timestamp from Sentinel item id: {item_id}")
    x = dt.datetime.strptime(m.group(1), "%Y%m%dT%H%M%S")
    return x.replace(tzinfo=dt.timezone.utc)


def mgrs_from_item_id(item_id: str) -> str | None:
    m = re.search(r"_T([0-9]{2}[A-Z]{3})_", item_id)
    return m.group(1) if m else None


def capture_period(x: dt.datetime) -> str:
    return f"{x.year:04d}-{x.month:02d}"


def parse_processing_baseline(value: Any) -> float | None:
    if value is None:
        return None
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def sentinel_dn_to_reflectance(arr: np.ndarray, processing_baseline: Any) -> np.ndarray:
    """Exact foundation_v1 Sentinel L2A DN -> BOA reflectance conversion."""
    x = arr.astype(np.float32, copy=False)
    nodata = x == 0
    baseline = parse_processing_baseline(processing_baseline)
    if baseline is not None and baseline >= 4.0:
        x = (x - 1000.0) / 10000.0
    else:
        x = x / 10000.0
    x[nodata] = 0.0
    x = np.nan_to_num(x, nan=0.0, posinf=1.5, neginf=-0.2)
    return np.clip(x, -0.2, 1.5)


def fill_bad_pixels_with_band_median(
    spectral_hwc: np.ndarray,
    bad_mask: np.ndarray,
) -> np.ndarray:
    out = spectral_hwc.copy()
    good = ~bad_mask
    if not np.any(good):
        raise ValueError("Sentinel crop contains no SCL-valid pixels")
    for c in range(out.shape[-1]):
        values = out[..., c][good]
        values = values[np.isfinite(values)]
        fill = float(np.median(values)) if values.size else 0.0
        out[..., c][bad_mask] = fill
    return out


def scl_quality(scl_window: np.ndarray) -> dict[str, float]:
    if scl_window.size == 0:
        return {
            "local_cloud_fraction": 1.0,
            "local_shadow_fraction": 0.0,
            "local_invalid_fraction": 1.0,
            "local_snow_fraction": 0.0,
            "local_bad_fraction": 1.0,
            "local_clear_fraction": 0.0,
        }
    cloud = np.isin(scl_window, tuple(CLOUD_CLASSES))
    shadow = np.isin(scl_window, tuple(SHADOW_CLASSES))
    invalid = np.isin(scl_window, tuple(INVALID_CLASSES))
    snow = np.isin(scl_window, tuple(SNOW_CLASSES))
    bad = cloud | shadow | invalid | snow
    return {
        "local_cloud_fraction": float(cloud.mean()),
        "local_shadow_fraction": float(shadow.mean()),
        "local_invalid_fraction": float(invalid.mean()),
        "local_snow_fraction": float(snow.mean()),
        "local_bad_fraction": float(bad.mean()),
        "local_clear_fraction": float(1.0 - bad.mean()),
    }


# -----------------------------------------------------------------------------
# Configuration / paths
# -----------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Paths:
    work: Path
    raw_b2: Path
    existing_gold: Path
    inventory: Path
    candidates: Path
    selection: Path
    context_raw: Path
    frames: Path
    embeddings: Path
    ledgers: Path
    final: Path
    manifests: Path


def make_paths(work_dir: str) -> Paths:
    work = Path(work_dir).expanduser().resolve()
    return Paths(
        work=ensure_dir(work),
        raw_b2=ensure_dir(work / "raw_b2"),
        existing_gold=ensure_dir(work / "existing_gold"),
        inventory=ensure_dir(work / "inventory"),
        candidates=ensure_dir(work / "candidates"),
        selection=ensure_dir(work / "selection"),
        context_raw=ensure_dir(work / "context_raw"),
        frames=ensure_dir(work / "frames"),
        embeddings=ensure_dir(work / "embeddings"),
        ledgers=ensure_dir(work / "ledgers"),
        final=ensure_dir(work / "final"),
        manifests=ensure_dir(work / "manifests"),
    )


# -----------------------------------------------------------------------------
# Disk-first staging
# -----------------------------------------------------------------------------

def remote_size_bytes(remote: str) -> int | None:
    try:
        payload = json.loads(run_capture(["rclone", "size", remote, "--json", "--fast-list"]))
        return int(payload.get("bytes", 0))
    except Exception as exc:
        log(f"WARN unable to obtain rclone size for {remote}: {exc}")
        return None


def stage_remote(
    remote: str,
    local: Path,
    *,
    transfers: int,
    checkers: int,
    multi_thread_streams: int,
    include_parquet_only: bool = False,
) -> None:
    rclone_exists()
    ensure_dir(local)
    cmd = [
        "rclone", "copy", remote, str(local),
        "--fast-list",
        "--transfers", str(transfers),
        "--checkers", str(checkers),
        "--multi-thread-streams", str(multi_thread_streams),
        "--multi-thread-cutoff", "64M",
        "--buffer-size", "32M",
        "--retries", "20",
        "--low-level-retries", "50",
        "--retries-sleep", "3s",
        "--stats", "30s",
        "--stats-one-line",
    ]
    if include_parquet_only:
        cmd += ["--include", "*.parquet", "--max-depth", "1"]
    run(cmd)


def phase_stage(args: argparse.Namespace, paths: Paths) -> None:
    raw_done = paths.manifests / "stage_b2.done.json"
    if args.resume and raw_done.exists():
        log("stage: B2 source already marked complete; skipping")
    else:
        source_bytes = remote_size_bytes(args.b2_source_remote)
        if source_bytes and not args.skip_disk_check:
            free = shutil.disk_usage(paths.work).free
            # Need the raw source itself plus some operating headroom.  Frame/context
            # caches are checked separately because context size is data-dependent.
            required = int(source_bytes * 1.08)
            if free < required:
                die(
                    f"Not enough free disk for disk-first B2 stage: "
                    f"free={free/1024**4:.2f} TiB, source={source_bytes/1024**4:.2f} TiB. "
                    "Rent a larger Vast volume or pass --skip-disk-check intentionally."
                )
        stage_remote(
            args.b2_source_remote,
            paths.raw_b2,
            transfers=args.rclone_transfers,
            checkers=args.rclone_checkers,
            multi_thread_streams=args.rclone_multi_thread_streams,
        )
        raw_done.write_text(json.dumps({
            "created_at_utc": iso_z(utcnow()),
            "remote": args.b2_source_remote,
            "remote_bytes": source_bytes,
        }, indent=2))

    if args.no_existing_gold:
        return
    gold_done = paths.manifests / "stage_existing_gold.done.json"
    if args.resume and gold_done.exists():
        log("stage: existing Gold already marked complete; skipping")
    else:
        stage_remote(
            args.existing_gold_remote,
            paths.existing_gold,
            transfers=max(8, min(args.rclone_transfers, 32)),
            checkers=max(16, min(args.rclone_checkers, 64)),
            multi_thread_streams=max(2, min(args.rclone_multi_thread_streams, 8)),
            include_parquet_only=True,
        )
        gold_done.write_text(json.dumps({
            "created_at_utc": iso_z(utcnow()),
            "remote": args.existing_gold_remote,
        }, indent=2))


# -----------------------------------------------------------------------------
# B2 inventory / manifest merge
# -----------------------------------------------------------------------------

def load_b2_manifest_metadata(raw_root: Path) -> dict[str, dict[str, Any]]:
    by_item: dict[str, dict[str, Any]] = {}
    manifest_dir = raw_root / "manifests"
    files = sorted(manifest_dir.glob("*.json")) if manifest_dir.exists() else []
    log(f"inventory: scanning {len(files):,} B2 selection manifest JSON files")
    for path in files:
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            log(f"WARN manifest parse failed {path}: {exc}")
            continue
        for scene in payload.get("scenes", []) or []:
            item_id = scene.get("item_id")
            if not item_id:
                continue
            old = by_item.get(item_id, {})
            merged = dict(old)
            for k, v in scene.items():
                if v is not None:
                    merged[k] = v
            by_item[item_id] = merged
    return by_item


def phase_inventory(args: argparse.Namespace, paths: Paths) -> None:
    out = paths.inventory / "scenes.parquet"
    if args.resume and out.exists():
        log(f"inventory: {out} exists; skipping")
        return

    metadata = load_b2_manifest_metadata(paths.raw_b2)
    l2a = paths.raw_b2 / "l2a"
    if not l2a.exists():
        die(f"Expected staged B2 l2a directory at {l2a}")

    scene_files: dict[str, dict[str, Path]] = defaultdict(dict)
    scene_meta_from_path: dict[str, dict[str, str]] = {}
    pattern = re.compile(
        r"mgrs_tile=([^/]+)/capture_date=([^/]+)/item_id=([^/]+)/([^/]+)$"
    )
    for tif in l2a.rglob("*.tif"):
        rel = tif.relative_to(l2a).as_posix()
        m = pattern.search(rel)
        if not m:
            continue
        tile, date, item_id, filename = m.groups()
        asset = Path(filename).stem
        scene_files[item_id][asset] = tif
        scene_meta_from_path[item_id] = {"mgrs_tile": tile, "capture_date": date}

    rows: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    for item_id, files in sorted(scene_files.items()):
        path_meta = scene_meta_from_path[item_id]
        meta = metadata.get(item_id, {})
        missing = sorted(set(REQUIRED_ASSETS) - set(files))
        capture_time_raw = meta.get("capture_time_utc")
        capture = (
            parse_datetime(capture_time_raw)
            if capture_time_raw
            else parse_item_capture_time(item_id)
        )
        mgrs_tile = str(meta.get("mgrs_tile") or path_meta["mgrs_tile"])
        bbox = meta.get("bbox")
        row: dict[str, Any] = {
            "item_id": item_id,
            "mgrs_tile": mgrs_tile,
            "capture_timestamp_utc": capture,
            "capture_date": capture.date().isoformat(),
            "capture_period": capture_period(capture),
            "eo_cloud_cover": (
                float(meta["eo_cloud_cover"])
                if meta.get("eo_cloud_cover") is not None
                else None
            ),
            "s2_processing_baseline": (
                str(meta["s2_processing_baseline"])
                if meta.get("s2_processing_baseline") is not None
                else None
            ),
            "platform": meta.get("platform"),
            "bbox": bbox,
            "complete": len(missing) == 0,
            "missing_assets": missing,
            "scene_dir": str(next(iter(files.values())).parent),
            "source_uri": (
                args.b2_source_remote.rstrip("/")
                + "/l2a/"
                + f"mgrs_tile={safe_component(mgrs_tile)}/"
                + f"capture_date={capture.date().isoformat()}/"
                + f"item_id={safe_component(item_id)}"
            ),
        }
        for asset in REQUIRED_ASSETS:
            row[f"path_{asset}"] = str(files[asset]) if asset in files else None
        rows.append(row)
        if missing:
            rejects.append({
                "item_id": item_id,
                "mgrs_tile": mgrs_tile,
                "capture_timestamp_utc": capture,
                "missing_assets": missing,
            })

    df = pl.DataFrame(rows, infer_schema_length=None).sort(
        ["capture_timestamp_utc", "mgrs_tile", "item_id"]
    )
    df.write_parquet(out, compression="zstd", compression_level=3, statistics=True)
    if rejects:
        pl.DataFrame(rejects, infer_schema_length=None).write_parquet(
            paths.inventory / "rejected_incomplete_scenes.parquet",
            compression="zstd",
            compression_level=3,
        )
    complete = df.filter(pl.col("complete") == True).height
    log(
        f"inventory: scenes={df.height:,}, complete={complete:,}, "
        f"incomplete={df.height-complete:,}, tiles={df['mgrs_tile'].n_unique():,}, "
        f"date={df['capture_date'].min()}..{df['capture_date'].max()}"
    )


# -----------------------------------------------------------------------------
# Existing Gold metadata / H3 universe
# -----------------------------------------------------------------------------

def load_h3_manifest(path: str | None) -> pl.DataFrame | None:
    if not path:
        return None
    p = Path(path).expanduser().resolve()
    if not p.exists():
        die(f"H3 manifest does not exist: {p}")
    if p.suffix.lower() == ".parquet":
        df = pl.read_parquet(p)
    elif p.suffix.lower() in {".csv", ".tsv"}:
        df = pl.read_csv(p, separator="\t" if p.suffix.lower() == ".tsv" else ",")
    else:
        values = [x.strip() for x in p.read_text().splitlines() if x.strip()]
        df = pl.DataFrame({"h3_cell": values})
    if "h3_cell" not in df.columns:
        if len(df.columns) == 1:
            df = df.rename({df.columns[0]: "h3_cell"})
        else:
            die(f"H3 manifest must contain h3_cell column: {p}")
    return df.select(pl.col("h3_cell").cast(pl.Utf8)).unique().sort("h3_cell")


def ensure_existing_gold_meta(args: argparse.Namespace, paths: Paths) -> pl.DataFrame:
    out = paths.selection / "existing_gold_meta.parquet"
    if out.exists() and args.resume:
        return pl.read_parquet(out)
    files = sorted(paths.existing_gold.rglob("*.parquet"))
    if not files:
        if args.no_existing_gold:
            return pl.DataFrame()
        die(
            "No existing Gold Sentinel parquet files found. Run phase=stage first, "
            "pass --no-existing-gold, or supply --h3-manifest."
        )
    log(f"existing Gold: scanning metadata from {len(files):,} parquet shards")
    lf = pl.scan_parquet(files)
    available_columns = set(lf.collect_schema().names())
    meta = (
        lf.select([c for c in EXISTING_META_COLUMNS if c in available_columns])
        .collect(engine="streaming")
        .sort(["h3_cell", "capture_timestamp_utc", "item_id"])
    )
    meta.write_parquet(out, compression="zstd", compression_level=3, statistics=True)
    log(
        f"existing Gold metadata: rows={meta.height:,}, "
        f"h3={meta['h3_cell'].n_unique():,}, months={meta['capture_period'].n_unique():,}"
    )
    return meta


def target_h3_cells(args: argparse.Namespace, paths: Paths) -> pl.DataFrame:
    custom = load_h3_manifest(args.h3_manifest)
    if custom is not None:
        return custom
    existing = ensure_existing_gold_meta(args, paths)
    if existing.is_empty():
        die("No target H3 universe. Supply --h3-manifest or stage existing Gold.")
    return existing.select("h3_cell").unique().sort("h3_cell")


# -----------------------------------------------------------------------------
# Spatial helpers: reproduce old Silver H3 context windows
# -----------------------------------------------------------------------------

_TRANSFORMER_CACHE: dict[str, Transformer] = {}
_TRANSFORMER_LOCK = threading.Lock()


def transformer_to_crs(crs_text: str) -> Transformer:
    with _TRANSFORMER_LOCK:
        t = _TRANSFORMER_CACHE.get(crs_text)
        if t is None:
            t = Transformer.from_crs("EPSG:4326", CRS.from_user_input(crs_text), always_xy=True)
            _TRANSFORMER_CACHE[crs_text] = t
        return t


def meters_to_crs_units(crs: CRS, meters: float) -> float:
    if meters == 0:
        return 0.0
    if not crs.is_projected:
        raise ValueError(f"Expected projected imagery CRS, got {crs.to_string()}")
    axis_info = crs.axis_info
    if not axis_info:
        return meters
    factor = axis_info[0].unit_conversion_factor
    if factor is None or factor <= 0:
        return meters
    return meters / float(factor)


def h3_context_bounds_in_crs(
    h3_cell: str,
    crs_text: str,
    context_margin_m: float = SENTINEL_CONTEXT_MARGIN_M,
) -> tuple[float, float, float, float]:
    crs = CRS.from_user_input(crs_text)
    transformer = transformer_to_crs(crs_text)
    boundary = h3.cell_to_boundary(h3_cell)
    xs: list[float] = []
    ys: list[float] = []
    for lat, lon in boundary:
        x, y = transformer.transform(lon, lat)
        xs.append(float(x))
        ys.append(float(y))
    margin_units = meters_to_crs_units(crs, context_margin_m)
    return (
        min(xs) - margin_units,
        min(ys) - margin_units,
        max(xs) + margin_units,
        max(ys) + margin_units,
    )


@dataclasses.dataclass(frozen=True)
class RasterWindowInfo:
    col_off: int
    row_off: int
    width: int
    height: int
    coverage_fraction: float

    @property
    def window(self) -> Window:
        return Window(self.col_off, self.row_off, self.width, self.height)


def window_from_requested_bounds(
    src: rasterio.io.DatasetReader,
    requested_bounds: tuple[float, float, float, float],
) -> RasterWindowInfo | None:
    left, bottom, right, top = requested_bounds
    if right <= left or top <= bottom:
        return None
    raw = from_bounds(left, bottom, right, top, transform=src.transform)
    req_col0 = math.floor(raw.col_off)
    req_row0 = math.floor(raw.row_off)
    req_col1 = math.ceil(raw.col_off + raw.width)
    req_row1 = math.ceil(raw.row_off + raw.height)
    full_width = max(0, req_col1 - req_col0)
    full_height = max(0, req_row1 - req_row0)
    if full_width == 0 or full_height == 0:
        return None
    col0 = max(0, req_col0)
    row0 = max(0, req_row0)
    col1 = min(src.width, req_col1)
    row1 = min(src.height, req_row1)
    width = max(0, col1 - col0)
    height = max(0, row1 - row0)
    if width == 0 or height == 0:
        return None
    coverage = (width * height) / float(full_width * full_height)
    return RasterWindowInfo(
        int(col0), int(row0), int(width), int(height),
        float(min(1.0, max(0.0, coverage))),
    )


def center_window(row_window: Window, size: int = SENTINEL_IMAGE_SIZE) -> Window:
    width = min(int(row_window.width), size)
    height = min(int(row_window.height), size)
    col_off = int(row_window.col_off + max(0, (row_window.width - width) // 2))
    row_off = int(row_window.row_off + max(0, (row_window.height - height) // 2))
    return Window(col_off=col_off, row_off=row_off, width=width, height=height)


# -----------------------------------------------------------------------------
# Candidate generation: SCL-heavy CPU phase
# -----------------------------------------------------------------------------

def scene_bbox(scene: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = scene.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 4:
        return tuple(map(float, bbox[:4]))  # type: ignore[return-value]
    b02 = scene.get("path_B02")
    if not b02:
        return None
    with rasterio.open(b02) as ds:
        if ds.crs is None:
            return None
        return transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21)


def map_scenes_to_h3(
    scenes: pl.DataFrame,
    h3_df: pl.DataFrame,
    bbox_pad_deg: float = 0.03,
) -> dict[str, list[str]]:
    cells = h3_df["h3_cell"].to_list()
    lat = np.empty(len(cells), dtype=np.float64)
    lon = np.empty(len(cells), dtype=np.float64)
    for i, cell in enumerate(cells):
        la, lo = h3.cell_to_latlng(cell)
        lat[i] = la
        lon[i] = lo

    tile_cells: dict[str, list[str]] = {}
    # Same MGRS tile always shares the same footprint; evaluate one scene per tile.
    for tile, group in scenes.group_by("mgrs_tile", maintain_order=True):
        tile_value = tile[0] if isinstance(tile, tuple) else tile
        rec = group.row(0, named=True)
        bbox = scene_bbox(rec)
        if bbox is None:
            tile_cells[str(tile_value)] = []
            continue
        left, bottom, right, top = bbox
        lat_mask = (lat >= bottom - bbox_pad_deg) & (lat <= top + bbox_pad_deg)
        if left <= right:
            lon_mask = (lon >= left - bbox_pad_deg) & (lon <= right + bbox_pad_deg)
        else:
            # Antimeridian-crossing tile bbox.
            lon_mask = (lon >= left - bbox_pad_deg) | (lon <= right + bbox_pad_deg)
        mask = lon_mask & lat_mask
        idx = np.nonzero(mask)[0]
        tile_cells[str(tile_value)] = [cells[int(i)] for i in idx]
    return tile_cells


def candidate_rows_for_scene(
    scene: dict[str, Any],
    h3_cells: list[str],
) -> list[dict[str, Any]]:
    if not h3_cells:
        return []
    b02_path = scene["path_B02"]
    scl_path = scene["path_SCL"]
    rows: list[dict[str, Any]] = []
    with rasterio.Env(GDAL_CACHEMAX=512):
        with rasterio.open(b02_path, sharing=False) as b02, rasterio.open(scl_path, sharing=False) as scl_ds:
            if b02.crs is None or scl_ds.crs is None:
                raise ValueError(f"Scene has missing CRS: {scene['item_id']}")
            if b02.crs != scl_ds.crs:
                raise ValueError(f"B02/SCL CRS mismatch: {scene['item_id']}")
            crs_text = b02.crs.to_string()
            # SCL is ~30 MB at native 20 m.  One full read amortizes thousands of
            # tiny windows and is cheap on a hundreds-of-GB Vast host.
            scl_full = scl_ds.read(1, out_dtype="uint8")

            for cell in h3_cells:
                if h3.get_resolution(cell) != TARGET_H3_RESOLUTION:
                    continue
                requested = h3_context_bounds_in_crs(cell, crs_text)
                base = window_from_requested_bounds(b02, requested)
                if base is None:
                    continue
                scl_win = window_from_requested_bounds(scl_ds, requested)
                if scl_win is None:
                    quality = scl_quality(np.empty((0, 0), dtype=np.uint8))
                else:
                    r0 = scl_win.row_off
                    r1 = r0 + scl_win.height
                    c0 = scl_win.col_off
                    c1 = c0 + scl_win.width
                    quality = scl_quality(scl_full[r0:r1, c0:c1])

                usable = quality["local_bad_fraction"] <= SENTINEL_MAX_LOCAL_BAD_FRACTION
                rows.append({
                    "source": "sentinel2",
                    "collection": SENTINEL_COLLECTION,
                    "item_id": scene["item_id"],
                    "capture_timestamp_utc": scene["capture_timestamp_utc"],
                    "capture_period": scene["capture_period"],
                    "h3_cell": cell,
                    "h3_resolution": TARGET_H3_RESOLUTION,
                    "source_uri": scene["source_uri"],
                    "mgrs_tile": scene["mgrs_tile"],
                    "raster_crs": crs_text,
                    "raster_width": int(b02.width),
                    "raster_height": int(b02.height),
                    "pixel_size_x": float(abs(b02.transform.a)),
                    "pixel_size_y": float(abs(b02.transform.e)),
                    "window_col_off": base.col_off,
                    "window_row_off": base.row_off,
                    "window_width": base.width,
                    "window_height": base.height,
                    "coverage_fraction": base.coverage_fraction,
                    "requires_mosaic": base.coverage_fraction < MIN_SINGLE_SOURCE_COVERAGE_FRACTION,
                    "eo_cloud_cover": scene.get("eo_cloud_cover"),
                    **quality,
                    "is_usable": usable,
                    "s2_processing_baseline": scene.get("s2_processing_baseline"),
                    "error": None,
                })
    return rows


def flush_candidate_buffer(buffer: list[dict[str, Any]], out_dir: Path, part_idx: int) -> int:
    if not buffer:
        return part_idx
    path = out_dir / f"part-{part_idx:05d}.parquet"
    df = pl.DataFrame(buffer, infer_schema_length=None).with_columns(
        pl.col("source").cast(pl.Utf8),
        pl.col("collection").cast(pl.Utf8),
        pl.col("item_id").cast(pl.Utf8),
        pl.col("capture_timestamp_utc").cast(pl.Datetime("us", "UTC")),
        pl.col("capture_period").cast(pl.Utf8),
        pl.col("h3_cell").cast(pl.Utf8),
        pl.col("h3_resolution").cast(pl.Int8),
        pl.col("source_uri").cast(pl.Utf8),
        pl.col("mgrs_tile").cast(pl.Utf8),
        pl.col("raster_crs").cast(pl.Utf8),
        pl.col("coverage_fraction").cast(pl.Float64),
        pl.col("requires_mosaic").cast(pl.Boolean),
        pl.col("eo_cloud_cover").cast(pl.Float64),
        pl.col("local_cloud_fraction").cast(pl.Float64),
        pl.col("local_shadow_fraction").cast(pl.Float64),
        pl.col("local_invalid_fraction").cast(pl.Float64),
        pl.col("local_snow_fraction").cast(pl.Float64),
        pl.col("local_bad_fraction").cast(pl.Float64),
        pl.col("local_clear_fraction").cast(pl.Float64),
        pl.col("is_usable").cast(pl.Boolean),
        pl.col("s2_processing_baseline").cast(pl.Utf8),
        pl.col("error").cast(pl.Utf8),
    )
    df.write_parquet(
        path,
        compression="zstd",
        compression_level=3,
        statistics=True,
    )
    log(f"candidates: wrote {len(buffer):,} rows -> {path.name}")
    buffer.clear()
    return part_idx + 1


def phase_candidates(args: argparse.Namespace, paths: Paths) -> None:
    done = paths.manifests / "candidates.done.json"
    if args.resume and done.exists():
        log("candidates: completion marker exists; skipping")
        return
    for p in paths.candidates.glob("part-*.parquet"):
        p.unlink()

    scenes = pl.read_parquet(paths.inventory / "scenes.parquet").filter(pl.col("complete") == True)
    h3_df = target_h3_cells(args, paths)
    log(f"candidates: target H3 cells={h3_df.height:,}; complete scenes={scenes.height:,}")
    tile_cells = map_scenes_to_h3(scenes, h3_df)

    tasks: list[tuple[dict[str, Any], list[str]]] = []
    skipped_zero = 0
    for scene in scenes.iter_rows(named=True):
        cells = tile_cells.get(scene["mgrs_tile"], [])
        if not cells:
            skipped_zero += 1
            continue
        tasks.append((scene, cells))
    log(
        f"candidates: scenes intersecting target universe={len(tasks):,}; "
        f"scenes skipped with zero nearby targets={skipped_zero:,}; workers={args.candidate_workers}"
    )

    buffer: list[dict[str, Any]] = []
    part_idx = 0
    total_rows = 0
    errors: list[dict[str, str]] = []
    started = time.monotonic()

    with cf.ThreadPoolExecutor(
        max_workers=args.candidate_workers,
        thread_name_prefix="scl-candidate",
    ) as pool:
        future_map = {
            pool.submit(candidate_rows_for_scene, scene, cells): scene["item_id"]
            for scene, cells in tasks
        }
        completed = 0
        for fut in cf.as_completed(future_map):
            item_id = future_map[fut]
            try:
                rows = fut.result()
                buffer.extend(rows)
                total_rows += len(rows)
            except Exception as exc:
                errors.append({"item_id": item_id, "error": repr(exc)})
                log(f"ERROR candidate scene {item_id}: {exc}")
                if args.strict:
                    raise
            completed += 1
            if len(buffer) >= args.candidate_rows_per_shard:
                part_idx = flush_candidate_buffer(buffer, paths.candidates, part_idx)
            if completed % 50 == 0 or completed == len(tasks):
                elapsed = max(time.monotonic() - started, 1e-9)
                log(
                    f"candidates progress: scenes={completed:,}/{len(tasks):,}, "
                    f"rows={total_rows:,}, scenes/s={completed/elapsed:.2f}"
                )
    part_idx = flush_candidate_buffer(buffer, paths.candidates, part_idx)
    if errors:
        pl.DataFrame(errors).write_parquet(paths.candidates / "errors.parquet")
    if args.strict and errors:
        die(f"Candidate generation had {len(errors)} scene errors")
    done.write_text(json.dumps({
        "created_at_utc": iso_z(utcnow()),
        "rows": total_rows,
        "shards": part_idx,
        "scene_errors": len(errors),
        "target_h3_cells": h3_df.height,
    }, indent=2))


# -----------------------------------------------------------------------------
# Exact monthly selection + temporal plan
# -----------------------------------------------------------------------------

def phase_select(args: argparse.Namespace, paths: Paths) -> None:
    selected_path = paths.selection / "selected_b2.parquet"
    plan_path = paths.selection / "frame_plan.parquet"
    if args.resume and selected_path.exists() and plan_path.exists():
        log("select: outputs exist; skipping")
        return

    candidate_files = sorted(paths.candidates.glob("part-*.parquet"))
    if not candidate_files:
        die("No candidate shards. Run phase=candidates first.")

    log(f"select: reading {len(candidate_files):,} candidate shards")
    candidates = pl.scan_parquet(candidate_files).filter(
        (pl.col("source") == "sentinel2")
        & (pl.col("is_usable") == True)
        & pl.col("error").is_null()
    )
    # Exact old _select_best_sentinel_per_month ordering.
    selected = (
        candidates
        .sort(
            [
                "h3_cell",
                "capture_period",
                "local_bad_fraction",
                "coverage_fraction",
                "eo_cloud_cover",
                "capture_timestamp_utc",
                "item_id",
            ],
            descending=[False, False, False, True, False, True, False],
            nulls_last=True,
        )
        .with_columns(
            pl.int_range(1, pl.len() + 1)
            .over(["h3_cell", "capture_period"])
            .cast(pl.Int32)
            .alias("candidate_rank")
        )
        .filter(pl.col("candidate_rank") == 1)
        .with_columns(
            pl.lit(True).alias("selected_in_period"),
            pl.col("capture_timestamp_utc").alias("valid_from_utc"),
        )
        .collect(engine="streaming")
        .sort(["h3_cell", "capture_timestamp_utc", "item_id"])
    )
    selected.write_parquet(selected_path, compression="zstd", compression_level=3, statistics=True)
    log(
        f"select: B2 winners={selected.height:,}, h3={selected['h3_cell'].n_unique():,}, "
        f"months={selected['capture_period'].n_unique():,}"
    )

    existing = pl.DataFrame()
    if not args.no_existing_gold:
        existing = ensure_existing_gold_meta(args, paths)

    if existing.is_empty():
        new_rows = selected.with_columns(pl.lit(True).alias("emit_target"))
        context_old = pl.DataFrame()
    else:
        existing_keys = existing.select(["h3_cell", "capture_period"]).unique()
        new_rows = (
            selected.join(existing_keys, on=["h3_cell", "capture_period"], how="anti")
            .with_columns(pl.lit(True).alias("emit_target"))
        )
        log(
            f"select: skipped {selected.height-new_rows.height:,} B2 H3/month winners "
            "because the month already exists in foundation_v1"
        )

        # Pull only the old rows capable of affecting a new target's <=400-day,
        # <=12-observation history.  Existing monthly selections remain canonical.
        min_new = (
            new_rows.group_by("h3_cell")
            .agg(pl.col("capture_timestamp_utc").min().alias("min_new_ts"))
        )
        context_old = (
            existing.join(min_new, on="h3_cell", how="inner")
            .filter(
                (pl.col("capture_timestamp_utc") < pl.col("min_new_ts"))
                & (
                    pl.col("capture_timestamp_utc")
                    >= pl.col("min_new_ts") - dt.timedelta(days=SENTINEL_LOOKBACK_DAYS)
                )
            )
            .drop("min_new_ts")
            .with_columns(
                pl.lit(False).alias("emit_target"),
                pl.lit("existing_context").alias("frame_origin"),
                pl.col("gcs_uri").alias("source_uri"),
            )
        )

    if new_rows.is_empty():
        log("select: no new H3/month targets remain after existing-Gold anti-join")
        pl.DataFrame().write_parquet(plan_path)
        return

    new_rows = new_rows.with_columns(
        pl.lit("b2").alias("frame_origin")
    )

    # Recompute validity over existing + new monthly selections.  We do not mutate
    # old shards; changed old intervals are emitted as a sidecar patch.
    if existing.is_empty():
        combined_meta = new_rows.select([
            "h3_cell", "capture_period", "capture_timestamp_utc", "item_id"
        ])
        valid = (
            combined_meta.sort(["h3_cell", "capture_timestamp_utc", "item_id"])
            .with_columns(
                pl.col("capture_timestamp_utc").alias("valid_from_utc"),
                pl.col("capture_timestamp_utc").shift(-1).over("h3_cell").alias("valid_to_utc_new"),
            )
        )
    else:
        combined_meta = pl.concat([
            existing.select(["h3_cell", "capture_period", "capture_timestamp_utc", "item_id"]),
            new_rows.select(["h3_cell", "capture_period", "capture_timestamp_utc", "item_id"]),
        ], how="vertical_relaxed")
        valid = (
            combined_meta.sort(["h3_cell", "capture_timestamp_utc", "item_id"])
            .with_columns(
                pl.col("capture_timestamp_utc").alias("valid_from_utc"),
                pl.col("capture_timestamp_utc").shift(-1).over("h3_cell").alias("valid_to_utc_new"),
            )
        )

    new_rows = (
        new_rows.drop([c for c in ["valid_to_utc"] if c in new_rows.columns])
        .join(
            valid.select(["h3_cell", "item_id", "valid_from_utc", "valid_to_utc_new"]),
            on=["h3_cell", "item_id"],
            how="left",
        )
        .rename({"valid_to_utc_new": "valid_to_utc"})
    )

    if not existing.is_empty():
        patch = (
            existing.select(["h3_cell", "item_id", "valid_to_utc"])
            .join(
                valid.select(["h3_cell", "item_id", "valid_to_utc_new"]),
                on=["h3_cell", "item_id"],
                how="inner",
            )
            .filter(
                (pl.col("valid_to_utc").is_null() & pl.col("valid_to_utc_new").is_not_null())
                | (
                    pl.col("valid_to_utc").is_not_null()
                    & pl.col("valid_to_utc_new").is_not_null()
                    & (pl.col("valid_to_utc") != pl.col("valid_to_utc_new"))
                )
            )
        )
        patch.write_parquet(
            paths.selection / "existing_validity_patch.parquet",
            compression="zstd",
            compression_level=3,
        )
        log(f"select: existing validity rows requiring patch={patch.height:,}")

    # Only H3s with new targets need context frames.
    if context_old.is_empty():
        plan = new_rows
    else:
        common = sorted(set(context_old.columns) & set(new_rows.columns))
        # Keep a stable superset by diagonal concat; missing fields are filled null.
        plan = pl.concat([context_old, new_rows], how="diagonal_relaxed")

    plan = (
        plan
        .with_columns(
            pl.col("h3_cell").map_elements(
                lambda x: stable_bucket(x, args.frame_buckets),
                return_dtype=pl.Int32,
            ).alias("frame_bucket")
        )
        .sort(["frame_bucket", "h3_cell", "capture_timestamp_utc", "item_id"])
    )
    plan.write_parquet(plan_path, compression="zstd", compression_level=3, statistics=True)
    new_rows.write_parquet(
        paths.selection / "new_targets.parquet",
        compression="zstd",
        compression_level=3,
        statistics=True,
    )
    log(
        f"select: frame plan rows={plan.height:,} "
        f"(context={plan.filter(pl.col('emit_target') == False).height:,}, "
        f"targets={plan.filter(pl.col('emit_target') == True).height:,}), "
        f"buckets={args.frame_buckets}"
    )


# -----------------------------------------------------------------------------
# Historical context hydration from Planetary Computer
# -----------------------------------------------------------------------------

def requests_session():
    import requests

    sess = getattr(_TLS, "requests_session", None)
    if sess is None:
        sess = requests.Session()
        sess.headers.update({"User-Agent": "CrimeNet-Sentinel-Vast/1.0"})
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=64,
            pool_maxsize=64,
            max_retries=0,
        )
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        _TLS.requests_session = sess
    return sess


def pc_scene_metadata(item_ids: list[str], batch_size: int = 100) -> dict[str, dict[str, Any]]:
    try:
        import pystac_client
    except ImportError as exc:
        raise RuntimeError("pystac-client is required for historical context hydration") from exc

    catalog = pystac_client.Client.open(STAC_URL)
    out: dict[str, dict[str, Any]] = {}
    for start in range(0, len(item_ids), batch_size):
        batch = item_ids[start : start + batch_size]
        search = catalog.search(collections=[SENTINEL_COLLECTION], ids=batch)
        for item in search.items():
            missing = [a for a in REQUIRED_ASSETS if a not in item.assets]
            if missing:
                log(f"WARN PC context item {item.id} missing assets {missing}")
                continue
            props = item.properties
            capture = item.datetime or parse_item_capture_time(item.id)
            out[item.id] = {
                "item_id": item.id,
                "mgrs_tile": (
                    str(props.get("s2:mgrs_tile") or mgrs_from_item_id(item.id) or "")
                    .removeprefix("T")
                ),
                "capture_timestamp_utc": parse_datetime(capture),
                "capture_date": parse_datetime(capture).date().isoformat(),
                "eo_cloud_cover": (
                    float(props["eo:cloud_cover"])
                    if props.get("eo:cloud_cover") is not None
                    else None
                ),
                "s2_processing_baseline": (
                    str(props["s2:processing_baseline"])
                    if props.get("s2:processing_baseline") is not None
                    else None
                ),
                "bbox": list(item.bbox) if item.bbox else None,
                "assets": {a: item.assets[a].href for a in REQUIRED_ASSETS},
            }
        log(
            f"context STAC metadata: {min(start+batch_size, len(item_ids)):,}/{len(item_ids):,} ids; "
            f"resolved={len(out):,}"
        )
    return out


def signed_pc_href(raw_href: str) -> str:
    try:
        import planetary_computer
    except ImportError as exc:
        raise RuntimeError("planetary-computer is required for context downloads") from exc
    return planetary_computer.sign(raw_href)


def download_pc_asset(
    raw_href: str,
    local_path: Path,
    *,
    attempts: int = 8,
    expected_min_bytes: int = 1024,
) -> int:
    ensure_dir(local_path.parent)
    for attempt in range(1, attempts + 1):
        existing = local_path.stat().st_size if local_path.exists() else 0
        try:
            href = signed_pc_href(raw_href)
            headers: dict[str, str] = {}
            mode = "wb"
            if existing > 0:
                headers["Range"] = f"bytes={existing}-"
            with requests_session().get(
                href,
                headers=headers,
                stream=True,
                timeout=(30, 300),
            ) as r:
                if existing > 0 and r.status_code == 206:
                    mode = "ab"
                elif existing > 0 and r.status_code == 200:
                    mode = "wb"
                r.raise_for_status()
                with local_path.open(mode) as f:
                    for chunk in r.iter_content(chunk_size=16 * 1024**2):
                        if chunk:
                            f.write(chunk)
            size = local_path.stat().st_size
            if size < expected_min_bytes:
                raise IOError(f"Downloaded file suspiciously small: {size} bytes")
            return size
        except Exception as exc:
            if attempt >= attempts:
                raise
            delay = min(60, 2 ** (attempt - 1))
            log(
                f"PC download retry {attempt}/{attempts} {local_path.name}: "
                f"{type(exc).__name__}: {exc}; sleep={delay}s"
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def phase_context(args: argparse.Namespace, paths: Paths) -> None:
    if args.no_historical_context or args.no_existing_gold:
        log("context: disabled")
        return
    plan_path = paths.selection / "frame_plan.parquet"
    if not plan_path.exists():
        die("No frame plan. Run phase=select first.")
    plan = pl.read_parquet(plan_path)
    if plan.is_empty() or "frame_origin" not in plan.columns:
        log("context: no frame plan rows")
        return
    context_rows = plan.filter(pl.col("frame_origin") == "existing_context")
    if context_rows.is_empty():
        log("context: no existing historical context required")
        return

    inventory_path = paths.inventory / "context_scenes.parquet"
    done = paths.manifests / "context.done.json"
    if args.resume and done.exists() and inventory_path.exists():
        log("context: completion marker exists; skipping downloads")
        return

    item_ids = sorted(set(context_rows["item_id"].to_list()))
    log(f"context: resolving {len(item_ids):,} unique old selected Sentinel scenes")
    meta = pc_scene_metadata(item_ids, batch_size=args.stac_id_batch_size)
    missing_items = sorted(set(item_ids) - set(meta))
    if missing_items:
        msg = f"Planetary Computer could not resolve {len(missing_items):,} required context item_ids"
        if args.strict_context:
            die(msg + f"; first={missing_items[:10]}")
        log("WARN " + msg)

    tasks: list[tuple[str, str, Path]] = []
    scene_rows: list[dict[str, Any]] = []
    for item_id in item_ids:
        scene = meta.get(item_id)
        if scene is None:
            continue
        scene_dir = (
            paths.context_raw
            / f"mgrs_tile={safe_component(scene['mgrs_tile'])}"
            / f"capture_date={scene['capture_date']}"
            / f"item_id={safe_component(item_id)}"
        )
        row = {
            "item_id": item_id,
            "mgrs_tile": scene["mgrs_tile"],
            "capture_timestamp_utc": scene["capture_timestamp_utc"],
            "capture_date": scene["capture_date"],
            "eo_cloud_cover": scene.get("eo_cloud_cover"),
            "s2_processing_baseline": scene.get("s2_processing_baseline"),
            "bbox": scene.get("bbox"),
            "scene_dir": str(scene_dir),
        }
        for asset in REQUIRED_ASSETS:
            href = scene["assets"][asset]
            suffix = Path(urlsplit(href).path).suffix or ".tif"
            local = scene_dir / f"{asset}{suffix.lower()}"
            row[f"path_{asset}"] = str(local)
            if not local.exists() or local.stat().st_size < 1024:
                tasks.append((href, asset, local))
        scene_rows.append(row)

    log(
        f"context: scenes={len(scene_rows):,}; asset files needing download={len(tasks):,}; "
        f"workers={args.context_download_workers}"
    )
    failures: list[tuple[str, str]] = []
    total_bytes = 0
    started = time.monotonic()

    def one(task: tuple[str, str, Path]) -> tuple[Path, int]:
        href, _asset, local = task
        return local, download_pc_asset(href, local)

    with cf.ThreadPoolExecutor(
        max_workers=args.context_download_workers,
        thread_name_prefix="pc-context",
    ) as pool:
        future_map = {pool.submit(one, t): t for t in tasks}
        complete = 0
        for fut in cf.as_completed(future_map):
            task = future_map[fut]
            try:
                local, n = fut.result()
                total_bytes += n
            except Exception as exc:
                failures.append((str(task[2]), repr(exc)))
                log(f"ERROR context asset {task[2]}: {exc}")
                if args.strict_context:
                    raise
            complete += 1
            if complete % 100 == 0 or complete == len(tasks):
                elapsed = max(time.monotonic() - started, 1e-9)
                log(
                    f"context download: {complete:,}/{len(tasks):,}; "
                    f"downloaded={total_bytes/1024**3:.1f} GiB; files/s={complete/elapsed:.2f}"
                )

    # Validate completeness after all resumable downloads.
    valid_rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for row in scene_rows:
        missing = [
            a for a in REQUIRED_ASSETS
            if not Path(row[f"path_{a}"]).exists() or Path(row[f"path_{a}"]).stat().st_size < 1024
        ]
        row["complete"] = not missing
        row["missing_assets"] = missing
        (valid_rows if not missing else invalid).append(row)

    pl.DataFrame(scene_rows, infer_schema_length=None).write_parquet(
        inventory_path,
        compression="zstd",
        compression_level=3,
        statistics=True,
    )
    if invalid:
        pl.DataFrame(invalid, infer_schema_length=None).write_parquet(
            paths.inventory / "context_incomplete.parquet",
            compression="zstd",
            compression_level=3,
        )
    if args.strict_context and invalid:
        die(f"Historical context hydration incomplete for {len(invalid):,} scenes")

    done.write_text(json.dumps({
        "created_at_utc": iso_z(utcnow()),
        "required_item_ids": len(item_ids),
        "resolved_item_ids": len(meta),
        "complete_scenes": len(valid_rows),
        "incomplete_scenes": len(invalid),
        "downloaded_bytes_this_run": total_bytes,
    }, indent=2))


# -----------------------------------------------------------------------------
# Frame cache: native GeoTIFFs -> exact 128x128 DN/SCL crops in Parquet
# -----------------------------------------------------------------------------

def scene_path_map(paths: Paths) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    b2 = pl.read_parquet(paths.inventory / "scenes.parquet")
    for row in b2.filter(pl.col("complete") == True).iter_rows(named=True):
        out[row["item_id"]] = row
    context_path = paths.inventory / "context_scenes.parquet"
    if context_path.exists():
        ctx = pl.read_parquet(context_path)
        for row in ctx.filter(pl.col("complete") == True).iter_rows(named=True):
            out[row["item_id"]] = row
    return out


class NativeSceneBundle:
    """One local native-resolution Sentinel L2A scene, all required band handles open."""

    def __init__(self, scene: dict[str, Any]):
        self.item_id = scene["item_id"]
        self.datasets: dict[str, rasterio.io.DatasetReader] = {}
        for asset in REQUIRED_ASSETS:
            p = scene.get(f"path_{asset}")
            if not p:
                self.close()
                raise FileNotFoundError(f"{self.item_id}: missing local path for {asset}")
            self.datasets[asset] = rasterio.open(p, sharing=False)
        base = self.datasets["B02"]
        if base.crs is None:
            self.close()
            raise ValueError(f"{self.item_id}: B02 has no CRS")
        for asset, ds in self.datasets.items():
            if ds.crs != base.crs:
                self.close()
                raise ValueError(f"{self.item_id}: CRS mismatch {asset}={ds.crs} B02={base.crs}")
        self.crs_text = base.crs.to_string()

    def close(self) -> None:
        for ds in getattr(self, "datasets", {}).values():
            with contextlib.suppress(Exception):
                ds.close()
        if hasattr(self, "datasets"):
            self.datasets.clear()

    def extract_dn_and_scl(self, h3_cell: str) -> tuple[np.ndarray, np.ndarray]:
        b02 = self.datasets["B02"]
        requested = h3_context_bounds_in_crs(h3_cell, self.crs_text)
        base = window_from_requested_bounds(b02, requested)
        if base is None:
            raise ValueError(f"{self.item_id}/{h3_cell}: no B02 coverage")
        target_base = center_window(base.window, SENTINEL_IMAGE_SIZE)
        target_bounds = window_bounds(target_base, b02.transform)

        channels: list[np.ndarray] = []
        for band in SPECTRAL_BANDS:
            ds = self.datasets[band]
            # Same physical 1.28 km target footprint for all native resolutions.
            win = from_bounds(*target_bounds, transform=ds.transform)
            arr = ds.read(
                1,
                window=win,
                out_shape=(SENTINEL_IMAGE_SIZE, SENTINEL_IMAGE_SIZE),
                resampling=Resampling.bilinear,
                boundless=True,
                fill_value=0,
                out_dtype="uint16",
            )
            channels.append(arr)
        spectral = np.stack(channels, axis=-1).astype(np.uint16, copy=False)

        scl_ds = self.datasets["SCL"]
        scl_win = from_bounds(*target_bounds, transform=scl_ds.transform)
        scl = scl_ds.read(
            1,
            window=scl_win,
            out_shape=(SENTINEL_IMAGE_SIZE, SENTINEL_IMAGE_SIZE),
            resampling=Resampling.nearest,
            boundless=True,
            fill_value=0,
            out_dtype="uint8",
        ).astype(np.uint8, copy=False)
        return spectral, scl


class ThreadSceneCache:
    def __init__(self, scenes: dict[str, dict[str, Any]], max_scenes: int):
        from collections import OrderedDict

        self.scenes = scenes
        self.max_scenes = max(1, max_scenes)
        self.cache: "OrderedDict[str, NativeSceneBundle]" = OrderedDict()

    def get(self, item_id: str) -> NativeSceneBundle:
        bundle = self.cache.pop(item_id, None)
        if bundle is not None:
            self.cache[item_id] = bundle
            return bundle
        scene = self.scenes.get(item_id)
        if scene is None:
            raise KeyError(f"No local scene inventory for item_id={item_id}")
        bundle = NativeSceneBundle(scene)
        self.cache[item_id] = bundle
        while len(self.cache) > self.max_scenes:
            _, old = self.cache.popitem(last=False)
            old.close()
        return bundle

    def close(self) -> None:
        for b in self.cache.values():
            b.close()
        self.cache.clear()


_FRAME_TLS = threading.local()
_FRAME_SCENES: dict[str, dict[str, Any]] = {}
_FRAME_CACHE_PER_THREAD = 4


def thread_frame_cache() -> ThreadSceneCache:
    cache = getattr(_FRAME_TLS, "scene_cache", None)
    if cache is None:
        cache = ThreadSceneCache(_FRAME_SCENES, _FRAME_CACHE_PER_THREAD)
        _FRAME_TLS.scene_cache = cache
    return cache


def prepare_h3_group_frames(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cache = thread_frame_cache()
    out: list[dict[str, Any]] = []
    with rasterio.Env(
        GDAL_CACHEMAX=1024,
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        NUM_THREADS="ALL_CPUS",
    ):
        for row in rows:
            bundle = cache.get(row["item_id"])
            spectral, scl = bundle.extract_dn_and_scl(row["h3_cell"])
            if spectral.nbytes != SPECTRAL_DN_BYTES or scl.nbytes != SCL_BYTES:
                raise RuntimeError("Prepared frame shape invariant failed")
            # Preserve uint16 DN and uint8 SCL losslessly; Parquet performs outer
            # ZSTD compression.  Radiometry/median-fill is performed just before GPU.
            out.append({
                "frame_bucket": int(row["frame_bucket"]),
                "source": "sentinel2",
                "h3_cell": row["h3_cell"],
                "h3_resolution": int(row.get("h3_resolution") or TARGET_H3_RESOLUTION),
                "capture_period": row["capture_period"],
                "capture_timestamp_utc": row["capture_timestamp_utc"],
                "valid_from_utc": row.get("valid_from_utc") or row["capture_timestamp_utc"],
                "valid_to_utc": row.get("valid_to_utc"),
                "item_id": row["item_id"],
                "source_uri": row.get("source_uri") or row.get("gcs_uri"),
                "coverage_fraction": row.get("coverage_fraction"),
                "requires_mosaic": row.get("requires_mosaic"),
                "local_bad_fraction": row.get("local_bad_fraction"),
                "local_clear_fraction": row.get("local_clear_fraction"),
                "s2_processing_baseline": row.get("s2_processing_baseline"),
                "frame_origin": row.get("frame_origin"),
                "emit_target": bool(row.get("emit_target", False)),
                "spectral_dn": memoryview(np.ascontiguousarray(spectral)).tobytes(),
                "scl": memoryview(np.ascontiguousarray(scl)).tobytes(),
            })
    return out


def write_frame_shard(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    table = pa.Table.from_pylist(rows)
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=1,
        use_dictionary=[
            "source", "h3_cell", "capture_period", "item_id", "frame_origin",
            "s2_processing_baseline",
        ],
        write_statistics=True,
        row_group_size=min(len(rows), 512),
    )


def phase_frames(args: argparse.Namespace, paths: Paths) -> None:
    done = paths.manifests / "frames.done.json"
    if args.resume and done.exists():
        log("frames: completion marker exists; skipping")
        return
    # Clean partial frame shards only when not resuming a verified completed store.
    for p in paths.frames.glob("bucket=*/part-*.parquet"):
        p.unlink()

    plan = pl.read_parquet(paths.selection / "frame_plan.parquet")
    if plan.is_empty():
        log("frames: empty plan")
        done.write_text(json.dumps({"created_at_utc": iso_z(utcnow()), "rows": 0}, indent=2))
        return

    global _FRAME_SCENES, _FRAME_CACHE_PER_THREAD
    _FRAME_SCENES = scene_path_map(paths)
    _FRAME_CACHE_PER_THREAD = args.frame_scene_cache_per_thread

    missing_items = sorted(set(plan["item_id"].to_list()) - set(_FRAME_SCENES))
    if missing_items:
        die(
            f"Frame plan references {len(missing_items):,} scenes not present locally. "
            f"Run phase=context or inspect incomplete inputs. First={missing_items[:10]}"
        )

    # Group rows by H3.  Sorting by fixed bucket then H3 lets us drain futures in
    # deterministic order and write already-sorted bucket shards.
    plan = plan.sort(["frame_bucket", "h3_cell", "capture_timestamp_utc", "item_id"])
    groups: list[tuple[int, str, list[dict[str, Any]]]] = []
    for key, group in plan.group_by(["frame_bucket", "h3_cell"], maintain_order=True):
        bucket, cell = key
        groups.append((int(bucket), str(cell), group.to_dicts()))

    log(
        f"frames: H3 groups={len(groups):,}, rows={plan.height:,}, workers={args.frame_workers}, "
        f"thread_scene_cache={args.frame_scene_cache_per_thread}"
    )
    buffers: dict[int, list[dict[str, Any]]] = defaultdict(list)
    part_idx: dict[int, int] = defaultdict(int)
    total_frames = 0
    total_raw_bytes = 0
    errors: list[dict[str, str]] = []
    started = time.monotonic()

    def flush_bucket(bucket: int, force: bool = False) -> None:
        nonlocal total_raw_bytes
        buf = buffers[bucket]
        while len(buf) >= args.frames_per_shard or (force and buf):
            take = min(args.frames_per_shard, len(buf))
            chunk = buf[:take]
            del buf[:take]
            d = ensure_dir(paths.frames / f"bucket={bucket:03d}")
            path = d / f"part-{part_idx[bucket]:05d}.parquet"
            write_frame_shard(chunk, path)
            total_raw_bytes += take * (SPECTRAL_DN_BYTES + SCL_BYTES)
            part_idx[bucket] += 1

    # Bounded out-of-order scheduler.  We consume groups in submission order to
    # preserve sorted bucket/H3 frame shards while still keeping many CPU workers busy.
    max_pending = max(args.frame_workers * 3, args.frame_workers)
    with cf.ThreadPoolExecutor(
        max_workers=args.frame_workers,
        thread_name_prefix="frame-prep",
    ) as pool:
        pending: dict[int, cf.Future[list[dict[str, Any]]]] = {}
        submit_idx = 0
        emit_idx = 0
        while emit_idx < len(groups):
            while submit_idx < len(groups) and len(pending) < max_pending:
                _bucket, _cell, rows = groups[submit_idx]
                pending[submit_idx] = pool.submit(prepare_h3_group_frames, rows)
                submit_idx += 1

            fut = pending.pop(emit_idx)
            bucket, cell, _ = groups[emit_idx]
            try:
                prepared = fut.result()
                buffers[bucket].extend(prepared)
                total_frames += len(prepared)
                flush_bucket(bucket, force=False)
            except Exception as exc:
                errors.append({"h3_cell": cell, "error": repr(exc)})
                log(f"ERROR frame prep h3={cell}: {exc}")
                if args.strict:
                    raise
            emit_idx += 1
            if emit_idx % 500 == 0 or emit_idx == len(groups):
                elapsed = max(time.monotonic() - started, 1e-9)
                log(
                    f"frames progress: h3={emit_idx:,}/{len(groups):,}, "
                    f"frames={total_frames:,}, h3/s={emit_idx/elapsed:.2f}, "
                    f"prepared_raw={total_raw_bytes/1024**3:.1f} GiB"
                )

    for bucket in range(args.frame_buckets):
        flush_bucket(bucket, force=True)
    if errors:
        pl.DataFrame(errors).write_parquet(paths.frames / "errors.parquet")
    if args.strict and errors:
        die(f"Frame preparation failed for {len(errors):,} H3 groups")

    frame_files = list(paths.frames.glob("bucket=*/part-*.parquet"))
    compressed_bytes = sum(p.stat().st_size for p in frame_files)
    done.write_text(json.dumps({
        "created_at_utc": iso_z(utcnow()),
        "frames": total_frames,
        "buckets": args.frame_buckets,
        "shards": len(frame_files),
        "uncompressed_dn_scl_bytes": total_frames * (SPECTRAL_DN_BYTES + SCL_BYTES),
        "parquet_bytes": compressed_bytes,
        "errors": len(errors),
    }, indent=2))
    log(
        f"frames complete: frames={total_frames:,}, parquet={compressed_bytes/1024**3:.1f} GiB, "
        f"shards={len(frame_files):,}"
    )

    if args.delete_local_raw_after_frames:
        # Local only.  Remote B2 raw source is deliberately untouched.
        log("frames: deleting local staged raw imagery after verified frame-cache build")
        shutil.rmtree(paths.raw_b2 / "l2a", ignore_errors=True)
        if paths.context_raw.exists():
            shutil.rmtree(paths.context_raw, ignore_errors=True)


# -----------------------------------------------------------------------------
# Multi-GPU OlmoEarth inference
# -----------------------------------------------------------------------------

def autocast_context(torch_mod, device, precision: str):
    precision = precision.lower()
    if precision == "fp32":
        return contextlib.nullcontext()
    if precision == "bf16":
        return torch_mod.autocast("cuda", dtype=torch_mod.bfloat16)
    if precision == "fp16":
        return torch_mod.autocast("cuda", dtype=torch_mod.float16)
    raise ValueError(f"Unsupported precision {precision!r}")


def load_olmoearth_for_gpu(gpu_rank: int):
    import torch

    try:
        from olmoearth_pretrain_minimal import ModelID, Normalizer, load_model_from_id
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import (
            MaskedOlmoEarthSample,
        )
    except ImportError as exc:
        raise RuntimeError(
            "OlmoEarth requires `olmoearth-pretrain-minimal`; install it before embedding"
        ) from exc

    torch.cuda.set_device(gpu_rank)
    device = torch.device(f"cuda:{gpu_rank}")
    full_model = load_model_from_id(ModelID.OLMOEARTH_V1_2_BASE, load_weights=True)
    encoder = full_model.encoder.eval().to(device)
    del full_model
    normalizer = Normalizer(std_multiplier=2.0)
    return encoder, normalizer, Modality, MaskedOlmoEarthSample, device


def olmo_timestamps(rows: list[dict[str, Any]]) -> np.ndarray:
    result = np.zeros((len(rows), 3), dtype=np.int64)
    for i, row in enumerate(rows):
        d = parse_datetime(row["capture_timestamp_utc"])
        result[i] = [d.day, d.month - 1, d.year]
    return result


def encode_olmo_batch(
    model,
    normalizer,
    Modality,
    MaskedOlmoEarthSample,
    sequences: list[np.ndarray],
    timestamp_sequences: list[np.ndarray],
    device,
    precision: str,
) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    if not sequences:
        return np.empty((0, 0), dtype=np.float32)
    t = sequences[0].shape[0]
    if any(x.shape[0] != t for x in sequences):
        raise ValueError("OlmoEarth batch must have a single common sequence length")

    # Exact old layout: [B,T,H,W,C] -> [B,H,W,T,C].
    batch = np.stack(sequences, axis=0)
    batch = np.transpose(batch, (0, 2, 3, 1, 4)).astype(np.float32, copy=False)
    normalized = normalizer.normalize(Modality.SENTINEL2_L2A, batch)
    normalized_np = np.asarray(normalized, dtype=np.float32)

    # Pin host tensors so H2D can be non-blocking.  This matters on multi-GPU Vast
    # hosts where CPU preparation should overlap CUDA execution.
    x_cpu = torch.from_numpy(normalized_np)
    ts_cpu = torch.from_numpy(np.stack(timestamp_sequences, axis=0)).long()
    if torch.cuda.is_available():
        x_cpu = x_cpu.pin_memory()
        ts_cpu = ts_cpu.pin_memory()
    x = x_cpu.to(device, non_blocking=True)
    ts = ts_cpu.to(device, non_blocking=True)

    mask = torch.zeros(
        x.shape[0], x.shape[1], x.shape[2], x.shape[3],
        dtype=torch.long,
        device=device,
    )
    sample = MaskedOlmoEarthSample(
        timestamps=ts,
        sentinel2_l2a=x,
        sentinel2_l2a_mask=mask,
    )
    with torch.inference_mode(), autocast_context(torch, device, precision):
        out = model(sample, patch_size=OLMO_PATCH_SIZE, input_res=OLMO_INPUT_RES_M, fast_pass=True)
        tokens = out["tokens_and_masks"].sentinel2_l2a.float()
        dims = tuple(range(1, tokens.ndim - 1))
        z = tokens.mean(dim=dims)
        z = F.normalize(z, p=2, dim=1)
    return z.cpu().numpy().astype(np.float32, copy=False)


def decode_prepared_frame(row: dict[str, Any]) -> np.ndarray:
    spectral_blob = row["spectral_dn"]
    scl_blob = row["scl"]
    if len(spectral_blob) != SPECTRAL_DN_BYTES:
        raise ValueError(
            f"Bad spectral frame bytes for {row['h3_cell']}/{row['item_id']}: "
            f"{len(spectral_blob)} != {SPECTRAL_DN_BYTES}"
        )
    if len(scl_blob) != SCL_BYTES:
        raise ValueError(
            f"Bad SCL frame bytes for {row['h3_cell']}/{row['item_id']}: "
            f"{len(scl_blob)} != {SCL_BYTES}"
        )
    dn = np.frombuffer(spectral_blob, dtype=np.uint16).reshape(SPECTRAL_DN_SHAPE)
    scl = np.frombuffer(scl_blob, dtype=np.uint8).reshape(SCL_SHAPE)
    bad_mask = np.isin(scl, tuple(BAD_CLASSES))
    spectral = sentinel_dn_to_reflectance(dn, row.get("s2_processing_baseline"))
    spectral = fill_bad_pixels_with_band_median(spectral, bad_mask)
    return spectral.astype(np.float32, copy=False)


def iter_bucket_rows(bucket_dir: Path) -> Iterator[dict[str, Any]]:
    for path in sorted(bucket_dir.glob("part-*.parquet")):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                yield row


def iter_h3_groups_from_bucket(bucket_dir: Path) -> Iterator[list[dict[str, Any]]]:
    current: str | None = None
    rows: list[dict[str, Any]] = []
    for row in iter_bucket_rows(bucket_dir):
        cell = row["h3_cell"]
        if current is None:
            current = cell
        if cell != current:
            yield rows
            rows = []
            current = cell
        rows.append(row)
    if rows:
        yield rows


def completed_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["h3_cell"]),
        str(row["capture_period"]),
        str(row["item_id"]),
        OLMOEARTH_MODEL_ID,
    )


def load_completed_keys(ledger_root: Path) -> set[tuple[str, str, str, str]]:
    files = sorted(ledger_root.glob("worker=*/part-*.parquet"))
    if not files:
        return set()
    df = pl.scan_parquet(files).select(
        "h3_cell", "capture_period", "item_id", "encoder_version"
    ).collect(engine="streaming")
    return {
        (r[0], r[1], r[2], r[3])
        for r in df.iter_rows()
    }


def prepare_sequence_jobs_for_h3(
    rows: list[dict[str, Any]],
    completed: set[tuple[str, str, str, str]],
) -> list[dict[str, Any]]:
    history: deque[tuple[dict[str, Any], np.ndarray]] = deque()
    jobs: list[dict[str, Any]] = []
    rows = sorted(rows, key=lambda r: (parse_datetime(r["capture_timestamp_utc"]), r["item_id"]))
    for row in rows:
        frame = decode_prepared_frame(row)
        ts = parse_datetime(row["capture_timestamp_utc"])
        cutoff = ts - dt.timedelta(days=SENTINEL_LOOKBACK_DAYS)
        while history and parse_datetime(history[0][0]["capture_timestamp_utc"]) < cutoff:
            history.popleft()
        history.append((row, frame))
        while len(history) > SENTINEL_MAX_TIMESTEPS:
            history.popleft()

        if not bool(row.get("emit_target", False)):
            continue
        if completed_key(row) in completed:
            continue
        seq_rows = [x[0] for x in history]
        seq = np.stack([x[1] for x in history], axis=0)
        jobs.append({
            "target_row": row,
            "sequence": seq,
            "timestamps": olmo_timestamps(seq_rows),
            "t": len(history),
        })
    return jobs


def embedding_output_row(row: dict[str, Any], z: np.ndarray, sequence_length: int) -> dict[str, Any]:
    return {
        "source": "sentinel2",
        "h3_cell": row["h3_cell"],
        "h3_resolution": int(row.get("h3_resolution") or TARGET_H3_RESOLUTION),
        "capture_period": row["capture_period"],
        "capture_timestamp_utc": row["capture_timestamp_utc"],
        "valid_from_utc": row.get("valid_from_utc") or row["capture_timestamp_utc"],
        "valid_to_utc": row.get("valid_to_utc"),
        "item_id": row["item_id"],
        "gcs_uri": row.get("source_uri"),
        "coverage_fraction": row.get("coverage_fraction"),
        "requires_mosaic": row.get("requires_mosaic"),
        "local_bad_fraction": row.get("local_bad_fraction"),
        "local_clear_fraction": row.get("local_clear_fraction"),
        "s2_processing_baseline": row.get("s2_processing_baseline"),
        "temporal_sequence_length": int(sequence_length),
        "sentinel_input_mode": "native13",
        "embedding": z.tolist(),
        "embedding_dim": int(z.shape[0]),
        "encoder_name": ENCODER_NAME,
        "encoder_version": OLMOEARTH_MODEL_ID,
    }


def write_embedding_and_ledger(
    output_path: Path,
    ledger_path: Path,
    rows: list[dict[str, Any]],
) -> tuple[int, int]:
    ensure_dir(output_path.parent)
    ensure_dir(ledger_path.parent)
    tmp_out = output_path.with_suffix(output_path.suffix + f".{uuid.uuid4().hex}.tmp")
    tmp_led = ledger_path.with_suffix(ledger_path.suffix + f".{uuid.uuid4().hex}.tmp")

    df = pl.DataFrame(rows, infer_schema_length=None).with_columns(
        pl.col("source").cast(pl.Utf8),
        pl.col("h3_cell").cast(pl.Utf8),
        pl.col("h3_resolution").cast(pl.Int8),
        pl.col("capture_period").cast(pl.Utf8),
        pl.col("capture_timestamp_utc").cast(pl.Datetime("us", "UTC")),
        pl.col("valid_from_utc").cast(pl.Datetime("us", "UTC")),
        pl.col("valid_to_utc").cast(pl.Datetime("us", "UTC")),
        pl.col("item_id").cast(pl.Utf8),
        pl.col("gcs_uri").cast(pl.Utf8),
        pl.col("coverage_fraction").cast(pl.Float64),
        pl.col("requires_mosaic").cast(pl.Boolean),
        pl.col("local_bad_fraction").cast(pl.Float64),
        pl.col("local_clear_fraction").cast(pl.Float64),
        pl.col("s2_processing_baseline").cast(pl.Utf8),
        pl.col("temporal_sequence_length").cast(pl.Int16),
        pl.col("sentinel_input_mode").cast(pl.Utf8),
        pl.col("embedding").cast(pl.List(pl.Float32)),
        pl.col("embedding_dim").cast(pl.Int32),
        pl.col("encoder_name").cast(pl.Utf8),
        pl.col("encoder_version").cast(pl.Utf8),
    ).select(OUTPUT_COLUMNS)
    df.write_parquet(
        tmp_out,
        compression="zstd",
        compression_level=3,
        statistics=True,
    )
    ledger = df.select(
        "h3_cell", "capture_period", "item_id", "encoder_version"
    )
    ledger.write_parquet(tmp_led, compression="zstd", compression_level=1, statistics=True)
    tmp_out.replace(output_path)
    tmp_led.replace(ledger_path)
    return output_path.stat().st_size, ledger_path.stat().st_size


class AsyncEmbeddingWriter:
    def __init__(self, paths: Paths, rank: int, rows_per_shard: int, writer_threads: int = 2):
        self.paths = paths
        self.rank = rank
        self.rows_per_shard = rows_per_shard
        self.pool = cf.ThreadPoolExecutor(
            max_workers=writer_threads,
            thread_name_prefix=f"gpu{rank}-writer",
        )
        self.pending: list[cf.Future[tuple[int, int]]] = []
        self.buffer: list[dict[str, Any]] = []
        self.next_part = self._discover_next_part()
        self.output_bytes = 0

    @property
    def out_dir(self) -> Path:
        return ensure_dir(self.paths.embeddings / f"worker={self.rank:02d}")

    @property
    def ledger_dir(self) -> Path:
        return ensure_dir(self.paths.ledgers / f"worker={self.rank:02d}")

    def _discover_next_part(self) -> int:
        out_dir = ensure_dir(self.paths.embeddings / f"worker={self.rank:02d}")
        ledger_dir = ensure_dir(self.paths.ledgers / f"worker={self.rank:02d}")
        # Delete any uncommitted embedding shard with no matching ledger.
        for out in out_dir.glob("part-*.parquet"):
            led = ledger_dir / out.name
            if not led.exists():
                log(f"worker {self.rank}: removing uncommitted shard {out}")
                out.unlink(missing_ok=True)
        for led in ledger_dir.glob("part-*.parquet"):
            out = out_dir / led.name
            if not out.exists():
                log(f"worker {self.rank}: removing orphan ledger {led}")
                led.unlink(missing_ok=True)
        indexes = []
        for led in ledger_dir.glob("part-*.parquet"):
            m = re.fullmatch(r"part-(\d{5})\.parquet", led.name)
            if m:
                indexes.append(int(m.group(1)))
        return max(indexes, default=-1) + 1

    def add(self, row: dict[str, Any]) -> None:
        self.buffer.append(row)
        if len(self.buffer) >= self.rows_per_shard:
            self.flush(force=False)

    def flush(self, force: bool = True) -> None:
        while len(self.buffer) >= self.rows_per_shard or (force and self.buffer):
            take = self.rows_per_shard if len(self.buffer) >= self.rows_per_shard else len(self.buffer)
            chunk = self.buffer[:take]
            del self.buffer[:take]
            idx = self.next_part
            self.next_part += 1
            out = self.out_dir / f"part-{idx:05d}.parquet"
            led = self.ledger_dir / f"part-{idx:05d}.parquet"
            self.pending.append(self.pool.submit(write_embedding_and_ledger, out, led, chunk))
            # Bound queued output memory / filesystem pressure.
            if len(self.pending) >= 4:
                fut = self.pending.pop(0)
                a, _ = fut.result()
                self.output_bytes += a

    def close(self) -> None:
        self.flush(force=True)
        for fut in self.pending:
            a, _ = fut.result()
            self.output_bytes += a
        self.pending.clear()
        self.pool.shutdown(wait=True)


def gpu_worker_main(
    rank: int,
    world_size: int,
    paths_dict: dict[str, str],
    config: dict[str, Any],
    result_queue,
) -> None:
    try:
        # Import torch only inside spawned CUDA workers.
        import torch

        torch.cuda.set_device(rank)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

        paths = Paths(**{k: Path(v) for k, v in paths_dict.items()})
        completed = load_completed_keys(paths.ledgers)
        model, normalizer, Modality, MaskedOlmoEarthSample, device = load_olmoearth_for_gpu(rank)
        writer = AsyncEmbeddingWriter(
            paths,
            rank,
            rows_per_shard=int(config["rows_per_output_shard"]),
            writer_threads=int(config["writer_threads_per_gpu"]),
        )

        assigned_buckets = [
            b for b in range(int(config["frame_buckets"]))
            if b % world_size == rank
        ]
        prep_threads = int(config["gpu_prep_threads"])
        initial_batch = int(config["gpu_batch_size"])
        precision = str(config["precision"])
        effective_batch = {t: initial_batch for t in range(1, SENTINEL_MAX_TIMESTEPS + 1)}
        buckets: dict[int, list[dict[str, Any]]] = {
            t: [] for t in range(1, SENTINEL_MAX_TIMESTEPS + 1)
        }
        encoded = 0
        target_jobs = 0
        started = time.monotonic()

        def flush_gpu_bucket(t: int, force: bool = False) -> None:
            nonlocal encoded
            jobs = buckets[t]
            if not jobs:
                return
            while len(jobs) >= effective_batch[t] or (force and jobs):
                bs = min(effective_batch[t], len(jobs))
                chunk = jobs[:bs]
                try:
                    torch.cuda.reset_peak_memory_stats(device)
                    z = encode_olmo_batch(
                        model,
                        normalizer,
                        Modality,
                        MaskedOlmoEarthSample,
                        [j["sequence"] for j in chunk],
                        [j["timestamps"] for j in chunk],
                        device,
                        precision,
                    )
                except torch.cuda.OutOfMemoryError:
                    if effective_batch[t] <= 1:
                        raise
                    old = effective_batch[t]
                    effective_batch[t] = max(1, old // 2)
                    torch.cuda.empty_cache()
                    log(f"GPU {rank}: OOM T={t}; batch {old}->{effective_batch[t]}")
                    continue
                del jobs[:bs]
                for job, emb in zip(chunk, z, strict=True):
                    writer.add(embedding_output_row(job["target_row"], emb, t))
                encoded += len(chunk)
                if encoded % 5000 < len(chunk):
                    elapsed = max(time.monotonic() - started, 1e-9)
                    alloc = torch.cuda.max_memory_allocated(device) / 1024**3
                    log(
                        f"GPU {rank}: encoded={encoded:,}, rate={encoded/elapsed:.2f}/s, "
                        f"T={t}, batch={effective_batch[t]}, peak_alloc={alloc:.1f} GiB"
                    )

        def consume_jobs(jobs: list[dict[str, Any]]) -> None:
            nonlocal target_jobs
            for job in jobs:
                t = int(job["t"])
                buckets[t].append(job)
                target_jobs += 1
                flush_gpu_bucket(t, force=False)

        with cf.ThreadPoolExecutor(
            max_workers=prep_threads,
            thread_name_prefix=f"gpu{rank}-prep",
        ) as prep_pool:
            max_pending = max(prep_threads * 4, 8)
            pending: set[cf.Future[list[dict[str, Any]]]] = set()

            def drain_one() -> None:
                if not pending:
                    return
                done, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                for fut in done:
                    pending.remove(fut)
                    consume_jobs(fut.result())

            for bucket_id in assigned_buckets:
                bucket_dir = paths.frames / f"bucket={bucket_id:03d}"
                if not bucket_dir.exists():
                    continue
                log(f"GPU {rank}: reading frame bucket {bucket_id:03d}")
                for h3_rows in iter_h3_groups_from_bucket(bucket_dir):
                    pending.add(prep_pool.submit(prepare_sequence_jobs_for_h3, h3_rows, completed))
                    while len(pending) >= max_pending:
                        drain_one()
                # Do not force GPU buckets at bucket boundaries; carry partially
                # filled T batches into the next assigned bucket for efficiency.
            while pending:
                drain_one()

        for t in range(1, SENTINEL_MAX_TIMESTEPS + 1):
            flush_gpu_bucket(t, force=True)
        writer.close()
        elapsed = max(time.monotonic() - started, 1e-9)
        result = {
            "rank": rank,
            "encoded": encoded,
            "target_jobs": target_jobs,
            "elapsed_s": elapsed,
            "embeddings_per_s": encoded / elapsed,
            "effective_batch_sizes": effective_batch,
            "output_bytes": writer.output_bytes,
            "assigned_buckets": assigned_buckets,
        }
        (paths.manifests / f"embed_worker_{rank:02d}.json").write_text(json.dumps(result, indent=2))
        result_queue.put(("ok", result))
    except BaseException as exc:
        tb = traceback.format_exc()
        result_queue.put(("error", {"rank": rank, "error": repr(exc), "traceback": tb}))
        raise


def prewarm_olmo_model_cache() -> None:
    """Populate model/checkpoint caches once before spawning N GPU workers."""
    try:
        from olmoearth_pretrain_minimal import ModelID, load_model_from_id
        log("embed: prewarming OlmoEarth model/checkpoint cache once on CPU")
        model = load_model_from_id(ModelID.OLMOEARTH_V1_2_BASE, load_weights=True)
        del model
    except Exception as exc:
        # The workers will surface the authoritative error.  This warning keeps
        # prewarm from masking environments where the package handles caching differently.
        log(f"WARN OlmoEarth cache prewarm failed: {exc}")


def phase_embed(args: argparse.Namespace, paths: Paths) -> None:
    done = paths.manifests / "embed.done.json"
    if args.resume and done.exists():
        log("embed: completion marker exists; skipping")
        return

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for embedding") from exc
    available = torch.cuda.device_count()
    if available <= 0:
        die("No CUDA GPUs visible")
    world_size = available if args.gpus <= 0 else min(args.gpus, available)
    if world_size <= 0:
        die("No GPUs selected")
    log(
        f"embed: visible_gpus={available}, using={world_size}, "
        f"prep_threads/gpu={args.gpu_prep_threads}, initial_batch={args.gpu_batch_size}, "
        f"precision={args.precision}"
    )
    if not args.skip_model_prewarm:
        prewarm_olmo_model_cache()

    # Do not fork after CUDA initialization.  Spawn gives every process a clean CUDA runtime.
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    paths_dict = {field.name: str(getattr(paths, field.name)) for field in dataclasses.fields(Paths)}
    config = {
        "frame_buckets": args.frame_buckets,
        "rows_per_output_shard": args.rows_per_output_shard,
        "writer_threads_per_gpu": args.writer_threads_per_gpu,
        "gpu_prep_threads": args.gpu_prep_threads,
        "gpu_batch_size": args.gpu_batch_size,
        "precision": args.precision,
    }
    procs: list[mp.Process] = []
    for rank in range(world_size):
        p = ctx.Process(
            target=gpu_worker_main,
            args=(rank, world_size, paths_dict, config, result_queue),
            name=f"olmo-gpu-{rank}",
        )
        p.start()
        procs.append(p)

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    remaining = world_size
    while remaining:
        try:
            status, payload = result_queue.get(timeout=30)
            if status == "ok":
                results.append(payload)
            else:
                errors.append(payload)
            remaining -= 1
        except queue.Empty:
            failed = [p for p in procs if p.exitcode not in (None, 0)]
            if failed:
                errors.append({
                    "rank": -1,
                    "error": "One or more GPU workers exited before reporting",
                    "workers": [(p.name, p.exitcode) for p in failed],
                })
                break

    for p in procs:
        p.join()
    bad_exit = [(p.name, p.exitcode) for p in procs if p.exitcode != 0]
    if errors or bad_exit:
        payload = {"errors": errors, "bad_exit": bad_exit}
        (paths.manifests / "embed.errors.json").write_text(json.dumps(payload, indent=2))
        die(f"Embedding failed: {payload}")

    total = sum(int(r["encoded"]) for r in results)
    wall = max((float(r["elapsed_s"]) for r in results), default=0.0)
    done.write_text(json.dumps({
        "created_at_utc": iso_z(utcnow()),
        "gpus": world_size,
        "encoded_rows": total,
        "wall_s_approx": wall,
        "aggregate_embeddings_per_s": (total / wall if wall > 0 else None),
        "workers": sorted(results, key=lambda x: x["rank"]),
    }, indent=2))
    log(
        f"embed complete: rows={total:,}, gpus={world_size}, "
        f"aggregate~={total/wall if wall else 0:.2f} embeddings/s"
    )


# -----------------------------------------------------------------------------
# Finalize / schema audit / B2 publish
# -----------------------------------------------------------------------------

def existing_shard_count(paths: Paths) -> int:
    indexes: list[int] = []
    for p in paths.existing_gold.glob("part-*.parquet"):
        m = re.fullmatch(r"part-(\d+)\.parquet", p.name)
        if m:
            indexes.append(int(m.group(1)))
    if not indexes:
        return 0
    return max(indexes) + 1


def clean_final_dir(final_sentinel: Path) -> None:
    ensure_dir(final_sentinel)
    for p in final_sentinel.iterdir():
        if p.is_file():
            p.unlink()


def phase_finalize(args: argparse.Namespace, paths: Paths) -> None:
    final_sentinel = ensure_dir(paths.final / "sentinel2")
    clean_final_dir(final_sentinel)
    worker_files = sorted(paths.embeddings.glob("worker=*/part-*.parquet"))
    if not worker_files:
        log("finalize: no new embedding shards")
    start_idx = args.part_start if args.part_start >= 0 else existing_shard_count(paths)

    output_rows = 0
    for i, src in enumerate(worker_files):
        dst = final_sentinel / f"part-{start_idx+i:05d}.parquet"
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        output_rows += pq.ParquetFile(src).metadata.num_rows

    # Schema compatibility audit against an existing shard when available.
    existing_files = sorted(paths.existing_gold.glob("part-*.parquet"))
    new_files = sorted(final_sentinel.glob("part-*.parquet"))
    schema_audit: dict[str, Any] = {"passed": True}
    if existing_files and new_files:
        old_schema = pl.read_parquet_schema(existing_files[0])
        new_schema = pl.read_parquet_schema(new_files[0])
        schema_audit = {
            "passed": old_schema == new_schema,
            "old": {k: str(v) for k, v in old_schema.items()},
            "new": {k: str(v) for k, v in new_schema.items()},
        }
        if not schema_audit["passed"]:
            (paths.manifests / "schema_mismatch.json").write_text(json.dumps(schema_audit, indent=2))
            die("New embedding schema does not exactly match existing foundation_v1 schema")

    validity_patch = paths.selection / "existing_validity_patch.parquet"
    if validity_patch.exists():
        shutil.copy2(validity_patch, final_sentinel / "_EXISTING_VALIDITY_PATCH.parquet")

    recipe = {
        "created_at_utc": iso_z(utcnow()),
        "format": "crimenet-foundation-v1-b2-vast-extension-v1",
        "scientific_contract": {
            "model": OLMOEARTH_MODEL_ID,
            "band_order": list(SPECTRAL_BANDS),
            "image_size": SENTINEL_IMAGE_SIZE,
            "input_res_m": OLMO_INPUT_RES_M,
            "patch_size": OLMO_PATCH_SIZE,
            "max_timesteps": SENTINEL_MAX_TIMESTEPS,
            "lookback_days": SENTINEL_LOOKBACK_DAYS,
            "h3_resolution": TARGET_H3_RESOLUTION,
            "sentinel_context_margin_m": SENTINEL_CONTEXT_MARGIN_M,
            "max_local_bad_fraction": SENTINEL_MAX_LOCAL_BAD_FRACTION,
            "cloud_handling": "SCL BAD_CLASSES -> per-band median fill before OlmoEarth normalization",
            "radiometry": "PB>=04.00: (DN-1000)/10000; earlier: DN/10000; DN=0 nodata",
            "pooling": "mean all Sentinel token dimensions except batch/embedding; L2 normalize",
            "precision": args.precision,
        },
        "execution": {
            "disk_first": True,
            "raw_rasters_packed_into_parquet": False,
            "prepared_frame_cache": "lossless uint16 spectral DN + uint8 SCL binary columns in ZSTD Parquet",
            "multi_gpu": True,
            "frame_buckets": args.frame_buckets,
            "initial_gpu_batch_size": args.gpu_batch_size,
            "existing_month_policy": "skip_existing_months",
            "historical_context_hydration": not args.no_historical_context and not args.no_existing_gold,
        },
        "source": {
            "b2_remote": args.b2_source_remote,
            "existing_gold_remote": None if args.no_existing_gold else args.existing_gold_remote,
        },
    }
    (final_sentinel / "_B2_VAST_RECIPE.json").write_text(json.dumps(recipe, indent=2))

    embed_manifest = {}
    embed_done = paths.manifests / "embed.done.json"
    if embed_done.exists():
        embed_manifest = json.loads(embed_done.read_text())
    frame_manifest = {}
    frame_done = paths.manifests / "frames.done.json"
    if frame_done.exists():
        frame_manifest = json.loads(frame_done.read_text())
    success = {
        "created_at_utc": iso_z(utcnow()),
        "output_rows": output_rows,
        "output_shards": len(worker_files),
        "part_start": start_idx,
        "part_end": (start_idx + len(worker_files) - 1 if worker_files else None),
        "schema_audit": schema_audit,
        "embed": embed_manifest,
        "frames": frame_manifest,
    }
    (final_sentinel / "_B2_VAST_SUCCESS.json").write_text(json.dumps(success, indent=2))
    log(
        f"finalize: rows={output_rows:,}, shards={len(worker_files):,}, "
        f"part range={start_idx}..{start_idx+len(worker_files)-1 if worker_files else start_idx}"
    )


def phase_publish(args: argparse.Namespace, paths: Paths) -> None:
    if not args.publish_remote:
        log("publish: --publish-remote empty; skipping")
        return
    rclone_exists()
    local = paths.final / "sentinel2"
    if not local.exists():
        die("No final output. Run phase=finalize first.")
    cmd = [
        "rclone", "copy", str(local), args.publish_remote,
        "--fast-list",
        "--transfers", str(args.publish_transfers),
        "--checkers", str(args.publish_checkers),
        "--multi-thread-streams", str(args.rclone_multi_thread_streams),
        "--multi-thread-cutoff", "64M",
        "--buffer-size", "32M",
        "--retries", "20",
        "--low-level-retries", "50",
        "--retries-sleep", "3s",
        "--stats", "30s",
        "--stats-one-line",
    ]
    run(cmd)
    if args.verify_publish:
        run([
            "rclone", "check", str(local), args.publish_remote,
            "--one-way", "--size-only",
            "--checkers", str(args.publish_checkers),
        ])
    log(f"publish complete -> {args.publish_remote}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    cpu = os.cpu_count() or 64
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Vast-optimized disk-first multi-GPU Sentinel-2 OlmoEarth embedding pipeline",
    )
    p.add_argument(
        "--phase",
        choices=["all", "stage", "inventory", "candidates", "select", "context", "frames", "embed", "finalize", "publish"],
        default="all",
    )
    p.add_argument("--work-dir", default="/workspace/crimenet-sentinel-vast")
    p.add_argument(
        "--b2-source-remote",
        default="b2:crimenet-data/bronze/imagery/sentinel2/national",
    )
    p.add_argument(
        "--existing-gold-remote",
        default="b2:crimenet-data/gold/imagery/embeddings/foundation_v1/sentinel2",
    )
    p.add_argument(
        "--publish-remote",
        default="b2:crimenet-data/gold/imagery/embeddings/foundation_v1_b2_staging/sentinel2",
    )
    p.add_argument("--h3-manifest", default=None)
    p.add_argument("--no-existing-gold", action="store_true")
    p.add_argument("--no-historical-context", action="store_true")
    p.add_argument("--strict-context", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip-disk-check", action="store_true")

    # Aggressive B2/NVMe staging defaults for large Vast hosts.
    p.add_argument("--rclone-transfers", type=int, default=96)
    p.add_argument("--rclone-checkers", type=int, default=192)
    p.add_argument("--rclone-multi-thread-streams", type=int, default=8)

    # Candidate/SCL phase.
    p.add_argument("--candidate-workers", type=int, default=min(160, max(32, cpu // 2)))
    p.add_argument("--candidate-rows-per-shard", type=int, default=250_000)

    # Context hydration.
    p.add_argument("--context-download-workers", type=int, default=min(96, max(32, cpu // 4)))
    p.add_argument("--stac-id-batch-size", type=int, default=100)

    # Prepared-frame cache.  64 fixed buckets are enough to load-balance 1-16 GPUs
    # while avoiding pathological tiny-file counts.
    p.add_argument("--frame-buckets", type=int, default=64)
    p.add_argument("--frame-workers", type=int, default=min(192, max(48, int(cpu * 0.75))))
    p.add_argument("--frame-scene-cache-per-thread", type=int, default=4)
    p.add_argument("--frames-per-shard", type=int, default=384)
    p.add_argument("--delete-local-raw-after-frames", action=argparse.BooleanOptionalAction, default=False)

    # GPU stage.
    p.add_argument("--gpus", type=int, default=0, help="0 = all visible GPUs")
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--gpu-batch-size", type=int, default=128)
    p.add_argument("--gpu-prep-threads", type=int, default=max(4, min(16, cpu // 32)))
    p.add_argument("--writer-threads-per-gpu", type=int, default=2)
    p.add_argument("--skip-model-prewarm", action="store_true")
    p.add_argument("--rows-per-output-shard", type=int, default=25_000)

    # Finalization/publish.
    p.add_argument("--part-start", type=int, default=-1, help="-1 = append after staged existing part index")
    p.add_argument("--publish-transfers", type=int, default=32)
    p.add_argument("--publish-checkers", type=int, default=64)
    p.add_argument("--verify-publish", action=argparse.BooleanOptionalAction, default=True)
    return p


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "rclone_transfers": args.rclone_transfers,
        "rclone_checkers": args.rclone_checkers,
        "rclone_multi_thread_streams": args.rclone_multi_thread_streams,
        "candidate_workers": args.candidate_workers,
        "candidate_rows_per_shard": args.candidate_rows_per_shard,
        "context_download_workers": args.context_download_workers,
        "frame_buckets": args.frame_buckets,
        "frame_workers": args.frame_workers,
        "frame_scene_cache_per_thread": args.frame_scene_cache_per_thread,
        "frames_per_shard": args.frames_per_shard,
        "gpu_batch_size": args.gpu_batch_size,
        "gpu_prep_threads": args.gpu_prep_threads,
        "rows_per_output_shard": args.rows_per_output_shard,
        "publish_transfers": args.publish_transfers,
        "publish_checkers": args.publish_checkers,
    }
    bad = [k for k, v in positive.items() if int(v) <= 0]
    if bad:
        die(f"These options must be positive: {bad}")


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    paths = make_paths(args.work_dir)
    raise_nofile_limit()
    log(f"work_dir={paths.work}")
    log(
        f"host resources: cpu={os.cpu_count()}, ram_hint=unbounded-by-script, "
        f"phase={args.phase}, resume={args.resume}"
    )

    phases = (
        ["stage", "inventory", "candidates", "select", "context", "frames", "embed", "finalize", "publish"]
        if args.phase == "all"
        else [args.phase]
    )
    dispatch = {
        "stage": phase_stage,
        "inventory": phase_inventory,
        "candidates": phase_candidates,
        "select": phase_select,
        "context": phase_context,
        "frames": phase_frames,
        "embed": phase_embed,
        "finalize": phase_finalize,
        "publish": phase_publish,
    }
    for phase in phases:
        log(f"========== PHASE {phase.upper()} ==========")
        dispatch[phase](args, paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
