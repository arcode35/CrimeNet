#!/usr/bin/env python3
"""
CrimeNet national NAIP H3 crop pipeline WITH DINOv3 compression-quality gate.

What this script does
---------------------
One script handles both the storage pipeline and an empirical ML-quality test.

QUALITY PHASE
  Planetary Computer latest NAIP source COGs
      -> representative H3-r9 512x512 RGB crops
      -> baseline: UNCOMPRESSED 512x512 pixels -> DINOv3 SAT-493M
      -> JPEG Q90 4:2:0 -> decode -> DINOv3
      -> JPEG Q95 4:4:4 -> decode -> DINOv3
      -> measure:
           * CLS cosine similarity
           * patch-mean cosine similarity
           * CLS||patch-mean (2048-d) cosine similarity
           * relative L2 embedding drift
           * pairwise similarity-matrix correlation
           * nearest-neighbor recall@K
           * pixel PSNR
           * actual bytes/crop
      -> quality gate

BUILD PHASE
  Planetary Computer latest NAIP source COG
      -> temporary local source COG
      -> H3-r9 512x512 RGB crops
      -> selected JPEG codec
      -> tar shard (many H3 JPEGs + metadata)
      -> Backblaze B2
      -> verify upload
      -> delete temporary source COG and local tar

The quality test deliberately compares every compressed representation against
the EXACT SAME already-resized 512x512 RGB pixels. Therefore it measures the
incremental loss from JPEG compression, not the separate information loss caused
by the native-NAIP -> 512x512 resize.

Default production codec
------------------------
JPEG quality 95, 4:4:4 (Pillow subsampling=0).

The quality phase also evaluates Q90 4:2:0 so you can see how much storage it
would save and how much additional DINO representation drift it creates.

DINOv3 representation
----------------------
Uses DINOv3 ViT-L/16 SAT-493M and extracts:
    concat(
        x_norm_clstoken,                  # 1024
        mean(x_norm_patchtokens, dim=1),  # 1024
    )                                    # 2048

SAT-493M normalization:
    mean = (0.430, 0.411, 0.296)
    std  = (0.213, 0.156, 0.143)

DINOv3 loading
--------------
Recommended:
    --dinov3-repo /path/to/facebookresearch/dinov3
    --dinov3-weights /path/to/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth

The official SAT checkpoint filename contains eadcf0ff; the current DINOv3
loader uses that identifier to instantiate the SAT-compatible ViT-L backbone.

You can alternatively use:
    --dinov3-weights sat493m

which asks the local DINOv3 repo for Weights.SAT493M. Whether the checkpoint can
be downloaded automatically depends on your local access/cache.

Required environment
--------------------
B2_KEY_ID
B2_APPLICATION_KEY
B2_ENDPOINT_URL
B2_BUCKET              optional, defaults to crimenet-data

Optional:
DINOV3_REPO
DINOV3_WEIGHTS

Dependencies
------------
uv add \
  boto3 requests pystac-client pystac planetary-computer \
  rasterio shapely pyshp h3 pillow numpy polars torch

You also need a local checkout of facebookresearch/dinov3 for the quality phase.

Recommended workflow
--------------------
1) QUALITY CHECK ONLY (downloads a small representative sample):
python scripts/seed_crimenet_national_naip_h3_quality_b2.py \
  --mode quality \
  --dinov3-repo "$HOME/dinov3" \
  --dinov3-weights "/path/to/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth" \
  --quality-items 12 \
  --quality-cells-per-item 24

2) QUALITY-GATED SMALL TEXAS BUILD:
python scripts/seed_crimenet_national_naip_h3_quality_b2.py \
  --mode quality-and-build \
  --states TX \
  --limit-items 10 \
  --dinov3-repo "$HOME/dinov3" \
  --dinov3-weights "/path/to/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"

3) FULL BUILD AFTER QUALITY PASSES:
python scripts/seed_crimenet_national_naip_h3_quality_b2.py \
  --mode build \
  --workers 3 \
  --production-codec jpeg95_444

Notes
-----
* "ML quality loss" here means DINO representation drift caused by JPEG.
  It does NOT claim to measure end-to-end crime-prediction accuracy, because
  current nationwide inference data does not have national crime labels.
* Before the full national run, ensure crop_h3_rgb() matches the existing
  CrimeNet Gold H3->512 crop semantics. The JPEG quality test itself remains
  valid because all codecs are compared after that common crop.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses
import datetime as dt
import io
import json
import math
import os
import random
import re
import sys
import tarfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

import boto3
import h3
import numpy as np
import planetary_computer
import polars as pl
import pystac_client
import rasterio
import requests
import shapefile
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from PIL import Image
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from shapely.geometry import Polygon, shape
from shapely.geometry.base import BaseGeometry


# =============================================================================
# Constants
# =============================================================================

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "naip"
ASSET_KEY = "image"

DEFAULT_START_YEAR = 2018
DEFAULT_END_YEAR = 2023

H3_RESOLUTION = 9
IMAGE_SIZE = 512

DEFAULT_B2_PREFIX = "silver/imagery/h3_crops/naip_national_v2"

STATE_BOUNDARY_URL = (
    "https://www2.census.gov/geo/tiger/GENZ2024/shp/"
    "cb_2024_us_state_20m.zip"
)

STATE_FIPS_50_DC = {
    "01","02","04","05","06","08","09","10","11","12","13","15","16",
    "17","18","19","20","21","22","23","24","25","26","27","28","29",
    "30","31","32","33","34","35","36","37","38","39","40","41","42",
    "44","45","46","47","48","49","50","51","53","54","55","56",
}

DATE_TOKEN = re.compile(r"^(19|20)\d{6}$")
USER_AGENT = "CrimeNet-national-NAIP-H3-quality/2.0"

SAT_MEAN = np.asarray((0.430, 0.411, 0.296), dtype=np.float32)
SAT_STD = np.asarray((0.213, 0.156, 0.143), dtype=np.float32)

# Supported durable candidates. The names are intentionally explicit.
CODECS = {
    "jpeg90_420": {"quality": 90, "subsampling": 2},
    "jpeg95_444": {"quality": 95, "subsampling": 0},
}

PRINT_LOCK = threading.Lock()
TLS = threading.local()

TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=64 * 1024**2,
    multipart_chunksize=128 * 1024**2,
    max_concurrency=2,
    use_threads=True,
)


# =============================================================================
# Data models
# =============================================================================

@dataclasses.dataclass(frozen=True)
class StateDomain:
    fips: str
    abbr: str
    geometry: BaseGeometry


@dataclasses.dataclass(frozen=True)
class SelectedItem:
    item_id: str
    state: str
    stable_tile_key: str
    capture_time_utc: str
    gsd_m: float | None
    geometry: dict[str, Any]
    clipped_geometry: dict[str, Any]
    unsigned_href: str

    @property
    def year(self) -> int:
        return parse_datetime(self.capture_time_utc).year


@dataclasses.dataclass(frozen=True)
class BuildRuntime:
    bucket: str
    prefix: str
    work_dir: Path
    h3_resolution: int
    image_size: int
    codec_name: str
    resampling: Resampling
    exact_intersections: bool
    limit_cells_per_item: int
    download_attempts: int
    upload_attempts: int
    keep_source_on_failure: bool


@dataclasses.dataclass
class QualitySample:
    state: str
    item_id: str
    h3_cell_id: str
    capture_time_utc: str
    gsd_m: float | None
    valid_fraction: float
    baseline_rgb: np.ndarray
    compressed_rgb: dict[str, np.ndarray]
    compressed_bytes: dict[str, int]
    psnr_db: dict[str, float]


# =============================================================================
# Generic helpers
# =============================================================================

def log(message: str) -> None:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with PRINT_LOCK:
        print(f"[{stamp}] {message}", flush=True)


def env_required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def parse_datetime(value: str) -> dt.datetime:
    out = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt.timezone.utc)
    return out.astimezone(dt.timezone.utc)


def iso_z(value: dt.datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def percentile(values: np.ndarray, q: float) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.percentile(values, q))


def get_session() -> requests.Session:
    session = getattr(TLS, "requests_session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=32,
            pool_maxsize=32,
            max_retries=0,
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        TLS.requests_session = session
    return session


def get_b2_client():
    client = getattr(TLS, "b2_client", None)
    if client is None:
        client = boto3.client(
            "s3",
            endpoint_url=env_required("B2_ENDPOINT_URL"),
            aws_access_key_id=env_required("B2_KEY_ID"),
            aws_secret_access_key=env_required("B2_APPLICATION_KEY"),
            config=Config(
                signature_version="s3v4",
                retries={"max_attempts": 20, "mode": "adaptive"},
                connect_timeout=30,
                read_timeout=300,
                max_pool_connections=64,
                s3={"addressing_style": "path"},
            ),
        )
        TLS.b2_client = client
    return client


# =============================================================================
# B2
# =============================================================================

def head_object(bucket: str, key: str) -> dict[str, Any] | None:
    try:
        return get_b2_client().head_object(Bucket=bucket, Key=key)
    except Exception as exc:
        response = getattr(exc, "response", None)
        status = (
            response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if response
            else None
        )
        if status in (403, 404) or "404" in str(exc) or "NoSuchKey" in str(exc):
            return None
        raise


def abort_multipart(bucket: str, key: str) -> None:
    client = get_b2_client()
    try:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": key}
        while True:
            response = client.list_multipart_uploads(**kwargs)
            for upload in response.get("Uploads", []):
                if upload.get("Key") == key:
                    client.abort_multipart_upload(
                        Bucket=bucket,
                        Key=key,
                        UploadId=upload["UploadId"],
                    )
            if not response.get("IsTruncated"):
                break
            kwargs["KeyMarker"] = response.get("NextKeyMarker")
            kwargs["UploadIdMarker"] = response.get("NextUploadIdMarker")
    except Exception as exc:
        log(f"WARN multipart cleanup failed {key}: {exc}")


def upload_verified(
    local_path: Path,
    bucket: str,
    key: str,
    content_type: str,
    attempts: int,
    metadata: dict[str, Any] | None = None,
) -> int:
    local_size = local_path.stat().st_size

    existing = head_object(bucket, key)
    if existing is not None and int(existing["ContentLength"]) > 0:
        return int(existing["ContentLength"])

    client = get_b2_client()

    for attempt in range(1, attempts + 1):
        try:
            if attempt > 1:
                abort_multipart(bucket, key)

            extra: dict[str, Any] = {"ContentType": content_type}
            if metadata:
                extra["Metadata"] = {
                    str(k): str(v)[:1024]
                    for k, v in metadata.items()
                    if v is not None
                }

            client.upload_file(
                str(local_path),
                bucket,
                key,
                ExtraArgs=extra,
                Config=TRANSFER_CONFIG,
            )

            remote = client.head_object(Bucket=bucket, Key=key)
            remote_size = int(remote["ContentLength"])

            if remote_size != local_size:
                raise IOError(
                    f"B2 size mismatch local={local_size}, "
                    f"remote={remote_size}: {key}"
                )
            return remote_size

        except Exception as exc:
            if attempt >= attempts:
                abort_multipart(bucket, key)
                raise
            delay = min(60, 2 ** (attempt - 1))
            log(
                f"B2 retry {attempt}/{attempts} {key}: "
                f"{type(exc).__name__}: {exc}; sleep={delay}s"
            )
            time.sleep(delay)

    raise AssertionError("unreachable")


def put_json(bucket: str, key: str, payload: dict[str, Any]) -> None:
    body = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")

    get_b2_client().put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
    )


# =============================================================================
# Census state boundaries
# =============================================================================

def download_small(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 6):
        try:
            response = get_session().get(url, timeout=(30, 120))
            response.raise_for_status()
            path.write_bytes(response.content)
            return
        except Exception:
            if attempt == 5:
                raise
            time.sleep(min(30, 2 ** (attempt - 1)))


def load_states(
    work_dir: Path,
    requested: set[str] | None,
) -> list[StateDomain]:
    root = work_dir / "_boundaries"
    root.mkdir(parents=True, exist_ok=True)

    zip_path = root / "cb_2024_us_state_20m.zip"
    extracted = root / "cb_2024_us_state_20m"

    if not zip_path.exists():
        log("download Census state boundary archive")
        download_small(STATE_BOUNDARY_URL, zip_path)

    if not extracted.exists():
        extracted.mkdir(parents=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extracted)

    shp_files = list(extracted.glob("*.shp"))
    if len(shp_files) != 1:
        raise RuntimeError(f"Expected one state shapefile, got {shp_files}")

    reader = shapefile.Reader(str(shp_files[0]))
    fields = [field[0] for field in reader.fields[1:]]

    states: list[StateDomain] = []

    for sr in reader.iterShapeRecords():
        attrs = dict(zip(fields, sr.record))
        fips = str(attrs.get("STATEFP", "")).zfill(2)

        if fips not in STATE_FIPS_50_DC:
            continue

        abbr = str(attrs.get("STUSPS", "")).upper()
        if requested and abbr not in requested:
            continue

        geom = shape(sr.shape.__geo_interface__)
        if not geom.is_empty:
            states.append(StateDomain(fips=fips, abbr=abbr, geometry=geom))

    states.sort(key=lambda s: s.fips)

    if requested:
        found = {s.abbr for s in states}
        missing = sorted(requested - found)
        if missing:
            raise SystemExit(f"Unknown states: {missing}")

    return states


# =============================================================================
# STAC latest-source selection
# =============================================================================

def stable_naip_tile_key(item_id: str) -> str:
    parts = item_id.split("_")

    first_date = None
    for i, token in enumerate(parts):
        if DATE_TOKEN.match(token):
            first_date = i
            break

    if first_date is not None:
        return "_".join(parts[: max(1, first_date - 1)])

    while parts and DATE_TOKEN.match(parts[-1]):
        parts.pop()

    return "_".join(parts)


def item_time(item) -> dt.datetime:
    value = item.datetime
    if value is None:
        raw = item.properties.get("datetime")
        if not raw:
            raise ValueError(f"Missing datetime: {item.id}")
        value = parse_datetime(str(raw))

    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)

    return value.astimezone(dt.timezone.utc)


def item_gsd(item) -> float | None:
    value = item.properties.get("gsd")
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


def choose_latest(items: list[Any]) -> list[Any]:
    latest: dict[str, Any] = {}

    for item in items:
        key = stable_naip_tile_key(item.id)
        current = latest.get(key)

        if current is None:
            latest[key] = item
            continue

        a = item_time(item)
        b = item_time(current)

        if a > b:
            latest[key] = item
        elif a == b:
            ag = item_gsd(item)
            bg = item_gsd(current)
            ags = ag if ag is not None else float("inf")
            bgs = bg if bg is not None else float("inf")
            if ags < bgs or (ags == bgs and item.id < current.id):
                latest[key] = item

    return sorted(
        latest.values(),
        key=lambda x: (
            stable_naip_tile_key(x.id),
            item_time(x),
            x.id,
        ),
    )


def discover_latest(
    states: list[StateDomain],
    start_year: int,
    end_year: int,
) -> tuple[list[SelectedItem], dict[str, Any]]:
    catalog = pystac_client.Client.open(STAC_URL)

    selected: list[SelectedItem] = []
    stats: dict[str, Any] = {}

    interval = (
        f"{start_year:04d}-01-01T00:00:00Z/"
        f"{end_year:04d}-12-31T23:59:59Z"
    )

    for index, state in enumerate(states, 1):
        log(f"discover {index}/{len(states)} {state.abbr}")

        search = catalog.search(
            collections=[COLLECTION],
            intersects=state.geometry.__geo_interface__,
            datetime=interval,
        )

        candidates = []

        for item in search.items():
            if not item.id.lower().startswith(state.abbr.lower() + "_"):
                continue
            if ASSET_KEY not in item.assets:
                continue
            if not item.geometry:
                continue
            candidates.append(item)

        chosen = choose_latest(candidates)
        state_selected = 0

        for item in chosen:
            source_geom = shape(item.geometry)
            clipped = source_geom.intersection(state.geometry)
            if clipped.is_empty:
                continue

            selected.append(
                SelectedItem(
                    item_id=item.id,
                    state=state.abbr,
                    stable_tile_key=stable_naip_tile_key(item.id),
                    capture_time_utc=iso_z(item_time(item)),
                    gsd_m=item_gsd(item),
                    geometry=item.geometry,
                    clipped_geometry=clipped.__geo_interface__,
                    unsigned_href=item.assets[ASSET_KEY].href,
                )
            )
            state_selected += 1

        stats[state.abbr] = {
            "catalog_candidates": len(candidates),
            "latest_tiles": state_selected,
        }

        log(
            f"  {state.abbr}: {len(candidates):,} candidates -> "
            f"{state_selected:,} latest"
        )

    selected.sort(
        key=lambda x: (
            x.state,
            x.stable_tile_key,
            x.capture_time_utc,
            x.item_id,
        )
    )

    return selected, stats


# =============================================================================
# Source COG download
# =============================================================================

def source_head(unsigned_href: str) -> tuple[str, int | None]:
    signed = planetary_computer.sign(unsigned_href)

    response = get_session().head(
        signed,
        allow_redirects=True,
        timeout=(30, 120),
    )
    response.raise_for_status()

    size = response.headers.get("Content-Length")
    return signed, int(size) if size is not None else None


def download_source_cog(
    rec: SelectedItem,
    local_path: Path,
    attempts: int,
) -> int:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    last_exc: Exception | None = None

    for attempt in range(1, attempts + 1):
        existing = local_path.stat().st_size if local_path.exists() else 0

        try:
            signed, expected = source_head(rec.unsigned_href)

            if expected is not None and existing == expected:
                return existing

            if expected is not None and existing > expected:
                local_path.unlink(missing_ok=True)
                existing = 0

            headers: dict[str, str] = {}
            mode = "wb"

            if existing > 0:
                headers["Range"] = f"bytes={existing}-"

            log(
                f"source {rec.state} {rec.item_id}: "
                f"resume={existing / 1024**2:.1f} MiB"
            )

            with get_session().get(
                signed,
                headers=headers,
                stream=True,
                timeout=(30, 300),
            ) as response:

                if existing > 0 and response.status_code == 206:
                    mode = "ab"
                elif existing > 0 and response.status_code == 200:
                    existing = 0
                    mode = "wb"

                response.raise_for_status()

                with local_path.open(mode) as out:
                    for chunk in response.iter_content(8 * 1024**2):
                        if chunk:
                            out.write(chunk)

            final_size = local_path.stat().st_size

            if expected is not None and final_size != expected:
                raise IOError(
                    f"download mismatch expected={expected}, got={final_size}"
                )

            return final_size

        except Exception as exc:
            last_exc = exc
            if attempt >= attempts:
                break

            delay = min(60, 2 ** (attempt - 1))
            log(
                f"source retry {attempt}/{attempts} {rec.item_id}: "
                f"{type(exc).__name__}: {exc}; sleep={delay}s"
            )
            time.sleep(delay)

    assert last_exc is not None
    raise last_exc


# =============================================================================
# H3 crop logic
# =============================================================================

def h3_polygon(cell: str) -> Polygon:
    return Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(cell)])


def item_h3_cells(
    rec: SelectedItem,
    resolution: int,
    exact_intersections: bool,
) -> list[str]:
    geom = shape(rec.clipped_geometry)

    centers = set(
        h3.geo_to_cells(
            geom.__geo_interface__,
            resolution,
        )
    )

    if not exact_intersections:
        return sorted(centers)

    candidates = set(centers)
    for cell in centers:
        candidates.update(h3.grid_disk(cell, 1))

    exact = [
        cell
        for cell in candidates
        if h3_polygon(cell).intersects(geom)
    ]
    exact.sort()
    return exact


def h3_square_bounds_source_crs(
    cell: str,
    src_crs,
) -> tuple[float, float, float, float]:
    polygon = h3_polygon(cell)
    min_lon, min_lat, max_lon, max_lat = polygon.bounds

    left, bottom, right, top = transform_bounds(
        "EPSG:4326",
        src_crs,
        min_lon,
        min_lat,
        max_lon,
        max_lat,
        densify_pts=21,
    )

    cx = (left + right) / 2.0
    cy = (bottom + top) / 2.0
    side = max(right - left, top - bottom)

    return (
        cx - side / 2.0,
        cy - side / 2.0,
        cx + side / 2.0,
        cy + side / 2.0,
    )


def to_uint8_rgb(data: np.ndarray) -> np.ndarray:
    if data.dtype != np.uint8:
        if np.issubdtype(data.dtype, np.integer):
            max_value = np.iinfo(data.dtype).max
            if max_value != 255:
                data = np.clip(
                    data.astype(np.float32) / max_value * 255.0,
                    0,
                    255,
                ).astype(np.uint8)
            else:
                data = data.astype(np.uint8)
        else:
            data = np.clip(data, 0, 255).astype(np.uint8)

    return np.moveaxis(data, 0, -1)


def crop_h3_rgb(
    src: rasterio.io.DatasetReader,
    cell: str,
    image_size: int,
    resampling: Resampling,
) -> tuple[np.ndarray, float]:
    """
    Common baseline crop.

    The compression-quality test occurs AFTER this function so the measured
    DINO drift isolates JPEG compression.

    If the existing CrimeNet Gold crop semantics differ, replace this function
    with the exact Gold implementation before the national build.
    """
    if src.count < 3:
        raise ValueError(f"Source has only {src.count} bands")

    bounds = h3_square_bounds_source_crs(cell, src.crs)
    window = from_bounds(*bounds, transform=src.transform)

    data = src.read(
        indexes=(1, 2, 3),
        window=window,
        out_shape=(3, image_size, image_size),
        boundless=True,
        fill_value=0,
        resampling=resampling,
    )

    masks = src.read_masks(
        indexes=(1, 2, 3),
        window=window,
        out_shape=(3, image_size, image_size),
        boundless=True,
        resampling=Resampling.nearest,
    )

    valid = np.all(masks > 0, axis=0)
    valid_fraction = float(valid.mean())

    return to_uint8_rgb(data), valid_fraction


# =============================================================================
# JPEG
# =============================================================================

def encode_jpeg(
    rgb: np.ndarray,
    quality: int,
    subsampling: int,
) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(
        buffer,
        format="JPEG",
        quality=quality,
        subsampling=subsampling,
        optimize=False,
        progressive=False,
    )
    return buffer.getvalue()


def decode_jpeg(payload: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(payload)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def codec_encode_decode(
    rgb: np.ndarray,
    codec_name: str,
) -> tuple[bytes, np.ndarray]:
    cfg = CODECS[codec_name]
    payload = encode_jpeg(
        rgb,
        quality=int(cfg["quality"]),
        subsampling=int(cfg["subsampling"]),
    )
    return payload, decode_jpeg(payload)


def psnr_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    a = reference.astype(np.float32)
    b = candidate.astype(np.float32)
    mse = float(np.mean((a - b) ** 2))

    if mse <= 0.0:
        return float("inf")

    return float(20.0 * math.log10(255.0 / math.sqrt(mse)))


# =============================================================================
# DINOv3 quality evaluator
# =============================================================================

def resolve_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def load_dinov3(
    repo_dir: str,
    weights_spec: str,
    device: str,
):
    import torch

    repo = Path(repo_dir).expanduser().resolve()
    if not (repo / "hubconf.py").exists():
        raise SystemExit(
            f"--dinov3-repo must point to a local facebookresearch/dinov3 "
            f"checkout containing hubconf.py: {repo}"
        )

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    if weights_spec.lower() == "sat493m":
        from dinov3.hub.backbones import Weights
        weights: Any = Weights.SAT493M
    else:
        candidate = Path(weights_spec).expanduser()
        if candidate.exists():
            weights = str(candidate.resolve())
        else:
            # Official loader also accepts URLs.
            weights = weights_spec

    log(f"load DINOv3 ViT-L/16 SAT-493M -> {device}")

    model = torch.hub.load(
        str(repo),
        "dinov3_vitl16",
        source="local",
        weights=weights,
    )

    model.eval()
    model.to(device)

    return model


def rgb_batch_to_dino_tensor(
    images: list[np.ndarray],
    device: str,
):
    import torch

    array = np.stack(images, axis=0).astype(np.float32) / 255.0
    array = (array - SAT_MEAN[None, None, None, :]) / SAT_STD[
        None, None, None, :
    ]
    array = np.transpose(array, (0, 3, 1, 2))
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    return tensor.to(device=device, dtype=torch.float32)


def dino_embeddings(
    model,
    images: list[np.ndarray],
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return:
        cls:       [N,1024]
        patchmean: [N,1024]
        concat:    [N,2048]

    Float32 inference is intentional during the quality audit so JPEG drift is
    not confounded by bf16/fp16 numerical noise.
    """
    import torch

    cls_parts = []
    mean_parts = []
    concat_parts = []

    for start in range(0, len(images), batch_size):
        batch_images = images[start : start + batch_size]
        tensor = rgb_batch_to_dino_tensor(batch_images, device)

        with torch.inference_mode():
            features = model.forward_features(tensor)

            cls = features["x_norm_clstoken"].float()
            patch = features["x_norm_patchtokens"].float()
            patch_mean = patch.mean(dim=1)
            concat = torch.cat([cls, patch_mean], dim=-1)

        cls_parts.append(cls.cpu().numpy())
        mean_parts.append(patch_mean.cpu().numpy())
        concat_parts.append(concat.cpu().numpy())

        del tensor, features, cls, patch, patch_mean, concat

        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps" and hasattr(torch, "mps"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    return (
        np.concatenate(cls_parts, axis=0),
        np.concatenate(mean_parts, axis=0),
        np.concatenate(concat_parts, axis=0),
    )


def row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = np.linalg.norm(a, axis=1)
    bn = np.linalg.norm(b, axis=1)
    denom = np.maximum(an * bn, 1e-12)
    return np.sum(a * b, axis=1) / denom


def relative_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.linalg.norm(a, axis=1), 1e-12)
    return np.linalg.norm(a - b, axis=1) / denom


def normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x / denom


def similarity_matrix_correlation(
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> float:
    n = baseline.shape[0]
    if n < 3:
        return float("nan")

    a = normalize_rows(baseline)
    b = normalize_rows(candidate)

    sim_a = a @ a.T
    sim_b = b @ b.T

    tri = np.triu_indices(n, k=1)
    x = sim_a[tri]
    y = sim_b[tri]

    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])


def nearest_neighbor_recall_at_k(
    baseline: np.ndarray,
    candidate: np.ndarray,
    k: int,
) -> float:
    n = baseline.shape[0]
    if n <= 1:
        return float("nan")

    k = max(1, min(k, n - 1))

    a = normalize_rows(baseline)
    b = normalize_rows(candidate)

    sim_a = a @ a.T
    sim_b = b @ b.T

    np.fill_diagonal(sim_a, -np.inf)
    np.fill_diagonal(sim_b, -np.inf)

    top_a = np.argpartition(-sim_a, kth=k - 1, axis=1)[:, :k]
    top_b = np.argpartition(-sim_b, kth=k - 1, axis=1)[:, :k]

    recalls = []

    for row in range(n):
        sa = set(int(v) for v in top_a[row])
        sb = set(int(v) for v in top_b[row])
        recalls.append(len(sa & sb) / k)

    return float(np.mean(recalls))


def metric_summary(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {
            "min": float("nan"),
            "p01": float("nan"),
            "p05": float("nan"),
            "median": float("nan"),
            "mean": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }

    return {
        "min": float(np.min(finite)),
        "p01": percentile(finite, 1),
        "p05": percentile(finite, 5),
        "median": percentile(finite, 50),
        "mean": float(np.mean(finite)),
        "p95": percentile(finite, 95),
        "p99": percentile(finite, 99),
        "max": float(np.max(finite)),
    }


# =============================================================================
# Quality sample selection and extraction
# =============================================================================

def stratified_quality_items(
    selected: list[SelectedItem],
    count: int,
    seed: int,
) -> list[SelectedItem]:
    """
    Round-robin across states to avoid selecting a quality sample from only the
    largest states.
    """
    rng = random.Random(seed)

    by_state: dict[str, list[SelectedItem]] = {}
    for rec in selected:
        by_state.setdefault(rec.state, []).append(rec)

    states = sorted(by_state)
    rng.shuffle(states)

    for state in states:
        rng.shuffle(by_state[state])

    chosen: list[SelectedItem] = []
    depth = 0

    while len(chosen) < count:
        added = False
        for state in states:
            values = by_state[state]
            if depth < len(values):
                chosen.append(values[depth])
                added = True
                if len(chosen) >= count:
                    break
        if not added:
            break
        depth += 1

    return chosen


def sample_cells_for_item(
    rec: SelectedItem,
    h3_resolution: int,
    count: int,
    seed: int,
) -> list[str]:
    cells = item_h3_cells(
        rec,
        resolution=h3_resolution,
        exact_intersections=False,
    )

    if len(cells) <= count:
        return cells

    local_seed = (
        seed
        ^ int.from_bytes(rec.item_id.encode("utf-8")[:8].ljust(8, b"\0"), "little")
    )
    rng = random.Random(local_seed)
    return sorted(rng.sample(cells, count))


def collect_quality_samples(
    quality_items: list[SelectedItem],
    work_dir: Path,
    h3_resolution: int,
    image_size: int,
    cells_per_item: int,
    resampling: Resampling,
    download_attempts: int,
    seed: int,
    min_valid_fraction: float,
) -> list[QualitySample]:
    root = work_dir / "quality_sources"
    root.mkdir(parents=True, exist_ok=True)

    samples: list[QualitySample] = []

    for index, rec in enumerate(quality_items, 1):
        log(
            f"quality source {index}/{len(quality_items)}: "
            f"{rec.state} {rec.item_id}"
        )

        item_dir = root / rec.state / safe_component(rec.item_id)
        item_dir.mkdir(parents=True, exist_ok=True)

        suffix = Path(urlsplit(rec.unsigned_href).path).suffix.lower() or ".tif"
        source_path = item_dir / f"source{suffix}"

        download_source_cog(
            rec,
            source_path,
            attempts=download_attempts,
        )

        cells = sample_cells_for_item(
            rec,
            h3_resolution=h3_resolution,
            count=cells_per_item,
            seed=seed,
        )

        with rasterio.open(source_path) as src:
            for cell in cells:
                try:
                    rgb, valid_fraction = crop_h3_rgb(
                        src,
                        cell,
                        image_size=image_size,
                        resampling=resampling,
                    )

                    if valid_fraction < min_valid_fraction:
                        continue

                    decoded: dict[str, np.ndarray] = {}
                    sizes: dict[str, int] = {}
                    psnrs: dict[str, float] = {}

                    for codec_name in CODECS:
                        payload, candidate = codec_encode_decode(
                            rgb,
                            codec_name,
                        )

                        decoded[codec_name] = candidate
                        sizes[codec_name] = len(payload)
                        psnrs[codec_name] = psnr_db(rgb, candidate)

                    samples.append(
                        QualitySample(
                            state=rec.state,
                            item_id=rec.item_id,
                            h3_cell_id=cell,
                            capture_time_utc=rec.capture_time_utc,
                            gsd_m=rec.gsd_m,
                            valid_fraction=valid_fraction,
                            baseline_rgb=rgb,
                            compressed_rgb=decoded,
                            compressed_bytes=sizes,
                            psnr_db=psnrs,
                        )
                    )

                except Exception as exc:
                    log(
                        f"WARN quality crop failed {rec.item_id} {cell}: "
                        f"{type(exc).__name__}: {exc}"
                    )

        # Quality sources are only temporary.
        source_path.unlink(missing_ok=True)
        try:
            item_dir.rmdir()
        except OSError:
            pass

    return samples


def run_quality_audit(
    selected: list[SelectedItem],
    args: argparse.Namespace,
    resampling: Resampling,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    repo = args.dinov3_repo or os.environ.get("DINOV3_REPO")
    weights = args.dinov3_weights or os.environ.get("DINOV3_WEIGHTS")

    if not repo:
        raise SystemExit(
            "Quality mode requires --dinov3-repo or DINOV3_REPO."
        )
    if not weights:
        raise SystemExit(
            "Quality mode requires --dinov3-weights or DINOV3_WEIGHTS."
        )

    quality_items = stratified_quality_items(
        selected,
        count=args.quality_items,
        seed=args.quality_seed,
    )

    log(
        f"quality audit: {len(quality_items)} source COGs x "
        f"up to {args.quality_cells_per_item} H3 cells"
    )

    samples = collect_quality_samples(
        quality_items=quality_items,
        work_dir=Path(args.work_dir).expanduser().resolve(),
        h3_resolution=args.h3_resolution,
        image_size=args.image_size,
        cells_per_item=args.quality_cells_per_item,
        resampling=resampling,
        download_attempts=args.download_attempts,
        seed=args.quality_seed,
        min_valid_fraction=args.quality_min_valid_fraction,
    )

    if len(samples) < args.quality_min_samples:
        raise RuntimeError(
            f"Only {len(samples)} valid quality samples were collected; "
            f"need at least {args.quality_min_samples}."
        )

    device = resolve_device(args.device)
    model = load_dinov3(
        repo_dir=repo,
        weights_spec=weights,
        device=device,
    )

    baseline_images = [sample.baseline_rgb for sample in samples]

    log(f"DINO baseline embeddings: N={len(samples):,}")
    base_cls, base_mean, base_concat = dino_embeddings(
        model,
        baseline_images,
        device=device,
        batch_size=args.dino_batch_size,
    )

    candidate_embeddings: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for codec_name in CODECS:
        log(f"DINO compressed embeddings: {codec_name}")
        candidate_images = [
            sample.compressed_rgb[codec_name]
            for sample in samples
        ]

        candidate_embeddings[codec_name] = dino_embeddings(
            model,
            candidate_images,
            device=device,
            batch_size=args.dino_batch_size,
        )

    del model

    report: dict[str, Any] = {
        "format": "crimenet-naip-jpeg-dinov3-quality-v1",
        "created_at_utc": iso_z(dt.datetime.now(dt.timezone.utc)),
        "sample_count": len(samples),
        "source_item_count": len({sample.item_id for sample in samples}),
        "state_count": len({sample.state for sample in samples}),
        "states": sorted({sample.state for sample in samples}),
        "image_size": args.image_size,
        "h3_resolution": args.h3_resolution,
        "crop_resampling": args.resampling,
        "dino_model": "dinov3_vitl16_sat493m",
        "dino_embedding": "x_norm_clstoken || mean(x_norm_patchtokens)",
        "dino_embedding_dim": int(base_concat.shape[1]),
        "dino_device": device,
        "dino_dtype": "float32_quality_audit",
        "sat_mean": SAT_MEAN.tolist(),
        "sat_std": SAT_STD.tolist(),
        "quality_gate": {
            "min_median_concat_cosine": args.min_median_concat_cosine,
            "min_p01_concat_cosine": args.min_p01_concat_cosine,
            "min_nn_recall_at_k": args.min_nn_recall,
            "nn_k": args.nn_k,
        },
        "codecs": {},
    }

    rows: list[dict[str, Any]] = []

    for codec_name, (cand_cls, cand_mean, cand_concat) in candidate_embeddings.items():
        cls_cos = row_cosine(base_cls, cand_cls)
        mean_cos = row_cosine(base_mean, cand_mean)
        concat_cos = row_cosine(base_concat, cand_concat)
        rel_l2 = relative_l2(base_concat, cand_concat)

        sim_corr = similarity_matrix_correlation(
            base_concat,
            cand_concat,
        )

        nn_recall = nearest_neighbor_recall_at_k(
            base_concat,
            cand_concat,
            k=args.nn_k,
        )

        jpeg_sizes = np.asarray(
            [sample.compressed_bytes[codec_name] for sample in samples],
            dtype=np.float64,
        )

        psnrs = np.asarray(
            [sample.psnr_db[codec_name] for sample in samples],
            dtype=np.float64,
        )

        cls_summary = metric_summary(cls_cos)
        mean_summary = metric_summary(mean_cos)
        concat_summary = metric_summary(concat_cos)
        l2_summary = metric_summary(rel_l2)
        size_summary = metric_summary(jpeg_sizes)
        psnr_summary = metric_summary(psnrs)

        passed = bool(
            concat_summary["median"] >= args.min_median_concat_cosine
            and concat_summary["p01"] >= args.min_p01_concat_cosine
            and nn_recall >= args.min_nn_recall
        )

        codec_cfg = CODECS[codec_name]

        report["codecs"][codec_name] = {
            "jpeg_quality": codec_cfg["quality"],
            "jpeg_subsampling": codec_cfg["subsampling"],
            "passed_quality_gate": passed,
            "cls_cosine": cls_summary,
            "patchmean_cosine": mean_summary,
            "concat_2048_cosine": concat_summary,
            "concat_relative_l2": l2_summary,
            "pairwise_similarity_correlation": sim_corr,
            f"nearest_neighbor_recall_at_{args.nn_k}": nn_recall,
            "jpeg_bytes": size_summary,
            "jpeg_kib_mean": float(np.mean(jpeg_sizes) / 1024.0),
            "jpeg_kib_median": float(np.median(jpeg_sizes) / 1024.0),
            "pixel_psnr_db": psnr_summary,
        }

        for index, sample in enumerate(samples):
            rows.append(
                {
                    "state": sample.state,
                    "item_id": sample.item_id,
                    "h3_cell_id": sample.h3_cell_id,
                    "capture_time_utc": sample.capture_time_utc,
                    "gsd_m": sample.gsd_m,
                    "valid_fraction": sample.valid_fraction,
                    "codec": codec_name,
                    "jpeg_bytes": sample.compressed_bytes[codec_name],
                    "psnr_db": sample.psnr_db[codec_name],
                    "cls_cosine": float(cls_cos[index]),
                    "patchmean_cosine": float(mean_cos[index]),
                    "concat_cosine": float(concat_cos[index]),
                    "concat_relative_l2": float(rel_l2[index]),
                }
            )

    production_codec = args.production_codec
    report["production_codec"] = production_codec
    report["production_codec_passed"] = bool(
        report["codecs"][production_codec]["passed_quality_gate"]
    )

    if args.projection_h3_count > 0:
        for codec_name in CODECS:
            mean_bytes = report["codecs"][codec_name]["jpeg_bytes"]["mean"]
            projected_bytes = mean_bytes * args.projection_h3_count
            report["codecs"][codec_name]["projection"] = {
                "h3_count": args.projection_h3_count,
                "payload_TB_decimal": projected_bytes / 1e12,
                "payload_TiB_binary": projected_bytes / 1024**4,
            }

    print("\n=== DINOv3 JPEG QUALITY AUDIT ===")
    print(f"Samples: {len(samples):,}")
    print(
        f"Sources/states: "
        f"{report['source_item_count']:,}/{report['state_count']:,}"
    )
    print(f"Embedding dim: {base_concat.shape[1]:,}")
    print(f"Device: {device}")

    for codec_name in CODECS:
        r = report["codecs"][codec_name]
        print(f"\n{codec_name}:")
        print(
            f"  mean JPEG: {r['jpeg_kib_mean']:.1f} KiB "
            f"(median {r['jpeg_kib_median']:.1f} KiB)"
        )
        print(
            f"  concat cosine: "
            f"median={r['concat_2048_cosine']['median']:.8f}, "
            f"p01={r['concat_2048_cosine']['p01']:.8f}, "
            f"min={r['concat_2048_cosine']['min']:.8f}"
        )
        print(
            f"  CLS cosine median: "
            f"{r['cls_cosine']['median']:.8f}"
        )
        print(
            f"  patch-mean cosine median: "
            f"{r['patchmean_cosine']['median']:.8f}"
        )
        print(
            f"  relative L2 median: "
            f"{r['concat_relative_l2']['median']:.6f}"
        )
        print(
            f"  similarity-matrix corr: "
            f"{r['pairwise_similarity_correlation']:.8f}"
        )
        print(
            f"  NN recall@{args.nn_k}: "
            f"{r[f'nearest_neighbor_recall_at_{args.nn_k}']:.6f}"
        )
        print(
            f"  PSNR median: "
            f"{r['pixel_psnr_db']['median']:.2f} dB"
        )
        print(
            f"  QUALITY GATE: "
            f"{'PASS' if r['passed_quality_gate'] else 'FAIL'}"
        )

        if "projection" in r:
            p = r["projection"]
            print(
                f"  projected payload @ {p['h3_count']:,} H3: "
                f"{p['payload_TB_decimal']:.2f} TB "
                f"({p['payload_TiB_binary']:.2f} TiB)"
            )

    return report, rows


def persist_quality_report(
    report: dict[str, Any],
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    work_dir = Path(args.work_dir).expanduser().resolve()
    output_dir = work_dir / "quality_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "quality_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    )

    rows_path = output_dir / "quality_samples.parquet"
    pl.DataFrame(rows).write_parquet(
        rows_path,
        compression="zstd",
        statistics=True,
    )

    quality_prefix = f"{args.b2_prefix.strip('/')}/quality"

    upload_verified(
        report_path,
        args.bucket,
        f"{quality_prefix}/quality_report.json",
        content_type="application/json",
        attempts=5,
    )

    upload_verified(
        rows_path,
        args.bucket,
        f"{quality_prefix}/quality_samples.parquet",
        content_type="application/vnd.apache.parquet",
        attempts=5,
    )

    log(f"quality report local: {report_path}")


# =============================================================================
# Durable tar shard build
# =============================================================================

def tar_add_bytes(
    tar: tarfile.TarFile,
    name: str,
    payload: bytes,
) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mtime = 0
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(payload))


def shard_key(
    prefix: str,
    rec: SelectedItem,
    codec_name: str,
) -> str:
    return (
        f"{prefix.strip('/')}/"
        f"codec={codec_name}/"
        f"state={rec.state}/"
        f"acquisition_year={rec.year}/"
        f"tile_key={safe_component(rec.stable_tile_key)}/"
        f"item_id={safe_component(rec.item_id)}.tar"
    )


def build_shard(
    rec: SelectedItem,
    source_path: Path,
    tar_path: Path,
    cfg: BuildRuntime,
) -> dict[str, Any]:
    cells = item_h3_cells(
        rec,
        resolution=cfg.h3_resolution,
        exact_intersections=cfg.exact_intersections,
    )

    if cfg.limit_cells_per_item > 0:
        cells = cells[: cfg.limit_cells_per_item]

    codec_cfg = CODECS[cfg.codec_name]

    metadata_lines: list[str] = []
    jpeg_bytes_total = 0
    successful = 0
    low_valid = 0
    failed = 0

    tar_path.parent.mkdir(parents=True, exist_ok=True)
    tar_path.unlink(missing_ok=True)

    with rasterio.open(source_path) as src:
        with tarfile.open(tar_path, mode="w") as tar:
            for index, cell in enumerate(cells, 1):
                try:
                    rgb, valid_fraction = crop_h3_rgb(
                        src,
                        cell,
                        image_size=cfg.image_size,
                        resampling=cfg.resampling,
                    )

                    payload = encode_jpeg(
                        rgb,
                        quality=int(codec_cfg["quality"]),
                        subsampling=int(codec_cfg["subsampling"]),
                    )

                    member = f"crops/{cell}.jpg"
                    tar_add_bytes(tar, member, payload)

                    jpeg_bytes_total += len(payload)
                    successful += 1

                    if valid_fraction < 0.95:
                        low_valid += 1

                    metadata_lines.append(
                        json.dumps(
                            {
                                "h3_cell_id": cell,
                                "h3_resolution": cfg.h3_resolution,
                                "image_size": cfg.image_size,
                                "source_item_id": rec.item_id,
                                "source_state": rec.state,
                                "capture_time_utc": rec.capture_time_utc,
                                "source_gsd_m": rec.gsd_m,
                                "codec": cfg.codec_name,
                                "jpeg_quality": codec_cfg["quality"],
                                "jpeg_subsampling": codec_cfg["subsampling"],
                                "resampling": cfg.resampling.name,
                                "valid_fraction": valid_fraction,
                                "member": member,
                                "jpeg_bytes": len(payload),
                                "status": "ok",
                            },
                            separators=(",", ":"),
                        )
                    )

                except Exception as exc:
                    failed += 1
                    metadata_lines.append(
                        json.dumps(
                            {
                                "h3_cell_id": cell,
                                "source_item_id": rec.item_id,
                                "status": "crop_error",
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                            separators=(",", ":"),
                        )
                    )

                if index % 250 == 0:
                    avg_kib = jpeg_bytes_total / max(1, successful) / 1024
                    log(
                        f"crop {rec.item_id}: "
                        f"{index:,}/{len(cells):,}; "
                        f"avg={avg_kib:.1f} KiB"
                    )

            metadata_payload = (
                "\n".join(metadata_lines) + ("\n" if metadata_lines else "")
            ).encode("utf-8")
            tar_add_bytes(tar, "metadata.jsonl", metadata_payload)

            manifest = {
                "format": "crimenet-naip-h3-jpeg-shard-v2",
                "source_item_id": rec.item_id,
                "state": rec.state,
                "stable_tile_key": rec.stable_tile_key,
                "capture_time_utc": rec.capture_time_utc,
                "source_gsd_m": rec.gsd_m,
                "h3_resolution": cfg.h3_resolution,
                "image_size": cfg.image_size,
                "codec": cfg.codec_name,
                "jpeg_quality": codec_cfg["quality"],
                "jpeg_subsampling": codec_cfg["subsampling"],
                "resampling": cfg.resampling.name,
                "cell_selection": (
                    "polygon_intersection"
                    if cfg.exact_intersections
                    else "center_containment"
                ),
                "requested_cells": len(cells),
                "successful_crops": successful,
                "failed_crops": failed,
                "low_valid_fraction_lt_0_95": low_valid,
                "jpeg_payload_bytes": jpeg_bytes_total,
                "average_jpeg_bytes": (
                    jpeg_bytes_total / successful
                    if successful
                    else None
                ),
            }

            tar_add_bytes(
                tar,
                "manifest.json",
                json.dumps(
                    manifest,
                    indent=2,
                    sort_keys=True,
                ).encode("utf-8"),
            )

    return {
        "requested_cells": len(cells),
        "successful_crops": successful,
        "failed_crops": failed,
        "low_valid_crops": low_valid,
        "jpeg_bytes": jpeg_bytes_total,
        "avg_jpeg_bytes": (
            jpeg_bytes_total / successful
            if successful
            else None
        ),
        "tar_bytes": tar_path.stat().st_size,
    }


def process_build_item(
    rec: SelectedItem,
    cfg: BuildRuntime,
) -> dict[str, Any]:
    started = time.perf_counter()
    key = shard_key(cfg.prefix, rec, cfg.codec_name)

    existing = head_object(cfg.bucket, key)
    if existing is not None and int(existing["ContentLength"]) > 0:
        return {
            "state": rec.state,
            "item_id": rec.item_id,
            "stable_tile_key": rec.stable_tile_key,
            "capture_time_utc": rec.capture_time_utc,
            "gsd_m": rec.gsd_m,
            "codec": cfg.codec_name,
            "b2_key": key,
            "status": "exists",
            "requested_cells": None,
            "successful_crops": None,
            "failed_crops": None,
            "low_valid_crops": None,
            "jpeg_bytes": None,
            "avg_jpeg_bytes": None,
            "tar_bytes": int(existing["ContentLength"]),
            "source_bytes": None,
            "elapsed_seconds": time.perf_counter() - started,
            "error": None,
        }

    item_dir = (
        cfg.work_dir
        / "staging"
        / rec.state
        / safe_component(rec.item_id)
    )
    item_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(urlsplit(rec.unsigned_href).path).suffix.lower() or ".tif"
    source_path = item_dir / f"source{suffix}"
    tar_path = item_dir / f"h3_crops_{cfg.codec_name}.tar"

    try:
        source_bytes = download_source_cog(
            rec,
            source_path,
            attempts=cfg.download_attempts,
        )

        stats = build_shard(
            rec,
            source_path,
            tar_path,
            cfg,
        )

        if stats["successful_crops"] == 0:
            raise RuntimeError("No successful H3 crops produced")

        remote_bytes = upload_verified(
            tar_path,
            cfg.bucket,
            key,
            content_type="application/x-tar",
            attempts=cfg.upload_attempts,
            metadata={
                "source_item_id": rec.item_id,
                "capture_time_utc": rec.capture_time_utc,
                "h3_resolution": cfg.h3_resolution,
                "image_size": cfg.image_size,
                "codec": cfg.codec_name,
                "crop_count": stats["successful_crops"],
            },
        )

        # Delete source and tar only after B2 size verification.
        source_path.unlink(missing_ok=True)
        tar_path.unlink(missing_ok=True)

        try:
            item_dir.rmdir()
        except OSError:
            pass

        return {
            "state": rec.state,
            "item_id": rec.item_id,
            "stable_tile_key": rec.stable_tile_key,
            "capture_time_utc": rec.capture_time_utc,
            "gsd_m": rec.gsd_m,
            "codec": cfg.codec_name,
            "b2_key": key,
            "status": "uploaded",
            "requested_cells": stats["requested_cells"],
            "successful_crops": stats["successful_crops"],
            "failed_crops": stats["failed_crops"],
            "low_valid_crops": stats["low_valid_crops"],
            "jpeg_bytes": stats["jpeg_bytes"],
            "avg_jpeg_bytes": stats["avg_jpeg_bytes"],
            "tar_bytes": remote_bytes,
            "source_bytes": source_bytes,
            "elapsed_seconds": time.perf_counter() - started,
            "error": None,
        }

    except Exception as exc:
        if not cfg.keep_source_on_failure:
            source_path.unlink(missing_ok=True)
            tar_path.unlink(missing_ok=True)

        return {
            "state": rec.state,
            "item_id": rec.item_id,
            "stable_tile_key": rec.stable_tile_key,
            "capture_time_utc": rec.capture_time_utc,
            "gsd_m": rec.gsd_m,
            "codec": cfg.codec_name,
            "b2_key": key,
            "status": "error",
            "requested_cells": None,
            "successful_crops": None,
            "failed_crops": None,
            "low_valid_crops": None,
            "jpeg_bytes": None,
            "avg_jpeg_bytes": None,
            "tar_bytes": (
                tar_path.stat().st_size
                if tar_path.exists()
                else None
            ),
            "source_bytes": (
                source_path.stat().st_size
                if source_path.exists()
                else None
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


BUILD_RESULT_SCHEMA = {
    "state": pl.Utf8,
    "item_id": pl.Utf8,
    "stable_tile_key": pl.Utf8,
    "capture_time_utc": pl.Utf8,
    "gsd_m": pl.Float64,
    "codec": pl.Utf8,
    "b2_key": pl.Utf8,
    "status": pl.Utf8,
    "requested_cells": pl.Int64,
    "successful_crops": pl.Int64,
    "failed_crops": pl.Int64,
    "low_valid_crops": pl.Int64,
    "jpeg_bytes": pl.Int64,
    "avg_jpeg_bytes": pl.Float64,
    "tar_bytes": pl.Int64,
    "source_bytes": pl.Int64,
    "elapsed_seconds": pl.Float64,
    "error": pl.Utf8,
}


def persist_build_results(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    if not rows:
        return

    work_dir = Path(args.work_dir).expanduser().resolve()
    local = work_dir / "build_results.parquet"

    pl.DataFrame(
        rows,
        schema=BUILD_RESULT_SCHEMA,
    ).write_parquet(
        local,
        compression="zstd",
        statistics=True,
    )

    upload_verified(
        local,
        args.bucket,
        (
            f"{args.b2_prefix.strip('/')}/"
            f"codec={args.production_codec}/manifests/build_results.parquet"
        ),
        content_type="application/vnd.apache.parquet",
        attempts=5,
    )

    local.unlink(missing_ok=True)


def run_build(
    selected: list[SelectedItem],
    args: argparse.Namespace,
    resampling: Resampling,
) -> int:
    run_items = (
        selected[: args.limit_items]
        if args.limit_items > 0
        else selected
    )

    cfg = BuildRuntime(
        bucket=args.bucket,
        prefix=args.b2_prefix,
        work_dir=Path(args.work_dir).expanduser().resolve(),
        h3_resolution=args.h3_resolution,
        image_size=args.image_size,
        codec_name=args.production_codec,
        resampling=resampling,
        exact_intersections=args.exact_intersections,
        limit_cells_per_item=args.limit_cells_per_item,
        download_attempts=args.download_attempts,
        upload_attempts=args.upload_attempts,
        keep_source_on_failure=not args.delete_source_on_failure,
    )

    log(
        f"build {len(run_items):,} latest NAIP sources; "
        f"codec={args.production_codec}; workers={args.workers}"
    )

    results: list[dict[str, Any]] = []

    total_crops = 0
    total_jpeg_bytes = 0
    total_tar_bytes = 0
    uploaded = 0
    existed = 0
    failed = 0

    with cf.ThreadPoolExecutor(
        max_workers=args.workers,
        thread_name_prefix="naip-h3",
    ) as pool:

        futures = {
            pool.submit(process_build_item, rec, cfg): rec
            for rec in run_items
        }

        for completed, future in enumerate(
            cf.as_completed(futures),
            1,
        ):
            rec = futures[future]

            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "state": rec.state,
                    "item_id": rec.item_id,
                    "stable_tile_key": rec.stable_tile_key,
                    "capture_time_utc": rec.capture_time_utc,
                    "gsd_m": rec.gsd_m,
                    "codec": args.production_codec,
                    "b2_key": shard_key(
                        args.b2_prefix,
                        rec,
                        args.production_codec,
                    ),
                    "status": "error",
                    "requested_cells": None,
                    "successful_crops": None,
                    "failed_crops": None,
                    "low_valid_crops": None,
                    "jpeg_bytes": None,
                    "avg_jpeg_bytes": None,
                    "tar_bytes": None,
                    "source_bytes": None,
                    "elapsed_seconds": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }

            results.append(row)

            if row["status"] == "uploaded":
                uploaded += 1
                total_crops += int(row["successful_crops"] or 0)
                total_jpeg_bytes += int(row["jpeg_bytes"] or 0)
                total_tar_bytes += int(row["tar_bytes"] or 0)
            elif row["status"] == "exists":
                existed += 1
            else:
                failed += 1

            if completed % 10 == 0 or completed == len(run_items):
                avg_kib = (
                    total_jpeg_bytes / total_crops / 1024
                    if total_crops
                    else 0.0
                )

                log(
                    f"progress {completed:,}/{len(run_items):,}: "
                    f"uploaded={uploaded:,}, exists={existed:,}, "
                    f"failed={failed:,}, crops={total_crops:,}, "
                    f"avg={avg_kib:.1f} KiB, "
                    f"tar={total_tar_bytes / 1024**3:.2f} GiB"
                )

    persist_build_results(results, args)

    print("\n=== BUILD COMPLETE ===")
    print(f"Uploaded shards: {uploaded:,}")
    print(f"Existing shards: {existed:,}")
    print(f"Failed source items: {failed:,}")
    print(f"Successful crops this run: {total_crops:,}")

    if total_crops:
        print(
            f"Measured mean JPEG size: "
            f"{total_jpeg_bytes / total_crops / 1024:.1f} KiB"
        )
        print(
            f"Tar storage this run: "
            f"{total_tar_bytes / 1024**3:.2f} GiB"
        )

    if failed:
        print(
            "\nRerun the same command to resume. Completed B2 shards are "
            "skipped and partial source downloads are retained by default."
        )
        return 2

    return 0


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "National latest-NAIP -> 512x512 H3 JPEG shards with a DINOv3 "
            "SAT-493M compression-quality audit."
        )
    )

    parser.add_argument(
        "--mode",
        choices=("quality", "quality-and-build", "build", "dry-run"),
        default="quality",
        help=(
            "quality: ML compression audit only; "
            "quality-and-build: audit then build only if production codec "
            "passes; build: production build without rerunning DINO audit; "
            "dry-run: STAC discovery only."
        ),
    )

    parser.add_argument(
        "--bucket",
        default=os.environ.get("B2_BUCKET", "crimenet-data"),
    )

    parser.add_argument(
        "--b2-prefix",
        default=DEFAULT_B2_PREFIX,
    )

    parser.add_argument(
        "--work-dir",
        default=str(Path.home() / "crimenet-naip-h3-quality"),
    )

    parser.add_argument(
        "--states",
        default="",
        help="Optional comma-separated state abbreviations.",
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=DEFAULT_START_YEAR,
    )

    parser.add_argument(
        "--end-year",
        type=int,
        default=DEFAULT_END_YEAR,
    )

    parser.add_argument(
        "--h3-resolution",
        type=int,
        default=H3_RESOLUTION,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=IMAGE_SIZE,
    )

    parser.add_argument(
        "--resampling",
        choices=("nearest", "bilinear", "bicubic", "lanczos"),
        default="bicubic",
    )

    parser.add_argument(
        "--production-codec",
        choices=tuple(CODECS),
        default="jpeg95_444",
        help="Durable codec used by build mode.",
    )

    # ML quality audit
    parser.add_argument(
        "--dinov3-repo",
        default="",
        help="Local facebookresearch/dinov3 checkout.",
    )

    parser.add_argument(
        "--dinov3-weights",
        default="",
        help=(
            "SAT-493M checkpoint path/URL, or literal 'sat493m'. "
            "Official local filename should contain eadcf0ff."
        ),
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
    )

    parser.add_argument(
        "--dino-batch-size",
        type=int,
        default=4,
        help="Quality-audit DINO batch. Increase on large CUDA GPUs.",
    )

    parser.add_argument(
        "--quality-items",
        type=int,
        default=12,
        help="Representative latest source COGs sampled across states.",
    )

    parser.add_argument(
        "--quality-cells-per-item",
        type=int,
        default=24,
    )

    parser.add_argument(
        "--quality-min-samples",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--quality-min-valid-fraction",
        type=float,
        default=0.98,
    )

    parser.add_argument(
        "--quality-seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--min-median-concat-cosine",
        type=float,
        default=0.999,
    )

    parser.add_argument(
        "--min-p01-concat-cosine",
        type=float,
        default=0.995,
    )

    parser.add_argument(
        "--nn-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--min-nn-recall",
        type=float,
        default=0.95,
        help=(
            "Quality gate on preservation of DINO nearest-neighbor structure."
        ),
    )

    parser.add_argument(
        "--projection-h3-count",
        type=int,
        default=0,
        help=(
            "Optional national H3 count. If nonzero, quality report projects "
            "JPEG payload storage from measured mean bytes/crop."
        ),
    )

    # Build
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--exact-intersections",
        action="store_true",
        help=(
            "Include every H3 polygon intersecting each COG footprint. "
            "Default center containment reduces edge duplication."
        ),
    )

    parser.add_argument(
        "--download-attempts",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--upload-attempts",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--limit-items",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--limit-cells-per-item",
        type=int,
        default=0,
        help="Testing only.",
    )

    parser.add_argument(
        "--delete-source-on-failure",
        action="store_true",
        help=(
            "Default retains partial source COGs so reruns can resume. "
            "Set this only when local disk pressure is more important."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    env_required("B2_KEY_ID")
    env_required("B2_APPLICATION_KEY")
    env_required("B2_ENDPOINT_URL")

    if args.start_year > args.end_year:
        raise SystemExit("--start-year must be <= --end-year")

    if args.workers <= 0:
        raise SystemExit("--workers must be positive")

    if args.dino_batch_size <= 0:
        raise SystemExit("--dino-batch-size must be positive")

    if args.quality_items <= 0:
        raise SystemExit("--quality-items must be positive")

    if args.quality_cells_per_item <= 0:
        raise SystemExit("--quality-cells-per-item must be positive")

    resampling_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "bicubic": Resampling.cubic,
        "lanczos": Resampling.lanczos,
    }
    resampling = resampling_map[args.resampling]

    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    get_b2_client().head_bucket(Bucket=args.bucket)
    log(f"B2 access OK: b2://{args.bucket}")

    requested: set[str] | None = None
    if args.states.strip():
        requested = {
            token.strip().upper()
            for token in args.states.split(",")
            if token.strip()
        }

    states = load_states(work_dir, requested)

    selected, discovery_stats = discover_latest(
        states,
        args.start_year,
        args.end_year,
    )

    log(f"selected latest source items: {len(selected):,}")

    put_json(
        args.bucket,
        f"{args.b2_prefix.strip('/')}/manifests/selection_summary.json",
        {
            "format": "crimenet-naip-h3-quality-selection-v2",
            "created_at_utc": iso_z(dt.datetime.now(dt.timezone.utc)),
            "selected_source_items": len(selected),
            "start_year": args.start_year,
            "end_year": args.end_year,
            "h3_resolution": args.h3_resolution,
            "image_size": args.image_size,
            "resampling": args.resampling,
            "production_codec": args.production_codec,
            "states": discovery_stats,
        },
    )

    if args.mode == "dry-run":
        print("\n=== DRY RUN ===")
        print(f"States: {len(states):,}")
        print(f"Latest source COGs: {len(selected):,}")
        print("No imagery downloaded.")
        return 0

    quality_report: dict[str, Any] | None = None

    if args.mode in ("quality", "quality-and-build"):
        quality_report, quality_rows = run_quality_audit(
            selected,
            args,
            resampling,
        )

        persist_quality_report(
            quality_report,
            quality_rows,
            args,
        )

        production_passed = bool(
            quality_report["production_codec_passed"]
        )

        print(
            f"\nProduction codec {args.production_codec}: "
            f"{'PASS' if production_passed else 'FAIL'}"
        )

        if args.mode == "quality":
            return 0 if production_passed else 3

        if not production_passed:
            print(
                "\nBuild blocked because the selected production codec failed "
                "the configured DINO quality gate."
            )
            return 3

    return run_build(
        selected,
        args,
        resampling,
    )


if __name__ == "__main__":
    raise SystemExit(main())
