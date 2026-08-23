"""
CrimeNet national NAIP bootstrap: mirror the latest available original NAIP
Cloud-Optimized GeoTIFF (COG) for each spatial NAIP tile from Microsoft
Planetary Computer into Backblaze B2.

This is the final inference-oriented ingestion path:

Planetary Computer STAC
    -> select newest NAIP acquisition per stable spatial tile
    -> download the original `image` COG byte-for-byte
    -> upload the unchanged COG to Backblaze B2
    -> verify remote size
    -> delete local staging file
    -> write manifests

There is NO H3 enumeration, NO raster decoding, NO clipping, NO reprojection,
NO resampling, and NO COG reconstruction during ingestion.

Later, a GPU/feature job can read H3-r9 windows directly from these native
COGs and apply the same 512x512 preprocessing used by the existing DINOv3
pipeline.

Required environment variables
------------------------------
B2_KEY_ID
B2_APPLICATION_KEY
B2_ENDPOINT_URL
B2_BUCKET             optional; defaults to crimenet-data

Python dependencies
-------------------
uv add boto3 requests pystac-client pystac planetary-computer shapely pyshp polars

Examples
--------
# Catalog/selection only:
python scripts/seed_crimenet_national_naip_latest_cogs_b2.py \
  --work-dir "$HOME/crimenet-naip-latest" \
  --dry-run

# Texas smoke test:
python scripts/seed_crimenet_national_naip_latest_cogs_b2.py \
  --states TX \
  --limit-items 5 \
  --workers 2

# Full national run:
python scripts/seed_crimenet_national_naip_latest_cogs_b2.py \
  --workers 4
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import threading
import time
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import boto3
import planetary_computer
import polars as pl
import pystac_client
import requests
import shapefile
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry


# =============================================================================
# Configuration
# =============================================================================

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "naip"
ASSET_KEY = "image"

# Planetary Computer currently advertises NAIP through 2023-12-31.
DEFAULT_START_YEAR = 2018
DEFAULT_END_YEAR = 2023

DEFAULT_B2_PREFIX = "bronze/imagery/naip/national/latest"

STATE_BOUNDARY_URL = (
    "https://www2.census.gov/geo/tiger/GENZ2024/shp/"
    "cb_2024_us_state_20m.zip"
)

# 50 states + DC. Planetary Computer's NAIP collection has no Alaska coverage,
# but keeping AK in the domain is useful: it will simply report zero selected.
STATE_FIPS_50_DC = {
    "01","02","04","05","06","08","09","10","11","12","13","15","16",
    "17","18","19","20","21","22","23","24","25","26","27","28","29",
    "30","31","32","33","34","35","36","37","38","39","40","41","42",
    "44","45","46","47","48","49","50","51","53","54","55","56",
}

DATE_TOKEN = re.compile(r"^(19|20)\d{6}$")
USER_AGENT = "CrimeNet-national-NAIP-latest-COGs/1.0"

PRINT_LOCK = threading.Lock()
TLS = threading.local()

TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=64 * 1024**2,
    multipart_chunksize=128 * 1024**2,
    max_concurrency=2,
    use_threads=True,
)


# =============================================================================
# Models
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
    bbox: list[float] | None
    geometry: dict[str, Any] | None
    unsigned_image_href: str
    source_type: str | None
    source_roles: list[str]
    source_file_size_hint: int | None
    source_etag_hint: str | None

    @property
    def year(self) -> int:
        return parse_datetime(self.capture_time_utc).year


# =============================================================================
# Generic helpers
# =============================================================================

def log(message: str) -> None:
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with PRINT_LOCK:
        print(f"[{now}] {message}", flush=True)


def env_required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso_z(value: dt.datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def parse_datetime(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def get_requests_session() -> requests.Session:
    session = getattr(TLS, "requests_session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=32,
            pool_maxsize=32,
            max_retries=0,  # retries are explicit below
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
# Backblaze helpers
# =============================================================================

def head_b2_object(bucket: str, key: str) -> dict[str, Any] | None:
    client = get_b2_client()
    try:
        return client.head_object(Bucket=bucket, Key=key)
    except Exception as exc:
        response = getattr(exc, "response", None)
        status = (
            response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if response else None
        )
        text = str(exc)
        if status in (403, 404) or "404" in text or "NoSuchKey" in text:
            return None
        raise


def abort_multipart_for_key(bucket: str, key: str) -> int:
    client = get_b2_client()
    aborted = 0

    try:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": key}

        while True:
            response = client.list_multipart_uploads(**kwargs)

            for upload in response.get("Uploads", []):
                if upload.get("Key") != key:
                    continue
                try:
                    client.abort_multipart_upload(
                        Bucket=bucket,
                        Key=key,
                        UploadId=upload["UploadId"],
                    )
                    aborted += 1
                except Exception as exc:
                    log(
                        f"WARN could not abort multipart "
                        f"{upload.get('UploadId')}: {exc}"
                    )

            if not response.get("IsTruncated"):
                break

            kwargs["KeyMarker"] = response.get("NextKeyMarker")
            kwargs["UploadIdMarker"] = response.get("NextUploadIdMarker")

    except Exception as exc:
        log(f"WARN multipart cleanup failed for {key}: {exc}")

    return aborted


def upload_verified(
    local_path: Path,
    bucket: str,
    key: str,
    metadata: dict[str, str],
    attempts: int,
) -> int:
    client = get_b2_client()
    local_size = local_path.stat().st_size

    existing = head_b2_object(bucket, key)
    if existing is not None:
        remote_size = int(existing["ContentLength"])
        if remote_size == local_size:
            return local_size
        raise RuntimeError(
            f"Remote object exists with different size: "
            f"b2://{bucket}/{key}; local={local_size}, remote={remote_size}"
        )

    for attempt in range(1, attempts + 1):
        try:
            if attempt > 1:
                aborted = abort_multipart_for_key(bucket, key)
                if aborted:
                    log(f"aborted {aborted} stale multipart upload(s): {key}")

            log(
                f"B2 upload {attempt}/{attempts} -> {key} "
                f"({local_size / 1024**2:.1f} MiB)"
            )

            client.upload_file(
                str(local_path),
                bucket,
                key,
                ExtraArgs={
                    "ContentType": "image/tiff",
                    "Metadata": {
                        k: str(v)[:1024]
                        for k, v in metadata.items()
                        if v is not None
                    },
                },
                Config=TRANSFER_CONFIG,
            )

            head = client.head_object(Bucket=bucket, Key=key)
            remote_size = int(head["ContentLength"])

            if remote_size != local_size:
                raise IOError(
                    f"B2 size verification failed for {key}: "
                    f"local={local_size}, remote={remote_size}"
                )

            return remote_size

        except Exception as exc:
            if attempt >= attempts:
                abort_multipart_for_key(bucket, key)
                raise

            delay = min(60, 2 ** (attempt - 1))
            log(
                f"B2 retry {attempt}/{attempts} for {key}: "
                f"{type(exc).__name__}: {exc}; sleeping {delay}s"
            )
            time.sleep(delay)

    raise AssertionError("unreachable")


def put_bytes_verified(
    bucket: str,
    key: str,
    body: bytes,
    content_type: str,
) -> None:
    client = get_b2_client()

    for attempt in range(1, 6):
        try:
            client.put_object(
                Bucket=bucket,
                Key=key,
                Body=body,
                ContentType=content_type,
            )
            head = client.head_object(Bucket=bucket, Key=key)
            if int(head["ContentLength"]) != len(body):
                raise IOError(f"Size verification failed for {key}")
            return
        except Exception:
            if attempt == 5:
                raise
            time.sleep(min(30, 2 ** (attempt - 1)))


# =============================================================================
# Census boundaries
# =============================================================================

def download_small(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, 6):
        try:
            with get_requests_session().get(url, timeout=(30, 120)) as response:
                response.raise_for_status()
                path.write_bytes(response.content)
            return
        except Exception:
            if attempt == 5:
                raise
            time.sleep(min(30, 2 ** (attempt - 1)))


def load_state_domains(
    work_dir: Path,
    requested_states: set[str] | None,
) -> list[StateDomain]:
    root = work_dir / "_boundaries"
    root.mkdir(parents=True, exist_ok=True)

    zip_path = root / "cb_2024_us_state_20m.zip"
    extracted = root / "cb_2024_us_state_20m"

    if not zip_path.exists():
        log(f"download Census state boundaries <- {STATE_BOUNDARY_URL}")
        download_small(STATE_BOUNDARY_URL, zip_path)

    if not extracted.exists():
        extracted.mkdir(parents=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extracted)

    shp_files = list(extracted.glob("*.shp"))
    if len(shp_files) != 1:
        raise RuntimeError(f"Expected one state .shp, found: {shp_files}")

    reader = shapefile.Reader(str(shp_files[0]))
    fields = [field[0] for field in reader.fields[1:]]

    result: list[StateDomain] = []

    for shape_record in reader.iterShapeRecords():
        attrs = dict(zip(fields, shape_record.record))
        fips = str(attrs.get("STATEFP", "")).zfill(2)

        if fips not in STATE_FIPS_50_DC:
            continue

        abbr = str(attrs.get("STUSPS", "")).upper()

        if requested_states and abbr not in requested_states:
            continue

        geom = shape(shape_record.shape.__geo_interface__)
        if geom.is_empty:
            continue

        result.append(StateDomain(fips=fips, abbr=abbr, geometry=geom))

    result.sort(key=lambda x: x.fips)

    if requested_states:
        found = {x.abbr for x in result}
        missing = sorted(requested_states - found)
        if missing:
            raise SystemExit(f"Unknown/unavailable state abbreviations: {missing}")

    if not result:
        raise RuntimeError("No U.S. state domains selected")

    return result


# =============================================================================
# NAIP selection
# =============================================================================

def stable_naip_tile_key(item_id: str) -> str:
    """
    Normalize repeated NAIP acquisitions over the same spatial tile.

    Common Planetary Computer NAIP id:
        fl_m_2608005_nw_17_060_20191215_20200113

    The trailing YYYYMMDD tokens are dates. The preceding numeric token commonly
    represents resolution/GSD and may differ between vintages, so remove it too.

    Result:
        fl_m_2608005_nw_17
    """
    parts = item_id.split("_")

    first_date_index = None
    for index, token in enumerate(parts):
        if DATE_TOKEN.match(token):
            first_date_index = index
            break

    if first_date_index is not None:
        return "_".join(parts[: max(1, first_date_index - 1)])

    while parts and DATE_TOKEN.match(parts[-1]):
        parts.pop()

    return "_".join(parts)


def item_datetime(item) -> dt.datetime:
    value = item.datetime
    if value is None:
        raw = item.properties.get("datetime")
        if not raw:
            raise ValueError(f"NAIP item lacks datetime: {item.id}")
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


def source_size_hint(asset) -> int | None:
    # STAC file extension commonly exposes file:size in asset.extra_fields.
    raw = asset.extra_fields.get("file:size")
    try:
        return int(raw) if raw is not None else None
    except Exception:
        return None


def source_etag_hint(asset) -> str | None:
    for key in ("file:checksum", "checksum:multihash", "etag"):
        value = asset.extra_fields.get(key)
        if value:
            return str(value)
    return None


def enumerate_state_candidates(
    catalog,
    state: StateDomain,
    start_year: int,
    end_year: int,
) -> list[Any]:
    interval = (
        f"{start_year:04d}-01-01T00:00:00Z/"
        f"{end_year:04d}-12-31T23:59:59Z"
    )

    log(f"STAC {state.abbr}: {interval}")

    search = catalog.search(
        collections=[COLLECTION],
        intersects=state.geometry.__geo_interface__,
        datetime=interval,
    )

    candidates = []

    for item in search.items():
        # NAIP item ids are state-prefixed. This removes neighboring-state items
        # returned only because their imagery touches the state boundary.
        if not item.id.lower().startswith(state.abbr.lower() + "_"):
            continue

        if ASSET_KEY not in item.assets:
            continue

        candidates.append(item)

    return candidates


def select_latest_per_tile(items: list[Any]) -> list[Any]:
    latest: dict[str, Any] = {}

    for item in items:
        tile_key = stable_naip_tile_key(item.id)
        current = latest.get(tile_key)

        if current is None:
            latest[tile_key] = item
            continue

        new_time = item_datetime(item)
        old_time = item_datetime(current)

        if new_time > old_time:
            latest[tile_key] = item
            continue

        if new_time == old_time:
            new_gsd = item_gsd(item)
            old_gsd = item_gsd(current)
            new_gsd_sort = new_gsd if new_gsd is not None else float("inf")
            old_gsd_sort = old_gsd if old_gsd is not None else float("inf")

            if new_gsd_sort < old_gsd_sort:
                latest[tile_key] = item

    return sorted(
        latest.values(),
        key=lambda x: (
            stable_naip_tile_key(x.id),
            item_datetime(x),
            x.id,
        ),
    )


def convert_selected_item(item, state_abbr: str) -> SelectedItem:
    asset = item.assets[ASSET_KEY]

    return SelectedItem(
        item_id=item.id,
        state=state_abbr,
        stable_tile_key=stable_naip_tile_key(item.id),
        capture_time_utc=iso_z(item_datetime(item)),
        gsd_m=item_gsd(item),
        bbox=list(item.bbox) if item.bbox else None,
        geometry=item.geometry,
        unsigned_image_href=asset.href,
        source_type=asset.media_type,
        source_roles=list(asset.roles or []),
        source_file_size_hint=source_size_hint(asset),
        source_etag_hint=source_etag_hint(asset),
    )


def discover_latest(
    domains: list[StateDomain],
    start_year: int,
    end_year: int,
) -> tuple[list[SelectedItem], dict[str, Any]]:
    catalog = pystac_client.Client.open(STAC_URL)

    selected_all: list[SelectedItem] = []
    stats: dict[str, Any] = {}

    for index, state in enumerate(domains, 1):
        log(f"discover {index}/{len(domains)}: {state.abbr}")

        candidates = enumerate_state_candidates(
            catalog,
            state,
            start_year,
            end_year,
        )
        chosen = select_latest_per_tile(candidates)

        selected = [
            convert_selected_item(item, state.abbr)
            for item in chosen
        ]

        selected_all.extend(selected)

        year_counts: dict[str, int] = {}
        for rec in selected:
            year = str(rec.year)
            year_counts[year] = year_counts.get(year, 0) + 1

        stats[state.abbr] = {
            "catalog_candidates": len(candidates),
            "selected_latest_tiles": len(selected),
            "selected_year_counts": dict(sorted(year_counts.items())),
        }

        log(
            f"  {state.abbr}: {len(candidates):,} candidates -> "
            f"{len(selected):,} latest tiles"
        )

    selected_all.sort(
        key=lambda x: (
            x.state,
            x.stable_tile_key,
            x.capture_time_utc,
            x.item_id,
        )
    )

    return selected_all, stats


# =============================================================================
# Direct COG transfer
# =============================================================================

def destination_key(prefix: str, rec: SelectedItem) -> str:
    # Preserve the original asset extension when possible.
    suffix = Path(urlsplit(rec.unsigned_image_href).path).suffix.lower() or ".tif"

    return (
        f"{prefix.strip('/')}/"
        f"state={rec.state}/"
        f"acquisition_year={rec.year}/"
        f"tile_key={safe_component(rec.stable_tile_key)}/"
        f"item_id={safe_component(rec.item_id)}/"
        f"image{suffix}"
    )


def source_head(unsigned_href: str, attempts: int = 5) -> dict[str, Any]:
    last_exc: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            signed_href = planetary_computer.sign(unsigned_href)
            response = get_requests_session().head(
                signed_href,
                allow_redirects=True,
                timeout=(30, 120),
            )
            response.raise_for_status()

            content_length = response.headers.get("Content-Length")
            return {
                "signed_href": signed_href,
                "content_length": (
                    int(content_length)
                    if content_length is not None
                    else None
                ),
                "etag": response.headers.get("ETag"),
                "last_modified": response.headers.get("Last-Modified"),
                "content_type": response.headers.get("Content-Type"),
                "accept_ranges": response.headers.get("Accept-Ranges"),
            }

        except Exception as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            time.sleep(min(30, 2 ** (attempt - 1)))

    assert last_exc is not None
    raise last_exc


def download_original_cog(
    rec: SelectedItem,
    local_path: Path,
    attempts: int,
) -> dict[str, Any]:
    """
    Copy the original Planetary Computer COG to local staging byte-for-byte.

    Partial local files are resumed with HTTP Range requests. Every retry
    obtains a fresh SAS URL.
    """
    local_path.parent.mkdir(parents=True, exist_ok=True)

    last_exc: Exception | None = None

    for attempt in range(1, attempts + 1):
        existing = local_path.stat().st_size if local_path.exists() else 0

        try:
            head = source_head(rec.unsigned_image_href, attempts=3)
            expected_size = head["content_length"]
            signed_href = head["signed_href"]

            if expected_size is not None and existing == expected_size:
                return {
                    "bytes": existing,
                    "source_etag": head["etag"],
                    "source_last_modified": head["last_modified"],
                    "source_content_type": head["content_type"],
                    "resumed_from_bytes": existing,
                    "download_skipped_local_complete": True,
                }

            if (
                expected_size is not None
                and existing > expected_size
            ):
                log(
                    f"WARN local stage larger than source; restarting "
                    f"{rec.item_id}"
                )
                local_path.unlink(missing_ok=True)
                existing = 0

            headers = {}
            mode = "wb"

            if existing > 0:
                headers["Range"] = f"bytes={existing}-"

            log(
                f"download {rec.state} {rec.year} {rec.item_id} "
                f"from={existing / 1024**2:.1f} MiB"
            )

            with get_requests_session().get(
                signed_href,
                headers=headers,
                stream=True,
                timeout=(30, 300),
            ) as response:

                if existing > 0 and response.status_code == 206:
                    mode = "ab"

                elif existing > 0 and response.status_code == 200:
                    # Server ignored Range: restart rather than duplicate bytes.
                    existing = 0
                    mode = "wb"

                response.raise_for_status()

                with local_path.open(mode) as out:
                    for chunk in response.iter_content(
                        chunk_size=8 * 1024**2
                    ):
                        if chunk:
                            out.write(chunk)

            final_size = local_path.stat().st_size

            if expected_size is not None and final_size != expected_size:
                raise IOError(
                    f"Source download size mismatch for {rec.item_id}: "
                    f"expected={expected_size}, got={final_size}"
                )

            if final_size < 1024:
                raise IOError(
                    f"Source COG suspiciously small: {final_size} bytes"
                )

            return {
                "bytes": final_size,
                "source_etag": head["etag"],
                "source_last_modified": head["last_modified"],
                "source_content_type": head["content_type"],
                "resumed_from_bytes": existing,
                "download_skipped_local_complete": False,
            }

        except Exception as exc:
            last_exc = exc

            if attempt >= attempts:
                break

            delay = min(60, 2 ** (attempt - 1))
            log(
                f"download retry {attempt}/{attempts} "
                f"{rec.item_id}: {type(exc).__name__}: {exc}; "
                f"sleeping {delay}s"
            )
            time.sleep(delay)

    assert last_exc is not None
    raise last_exc


@dataclasses.dataclass(frozen=True)
class TransferConfigRuntime:
    bucket: str
    prefix: str
    work_dir: Path
    download_attempts: int
    upload_attempts: int
    keep_local_on_failure: bool


def process_one(
    rec: SelectedItem,
    cfg: TransferConfigRuntime,
) -> dict[str, Any]:
    started = time.perf_counter()

    key = destination_key(cfg.prefix, rec)

    # A finalized remote object is the resume marker. We don't know the source
    # size without signing/HEAD, so compare against STAC file:size when present;
    # otherwise accept a positive finalized object and retain manifest metadata.
    existing = head_b2_object(cfg.bucket, key)

    if existing is not None and int(existing["ContentLength"]) > 0:
        return {
            "state": rec.state,
            "stable_tile_key": rec.stable_tile_key,
            "item_id": rec.item_id,
            "capture_time_utc": rec.capture_time_utc,
            "gsd_m": rec.gsd_m,
            "b2_key": key,
            "status": "exists",
            "bytes": int(existing["ContentLength"]),
            "elapsed_seconds": time.perf_counter() - started,
            "source_etag": None,
            "source_last_modified": None,
            "error": None,
        }

    item_dir = (
        cfg.work_dir
        / "staging"
        / rec.state
        / safe_component(rec.stable_tile_key)
        / safe_component(rec.item_id)
    )
    item_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(urlsplit(rec.unsigned_image_href).path).suffix.lower() or ".tif"
    local_path = item_dir / f"image{suffix}"

    try:
        dl = download_original_cog(
            rec,
            local_path,
            attempts=cfg.download_attempts,
        )

        uploaded_bytes = upload_verified(
            local_path=local_path,
            bucket=cfg.bucket,
            key=key,
            metadata={
                "source": "microsoft-planetary-computer",
                "collection": COLLECTION,
                "asset": ASSET_KEY,
                "item_id": rec.item_id,
                "state": rec.state,
                "stable_tile_key": rec.stable_tile_key,
                "capture_time_utc": rec.capture_time_utc,
                "gsd_m": "" if rec.gsd_m is None else rec.gsd_m,
                "source_etag": dl.get("source_etag") or "",
                "source_last_modified": dl.get("source_last_modified") or "",
                "copy_semantics": "original_cog_byte_for_byte",
            },
            attempts=cfg.upload_attempts,
        )

        # Delete only after verified B2 upload.
        local_path.unlink(missing_ok=True)

        try:
            item_dir.rmdir()
            item_dir.parent.rmdir()
        except OSError:
            pass

        elapsed = time.perf_counter() - started

        return {
            "state": rec.state,
            "stable_tile_key": rec.stable_tile_key,
            "item_id": rec.item_id,
            "capture_time_utc": rec.capture_time_utc,
            "gsd_m": rec.gsd_m,
            "b2_key": key,
            "status": "uploaded",
            "bytes": uploaded_bytes,
            "elapsed_seconds": elapsed,
            "source_etag": dl.get("source_etag"),
            "source_last_modified": dl.get("source_last_modified"),
            "error": None,
        }

    except Exception as exc:
        if not cfg.keep_local_on_failure:
            local_path.unlink(missing_ok=True)

        return {
            "state": rec.state,
            "stable_tile_key": rec.stable_tile_key,
            "item_id": rec.item_id,
            "capture_time_utc": rec.capture_time_utc,
            "gsd_m": rec.gsd_m,
            "b2_key": key,
            "status": "error",
            "bytes": (
                local_path.stat().st_size
                if local_path.exists()
                else None
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "source_etag": None,
            "source_last_modified": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


# =============================================================================
# Manifests
# =============================================================================

def selected_manifest_rows(
    selected: list[SelectedItem],
    prefix: str,
) -> list[dict[str, Any]]:
    return [
        {
            "state": rec.state,
            "stable_tile_key": rec.stable_tile_key,
            "item_id": rec.item_id,
            "capture_time_utc": parse_datetime(rec.capture_time_utc),
            "acquisition_year": rec.year,
            "gsd_m": rec.gsd_m,
            "bbox": rec.bbox,
            "geometry_json": (
                json.dumps(rec.geometry, separators=(",", ":"))
                if rec.geometry is not None
                else None
            ),
            "source_href_unsigned": rec.unsigned_image_href,
            "source_type": rec.source_type,
            "source_file_size_hint": rec.source_file_size_hint,
            "source_etag_hint": rec.source_etag_hint,
            "b2_key": destination_key(prefix, rec),
        }
        for rec in selected
    ]


def upload_selection_manifest(
    selected: list[SelectedItem],
    stats: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    prefix = args.b2_prefix.strip("/")
    manifest_root = f"{prefix}/manifests"

    summary = {
        "format": "crimenet-national-naip-latest-cog-selection-v1",
        "created_at_utc": iso_z(utcnow()),
        "collection": COLLECTION,
        "asset": ASSET_KEY,
        "catalog_window": {
            "start_year": args.start_year,
            "end_year": args.end_year,
        },
        "selection": "latest acquisition per stable NAIP spatial tile",
        "copy_semantics": (
            "original Planetary Computer image COG copied byte-for-byte; "
            "no raster transformation"
        ),
        "selected_item_count": len(selected),
        "state_stats": stats,
    }

    put_bytes_verified(
        args.bucket,
        f"{manifest_root}/selection_summary.json",
        json.dumps(summary, indent=2, sort_keys=True).encode("utf-8"),
        "application/json",
    )

    rows = selected_manifest_rows(selected, args.b2_prefix)

    if rows:
        df = pl.DataFrame(rows)
    else:
        df = pl.DataFrame(
            schema={
                "state": pl.Utf8,
                "stable_tile_key": pl.Utf8,
                "item_id": pl.Utf8,
                "capture_time_utc": pl.Datetime("us", "UTC"),
                "acquisition_year": pl.Int64,
                "gsd_m": pl.Float64,
                "bbox": pl.List(pl.Float64),
                "geometry_json": pl.Utf8,
                "source_href_unsigned": pl.Utf8,
                "source_type": pl.Utf8,
                "source_file_size_hint": pl.Int64,
                "source_etag_hint": pl.Utf8,
                "b2_key": pl.Utf8,
            }
        )

    local_manifest = (
        Path(args.work_dir).expanduser().resolve()
        / "selection_items.parquet"
    )
    df.write_parquet(
        local_manifest,
        compression="zstd",
        statistics=True,
    )

    upload_verified_generic(
        local_manifest,
        args.bucket,
        f"{manifest_root}/selection_items.parquet",
        content_type="application/vnd.apache.parquet",
    )

    local_manifest.unlink(missing_ok=True)


def upload_verified_generic(
    local_path: Path,
    bucket: str,
    key: str,
    content_type: str,
) -> None:
    client = get_b2_client()
    size = local_path.stat().st_size

    for attempt in range(1, 6):
        try:
            client.upload_file(
                str(local_path),
                bucket,
                key,
                ExtraArgs={"ContentType": content_type},
                Config=TRANSFER_CONFIG,
            )
            head = client.head_object(Bucket=bucket, Key=key)
            if int(head["ContentLength"]) != size:
                raise IOError(f"Manifest verification failed for {key}")
            return
        except Exception:
            if attempt == 5:
                raise
            time.sleep(min(30, 2 ** (attempt - 1)))


def upload_completion_manifest(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    prefix = args.b2_prefix.strip("/")
    manifest_root = f"{prefix}/manifests"

    schema = {
        "state": pl.Utf8,
        "stable_tile_key": pl.Utf8,
        "item_id": pl.Utf8,
        "capture_time_utc": pl.Utf8,
        "gsd_m": pl.Float64,
        "b2_key": pl.Utf8,
        "status": pl.Utf8,
        "bytes": pl.Int64,
        "elapsed_seconds": pl.Float64,
        "source_etag": pl.Utf8,
        "source_last_modified": pl.Utf8,
        "error": pl.Utf8,
    }

    df = pl.DataFrame(rows, schema=schema)

    local = (
        Path(args.work_dir).expanduser().resolve()
        / "transfer_results.parquet"
    )
    df.write_parquet(local, compression="zstd", statistics=True)

    upload_verified_generic(
        local,
        args.bucket,
        f"{manifest_root}/transfer_results.parquet",
        content_type="application/vnd.apache.parquet",
    )
    local.unlink(missing_ok=True)

    status_counts = {
        row["status"]: int(count)
        for row, count in (
            df.group_by("status")
            .len()
            .select("status", "len")
            .iter_rows()
        )
    } if df.height else {}

    total_bytes = (
        int(df["bytes"].fill_null(0).sum())
        if df.height
        else 0
    )

    summary = {
        "format": "crimenet-national-naip-latest-cog-transfer-v1",
        "finished_at_utc": iso_z(utcnow()),
        "status_counts": status_counts,
        "verified_or_staged_bytes": total_bytes,
        "result_rows": len(rows),
    }

    put_bytes_verified(
        args.bucket,
        f"{manifest_root}/transfer_summary.json",
        json.dumps(summary, indent=2, sort_keys=True).encode("utf-8"),
        "application/json",
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mirror the latest available original Planetary Computer NAIP "
            "COG for every U.S. spatial tile into Backblaze B2."
        )
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
        default=str(Path.home() / "crimenet-naip-latest"),
        help=(
            "Persistent local staging directory. Partial downloads survive "
            "failures and are resumed on rerun."
        ),
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
        "--states",
        default="",
        help="Optional comma-separated states, e.g. TX,CA,NY.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help=(
            "Concurrent COG transfers. Start with 4 on a Mac; increase to 6-8 "
            "only if network and local SSD throughput remain healthy."
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
        help="Testing only: transfer only the first N selected items.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover/select and write manifests, but do not copy imagery.",
    )

    parser.add_argument(
        "--delete-local-on-failure",
        action="store_true",
        help=(
            "By default partial local downloads are retained for resume. "
            "Set this only if local disk pressure matters more than resume speed."
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

    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    # Fail fast on B2 auth.
    get_b2_client().head_bucket(Bucket=args.bucket)
    log(f"B2 access OK: b2://{args.bucket}")

    requested_states: set[str] | None = None
    if args.states.strip():
        requested_states = {
            token.strip().upper()
            for token in args.states.split(",")
            if token.strip()
        }

    domains = load_state_domains(work_dir, requested_states)

    selected, stats = discover_latest(
        domains=domains,
        start_year=args.start_year,
        end_year=args.end_year,
    )

    log(f"selected latest NAIP COGs: {len(selected):,}")

    upload_selection_manifest(
        selected=selected,
        stats=stats,
        args=args,
    )

    if args.limit_items > 0:
        transfer_set = selected[: args.limit_items]
    else:
        transfer_set = selected

    if args.dry_run:
        size_hints = [
            rec.source_file_size_hint
            for rec in transfer_set
            if rec.source_file_size_hint is not None
        ]

        hinted_bytes = sum(size_hints)

        print("\n=== DRY RUN SUMMARY ===")
        print(f"States searched: {len(domains):,}")
        print(f"Selected latest source COGs: {len(selected):,}")
        print(f"Items in this run: {len(transfer_set):,}")
        print(
            f"STAC file:size hints available: "
            f"{len(size_hints):,}/{len(transfer_set):,}"
        )
        if size_hints:
            print(
                f"Hinted bytes for this run: "
                f"{hinted_bytes / 1024**4:.2f} TiB"
            )
        print("No imagery bytes copied.")
        return 0

    cfg = TransferConfigRuntime(
        bucket=args.bucket,
        prefix=args.b2_prefix,
        work_dir=work_dir,
        download_attempts=args.download_attempts,
        upload_attempts=args.upload_attempts,
        keep_local_on_failure=not args.delete_local_on_failure,
    )

    log(
        f"begin transfer: {len(transfer_set):,} COGs, "
        f"workers={args.workers}"
    )

    results: list[dict[str, Any]] = []

    uploaded = 0
    existed = 0
    failed = 0
    bytes_seen = 0

    with cf.ThreadPoolExecutor(
        max_workers=args.workers,
        thread_name_prefix="naip-cog",
    ) as pool:

        futures = {
            pool.submit(process_one, rec, cfg): rec
            for rec in transfer_set
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
                    "stable_tile_key": rec.stable_tile_key,
                    "item_id": rec.item_id,
                    "capture_time_utc": rec.capture_time_utc,
                    "gsd_m": rec.gsd_m,
                    "b2_key": destination_key(args.b2_prefix, rec),
                    "status": "error",
                    "bytes": None,
                    "elapsed_seconds": None,
                    "source_etag": None,
                    "source_last_modified": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }

            results.append(row)

            if row["status"] == "uploaded":
                uploaded += 1
            elif row["status"] == "exists":
                existed += 1
            else:
                failed += 1

            if row.get("bytes"):
                bytes_seen += int(row["bytes"])

            if (
                completed % 25 == 0
                or completed == len(transfer_set)
            ):
                avg_mib = (
                    bytes_seen
                    / max(1, uploaded + existed)
                    / 1024**2
                )
                projected_tib = (
                    avg_mib
                    * len(transfer_set)
                    / 1024**2
                )

                log(
                    f"progress {completed:,}/{len(transfer_set):,}; "
                    f"uploaded={uploaded:,}, exists={existed:,}, "
                    f"failed={failed:,}; "
                    f"observed={bytes_seen / 1024**3:.2f} GiB; "
                    f"avg={avg_mib:.1f} MiB/COG; "
                    f"projected={projected_tib:.2f} TiB"
                )

    upload_completion_manifest(results, args)

    print("\n=== COMPLETE ===")
    print(f"Uploaded: {uploaded:,}")
    print(f"Already existed: {existed:,}")
    print(f"Failed: {failed:,}")
    print(f"Observed bytes: {bytes_seen / 1024**4:.2f} TiB")

    if failed:
        print(
            "\nRerun the same command. Finalized B2 objects are skipped and "
            "partial local source downloads are resumed."
        )
        return 2

    put_bytes_verified(
        args.bucket,
        f"{args.b2_prefix.strip('/')}/_BOOTSTRAP_SUCCESS.json",
        json.dumps(
            {
                "finished_at_utc": iso_z(utcnow()),
                "selected_item_count": len(selected),
                "transferred_item_count": len(transfer_set),
                "copy_semantics": "original_cog_byte_for_byte",
                "catalog_window": {
                    "start_year": args.start_year,
                    "end_year": args.end_year,
                },
            },
            indent=2,
            sort_keys=True,
        ).encode("utf-8"),
        "application/json",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
