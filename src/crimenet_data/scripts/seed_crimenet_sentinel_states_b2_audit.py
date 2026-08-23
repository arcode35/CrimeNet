#!/usr/bin/env python3
"""
Bootstrap recent nationwide Sentinel-2 L2A source imagery from Microsoft
Planetary Computer into Backblaze B2 for CrimeNet.

Design goals
------------
- Enumerate recent Sentinel-2 L2A scenes intersecting the 50 states + DC.
- Keep several recent/usable candidate scenes per MGRS tile so later H3-level
  SCL filtering can choose the best observation for each cell.
- Download the complete OlmoEarth spectral set at native Sentinel resolutions:
    B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12 + SCL
- Never create one image object per H3 cell.
- Preserve the original Planetary Computer COGs in B2.
- Resume safely.
- Persist local staging files across failures.
- Harden large B2 uploads with multipart retry + verification.
- Abort orphaned multipart uploads for the object being retried.

Required environment variables
------------------------------
B2_KEY_ID
B2_APPLICATION_KEY
B2_ENDPOINT_URL       e.g. https://s3.us-east-005.backblazeb2.com
B2_BUCKET             optional, defaults to crimenet-data

Optional:
PC_SDK_SUBSCRIPTION_KEY

Dependencies
------------
pip install boto3 requests pystac-client pystac planetary-computer shapely pyshp
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses
import datetime as dt
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
import requests
import shapefile
import pystac_client
from botocore.config import Config
from boto3.s3.transfer import TransferConfig
from shapely.geometry import shape

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
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
TERRITORY_FIPS = {"60", "66", "69", "72", "78"}

REQUIRED_ASSETS = (
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B11", "B12", "SCL",
)

ASSET_GSD_M = {
    "B01": 60, "B02": 10, "B03": 10, "B04": 10,
    "B05": 20, "B06": 20, "B07": 20, "B08": 10,
    "B8A": 20, "B09": 60, "B11": 20, "B12": 20,
    "SCL": 20,
}

USER_AGENT = "CrimeNet-national-sentinel-bootstrap/1.0"

TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=64 * 1024**2,
    multipart_chunksize=128 * 1024**2,
    max_concurrency=2,
    use_threads=True,
)

_print_lock = threading.Lock()
_tls = threading.local()


def log(msg: str) -> None:
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with _print_lock:
        print(f"[{now}] {msg}", flush=True)


def env_required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso_z(x: dt.datetime) -> str:
    if x.tzinfo is None:
        x = x.replace(tzinfo=dt.timezone.utc)
    return x.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def parse_datetime(value: str) -> dt.datetime:
    v = value.strip().replace("Z", "+00:00")
    out = dt.datetime.fromisoformat(v)
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt.timezone.utc)
    return out.astimezone(dt.timezone.utc)


def requests_session() -> requests.Session:
    sess = getattr(_tls, "requests_session", None)
    if sess is None:
        sess = requests.Session()
        sess.headers.update({"User-Agent": USER_AGENT})
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=32,
            pool_maxsize=32,
            max_retries=0,
        )
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        _tls.requests_session = sess
    return sess


def make_s3_client():
    return boto3.client(
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


def object_size(s3, bucket: str, key: str) -> int | None:
    try:
        h = s3.head_object(Bucket=bucket, Key=key)
        return int(h["ContentLength"])
    except Exception as exc:
        response = getattr(exc, "response", None)
        status = None
        if response:
            status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status in (403, 404):
            return None
        text = str(exc)
        if "404" in text or "Not Found" in text or "NoSuchKey" in text:
            return None
        raise


def abort_multipart_for_key(s3, bucket: str, key: str) -> int:
    aborted = 0
    try:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": key}
        while True:
            r = s3.list_multipart_uploads(**kwargs)
            for up in r.get("Uploads", []):
                if up.get("Key") != key:
                    continue
                try:
                    s3.abort_multipart_upload(
                        Bucket=bucket,
                        Key=key,
                        UploadId=up["UploadId"],
                    )
                    aborted += 1
                except Exception as exc:
                    log(f"WARN could not abort multipart {up.get('UploadId')}: {exc}")
            if not r.get("IsTruncated"):
                break
            kwargs["KeyMarker"] = r.get("NextKeyMarker")
            kwargs["UploadIdMarker"] = r.get("NextUploadIdMarker")
    except Exception as exc:
        log(f"WARN multipart cleanup failed for {key}: {exc}")
    return aborted


def upload_file_verified(
    s3,
    bucket: str,
    path: Path,
    key: str,
    metadata: dict[str, str] | None = None,
    attempts: int = 8,
) -> None:
    local_size = path.stat().st_size
    existing = object_size(s3, bucket, key)
    if existing == local_size:
        return
    if existing is not None and existing != local_size:
        raise RuntimeError(
            f"Remote object exists with wrong size: b2://{bucket}/{key}; "
            f"remote={existing}, local={local_size}"
        )

    extra = {"Metadata": {k: str(v) for k, v in (metadata or {}).items()}}

    for attempt in range(1, attempts + 1):
        try:
            if attempt > 1:
                n = abort_multipart_for_key(s3, bucket, key)
                if n:
                    log(f"aborted {n} stale multipart upload(s) for {key}")

            log(
                f"B2 upload attempt {attempt}/{attempts} -> "
                f"b2://{bucket}/{key} ({local_size / 1024**2:.1f} MiB)"
            )
            s3.upload_file(
                str(path),
                bucket,
                key,
                ExtraArgs=extra,
                Config=TRANSFER_CONFIG,
            )

            remote_size = object_size(s3, bucket, key)
            if remote_size != local_size:
                raise IOError(
                    f"B2 size verification failed: local={local_size}, "
                    f"remote={remote_size}"
                )
            return
        except Exception as exc:
            if attempt >= attempts:
                abort_multipart_for_key(s3, bucket, key)
                raise
            delay = min(60, 2 ** (attempt - 1))
            log(
                f"B2 upload retry {attempt}/{attempts} for {key}: "
                f"{type(exc).__name__}: {exc}; sleeping {delay}s"
            )
            time.sleep(delay)


def put_json_verified(s3, bucket: str, key: str, payload: Any) -> None:
    body = json.dumps(payload, sort_keys=True, indent=2).encode("utf-8")
    for attempt in range(1, 6):
        try:
            s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=body,
                ContentType="application/json",
            )
            size = object_size(s3, bucket, key)
            if size != len(body):
                raise IOError(
                    f"JSON verification failed for {key}: "
                    f"remote={size}, expected={len(body)}"
                )
            return
        except Exception:
            if attempt == 5:
                raise
            time.sleep(min(30, 2 ** (attempt - 1)))


def download_small(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 6):
        try:
            with requests_session().get(url, timeout=(30, 120)) as r:
                r.raise_for_status()
                path.write_bytes(r.content)
            return
        except Exception:
            if attempt == 5:
                raise
            time.sleep(min(30, 2 ** (attempt - 1)))


def load_state_geometries(
    work_dir: Path,
    include_territories: bool,
    selected_states: set[str] | None = None,
) -> list[tuple[str, str, dict[str, Any]]]:
    boundary_dir = work_dir / "_boundaries"
    boundary_dir.mkdir(parents=True, exist_ok=True)
    zip_path = boundary_dir / "cb_2024_us_state_20m.zip"
    extracted = boundary_dir / "cb_2024_us_state_20m"

    if not zip_path.exists():
        log(f"download Census state boundaries <- {STATE_BOUNDARY_URL}")
        download_small(STATE_BOUNDARY_URL, zip_path)

    if not extracted.exists():
        extracted.mkdir(parents=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extracted)

    shp_files = list(extracted.glob("*.shp"))
    if len(shp_files) != 1:
        raise RuntimeError(f"Expected one .shp in {extracted}, found {shp_files}")

    reader = shapefile.Reader(str(shp_files[0]))
    field_names = [f[0] for f in reader.fields[1:]]

    out = []
    for sr in reader.iterShapeRecords():
        attrs = dict(zip(field_names, sr.record))
        fips = str(attrs.get("STATEFP", "")).zfill(2)
        abbr = str(attrs.get("STUSPS", fips))
        allowed = fips in STATE_FIPS_50_DC
        if include_territories:
            allowed = allowed or fips in TERRITORY_FIPS
        if not allowed:
            continue

        abbr_upper = abbr.upper()
        if selected_states is not None and abbr_upper not in selected_states:
            continue

        geom = shape(sr.shape.__geo_interface__)
        if geom.is_empty:
            continue
        out.append((fips, abbr, geom.__geo_interface__))

    out.sort(key=lambda x: x[0])
    if not out:
        raise RuntimeError("No Census state geometries loaded")

    if selected_states is not None:
        found = {abbr.upper() for _, abbr, _ in out}
        missing = sorted(selected_states - found)
        if missing:
            raise RuntimeError(
                "Requested state/territory abbreviations were not available in the "
                f"selected Census scope: {missing}. "
                "Use USPS abbreviations such as TX, CA, NY, DC."
            )

    return out


@dataclasses.dataclass(frozen=True)
class Scene:
    item_id: str
    mgrs_tile: str
    capture_time: str
    cloud_cover: float
    processing_baseline: str | None
    platform: str | None
    raw_assets: dict[str, str]
    bbox: list[float] | None

    @property
    def date(self) -> str:
        return self.capture_time[:10]


def mgrs_from_item(item) -> str | None:
    props = item.properties
    tile = props.get("s2:mgrs_tile")
    if tile:
        tile = str(tile)
        return tile[1:] if tile.startswith("T") else tile
    m = re.search(r"_T([0-9]{2}[A-Z]{3})_", item.id)
    return m.group(1) if m else None


def item_to_scene(item) -> Scene | None:
    mgrs = mgrs_from_item(item)
    if not mgrs:
        return None

    missing = [a for a in REQUIRED_ASSETS if a not in item.assets]
    if missing:
        log(f"WARN {item.id} missing required assets: {missing}")
        return None

    props = item.properties
    capture = item.datetime
    if capture is None:
        raw = props.get("datetime")
        if not raw:
            return None
        capture = parse_datetime(str(raw))

    cloud = props.get("eo:cloud_cover")
    try:
        cloud_f = float(cloud) if cloud is not None else 100.0
    except Exception:
        cloud_f = 100.0

    return Scene(
        item_id=item.id,
        mgrs_tile=mgrs,
        capture_time=iso_z(capture),
        cloud_cover=cloud_f,
        processing_baseline=(
            str(props.get("s2:processing_baseline"))
            if props.get("s2:processing_baseline") is not None
            else None
        ),
        platform=(
            str(props.get("platform"))
            if props.get("platform") is not None
            else None
        ),
        raw_assets={a: item.assets[a].href for a in REQUIRED_ASSETS},
        bbox=list(item.bbox) if item.bbox else None,
    )


def enumerate_recent_us_scenes(
    work_dir: Path,
    start: dt.datetime,
    end: dt.datetime,
    include_territories: bool,
    selected_states: set[str] | None = None,
) -> tuple[list[Scene], dict[str, set[str]]]:
    """Return unique scenes plus the states/DC each scene intersects."""
    states = load_state_geometries(
        work_dir,
        include_territories,
        selected_states=selected_states,
    )
    catalog = pystac_client.Client.open(STAC_URL)
    by_id: dict[str, Scene] = {}
    scene_states: dict[str, set[str]] = {}
    interval = f"{iso_z(start)}/{iso_z(end)}"

    for idx, (fips, abbr, geom) in enumerate(states, 1):
        state_code = abbr.upper()
        log(f"STAC search {idx}/{len(states)}: {abbr} ({fips}) {interval}")
        search = catalog.search(
            collections=[COLLECTION],
            intersects=geom,
            datetime=interval,
        )
        state_count = 0
        for item in search.items():
            scene = item_to_scene(item)
            if scene is None:
                continue
            by_id.setdefault(scene.item_id, scene)
            scene_states.setdefault(scene.item_id, set()).add(state_code)
            state_count += 1
        log(
            f"  {abbr}: {state_count:,} returned; "
            f"{len(by_id):,} unique scenes cumulative"
        )

    return list(by_id.values()), scene_states


def selection_score(
    scene: Scene,
    end: dt.datetime,
    cloud_penalty_days_per_percent: float,
) -> float:
    t = parse_datetime(scene.capture_time)
    age_days = max(0.0, (end - t).total_seconds() / 86400.0)
    cloud = min(max(scene.cloud_cover, 0.0), 100.0)
    return age_days + cloud_penalty_days_per_percent * cloud


def select_candidates(
    scenes: list[Scene],
    end: dt.datetime,
    candidates_per_tile: int,
    max_scene_cloud: float,
    cloud_penalty_days_per_percent: float,
) -> list[Scene]:
    by_tile: dict[str, list[Scene]] = {}
    for s in scenes:
        by_tile.setdefault(s.mgrs_tile, []).append(s)

    selected: list[Scene] = []
    for tile, group in sorted(by_tile.items()):
        eligible = [s for s in group if s.cloud_cover <= max_scene_cloud]
        fallback = [s for s in group if s.cloud_cover > max_scene_cloud]

        def key(s: Scene):
            return (
                selection_score(s, end, cloud_penalty_days_per_percent),
                -parse_datetime(s.capture_time).timestamp(),
                s.item_id,
            )

        eligible.sort(key=key)
        fallback.sort(key=key)

        chosen = eligible[:candidates_per_tile]
        if len(chosen) < candidates_per_tile:
            chosen += fallback[: candidates_per_tile - len(chosen)]
        selected.extend(chosen)

    selected.sort(key=lambda s: (s.mgrs_tile, s.capture_time, s.item_id))
    return selected


def safe_component(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def b2_scene_prefix(root_prefix: str, scene: Scene) -> str:
    root = root_prefix.strip("/")
    return (
        f"{root}/l2a/"
        f"mgrs_tile={safe_component(scene.mgrs_tile)}/"
        f"capture_date={scene.date}/"
        f"item_id={safe_component(scene.item_id)}"
    )


def asset_filename(asset_key: str, raw_href: str) -> str:
    suffix = Path(urlsplit(raw_href).path).suffix or ".tif"
    return f"{asset_key}{suffix.lower()}"


def download_asset_resumable(
    raw_href: str,
    local_path: Path,
    expected_min_bytes: int = 1024,
    attempts: int = 8,
) -> int:
    local_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, attempts + 1):
        existing = local_path.stat().st_size if local_path.exists() else 0
        try:
            href = planetary_computer.sign(raw_href)
            headers = {}
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
                    for chunk in r.iter_content(chunk_size=8 * 1024**2):
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
                f"download retry {attempt}/{attempts}: "
                f"{type(exc).__name__}: {exc}; sleeping {delay}s"
            )
            time.sleep(delay)

    raise AssertionError("unreachable")


@dataclasses.dataclass
class AssetTask:
    scene: Scene
    asset_key: str
    root_prefix: str
    work_dir: Path
    bucket: str
    preflight_checked: bool = False
    preflight_remote_size: int | None = None


def process_asset(task: AssetTask) -> dict[str, Any]:
    scene = task.scene
    raw_href = scene.raw_assets[task.asset_key]
    filename = asset_filename(task.asset_key, raw_href)
    scene_prefix = b2_scene_prefix(task.root_prefix, scene)
    key = f"{scene_prefix}/{filename}"

    local = (
        task.work_dir
        / "assets"
        / safe_component(scene.mgrs_tile)
        / safe_component(scene.item_id)
        / filename
    )
    local.parent.mkdir(parents=True, exist_ok=True)

    s3 = make_s3_client()
    remote_size = (
        task.preflight_remote_size
        if task.preflight_checked
        else object_size(s3, task.bucket, key)
    )
    if remote_size is not None and remote_size > 0:
        local.unlink(missing_ok=True)
        return {
            "item_id": scene.item_id,
            "mgrs_tile": scene.mgrs_tile,
            "asset": task.asset_key,
            "key": key,
            "bytes": remote_size,
            "status": "exists",
        }

    log(f"download {scene.mgrs_tile} {scene.date} {scene.item_id} {task.asset_key}")
    size = download_asset_resumable(raw_href, local)

    upload_file_verified(
        s3,
        task.bucket,
        local,
        key,
        metadata={
            "source": "microsoft-planetary-computer",
            "collection": COLLECTION,
            "item_id": scene.item_id,
            "mgrs_tile": scene.mgrs_tile,
            "asset": task.asset_key,
            "capture_time_utc": scene.capture_time,
            "eo_cloud_cover": f"{scene.cloud_cover:.6f}",
            "native_gsd_m": str(ASSET_GSD_M[task.asset_key]),
        },
    )

    local.unlink(missing_ok=True)
    return {
        "item_id": scene.item_id,
        "mgrs_tile": scene.mgrs_tile,
        "asset": task.asset_key,
        "key": key,
        "bytes": size,
        "status": "uploaded",
    }


def asset_task_key(scene: Scene, asset_key: str, root_prefix: str) -> str:
    raw_href = scene.raw_assets[asset_key]
    return (
        f"{b2_scene_prefix(root_prefix, scene)}/"
        f"{asset_filename(asset_key, raw_href)}"
    )


def audit_existing_b2_coverage(
    *,
    selected: list[Scene],
    scene_states: dict[str, set[str]],
    requested_states: list[str] | None,
    bucket: str,
    root_prefix: str,
    workers: int,
) -> tuple[dict[str, int | None], dict[str, dict[str, int | float]]]:
    """HEAD every selected B2 asset once and report exact completion by state/DC."""
    if not selected:
        return {}, {}

    state_order = (
        list(requested_states)
        if requested_states is not None
        else sorted({st for states in scene_states.values() for st in states})
    )

    # A selected scene can intersect multiple states. Count a B2 object once globally,
    # but let that same object satisfy every state whose geometry the scene intersects.
    state_keys: dict[str, set[str]] = {st: set() for st in state_order}
    state_scene_keys: dict[str, dict[str, set[str]]] = {
        st: {} for st in state_order
    }
    all_keys: set[str] = set()

    for scene in selected:
        states = scene_states.get(scene.item_id, set())
        keys = {
            asset_task_key(scene, asset_key, root_prefix)
            for asset_key in REQUIRED_ASSETS
        }
        all_keys.update(keys)
        for st in states:
            if st not in state_keys:
                continue
            state_keys[st].update(keys)
            state_scene_keys[st][scene.item_id] = keys

    log(
        f"B2 preflight audit: HEAD {len(all_keys):,} unique selected asset objects "
        f"with {workers} workers"
    )

    sizes: dict[str, int | None] = {}
    tls = threading.local()

    def head_one(key: str) -> tuple[str, int | None]:
        client = getattr(tls, "s3", None)
        if client is None:
            client = make_s3_client()
            tls.s3 = client
        size = object_size(client, bucket, key)
        return key, size if size is not None and size > 0 else None

    with cf.ThreadPoolExecutor(
        max_workers=max(1, workers),
        thread_name_prefix="b2-audit",
    ) as pool:
        futures = [pool.submit(head_one, key) for key in sorted(all_keys)]
        completed = 0
        for fut in cf.as_completed(futures):
            key, size = fut.result()
            sizes[key] = size
            completed += 1
            if completed % 1000 == 0 or completed == len(futures):
                existing_now = sum(v is not None for v in sizes.values())
                log(
                    f"B2 audit progress {completed:,}/{len(futures):,}; "
                    f"existing={existing_now:,}, missing={completed-existing_now:,}"
                )

    report: dict[str, dict[str, int | float]] = {}
    log("B2 EXISTING COVERAGE BY STATE/DC")
    for st in state_order:
        keys = state_keys.get(st, set())
        existing = sum(sizes.get(key) is not None for key in keys)
        required = len(keys)
        missing = required - existing
        pct = (100.0 * existing / required) if required else 0.0

        scenes = state_scene_keys.get(st, {})
        complete_scenes = sum(
            1
            for scene_keys in scenes.values()
            if scene_keys and all(sizes.get(key) is not None for key in scene_keys)
        )
        scene_count = len(scenes)

        report[st] = {
            "required_assets": required,
            "existing_assets": existing,
            "missing_assets": missing,
            "asset_completion_pct": pct,
            "selected_scenes": scene_count,
            "complete_scenes": complete_scenes,
        }

        status = (
            "COMPLETE"
            if required > 0 and missing == 0
            else ("NOT STARTED" if existing == 0 else "PARTIAL")
        )
        log(
            f"  {st:>2}: {existing:>6,}/{required:<6,} assets "
            f"({pct:6.2f}%) | scenes {complete_scenes:,}/{scene_count:,} complete "
            f"| {status}"
        )

    fully_complete = [
        st for st in state_order
        if int(report[st]["required_assets"]) > 0
        and int(report[st]["missing_assets"]) == 0
    ]
    partial = [
        st for st in state_order
        if int(report[st]["existing_assets"]) > 0
        and int(report[st]["missing_assets"]) > 0
    ]
    not_started = [
        st for st in state_order
        if int(report[st]["existing_assets"]) == 0
    ]

    log(f"Fully complete states/DC: {fully_complete or 'none'}")
    log(f"Partially complete states/DC: {partial or 'none'}")
    log(f"Not started states/DC: {not_started or 'none'}")

    return sizes, report


def scene_manifest_entry(scene: Scene, root_prefix: str) -> dict[str, Any]:
    prefix = b2_scene_prefix(root_prefix, scene)
    assets = {
        a: {
            "b2_key": f"{prefix}/{asset_filename(a, scene.raw_assets[a])}",
            "native_gsd_m": ASSET_GSD_M[a],
        }
        for a in REQUIRED_ASSETS
    }
    return {
        "item_id": scene.item_id,
        "mgrs_tile": scene.mgrs_tile,
        "capture_time_utc": scene.capture_time,
        "capture_date": scene.date,
        "eo_cloud_cover": scene.cloud_cover,
        "s2_processing_baseline": scene.processing_baseline,
        "platform": scene.platform,
        "bbox": scene.bbox,
        "assets": assets,
    }


def write_local_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Seed recent Sentinel-2 L2A imagery for all U.S. states or a selected subset into Backblaze B2."
    )
    p.add_argument("--bucket", default=os.environ.get("B2_BUCKET", "crimenet-data"))
    p.add_argument(
        "--prefix",
        default="bronze/imagery/sentinel2/national",
    )
    p.add_argument(
        "--work-dir",
        default="/workspace/crimenet-sentinel-national",
    )
    p.add_argument("--lookback-days", type=int, default=60)
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--candidates-per-tile", type=int, default=4)
    p.add_argument("--max-scene-cloud", type=float, default=70.0)
    p.add_argument(
        "--cloud-penalty-days-per-percent",
        type=float,
        default=0.35,
    )
    p.add_argument("--download-workers", type=int, default=16)
    p.add_argument("--include-territories", action="store_true")
    p.add_argument(
        "--states",
        nargs="+",
        default=None,
        metavar="ST",
        help=(
            "Optional USPS state/DC abbreviations. Example: "
            "--states TX NY WA FL CA IL GA MD PA NC DC. "
            "Duplicates are removed automatically."
        ),
    )
    p.add_argument(
        "--audit-only",
        action="store_true",
        help=(
            "Enumerate/select the requested coverage, HEAD the corresponding B2 "
            "objects, print exact per-state completion, then exit without downloading imagery."
        ),
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    env_required("B2_KEY_ID")
    env_required("B2_APPLICATION_KEY")
    env_required("B2_ENDPOINT_URL")

    if args.lookback_days <= 0:
        raise SystemExit("--lookback-days must be positive")
    if args.candidates_per_tile <= 0:
        raise SystemExit("--candidates-per-tile must be positive")
    if args.download_workers <= 0:
        raise SystemExit("--download-workers must be positive")

    selected_states: set[str] | None = None
    selected_state_list: list[str] | None = None
    if args.states:
        selected_state_list = []
        seen_states: set[str] = set()
        for raw in args.states:
            code = raw.strip().upper()
            if not re.fullmatch(r"[A-Z]{2}", code):
                raise SystemExit(
                    f"Invalid --states value {raw!r}; use two-letter USPS abbreviations."
                )
            if code not in seen_states:
                seen_states.add(code)
                selected_state_list.append(code)
        selected_states = set(selected_state_list)

    scope_slug = (
        "all_50_states_dc"
        if selected_state_list is None
        else "states_" + "-".join(code.lower() for code in selected_state_list)
    )

    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    log(f"persistent staging dir: {work_dir}")

    end = parse_datetime(args.end_date) if args.end_date else utcnow()
    start = (
        parse_datetime(args.start_date)
        if args.start_date
        else end - dt.timedelta(days=args.lookback_days)
    )
    if start >= end:
        raise SystemExit("start must be before end")

    s3 = make_s3_client()
    s3.head_bucket(Bucket=args.bucket)
    log(f"B2 access OK: b2://{args.bucket}")

    coverage_label = (
        "50 states + DC"
        if selected_state_list is None
        else ", ".join(selected_state_list)
    )
    log(
        f"Enumerating {COLLECTION}: {iso_z(start)} -> {iso_z(end)}; "
        f"coverage={coverage_label}"
    )
    all_scenes, scene_states = enumerate_recent_us_scenes(
        work_dir,
        start,
        end,
        args.include_territories,
        selected_states=selected_states,
    )
    log(
        f"Unique recent scenes intersecting requested coverage "
        f"({coverage_label}): {len(all_scenes):,}"
    )

    selected = select_candidates(
        all_scenes,
        end=end,
        candidates_per_tile=args.candidates_per_tile,
        max_scene_cloud=args.max_scene_cloud,
        cloud_penalty_days_per_percent=args.cloud_penalty_days_per_percent,
    )

    tile_count = len({s.mgrs_tile for s in selected})
    log(
        f"Selected {len(selected):,} candidate scenes across "
        f"{tile_count:,} MGRS tiles"
    )

    manifest = {
        "format": "crimenet-national-sentinel-bootstrap-v1",
        "created_at_utc": iso_z(utcnow()),
        "collection": COLLECTION,
        "coverage": (
            "50_states_plus_dc"
            + ("_plus_territories" if args.include_territories else "")
            if selected_state_list is None
            else "selected_states_plus_dc"
        ),
        "requested_states": selected_state_list,
        "scope_slug": scope_slug,
        "start_utc": iso_z(start),
        "end_utc": iso_z(end),
        "selection": {
            "candidates_per_mgrs_tile": args.candidates_per_tile,
            "preferred_max_scene_cloud_pct": args.max_scene_cloud,
            "cloud_penalty_days_per_percent": args.cloud_penalty_days_per_percent,
            "note": (
                "Scene cloud is only a coarse prefilter. H3-level production "
                "selection should use local SCL/coverage from these candidates."
            ),
        },
        "required_assets": list(REQUIRED_ASSETS),
        "native_gsd_m": ASSET_GSD_M,
        "unique_catalog_scenes": len(all_scenes),
        "selected_scene_count": len(selected),
        "selected_mgrs_tile_count": tile_count,
        "scenes": [
            {
                **scene_manifest_entry(s, args.prefix),
                "intersecting_states": sorted(scene_states.get(s.item_id, set())),
            }
            for s in selected
        ],
    }

    local_manifest = work_dir / f"sentinel_selection_{scope_slug}.json"
    write_local_manifest(local_manifest, manifest)

    manifest_key = (
        f"{args.prefix.strip('/')}/manifests/"
        f"selection_{scope_slug}_{start.date().isoformat()}_{end.date().isoformat()}.json"
    )
    put_json_verified(s3, args.bucket, manifest_key, manifest)
    log(f"manifest -> b2://{args.bucket}/{manifest_key}")

    if args.dry_run:
        log("DRY RUN complete; no imagery assets copied and no B2 asset audit performed.")
        return 0

    audit_sizes, state_audit = audit_existing_b2_coverage(
        selected=selected,
        scene_states=scene_states,
        requested_states=selected_state_list,
        bucket=args.bucket,
        root_prefix=args.prefix,
        workers=args.download_workers,
    )

    audit_payload = {
        "format": "crimenet-sentinel-b2-state-coverage-audit-v1",
        "created_at_utc": iso_z(utcnow()),
        "scope_slug": scope_slug,
        "requested_states": selected_state_list,
        "selection_manifest_key": manifest_key,
        "selected_scene_count": len(selected),
        "selected_mgrs_tile_count": tile_count,
        "unique_asset_objects": len(audit_sizes),
        "existing_asset_objects": sum(v is not None for v in audit_sizes.values()),
        "missing_asset_objects": sum(v is None for v in audit_sizes.values()),
        "states": state_audit,
    }
    local_audit = work_dir / f"b2_coverage_audit_{scope_slug}.json"
    write_local_manifest(local_audit, audit_payload)
    audit_key = (
        f"{args.prefix.strip('/')}/manifests/"
        f"b2_coverage_audit_{scope_slug}_{start.date().isoformat()}_{end.date().isoformat()}.json"
    )
    put_json_verified(s3, args.bucket, audit_key, audit_payload)
    log(f"B2 coverage audit -> b2://{args.bucket}/{audit_key}")

    if args.audit_only:
        log("AUDIT ONLY complete; no imagery assets downloaded or uploaded.")
        return 0

    tasks = []
    for s in selected:
        for a in REQUIRED_ASSETS:
            key = asset_task_key(s, a, args.prefix)
            tasks.append(
                AssetTask(
                    scene=s,
                    asset_key=a,
                    root_prefix=args.prefix,
                    work_dir=work_dir,
                    bucket=args.bucket,
                    preflight_checked=True,
                    preflight_remote_size=audit_sizes.get(key),
                )
            )

    log(
        f"Asset tasks: {len(tasks):,} "
        f"({len(selected):,} scenes × {len(REQUIRED_ASSETS)} assets); "
        f"workers={args.download_workers}; "
        f"preflight_existing={sum(v is not None for v in audit_sizes.values()):,}; "
        f"preflight_missing={sum(v is None for v in audit_sizes.values()):,}"
    )

    uploaded = 0
    existed = 0
    bytes_total = 0
    failures: list[dict[str, str]] = []

    with cf.ThreadPoolExecutor(max_workers=args.download_workers) as pool:
        futures = {pool.submit(process_asset, t): t for t in tasks}
        completed = 0
        for fut in cf.as_completed(futures):
            task = futures[fut]
            completed += 1
            try:
                result = fut.result()
                bytes_total += int(result["bytes"])
                if result["status"] == "uploaded":
                    uploaded += 1
                else:
                    existed += 1
            except Exception as exc:
                failures.append(
                    {
                        "item_id": task.scene.item_id,
                        "mgrs_tile": task.scene.mgrs_tile,
                        "asset": task.asset_key,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                log(
                    f"ERROR {task.scene.item_id} {task.asset_key}: "
                    f"{type(exc).__name__}: {exc}"
                )

            if completed % 100 == 0 or completed == len(tasks):
                log(
                    f"progress {completed:,}/{len(tasks):,}; "
                    f"uploaded={uploaded:,}, exists={existed:,}, "
                    f"failures={len(failures):,}, "
                    f"verified_bytes={bytes_total / 1024**3:.2f} GiB"
                )

    completion = {
        "format": "crimenet-national-sentinel-bootstrap-completion-v1",
        "finished_at_utc": iso_z(utcnow()),
        "selection_manifest_key": manifest_key,
        "scope_slug": scope_slug,
        "requested_states": selected_state_list,
        "asset_tasks": len(tasks),
        "uploaded": uploaded,
        "already_existed": existed,
        "failures": failures,
        "verified_bytes": bytes_total,
    }

    completion_key = (
        f"{args.prefix.strip('/')}/manifests/"
        f"completion_{scope_slug}_{start.date().isoformat()}_{end.date().isoformat()}.json"
    )
    put_json_verified(s3, args.bucket, completion_key, completion)

    if failures:
        log(
            f"FINISHED WITH {len(failures)} FAILURE(S). "
            "Rerun the same command; verified objects will be skipped."
        )
        return 2

    success_key = (
        f"{args.prefix.strip('/')}/"
        f"_BOOTSTRAP_SUCCESS_{scope_slug}.json"
    )
    put_json_verified(
        s3,
        args.bucket,
        success_key,
        {
            "finished_at_utc": iso_z(utcnow()),
            "scope_slug": scope_slug,
            "requested_states": selected_state_list,
            "selection_manifest_key": manifest_key,
            "completion_manifest_key": completion_key,
            "selected_scene_count": len(selected),
            "selected_mgrs_tile_count": tile_count,
            "asset_count": len(tasks),
        },
    )

    log(
        f"SUCCESS: {len(selected):,} scenes, {len(tasks):,} assets, "
        f"{bytes_total / 1024**3:.2f} GiB verified"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
