from __future__ import annotations

import argparse
import logging
import math
import os
import tempfile
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

import h3
import planetary_computer
import polars as pl
import rasterio
from google.api_core import exceptions as gcs_exceptions
from google.cloud import storage
from pystac import Item
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.shutil import copy as rio_copy
from rasterio.transform import array_bounds
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform_bounds
from shapely.geometry import Polygon, shape
from shapely.strtree import STRtree


# =============================================================================
# Configuration
# =============================================================================

EVENT_SPINE_ROOT = "gs://crimenet/gold/event_spine"
INTEGRATION_SAMPLES_ROOT = "gs://crimenet/gold/integration_samples"
PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# New prefix so this optimized representation never collides with the earlier
# whole-source-object copy experiment.
DEFAULT_GCS_ROOT = "gs://crimenet/bronze/imagery/optimized"
TARGET_H3_RESOLUTION = 9

NAIP_ASSET = "image"
SENTINEL_ASSETS = (
    "B02",  # blue, 10 m
    "B03",  # green, 10 m
    "B04",  # red, 10 m
    "B08",  # NIR, 10 m
    "B11",  # SWIR1, 20 m -> resampled to 10 m
    "B12",  # SWIR2, 20 m -> resampled to 10 m
    "SCL",  # scene classification, 20 m -> nearest-neighbor to 10 m
)
SENTINEL_REFERENCE_ASSET = "B02"

# Raw Sentinel-2 L2A bands are integer-quantized. Keeping uint16 preserves the
# source values and also accommodates SCL in the same multiband output.
SENTINEL_OUTPUT_DTYPE = "uint16"

# Explicit schemas avoid Polars' mixed NAIP/Sentinel inference failure.
ITEM_MANIFEST_SCHEMA = {
    "collection": pl.Utf8,
    "item_id": pl.Utf8,
    "capture_timestamp_utc": pl.Datetime("us", "UTC"),
    "source_cities": pl.List(pl.Utf8),
    "h3_cell_count": pl.Int64,
    "h3_cells": pl.List(pl.Utf8),
    "gcs_uri": pl.Utf8,
    "status": pl.Utf8,
    "already_existed": pl.Boolean,
    "output_bytes": pl.Int64,
    "elapsed_seconds": pl.Float64,
    "effective_mib_per_second": pl.Float64,
    "eo_cloud_cover": pl.Float64,
    "s2_mgrs_tile": pl.Utf8,
    "s2_processing_baseline": pl.Utf8,
    "bbox_min_lon": pl.Float64,
    "bbox_min_lat": pl.Float64,
    "bbox_max_lon": pl.Float64,
    "bbox_max_lat": pl.Float64,
    "error": pl.Utf8,
}

H3_MANIFEST_SCHEMA = {
    "source_city": pl.Utf8,
    "h3_resolution": pl.Int8,
    "h3_cell": pl.Utf8,
    "h3_cell_int": pl.Int64,
    "latitude": pl.Float64,
    "longitude": pl.Float64,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("crimenet.imagery.optimized")


# =============================================================================
# Models / indexes
# =============================================================================

@dataclass(frozen=True)
class CityDomain:
    source_city: str
    cells: frozenset[str]
    min_timestamp: datetime
    max_timestamp: datetime

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        min_lon = 180.0
        min_lat = 90.0
        max_lon = -180.0
        max_lat = -90.0

        for cell in self.cells:
            for lat, lon in h3.cell_to_boundary(cell):
                min_lon = min(min_lon, lon)
                max_lon = max(max_lon, lon)
                min_lat = min(min_lat, lat)
                max_lat = max(max_lat, lat)

        return min_lon, min_lat, max_lon, max_lat


@dataclass
class H3SpatialIndex:
    cells: list[str]
    polygons: list[Polygon]
    tree: STRtree
    cell_cities: dict[str, tuple[str, ...]]

    def intersecting_cells(self, item: Item) -> list[str]:
        if not item.geometry:
            return []
        geom = shape(item.geometry)
        idxs = self.tree.query(geom, predicate="intersects")
        return [self.cells[int(i)] for i in idxs]

    def cities_for_cells(self, cells: Iterable[str]) -> list[str]:
        cities: set[str] = set()
        for cell in cells:
            cities.update(self.cell_cities.get(cell, ()))
        return sorted(cities)


# =============================================================================
# Thread-local clients
# =============================================================================

_thread_local = threading.local()


def get_gcs_client() -> storage.Client:
    client = getattr(_thread_local, "gcs_client", None)
    if client is None:
        client = storage.Client()
        _thread_local.gcs_client = client
    return client


# =============================================================================
# GDAL / HTTP tuning
# =============================================================================


def configure_gdal_for_remote_cogs(http_connections: int) -> None:
    """
    Configure GDAL for efficient HTTP range reads from Planetary Computer COGs.

    These are process-global. Set them once before any worker opens a raster.
    """
    settings = {
        # Prevent GDAL from trying to list sibling objects for each HTTPS file.
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        # Parallel/multiplexed range reads where libcurl/server support it.
        "GDAL_HTTP_MULTIRANGE": "YES",
        "GDAL_HTTP_MULTIPLEX": "YES",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "GDAL_HTTP_VERSION": "2TLS",
        # Retry transient throttling / transport failures inside GDAL.
        "GDAL_HTTP_MAX_RETRY": "5",
        "GDAL_HTTP_RETRY_DELAY": "1",
        "GDAL_HTTP_TIMEOUT": "180",
        # Keep a meaningful curl range cache for tiled GeoTIFF blocks.
        "CPL_VSIL_CURL_CACHE_SIZE": str(128 * 1024 * 1024),
        "VSI_CACHE": "TRUE",
        "VSI_CACHE_SIZE": str(64 * 1024 * 1024),
        # Cap libcurl connection growth when many source-item workers run.
        "GDAL_HTTP_MAX_TOTAL_CONNECTIONS": str(max(8, http_connections)),
        "GDAL_HTTP_MAX_CACHED_CONNECTIONS": str(max(8, http_connections)),
    }
    for key, value in settings.items():
        os.environ.setdefault(key, value)


# =============================================================================
# Generic helpers
# =============================================================================


def parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got {uri}")
    path = uri[5:]
    bucket, _, blob = path.partition("/")
    return bucket, blob.rstrip("/")


def safe_id(value: str) -> str:
    return value.replace("/", "_")


def utc_datetime(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def write_manifest(df: pl.DataFrame, local_name: str, destination_uri: str) -> None:
    path = Path(local_name)
    df.write_parquet(path, compression="zstd", statistics=True)

    bucket_name, blob_name = parse_gs_uri(destination_uri)
    get_gcs_client().bucket(bucket_name).blob(blob_name).upload_from_filename(str(path))
    path.unlink(missing_ok=True)
    log.info("Wrote manifest: %s", destination_uri)


def upload_file(local_path: str, destination_uri: str) -> int:
    """Upload one finished clipped raster and return the uploaded byte count."""
    client = get_gcs_client()
    bucket_name, blob_name = parse_gs_uri(destination_uri)
    blob = client.bucket(bucket_name).blob(blob_name)
    blob.chunk_size = 32 * 1024 * 1024  # 32 MiB resumable upload chunks

    size = os.path.getsize(local_path)
    blob.upload_from_filename(
        local_path,
        content_type="image/tiff",
        timeout=600,
    )
    return size


def list_existing_outputs(gcs_root: str) -> dict[str, int]:
    """List optimized output objects once, avoiding one metadata request per item."""
    client = get_gcs_client()
    bucket_name, root_blob = parse_gs_uri(gcs_root)
    prefix = f"{root_blob.rstrip('/')}/" if root_blob else ""
    existing: dict[str, int] = {}
    for blob in client.list_blobs(bucket_name, prefix=prefix):
        if blob.name.endswith(".tif"):
            existing[blob.name] = int(blob.size or 0)
    return existing


# =============================================================================
# H3 helpers
# =============================================================================


def int_h3_to_string(value: int) -> str:
    return h3.int_to_str(int(value))


def normalize_h3(cell: str, target_resolution: int) -> Iterable[str]:
    source_resolution = h3.get_resolution(cell)

    if source_resolution == target_resolution:
        yield cell
        return
    if source_resolution < target_resolution:
        yield from h3.cell_to_children(cell, target_resolution)
        return
    yield h3.cell_to_parent(cell, target_resolution)


def h3_polygon(cell: str) -> Polygon:
    return Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(cell)])


def build_spatial_index(h3_df: pl.DataFrame) -> H3SpatialIndex:
    cell_to_cities: dict[str, set[str]] = defaultdict(set)
    for row in h3_df.iter_rows(named=True):
        cell_to_cities[row["h3_cell"]].add(row["source_city"])

    cells = sorted(cell_to_cities)
    polygons = [h3_polygon(cell) for cell in cells]

    return H3SpatialIndex(
        cells=cells,
        polygons=polygons,
        tree=STRtree(polygons),
        cell_cities={cell: tuple(sorted(cities)) for cell, cities in cell_to_cities.items()},
    )


def geographic_bbox_for_cells(cells: Iterable[str]) -> tuple[float, float, float, float]:
    min_lon = 180.0
    min_lat = 90.0
    max_lon = -180.0
    max_lat = -90.0

    found = False
    for cell in cells:
        found = True
        for lat, lon in h3.cell_to_boundary(cell):
            min_lon = min(min_lon, lon)
            max_lon = max(max_lon, lon)
            min_lat = min(min_lat, lat)
            max_lat = max(max_lat, lat)

    if not found:
        raise ValueError("No H3 cells supplied")
    return min_lon, min_lat, max_lon, max_lat


# =============================================================================
# CrimeNet domain loading
# =============================================================================


def source_configuration(table: str) -> tuple[str, str]:
    if table == "integration_samples":
        return INTEGRATION_SAMPLES_ROOT, "sample_timestamp_utc"
    if table == "event_spine":
        return EVENT_SPINE_ROOT, "occurrence_timestamp_utc"
    raise ValueError(table)


def load_crimenet_domain(
    table: str,
    target_resolution: int,
    city_filter: Optional[list[str]],
) -> tuple[list[CityDomain], pl.DataFrame]:
    root, timestamp_column = source_configuration(table)
    gcp = pl.CredentialProviderGCP()

    lf = (
        pl.scan_delta(root, credential_provider=gcp)
        .select("source_city", "osm_h3_cell_id", timestamp_column)
        .filter(
            pl.col("source_city").is_not_null()
            & pl.col("osm_h3_cell_id").is_not_null()
            & pl.col(timestamp_column).is_not_null()
        )
    )

    if city_filter:
        lf = lf.filter(pl.col("source_city").is_in(city_filter))

    time_ranges = (
        lf.group_by("source_city")
        .agg(
            pl.col(timestamp_column).min().alias("min_timestamp"),
            pl.col(timestamp_column).max().alias("max_timestamp"),
        )
        .collect()
    )

    spatial = lf.select("source_city", "osm_h3_cell_id").unique().collect()
    log.info(f"Unique source H3 cells: {spatial.height:,}")

    resolution_counts: Counter[int] = Counter()
    cells_by_city: dict[str, set[str]] = defaultdict(set)

    for city, raw_cell in spatial.iter_rows():
        cell = int_h3_to_string(raw_cell)
        resolution_counts[h3.get_resolution(cell)] += 1
        cells_by_city[city].update(normalize_h3(cell, target_resolution))

    log.info("Source H3 resolution distribution: %s", dict(sorted(resolution_counts.items())))

    for resolution, count in sorted(resolution_counts.items()):
        if resolution < target_resolution:
            multiplier = 7 ** (target_resolution - resolution)
            log.warning(
                f"{count:,} cells at H3-{resolution} expand by approximately "
                f"{multiplier}x to H3-{target_resolution}"
            )

    ranges = {
        row["source_city"]: (row["min_timestamp"], row["max_timestamp"])
        for row in time_ranges.iter_rows(named=True)
    }

    domains: list[CityDomain] = []
    h3_records: list[dict] = []

    for city, cells in sorted(cells_by_city.items()):
        min_timestamp, max_timestamp = ranges[city]
        domain = CityDomain(
            source_city=city,
            cells=frozenset(cells),
            min_timestamp=min_timestamp,
            max_timestamp=max_timestamp,
        )
        domains.append(domain)

        log.info(
            f"{city}: {len(cells):,} H3-{target_resolution} cells; "
            f"bbox={tuple(round(v, 5) for v in domain.bbox)}; "
            f"time={min_timestamp} -> {max_timestamp}"
        )

        for cell in cells:
            lat, lon = h3.cell_to_latlng(cell)
            h3_records.append(
                {
                    "source_city": city,
                    "h3_resolution": target_resolution,
                    "h3_cell": cell,
                    "h3_cell_int": h3.str_to_int(cell),
                    "latitude": lat,
                    "longitude": lon,
                }
            )

    h3_df = pl.DataFrame(h3_records, schema=H3_MANIFEST_SCHEMA)
    return domains, h3_df


# =============================================================================
# STAC discovery / selection
# =============================================================================


def get_catalog() -> Client:
    # Keep search metadata unsigned. Each worker signs its item immediately before
    # raster access so long runs do not die on stale SAS URLs.
    return Client.open(PC_STAC_URL)


def search_start(
    domain_start: datetime,
    source_start: datetime,
    lookback_years: int,
) -> datetime:
    candidate = utc_datetime(domain_start) - timedelta(days=366 * lookback_years)
    return max(candidate, source_start)


def item_intersects_domain(item: Item, tree: STRtree) -> bool:
    if not item.geometry:
        return False
    return len(tree.query(shape(item.geometry), predicate="intersects")) > 0


def search_naip(
    catalog: Client,
    domain: CityDomain,
    lookback_years: int,
) -> list[Item]:
    start = search_start(
        domain.min_timestamp,
        datetime(2010, 1, 1, tzinfo=timezone.utc),
        lookback_years,
    )

    search = catalog.search(
        collections=["naip"],
        bbox=list(domain.bbox),
        datetime=f"{start.isoformat()}/{utc_datetime(domain.max_timestamp).isoformat()}",
    )
    items = list(search.items())
    log.info(f"{domain.source_city}: STAC returned {len(items):,} NAIP candidates")

    local_tree = STRtree([h3_polygon(cell) for cell in domain.cells])
    filtered = [item for item in items if item_intersects_domain(item, local_tree)]
    log.info(
        f"{domain.source_city}: {len(filtered):,} NAIP items intersect CrimeNet H3 domain"
    )
    return filtered


def sentinel_period(dt: datetime, cadence: str) -> str:
    if cadence == "monthly":
        return f"{dt.year:04d}-{dt.month:02d}"
    if cadence == "quarterly":
        quarter = ((dt.month - 1) // 3) + 1
        return f"{dt.year:04d}-Q{quarter}"
    if cadence == "yearly":
        return f"{dt.year:04d}"
    raise ValueError(cadence)


def search_sentinel2(
    catalog: Client,
    domain: CityDomain,
    max_cloud_cover: float,
    cadence: str,
    lookback_years: int,
) -> list[Item]:
    sentinel_start = datetime(2015, 6, 23, tzinfo=timezone.utc)
    start = search_start(domain.min_timestamp, sentinel_start, lookback_years)

    if start > utc_datetime(domain.max_timestamp):
        return []

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=list(domain.bbox),
        datetime=f"{start.isoformat()}/{utc_datetime(domain.max_timestamp).isoformat()}",
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
    )
    candidates = list(search.items())
    log.info(
        f"{domain.source_city}: STAC returned {len(candidates):,} Sentinel-2 "
        f"candidates (< {max_cloud_cover:.1f}% cloud)"
    )

    local_tree = STRtree([h3_polygon(cell) for cell in domain.cells])
    candidates = [item for item in candidates if item_intersects_domain(item, local_tree)]

    # Best granule-level cloud cover per MGRS tile and temporal bucket.
    best: dict[tuple[str, str], Item] = {}
    for item in candidates:
        if item.datetime is None:
            continue
        mgrs = item.properties.get("s2:mgrs_tile")
        if not mgrs:
            continue

        key = (str(mgrs), sentinel_period(item.datetime, cadence))
        cloud = float(item.properties.get("eo:cloud_cover", 100.0))
        current = best.get(key)
        if current is None or cloud < float(current.properties.get("eo:cloud_cover", 100.0)):
            best[key] = item

    selected = list(best.values())
    log.info(
        f"{domain.source_city}: selected {len(selected):,} Sentinel-2 scenes at {cadence} cadence"
    )
    return selected


def item_key(item: Item) -> tuple[str, str]:
    return item.collection_id, item.id


def globally_dedupe_items(
    city_items: dict[str, list[Item]],
) -> tuple[dict[tuple[str, str], Item], dict[tuple[str, str], set[str]]]:
    unique: dict[tuple[str, str], Item] = {}
    cities: dict[tuple[str, str], set[str]] = defaultdict(set)

    for city, items in city_items.items():
        for item in items:
            key = item_key(item)
            unique[key] = item
            cities[key].add(city)
    return unique, cities


# =============================================================================
# Raster clipping helpers
# =============================================================================


def clamp_window(window: Window, width: int, height: int) -> Window:
    full = Window(0, 0, width, height)
    clipped = window.intersection(full)
    # Integer offsets/lengths avoid subpixel ambiguity in block copying.
    return Window(
        col_off=max(0, math.floor(clipped.col_off)),
        row_off=max(0, math.floor(clipped.row_off)),
        width=max(1, math.ceil(clipped.width)),
        height=max(1, math.ceil(clipped.height)),
    ).intersection(full)


def source_window_for_cells(
    cells: list[str],
    src: rasterio.io.DatasetReader,
    margin_m: float,
) -> Window:
    geo_bounds = geographic_bbox_for_cells(cells)
    projected = transform_bounds("EPSG:4326", src.crs, *geo_bounds, densify_pts=21)

    left, bottom, right, top = projected
    if src.crs and src.crs.is_projected:
        left -= margin_m
        bottom -= margin_m
        right += margin_m
        top += margin_m

    # Intersect requested bounds with the source's true raster bounds.
    left = max(left, src.bounds.left)
    bottom = max(bottom, src.bounds.bottom)
    right = min(right, src.bounds.right)
    top = min(top, src.bounds.top)

    if left >= right or bottom >= top:
        raise ValueError("Requested H3 coverage does not overlap raster bounds")

    return clamp_window(from_bounds(left, bottom, right, top, src.transform), src.width, src.height)


def compression_profile(compression: str) -> dict:
    c = compression.upper()
    if c == "ZSTD":
        return {"compress": "ZSTD", "zstd_level": 3, "predictor": 2}
    if c == "DEFLATE":
        return {"compress": "DEFLATE", "zlevel": 4, "predictor": 2}
    if c == "LZW":
        return {"compress": "LZW", "predictor": 2}
    raise ValueError(f"Unsupported compression: {compression}")


def maybe_convert_to_cog(
    source_path: str,
    output_path: str,
    compression: str,
    categorical: bool,
) -> str:
    """Convert tiled GTiff to a proper COG when the GDAL COG driver exists."""
    with rasterio.Env() as env:
        has_cog = "COG" in env.drivers()

    if not has_cog:
        os.replace(source_path, output_path)
        return output_path

    try:
        rio_copy(
            source_path,
            output_path,
            driver="COG",
            compress=compression.upper(),
            blocksize=512,
            overview_resampling="nearest" if categorical else "average",
            BIGTIFF="IF_SAFER",
        )
        os.unlink(source_path)
        return output_path
    except Exception as exc:
        log.warning("COG conversion failed (%s); retaining tiled GeoTIFF", exc)
        os.replace(source_path, output_path)
        return output_path


def write_naip_clip(
    item: Item,
    cells: list[str],
    output_path: str,
    margin_m: float,
    compression: str,
    make_cog: bool,
) -> None:
    signed = planetary_computer.sign(item)
    if NAIP_ASSET not in signed.assets:
        raise KeyError(f"NAIP item {item.id} lacks '{NAIP_ASSET}' asset")

    href = signed.assets[NAIP_ASSET].href
    tmp_gtiff = output_path + ".working.tif"

    with rasterio.open(href, sharing=False) as src:
        src_window = source_window_for_cells(cells, src, margin_m)
        dst_transform = src.window_transform(src_window)
        width = int(src_window.width)
        height = int(src_window.height)

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            width=width,
            height=height,
            transform=dst_transform,
            tiled=True,
            blockxsize=min(512, max(16, 2 ** int(math.floor(math.log2(max(16, min(512, width))))))),
            blockysize=min(512, max(16, 2 ** int(math.floor(math.log2(max(16, min(512, height))))))),
            BIGTIFF="IF_SAFER",
            **compression_profile(compression),
        )

        # GeoTIFF tile dimensions should be multiples of 16. For tiny edge-case
        # clips, disabling tiling is safer than emitting an invalid tile shape.
        if width < 16 or height < 16:
            profile.pop("blockxsize", None)
            profile.pop("blockysize", None)
            profile["tiled"] = False

        with rasterio.open(tmp_gtiff, "w", **profile) as dst:
            for _, dst_window in dst.block_windows():
                read_window = Window(
                    col_off=src_window.col_off + dst_window.col_off,
                    row_off=src_window.row_off + dst_window.row_off,
                    width=dst_window.width,
                    height=dst_window.height,
                )
                data = src.read(window=read_window)
                dst.write(data, window=dst_window)

            # Preserve useful source metadata without relying on expiring URLs.
            dst.update_tags(
                source_collection="naip",
                source_item_id=item.id,
                capture_timestamp_utc=utc_datetime(item.datetime).isoformat() if item.datetime else "",
                h3_resolution=TARGET_H3_RESOLUTION,
            )

    if make_cog:
        maybe_convert_to_cog(tmp_gtiff, output_path, compression, categorical=False)
    else:
        os.replace(tmp_gtiff, output_path)


def write_sentinel_clip(
    item: Item,
    cells: list[str],
    output_path: str,
    margin_m: float,
    compression: str,
    make_cog: bool,
) -> None:
    signed = planetary_computer.sign(item)
    for asset_key in SENTINEL_ASSETS:
        if asset_key not in signed.assets:
            raise KeyError(f"Sentinel item {item.id} lacks '{asset_key}' asset")

    tmp_gtiff = output_path + ".working.tif"

    # Keep all seven remote datasets open once per source item. This is the core
    # speedup versus reopening every band for every H3 cell.
    datasets: dict[str, rasterio.io.DatasetReader] = {}
    vrts: dict[str, WarpedVRT] = {}
    try:
        for key in SENTINEL_ASSETS:
            datasets[key] = rasterio.open(signed.assets[key].href, sharing=False)

        ref = datasets[SENTINEL_REFERENCE_ASSET]
        ref_window = source_window_for_cells(cells, ref, margin_m)
        dst_transform = ref.window_transform(ref_window)
        width = int(ref_window.width)
        height = int(ref_window.height)

        # WarpedVRT aligns 20 m SWIR/SCL assets to the B02 10 m reference grid,
        # while preserving 10 m assets on that same grid.
        for key, src in datasets.items():
            resampling = Resampling.nearest if key == "SCL" else Resampling.bilinear
            vrts[key] = WarpedVRT(
                src,
                crs=ref.crs,
                transform=dst_transform,
                width=width,
                height=height,
                resampling=resampling,
                dtype=SENTINEL_OUTPUT_DTYPE,
            )

        profile = {
            "driver": "GTiff",
            "dtype": SENTINEL_OUTPUT_DTYPE,
            "count": len(SENTINEL_ASSETS),
            "width": width,
            "height": height,
            "crs": ref.crs,
            "transform": dst_transform,
            "nodata": 0,
            "tiled": True,
            "blockxsize": 512 if width >= 512 else max(16, (width // 16) * 16),
            "blockysize": 512 if height >= 512 else max(16, (height // 16) * 16),
            "BIGTIFF": "IF_SAFER",
            **compression_profile(compression),
        }
        if width < 16 or height < 16:
            profile.pop("blockxsize", None)
            profile.pop("blockysize", None)
            profile["tiled"] = False

        with rasterio.open(tmp_gtiff, "w", **profile) as dst:
            for band_index, key in enumerate(SENTINEL_ASSETS, start=1):
                dst.set_band_description(band_index, key)

            # Iterate output tiles. Each VRT read becomes targeted COG range reads
            # against the remote source asset rather than a full-file download.
            for _, block_window in dst.block_windows():
                for band_index, key in enumerate(SENTINEL_ASSETS, start=1):
                    data = vrts[key].read(1, window=block_window, out_dtype=SENTINEL_OUTPUT_DTYPE)
                    dst.write(data, band_index, window=block_window)

            dst.update_tags(
                source_collection="sentinel-2-l2a",
                source_item_id=item.id,
                capture_timestamp_utc=utc_datetime(item.datetime).isoformat() if item.datetime else "",
                s2_mgrs_tile=str(item.properties.get("s2:mgrs_tile", "")),
                s2_processing_baseline=str(item.properties.get("s2:processing_baseline", "")),
                eo_cloud_cover=str(item.properties.get("eo:cloud_cover", "")),
                h3_resolution=TARGET_H3_RESOLUTION,
                band_order=",".join(SENTINEL_ASSETS),
            )

    finally:
        for vrt in vrts.values():
            vrt.close()
        for src in datasets.values():
            src.close()

    if make_cog:
        # Mixed spectral + categorical data: use nearest overviews to avoid
        # corrupting SCL classes. Full-resolution training pixels are unchanged.
        maybe_convert_to_cog(tmp_gtiff, output_path, compression, categorical=True)
    else:
        os.replace(tmp_gtiff, output_path)


# =============================================================================
# Item processing
# =============================================================================


def output_uri(gcs_root: str, collection: str, item_id: str) -> str:
    if collection == "naip":
        source = "naip"
        filename = "image.tif"
    elif collection == "sentinel-2-l2a":
        source = "sentinel2"
        filename = "stack_B02_B03_B04_B08_B11_B12_SCL.tif"
    else:
        source = safe_id(collection)
        filename = "image.tif"

    return (
        f"{gcs_root.rstrip('/')}/{source}/"
        f"item_id={safe_id(item_id)}/{filename}"
    )


def result_row_base(
    item: Item,
    cells: list[str],
    spatial_index: H3SpatialIndex,
    gcs_uri: str,
) -> dict:
    bbox = item.bbox or [None, None, None, None]
    cloud = item.properties.get("eo:cloud_cover")
    baseline = item.properties.get("s2:processing_baseline")

    return {
        "collection": str(item.collection_id),
        "item_id": item.id,
        "capture_timestamp_utc": utc_datetime(item.datetime),
        "source_cities": spatial_index.cities_for_cells(cells),
        "h3_cell_count": len(cells),
        "h3_cells": sorted(cells),
        "gcs_uri": gcs_uri,
        "status": None,
        "already_existed": None,
        "output_bytes": None,
        "elapsed_seconds": None,
        "effective_mib_per_second": None,
        "eo_cloud_cover": float(cloud) if cloud is not None else None,
        "s2_mgrs_tile": str(item.properties.get("s2:mgrs_tile")) if item.properties.get("s2:mgrs_tile") is not None else None,
        "s2_processing_baseline": str(baseline) if baseline is not None else None,
        "bbox_min_lon": float(bbox[0]) if bbox[0] is not None else None,
        "bbox_min_lat": float(bbox[1]) if bbox[1] is not None else None,
        "bbox_max_lon": float(bbox[2]) if bbox[2] is not None else None,
        "bbox_max_lat": float(bbox[3]) if bbox[3] is not None else None,
        "error": None,
    }


def process_item(
    item: Item,
    spatial_index: H3SpatialIndex,
    gcs_root: str,
    temp_root: str,
    naip_margin_m: float,
    sentinel_margin_m: float,
    compression: str,
    make_cog: bool,
    overwrite: bool,
    existing_outputs: dict[str, int],
) -> dict:
    started = time.perf_counter()
    cells = spatial_index.intersecting_cells(item)
    uri = output_uri(gcs_root, item.collection_id, item.id)
    row = result_row_base(item, cells, spatial_index, uri)

    if not cells:
        row["status"] = "no_h3_overlap"
        row["elapsed_seconds"] = time.perf_counter() - started
        return row

    # Existing outputs were listed once before the worker pool started.
    # This avoids hundreds/thousands of individual GCS metadata requests.
    if not overwrite:
        _, blob_name = parse_gs_uri(uri)
        if blob_name in existing_outputs:
            row["status"] = "exists"
            row["already_existed"] = True
            row["output_bytes"] = existing_outputs[blob_name]
            row["elapsed_seconds"] = time.perf_counter() - started
            return row

    item_dir = Path(temp_root) / safe_id(item.id)
    item_dir.mkdir(parents=True, exist_ok=True)
    local_output = str(item_dir / "clip.tif")

    try:
        if item.collection_id == "naip":
            write_naip_clip(
                item=item,
                cells=cells,
                output_path=local_output,
                margin_m=naip_margin_m,
                compression=compression,
                make_cog=make_cog,
            )
        elif item.collection_id == "sentinel-2-l2a":
            write_sentinel_clip(
                item=item,
                cells=cells,
                output_path=local_output,
                margin_m=sentinel_margin_m,
                compression=compression,
                make_cog=make_cog,
            )
        else:
            raise ValueError(f"Unsupported collection: {item.collection_id}")

        output_bytes = upload_file(local_output, uri)
        elapsed = time.perf_counter() - started

        row["status"] = "uploaded"
        row["already_existed"] = False
        row["output_bytes"] = output_bytes
        row["elapsed_seconds"] = elapsed
        row["effective_mib_per_second"] = (
            (output_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else None
        )
        return row

    except Exception as exc:
        row["status"] = "error"
        row["already_existed"] = False
        row["elapsed_seconds"] = time.perf_counter() - started
        row["error"] = repr(exc)
        return row

    finally:
        try:
            if os.path.exists(local_output):
                os.unlink(local_output)
            working = local_output + ".working.tif"
            if os.path.exists(working):
                os.unlink(working)
            item_dir.rmdir()
        except OSError:
            pass


# =============================================================================
# Pipeline
# =============================================================================


def run_downloads(
    items: dict[tuple[str, str], Item],
    spatial_index: H3SpatialIndex,
    args: argparse.Namespace,
) -> pl.DataFrame:
    if args.dry_run:
        rows: list[dict] = []
        for item in items.values():
            cells = spatial_index.intersecting_cells(item)
            row = result_row_base(item, cells, spatial_index, output_uri(args.gcs_root, item.collection_id, item.id))
            row["status"] = "planned"
            rows.append(row)
        return pl.DataFrame(rows, schema=ITEM_MANIFEST_SCHEMA)

    temp_parent = args.temp_dir if args.temp_dir else None

    existing_outputs: dict[str, int] = {}
    if not args.overwrite:
        log.info("Listing existing optimized GCS outputs once for fast resume...")
        existing_outputs = list_existing_outputs(args.gcs_root)
        log.info(f"Found {len(existing_outputs):,} existing optimized raster objects")

    with tempfile.TemporaryDirectory(prefix="crimenet_imagery_", dir=temp_parent) as temp_root:
        results: list[dict] = []
        total = len(items)

        log.info(
            f"Processing {total:,} source items with {args.workers} concurrent workers "
            f"(remote range reads + local compression + GCS upload)"
        )

        with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="imagery") as executor:
            futures = {
                executor.submit(
                    process_item,
                    item,
                    spatial_index,
                    args.gcs_root,
                    temp_root,
                    args.naip_margin_m,
                    args.sentinel_margin_m,
                    args.compression,
                    args.make_cog,
                    args.overwrite,
                    existing_outputs,
                ): key
                for key, item in items.items()
            }

            for completed, future in enumerate(as_completed(futures), start=1):
                key = futures[future]
                collection, item_id = key
                try:
                    row = future.result()
                except Exception as exc:  # defensive: process_item should absorb errors
                    row = {
                        "collection": collection,
                        "item_id": item_id,
                        "capture_timestamp_utc": None,
                        "source_cities": [],
                        "h3_cell_count": 0,
                        "h3_cells": [],
                        "gcs_uri": output_uri(args.gcs_root, collection, item_id),
                        "status": "error",
                        "already_existed": False,
                        "output_bytes": None,
                        "elapsed_seconds": None,
                        "effective_mib_per_second": None,
                        "eo_cloud_cover": None,
                        "s2_mgrs_tile": None,
                        "s2_processing_baseline": None,
                        "bbox_min_lon": None,
                        "bbox_min_lat": None,
                        "bbox_max_lon": None,
                        "bbox_max_lat": None,
                        "error": repr(exc),
                    }

                results.append(row)

                elapsed = row.get("elapsed_seconds")
                size = row.get("output_bytes")
                size_text = f"{size / (1024**2):.1f} MiB" if size else "-"
                elapsed_text = f"{elapsed:.1f}s" if elapsed is not None else "-"

                log.info(
                    f"[{completed:,}/{total:,}] {row['status']:>10} | {collection} | "
                    f"{item_id} | H3={row.get('h3_cell_count', 0):,} | "
                    f"{size_text} | {elapsed_text}"
                )

        return pl.DataFrame(results, schema=ITEM_MANIFEST_SCHEMA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimized CrimeNet NAIP + Sentinel-2 ingestion: concurrent COG range reads, "
            "source-item clipping, Sentinel band stacking, deduplication, and resumable GCS output."
        )
    )

    parser.add_argument(
        "--table",
        choices=("integration_samples", "event_spine"),
        default="integration_samples",
    )
    parser.add_argument("--target-h3-resolution", type=int, default=TARGET_H3_RESOLUTION)
    parser.add_argument("--gcs-root", default=DEFAULT_GCS_ROOT)
    parser.add_argument("--cities", nargs="*", default=None)

    parser.add_argument(
        "--sentinel-cadence",
        choices=("monthly", "quarterly", "yearly"),
        default="monthly",
    )
    parser.add_argument("--sentinel-max-cloud-cover", type=float, default=20.0)
    parser.add_argument(
        "--history-lookback-years",
        type=int,
        default=5,
        help="Include imagery before the first model timestamp so early observations have leakage-safe history.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Concurrent source-item workers. 6 is a strong default for a laptop; try 8-12 on a fast cloud VM.",
    )
    parser.add_argument(
        "--http-connections",
        type=int,
        default=32,
        help="GDAL/libcurl connection cap shared by remote COG range reads.",
    )
    parser.add_argument(
        "--naip-margin-m",
        type=float,
        default=250.0,
        help="Extra context retained around the union of H3 cells inside each NAIP source tile.",
    )
    parser.add_argument(
        "--sentinel-margin-m",
        type=float,
        default=750.0,
        help="Extra context retained around H3 coverage; supports ~1.5 km Sentinel context windows later.",
    )
    parser.add_argument(
        "--compression",
        choices=("ZSTD", "DEFLATE", "LZW"),
        default="ZSTD",
    )
    parser.add_argument(
        "--make-cog",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Convert clipped outputs to Cloud-Optimized GeoTIFFs (default: true).",
    )
    parser.add_argument("--temp-dir", default=None)

    parser.add_argument("--skip-naip", action="store_true")
    parser.add_argument("--skip-sentinel", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_gdal_for_remote_cogs(args.http_connections)

    domains, h3_df = load_crimenet_domain(
        table=args.table,
        target_resolution=args.target_h3_resolution,
        city_filter=args.cities,
    )
    if not domains:
        raise RuntimeError("No CrimeNet domains matched the requested filters")

    spatial_index = build_spatial_index(h3_df)
    catalog = get_catalog()

    naip_by_city: dict[str, list[Item]] = {}
    sentinel_by_city: dict[str, list[Item]] = {}

    for domain in domains:
        log.info("Discovering imagery for %s", domain.source_city)

        if not args.skip_naip:
            naip_by_city[domain.source_city] = search_naip(
                catalog,
                domain,
                lookback_years=args.history_lookback_years,
            )

        if not args.skip_sentinel:
            sentinel_by_city[domain.source_city] = search_sentinel2(
                catalog,
                domain,
                max_cloud_cover=args.sentinel_max_cloud_cover,
                cadence=args.sentinel_cadence,
                lookback_years=args.history_lookback_years,
            )

    all_by_city: dict[str, list[Item]] = {}
    for domain in domains:
        city = domain.source_city
        all_by_city[city] = naip_by_city.get(city, []) + sentinel_by_city.get(city, [])

    unique_items, _item_cities = globally_dedupe_items(all_by_city)
    collection_counts = Counter(collection for collection, _ in unique_items)

    log.info(
        "Unique source items after global deduplication: %s",
        dict(collection_counts),
    )

    item_manifest = run_downloads(unique_items, spatial_index, args)

    if args.dry_run:
        planned_cells = int(item_manifest["h3_cell_count"].sum()) if item_manifest.height else 0
        print("\n=== DRY RUN SUMMARY ===")
        print(f"Unique H3-{args.target_h3_resolution} cells: {h3_df['h3_cell'].n_unique():,}")
        print(f"Unique source items: {len(unique_items):,}")
        print(f"By collection: {dict(collection_counts)}")
        print(f"Total item-to-H3 coverage links: {planned_cells:,}")
        print(f"Workers for real run: {args.workers}")
        print("No raster bytes were read and no GCS writes/existence checks were performed.")
        return

    manifest_root = f"{args.gcs_root.rstrip('/')}/manifests"
    write_manifest(
        h3_df,
        "h3_r9_cells.parquet",
        f"{manifest_root}/h3_r9_cells.parquet",
    )
    write_manifest(
        item_manifest,
        "imagery_items.parquet",
        f"{manifest_root}/imagery_items.parquet",
    )

    status_counts = item_manifest.group_by("status").len().sort("status")
    print("\n=== COMPLETE ===")
    print(status_counts)

    failures = item_manifest.filter(pl.col("status") == "error")
    if failures.height:
        print(f"\n{failures.height:,} source items failed. Rerun the same command to resume; existing outputs are skipped.")
        print(failures.select("collection", "item_id", "error").head(20))


if __name__ == "__main__":
    main()