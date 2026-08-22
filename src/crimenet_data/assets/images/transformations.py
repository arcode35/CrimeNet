from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import math
from typing import Callable, Iterable

import h3
import numpy as np
import polars as pl
import rasterio
from pyproj import CRS, Transformer
from rasterio.windows import Window, from_bounds


# Sentinel-2 L2A Scene Classification Layer (SCL) classes.
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

SENTINEL_SCL_BAND_INDEX = 7

CLOUD_CLASSES = {
    SCL_CLOUD_MEDIUM_PROBABILITY,
    SCL_CLOUD_HIGH_PROBABILITY,
    SCL_THIN_CIRRUS,
}
SHADOW_CLASSES = {SCL_CLOUD_SHADOW}
INVALID_CLASSES = {SCL_NO_DATA, SCL_SATURATED_OR_DEFECTIVE}
SNOW_CLASSES = {SCL_SNOW_OR_ICE}
BAD_CLASSES = CLOUD_CLASSES | SHADOW_CLASSES | INVALID_CLASSES | SNOW_CLASSES


CANDIDATE_SCHEMA: dict[str, pl.DataType] = {
    "source": pl.Utf8,
    "collection": pl.Utf8,
    "item_id": pl.Utf8,
    "capture_timestamp_utc": pl.Datetime("us", "UTC"),
    "capture_period": pl.Utf8,
    "source_cities": pl.List(pl.Utf8),
    "h3_cell": pl.Utf8,
    "h3_resolution": pl.Int8,
    "gcs_uri": pl.Utf8,
    "raster_crs": pl.Utf8,
    "raster_width": pl.Int64,
    "raster_height": pl.Int64,
    "raster_band_count": pl.Int16,
    "pixel_size_x": pl.Float64,
    "pixel_size_y": pl.Float64,
    "window_col_off": pl.Int64,
    "window_row_off": pl.Int64,
    "window_width": pl.Int64,
    "window_height": pl.Int64,
    "coverage_fraction": pl.Float64,
    "requires_mosaic": pl.Boolean,
    "eo_cloud_cover": pl.Float64,
    "local_cloud_fraction": pl.Float64,
    "local_shadow_fraction": pl.Float64,
    "local_invalid_fraction": pl.Float64,
    "local_snow_fraction": pl.Float64,
    "local_bad_fraction": pl.Float64,
    "local_clear_fraction": pl.Float64,
    "is_usable": pl.Boolean,
    "s2_mgrs_tile": pl.Utf8,
    "s2_processing_baseline": pl.Utf8,
    "error": pl.Utf8,
}

TEMPORAL_INDEX_SCHEMA: dict[str, pl.DataType] = {
    **CANDIDATE_SCHEMA,
    "selected_in_period": pl.Boolean,
    "candidate_rank": pl.Int32,
    "valid_from_utc": pl.Datetime("us", "UTC"),
    "valid_to_utc": pl.Datetime("us", "UTC"),
}


@dataclass(frozen=True)
class ImageryPreprocessSettings:
    target_h3_resolution: int = 9
    workers: int = 8
    # NAIP: retain the full H3 footprint plus a modest local context margin.
    naip_context_margin_m: float = 32.0
    # Sentinel: H3 width is ~350 m at r9; +600 m each side yields ~1.5 km context.
    sentinel_context_margin_m: float = 600.0
    sentinel_max_local_bad_fraction: float = 0.20
    min_single_source_coverage_fraction: float = 0.995


@dataclass(frozen=True)
class RasterWindowInfo:
    col_off: int
    row_off: int
    width: int
    height: int
    coverage_fraction: float


# -----------------------------------------------------------------------------
# URI / CRS / geometry helpers
# -----------------------------------------------------------------------------


def gcs_uri_to_vsigs(uri: str) -> str:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got {uri!r}")
    return "/vsigs/" + uri[5:]


@lru_cache(maxsize=128)
def _transformer_to_crs(crs_text: str) -> Transformer:
    return Transformer.from_crs("EPSG:4326", CRS.from_user_input(crs_text), always_xy=True)


def _meters_to_crs_units(crs: CRS, meters: float) -> float:
    if meters == 0:
        return 0.0
    if not crs.is_projected:
        raise ValueError(
            f"Expected projected imagery CRS for meter-based context windows; got {crs.to_string()}"
        )

    axis_info = crs.axis_info
    if not axis_info:
        return meters

    # pyproj's unit_conversion_factor converts the CRS axis unit to meters.
    meters_per_unit = axis_info[0].unit_conversion_factor or 1.0
    return meters / float(meters_per_unit)


@lru_cache(maxsize=500_000)
def h3_context_bounds_in_crs(
    h3_cell: str,
    crs_text: str,
    context_margin_m: float,
) -> tuple[float, float, float, float]:
    """
    Exact H3 polygon bounds transformed into the raster CRS, then expanded by
    a configurable meter-based context margin.
    """
    crs = CRS.from_user_input(crs_text)
    transformer = _transformer_to_crs(crs_text)

    # h3 returns (lat, lon); pyproj with always_xy expects (lon, lat).
    boundary = h3.cell_to_boundary(h3_cell)
    xs: list[float] = []
    ys: list[float] = []

    for lat, lon in boundary:
        x, y = transformer.transform(lon, lat)
        xs.append(float(x))
        ys.append(float(y))

    margin_units = _meters_to_crs_units(crs, context_margin_m)

    return (
        min(xs) - margin_units,
        min(ys) - margin_units,
        max(xs) + margin_units,
        max(ys) + margin_units,
    )


def _window_from_requested_bounds(
    src: rasterio.io.DatasetReader,
    requested_bounds: tuple[float, float, float, float],
) -> RasterWindowInfo | None:
    left, bottom, right, top = requested_bounds
    if right <= left or top <= bottom:
        return None

    # Convert requested bounds into a pixel window, rounding *outward* so the
    # requested spatial footprint is fully represented.
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

    coverage_fraction = (width * height) / float(full_width * full_height)

    return RasterWindowInfo(
        col_off=int(col0),
        row_off=int(row0),
        width=int(width),
        height=int(height),
        coverage_fraction=float(min(1.0, max(0.0, coverage_fraction))),
    )


# -----------------------------------------------------------------------------
# SCL quality metrics
# -----------------------------------------------------------------------------


def _fraction(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 1.0
    return float(mask.mean())


def _scl_quality(scl_window: np.ndarray) -> dict[str, float]:
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
        "local_cloud_fraction": _fraction(cloud),
        "local_shadow_fraction": _fraction(shadow),
        "local_invalid_fraction": _fraction(invalid),
        "local_snow_fraction": _fraction(snow),
        "local_bad_fraction": _fraction(bad),
        "local_clear_fraction": float(1.0 - _fraction(bad)),
    }


# -----------------------------------------------------------------------------
# Per-item preprocessing
# -----------------------------------------------------------------------------


def _capture_period(source: str, capture_timestamp: datetime | None) -> str | None:
    if capture_timestamp is None:
        return None
    if source == "sentinel2":
        return f"{capture_timestamp.year:04d}-{capture_timestamp.month:02d}"
    return capture_timestamp.date().isoformat()


def _base_candidate_row(
    manifest_row: dict,
    *,
    source: str,
    h3_cell: str,
    src: rasterio.io.DatasetReader,
    window: RasterWindowInfo,
    settings: ImageryPreprocessSettings,
) -> dict:
    capture_timestamp = manifest_row.get("capture_timestamp_utc")
    return {
        "source": source,
        "collection": manifest_row["collection"],
        "item_id": manifest_row["item_id"],
        "capture_timestamp_utc": capture_timestamp,
        "capture_period": _capture_period(source, capture_timestamp),
        "source_cities": manifest_row.get("source_cities") or [],
        "h3_cell": h3_cell,
        "h3_resolution": h3.get_resolution(h3_cell),
        "gcs_uri": manifest_row["gcs_uri"],
        "raster_crs": src.crs.to_string() if src.crs else None,
        "raster_width": int(src.width),
        "raster_height": int(src.height),
        "raster_band_count": int(src.count),
        "pixel_size_x": float(abs(src.transform.a)),
        "pixel_size_y": float(abs(src.transform.e)),
        "window_col_off": window.col_off,
        "window_row_off": window.row_off,
        "window_width": window.width,
        "window_height": window.height,
        "coverage_fraction": window.coverage_fraction,
        "requires_mosaic": (
            window.coverage_fraction < settings.min_single_source_coverage_fraction
        ),
        "eo_cloud_cover": (
            float(manifest_row["eo_cloud_cover"])
            if manifest_row.get("eo_cloud_cover") is not None
            else None
        ),
        "local_cloud_fraction": None,
        "local_shadow_fraction": None,
        "local_invalid_fraction": None,
        "local_snow_fraction": None,
        "local_bad_fraction": None,
        "local_clear_fraction": None,
        "is_usable": True,
        "s2_mgrs_tile": manifest_row.get("s2_mgrs_tile"),
        "s2_processing_baseline": manifest_row.get("s2_processing_baseline"),
        "error": None,
    }


def _open_raster(uri: str):
    # Optimized GCS outputs are COGs. /vsigs/ lets GDAL perform authenticated
    # range reads directly against GCS rather than downloading the full object.
    return rasterio.open(gcs_uri_to_vsigs(uri))


def preprocess_naip_item(
    manifest_row: dict,
    settings: ImageryPreprocessSettings,
) -> list[dict]:
    rows: list[dict] = []

    with _open_raster(manifest_row["gcs_uri"]) as src:
        if src.crs is None:
            raise ValueError(f"NAIP raster has no CRS: {manifest_row['gcs_uri']}")
        if src.count < 3:
            raise ValueError(
                f"Expected >=3 NAIP bands, got {src.count}: {manifest_row['gcs_uri']}"
            )

        crs_text = src.crs.to_string()

        for cell in manifest_row.get("h3_cells") or []:
            if h3.get_resolution(cell) != settings.target_h3_resolution:
                raise ValueError(
                    f"Expected H3-{settings.target_h3_resolution}, got H3-{h3.get_resolution(cell)}: {cell}"
                )

            requested = h3_context_bounds_in_crs(
                cell,
                crs_text,
                settings.naip_context_margin_m,
            )
            window = _window_from_requested_bounds(src, requested)
            if window is None:
                continue

            row = _base_candidate_row(
                manifest_row,
                source="naip",
                h3_cell=cell,
                src=src,
                window=window,
                settings=settings,
            )
            # NAIP acquisition is deliberately cloud-constrained. We do not
            # invent a cloud score here; coverage is the relevant ingestion QA.
            row["is_usable"] = window.coverage_fraction > 0.0
            rows.append(row)

    return rows


def preprocess_sentinel_item(
    manifest_row: dict,
    settings: ImageryPreprocessSettings,
) -> list[dict]:
    rows: list[dict] = []

    with _open_raster(manifest_row["gcs_uri"]) as src:
        if src.crs is None:
            raise ValueError(f"Sentinel raster has no CRS: {manifest_row['gcs_uri']}")
        if src.count < SENTINEL_SCL_BAND_INDEX:
            raise ValueError(
                f"Expected 7-band Sentinel stack, got {src.count}: {manifest_row['gcs_uri']}"
            )

        # The optimized raster is already clipped to CrimeNet coverage. Read
        # the SCL band once, then score every H3 window in memory. This is much
        # faster than thousands of tiny GCS range reads per source scene.
        scl = src.read(SENTINEL_SCL_BAND_INDEX, out_dtype="uint8")
        crs_text = src.crs.to_string()

        for cell in manifest_row.get("h3_cells") or []:
            if h3.get_resolution(cell) != settings.target_h3_resolution:
                raise ValueError(
                    f"Expected H3-{settings.target_h3_resolution}, got H3-{h3.get_resolution(cell)}: {cell}"
                )

            requested = h3_context_bounds_in_crs(
                cell,
                crs_text,
                settings.sentinel_context_margin_m,
            )
            window = _window_from_requested_bounds(src, requested)
            if window is None:
                continue

            row0 = window.row_off
            row1 = row0 + window.height
            col0 = window.col_off
            col1 = col0 + window.width
            quality = _scl_quality(scl[row0:row1, col0:col1])

            row = _base_candidate_row(
                manifest_row,
                source="sentinel2",
                h3_cell=cell,
                src=src,
                window=window,
                settings=settings,
            )
            row.update(quality)
            row["is_usable"] = bool(
                quality["local_bad_fraction"]
                <= settings.sentinel_max_local_bad_fraction
            )
            rows.append(row)

    return rows


def _error_rows_for_item(manifest_row: dict, error: Exception) -> list[dict]:
    source = "naip" if manifest_row.get("collection") == "naip" else "sentinel2"
    capture_timestamp = manifest_row.get("capture_timestamp_utc")
    message = repr(error)

    rows = []
    for cell in manifest_row.get("h3_cells") or []:
        rows.append(
            {
                "source": source,
                "collection": manifest_row.get("collection"),
                "item_id": manifest_row.get("item_id"),
                "capture_timestamp_utc": capture_timestamp,
                "capture_period": _capture_period(source, capture_timestamp),
                "source_cities": manifest_row.get("source_cities") or [],
                "h3_cell": cell,
                "h3_resolution": h3.get_resolution(cell),
                "gcs_uri": manifest_row.get("gcs_uri"),
                "raster_crs": None,
                "raster_width": None,
                "raster_height": None,
                "raster_band_count": None,
                "pixel_size_x": None,
                "pixel_size_y": None,
                "window_col_off": None,
                "window_row_off": None,
                "window_width": None,
                "window_height": None,
                "coverage_fraction": None,
                "requires_mosaic": None,
                "eo_cloud_cover": manifest_row.get("eo_cloud_cover"),
                "local_cloud_fraction": None,
                "local_shadow_fraction": None,
                "local_invalid_fraction": None,
                "local_snow_fraction": None,
                "local_bad_fraction": None,
                "local_clear_fraction": None,
                "is_usable": False,
                "s2_mgrs_tile": manifest_row.get("s2_mgrs_tile"),
                "s2_processing_baseline": manifest_row.get("s2_processing_baseline"),
                "error": message,
            }
        )
    return rows


def preprocess_item_manifest(
    item_manifest: pl.DataFrame,
    settings: ImageryPreprocessSettings,
    log: Callable[[str], None] | None = None,
) -> pl.DataFrame:
    """
    Expand optimized source imagery into one row per (source item, H3 cell),
    recording the precise crop window and local Sentinel quality metrics.

    Work is grouped by source item so each COG is opened once. Sentinel's SCL
    band is read once per item and reused for every covered H3 cell.
    """
    logger = log or (lambda _: None)

    required = {
        "collection",
        "item_id",
        "capture_timestamp_utc",
        "source_cities",
        "h3_cells",
        "gcs_uri",
        "status",
        "eo_cloud_cover",
        "s2_mgrs_tile",
        "s2_processing_baseline",
    }
    missing = required - set(item_manifest.columns)
    if missing:
        raise ValueError(f"Imagery manifest is missing required columns: {sorted(missing)}")

    manifest = (
        item_manifest
        .filter(pl.col("status").is_in(["uploaded", "exists"]))
        .filter(pl.col("gcs_uri").is_not_null())
        .filter(pl.col("h3_cells").list.len() > 0)
        .unique(subset=["collection", "item_id"], keep="first")
    )

    records = manifest.to_dicts()
    total = len(records)
    logger(f"Preprocessing {total:,} optimized source rasters with {settings.workers} workers")

    all_rows: list[dict] = []
    completed = 0

    def run_one(record: dict) -> list[dict]:
        collection = record["collection"]
        if collection == "naip":
            return preprocess_naip_item(record, settings)
        if collection == "sentinel-2-l2a":
            return preprocess_sentinel_item(record, settings)
        raise ValueError(f"Unsupported imagery collection: {collection}")

    with ThreadPoolExecutor(max_workers=settings.workers) as pool:
        future_to_record = {
            pool.submit(run_one, record): record
            for record in records
        }

        for future in as_completed(future_to_record):
            record = future_to_record[future]
            completed += 1
            try:
                rows = future.result()
                all_rows.extend(rows)
                if completed % 25 == 0 or completed == total:
                    logger(
                        f"[{completed:,}/{total:,}] {record['collection']} | "
                        f"{record['item_id']} | emitted {len(rows):,} H3 candidates"
                    )
            except Exception as exc:
                error_rows = _error_rows_for_item(record, exc)
                all_rows.extend(error_rows)
                logger(
                    f"ERROR [{completed:,}/{total:,}] {record['collection']} | "
                    f"{record['item_id']} | {exc!r}"
                )

    if not all_rows:
        return pl.DataFrame(schema=CANDIDATE_SCHEMA)

    return pl.DataFrame(all_rows, schema=CANDIDATE_SCHEMA)


# -----------------------------------------------------------------------------
# Candidate ranking / temporal validity index
# -----------------------------------------------------------------------------


def _select_best_naip_per_capture(candidates: pl.DataFrame) -> pl.DataFrame:
    # Neighboring NAIP source tiles can overlap the same H3. Prefer the source
    # that covers the largest fraction of the requested H3+context window.
    return (
        candidates
        .filter((pl.col("source") == "naip") & pl.col("is_usable"))
        .sort(
            ["h3_cell", "capture_timestamp_utc", "coverage_fraction", "item_id"],
            descending=[False, False, True, False],
            nulls_last=True,
        )
        .with_columns(
            pl.col("coverage_fraction")
            .rank(method="ordinal", descending=True)
            .over(["h3_cell", "capture_timestamp_utc"])
            .cast(pl.Int32)
            .alias("candidate_rank")
        )
        .filter(pl.col("candidate_rank") == 1)
    )


def _select_best_sentinel_per_month(candidates: pl.DataFrame) -> pl.DataFrame:
    # The actual local quality signal is SCL-based. Scene-level eo_cloud_cover
    # is only a tie-breaker after local bad-pixel fraction.
    return (
        candidates
        .filter((pl.col("source") == "sentinel2") & pl.col("is_usable"))
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
    )


def build_temporal_index(candidates: pl.DataFrame) -> pl.DataFrame:
    """
    Build a compact, leakage-safe H3 imagery dimension.

    NAIP: one best source for each H3/acquisition timestamp.
    Sentinel: one best locally-clear source for each H3/month among the source
    scenes that were actually ingested.

    Each selected image becomes valid only at its capture timestamp and stays
    valid until the next selected image for that H3/source. Downstream model
    observations can therefore use an as-of join on valid_from_utc without
    looking into the future.

    Rows marked requires_mosaic=True are intentionally retained. A later crop
    loader should combine neighboring source items when full context is needed.
    """
    good = candidates.filter(pl.col("error").is_null())

    naip = _select_best_naip_per_capture(good)
    sentinel = _select_best_sentinel_per_month(good)

    selected = pl.concat([naip, sentinel], how="diagonal_relaxed")

    if selected.is_empty():
        return pl.DataFrame(schema=TEMPORAL_INDEX_SCHEMA)

    selected = (
        selected
        .with_columns(pl.lit(True).alias("selected_in_period"))
        .sort(["source", "h3_cell", "capture_timestamp_utc", "item_id"])
        .with_columns(
            pl.col("capture_timestamp_utc").alias("valid_from_utc"),
            pl.col("capture_timestamp_utc")
            .shift(-1)
            .over(["source", "h3_cell"])
            .alias("valid_to_utc"),
        )
    )

    # Keep only the declared schema/order. Polars will preserve the UTC dtype.
    return selected.select(list(TEMPORAL_INDEX_SCHEMA.keys())).cast(TEMPORAL_INDEX_SCHEMA)


# -----------------------------------------------------------------------------
# Downstream as-of mapping helper
# -----------------------------------------------------------------------------


def map_observations_to_imagery_asof(
    observations: pl.DataFrame,
    temporal_index: pl.DataFrame,
    *,
    observation_h3_column: str,
    observation_time_column: str,
    source: str,
) -> pl.DataFrame:
    """
    Leakage-safe mapping for downstream CrimeNet event/integration observations.

    This intentionally does not read raster pixels. It maps each observation to
    the latest selected image whose capture time is <= the observation time.
    """
    idx = (
        temporal_index
        .filter(pl.col("source") == source)
        .sort(["h3_cell", "valid_from_utc"])
    )

    obs = (
        observations
        .with_columns(
            pl.col(observation_h3_column)
            .cast(pl.Int64)
            .map_elements(h3.int_to_str, return_dtype=pl.Utf8)
            .alias("_imagery_h3_cell")
        )
        .sort(["_imagery_h3_cell", observation_time_column])
    )

    mapped = obs.join_asof(
        idx,
        left_on=observation_time_column,
        right_on="valid_from_utc",
        by_left="_imagery_h3_cell",
        by_right="h3_cell",
        strategy="backward",
        check_sortedness=False,
    )

    return mapped.drop("_imagery_h3_cell")