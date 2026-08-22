import json
import os
import re
import threading
from collections import OrderedDict, deque
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import time
from typing import Any

import dagster as dg
from google.cloud import storage
import numpy as np
import planetary_computer as pc
import polars as pl
import pystac
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, transform as window_transform
import requests
from requests.adapters import HTTPAdapter
import torch
import torch.nn.functional as F
from urllib3.util.retry import Retry

from .silver import (
    SILVER_H3_TEMPORAL_INDEX_URI,
    silver_imagery_h3_temporal_index,
)
from .transformations import BAD_CLASSES, gcs_uri_to_vsigs


# -----------------------------------------------------------------------------
# Durable recipe
# -----------------------------------------------------------------------------

GOLD_IMAGERY_EMBEDDINGS_PREFIX = (
    "gs://crimenet/gold/imagery/embeddings/foundation_v1"
)

DINO_MODEL_ID = "facebook/dinov3-vitl16-pretrain-sat493m"
OLMOEARTH_MODEL_ID = "OLMOEARTH_V1_2_BASE"

STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"
SENTINEL_COLLECTION = "sentinel-2-l2a"

# Existing CrimeNet stack:
#   1 B02, 2 B03, 3 B04, 4 B08, 5 B11, 6 B12, 7 SCL
# Patched stacks append:
#   8 B05, 9 B06, 10 B07, 11 B8A, 12 B01, 13 B09
SENTINEL_SCL_BAND = 7
SENTINEL_PATCHED_OLMO_INDEXES = [1, 2, 3, 4, 8, 9, 10, 11, 5, 6, 12, 13]
SENTINEL_EXISTING_LOCAL_INDEXES = {
    "B02": 1,
    "B03": 2,
    "B04": 3,
    "B08": 4,
    "B11": 5,
    "B12": 6,
}
SENTINEL_MISSING_ASSET_KEYS = ["B05", "B06", "B07", "B8A", "B01", "B09"]
SENTINEL_OLMO_BAND_ORDER = [
    "B02", "B03", "B04", "B08",
    "B05", "B06", "B07", "B8A",
    "B11", "B12", "B01", "B09",
]

TEMPORAL_COLUMNS = [
    "source",
    "h3_cell",
    "h3_resolution",
    "capture_period",
    "capture_timestamp_utc",
    "valid_from_utc",
    "valid_to_utc",
    "item_id",
    "gcs_uri",
    "window_col_off",
    "window_row_off",
    "window_width",
    "window_height",
    "coverage_fraction",
    "requires_mosaic",
    "local_bad_fraction",
    "local_clear_fraction",
    "is_usable",
    "selected_in_period",
    "s2_processing_baseline",
    "error",
]


class ImageryEmbeddingConfig(dg.Config):
    output_prefix: str = GOLD_IMAGERY_EMBEDDINGS_PREFIX
    device: str = "auto"  # auto | cpu | cuda | mps
    precision: str = "bf16"  # bf16 | fp16 | fp32

    # NAIP / DINOv3-SAT
    naip_image_size: int = 512
    naip_batch_size: int = 8

    # Sentinel / OlmoEarth
    sentinel_image_size: int = 128
    sentinel_batch_size: int = 4
    sentinel_max_timesteps: int = 12
    sentinel_lookback_days: int = 400
    sentinel_scene_cache_size: int = 32

    # Output / control
    rows_per_shard: int = 5000
    overwrite: bool = False
    resume: bool = True
    limit_naip_rows: int = 0
    limit_sentinel_h3_cells: int = 0

    # Network
    stac_timeout_seconds: int = 60


# -----------------------------------------------------------------------------
# GCS helpers
# -----------------------------------------------------------------------------


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got {uri!r}")
    bucket, _, blob = uri[5:].partition("/")
    if not bucket or not blob:
        raise ValueError(f"Invalid GCS URI: {uri!r}")
    return bucket, blob


def _join_gs(prefix: str, suffix: str) -> str:
    return prefix.rstrip("/") + "/" + suffix.lstrip("/")


def _gcs_blob(uri: str):
    bucket_name, blob_name = _parse_gs_uri(uri)
    return storage.Client().bucket(bucket_name).blob(blob_name)


def _gcs_exists(uri: str) -> bool:
    return _gcs_blob(uri).exists()


def _upload_bytes(uri: str, payload: bytes, content_type: str) -> None:
    _gcs_blob(uri).upload_from_string(payload, content_type=content_type)


def _write_parquet_shard(uri: str, rows: list[dict]) -> None:
    if not rows:
        return

    df = pl.DataFrame(rows).with_columns(
        pl.col("embedding").cast(pl.List(pl.Float32)),
        pl.col("embedding_dim").cast(pl.Int32),
        pl.col("h3_resolution").cast(pl.Int8),
        pl.col("temporal_sequence_length").cast(pl.Int16),
    )

    with tempfile.TemporaryDirectory(prefix="crimenet_foundation_embeddings_") as tmp:
        local = Path(tmp) / "part.parquet"
        df.write_parquet(
            local,
            compression="zstd",
            compression_level=3,
            statistics=True,
        )
        _gcs_blob(uri).upload_from_filename(
            str(local),
            content_type="application/octet-stream",
        )


def _delete_prefix(prefix: str) -> None:
    bucket_name, probe = _parse_gs_uri(prefix.rstrip("/") + "/_probe")
    blob_prefix = probe.rsplit("/", 1)[0] + "/"
    client = storage.Client()
    for blob in client.list_blobs(bucket_name, prefix=blob_prefix):
        blob.delete()


def _read_temporal_index(uri: str) -> pl.DataFrame:
    return pl.read_parquet(
        uri,
        columns=TEMPORAL_COLUMNS,
        credential_provider=pl.CredentialProviderGCP(),
    )


def _read_json_if_exists(uri: str) -> dict | None:
    blob = _gcs_blob(uri)
    if not blob.exists():
        return None
    return json.loads(blob.download_as_text())


def _existing_shard_count(prefix: str, source: str) -> int:
    bucket_name, probe = _parse_gs_uri(_join_gs(prefix, f"{source}/_probe"))
    blob_prefix = probe.rsplit("/", 1)[0] + "/part-"
    client = storage.Client()
    indexes: list[int] = []
    for blob in client.list_blobs(bucket_name, prefix=blob_prefix):
        name = blob.name.rsplit("/", 1)[-1]
        match = re.fullmatch(r"part-(\d{5})\.parquet", name)
        if match:
            indexes.append(int(match.group(1)))

    if not indexes:
        return 0
    indexes.sort()
    expected = list(range(indexes[-1] + 1))
    if indexes != expected:
        raise RuntimeError(
            f"Non-contiguous {source} embedding shards under {prefix}: {indexes[:20]}..."
        )
    return len(indexes)


# -----------------------------------------------------------------------------
# Runtime helpers
# -----------------------------------------------------------------------------


def _ensure_gdal_gcs_credentials() -> None:
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        return

    adc = Path.home() / ".config/gcloud/application_default_credentials.json"
    if adc.exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(adc)
        return

    raise RuntimeError(
        "GDAL /vsigs/ cannot find GCS credentials. Run "
        "`gcloud auth application-default login` or set "
        "GOOGLE_APPLICATION_CREDENTIALS."
    )


def _resolve_device(requested: str) -> torch.device:
    requested = requested.lower().strip()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("device='cuda' requested but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError("device='mps' requested but Apple MPS is unavailable")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device {requested!r}")


def _autocast_context(device: torch.device, precision: str):
    precision = precision.lower()
    if precision == "fp32" or device.type not in {"cuda", "mps"}:
        return nullcontext()

    if device.type == "cuda":
        if precision == "bf16":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        if precision == "fp16":
            return torch.autocast("cuda", dtype=torch.float16)
    elif device.type == "mps" and precision == "fp16":
        return torch.autocast("mps", dtype=torch.float16)

    return nullcontext()


def _raster_env() -> rasterio.Env:
    return rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_MULTIRANGE="YES",
        GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
        GDAL_HTTP_MAX_RETRY="5",
        GDAL_HTTP_RETRY_DELAY="1",
        GDAL_HTTP_VERSION="2",
        CPL_VSIL_CURL_CHUNK_SIZE="1048576",
        VSI_CACHE="TRUE",
        VSI_CACHE_SIZE="134217728",
    )


def _window_from_row(row: dict) -> Window:
    return Window(
        col_off=int(row["window_col_off"]),
        row_off=int(row["window_row_off"]),
        width=int(row["window_width"]),
        height=int(row["window_height"]),
    )


def _center_window(row_window: Window, size: int) -> Window:
    """Return a native-pixel square centered inside the silver context window."""
    width = min(int(row_window.width), size)
    height = min(int(row_window.height), size)
    col_off = int(row_window.col_off + max(0, (row_window.width - width) // 2))
    row_off = int(row_window.row_off + max(0, (row_window.height - height) // 2))
    return Window(col_off=col_off, row_off=row_off, width=width, height=height)


# -----------------------------------------------------------------------------
# NAIP -> DINOv3 SAT-493M
# -----------------------------------------------------------------------------


def _naip_rgb_uint8(src: rasterio.io.DatasetReader, row: dict, size: int) -> np.ndarray:
    if src.count < 3:
        raise ValueError(f"NAIP source has {src.count} bands; expected >=3")

    window = _window_from_row(row)
    arr = src.read(
        [1, 2, 3],
        window=window,
        out_shape=(3, size, size),
        resampling=Resampling.bilinear,
        boundless=True,
        fill_value=0,
    )
    if arr.size == 0:
        raise ValueError("NAIP crop is empty")

    if np.issubdtype(arr.dtype, np.integer):
        max_value = float(np.iinfo(arr.dtype).max)
        x = arr.astype(np.float32) / max_value
    else:
        x = arr.astype(np.float32, copy=False)
        finite = x[np.isfinite(x)]
        if finite.size == 0:
            raise ValueError("NAIP crop contains no finite pixels")
        if float(np.nanmax(finite)) > 1.5:
            x /= 255.0

    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0).round().astype(np.uint8)
    return np.moveaxis(x, 0, -1)  # HWC


def _load_dino(device: torch.device):
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "DINOv3 requires transformers. Install with `pip install -U transformers`."
        ) from exc

    token = os.getenv("HF_TOKEN") or None
    processor = AutoImageProcessor.from_pretrained(DINO_MODEL_ID, token=token)
    model = AutoModel.from_pretrained(
        DINO_MODEL_ID,
        token=token,
        torch_dtype="auto",
    ).eval().to(device)
    return processor, model


def _encode_dino_batch(
    processor,
    model,
    images: list[np.ndarray],
    device: torch.device,
    precision: str,
    image_size: int,
) -> np.ndarray:
    inputs = processor(
        images=images,
        return_tensors="pt",
        do_resize=False,
    )
    pixel_values = inputs["pixel_values"].to(device, non_blocking=True)

    with torch.inference_mode(), _autocast_context(device, precision):
        outputs = model(pixel_values=pixel_values)

        cls = outputs.pooler_output.float()
        hidden = outputs.last_hidden_state.float()
        num_registers = int(getattr(model.config, "num_register_tokens", 4) or 0)
        patch_tokens = hidden[:, 1 + num_registers :, :]
        patch_mean = patch_tokens.mean(dim=1)

        # Meta documents both CLS and mean-patch representations for downstream use.
        z = torch.cat([cls, patch_mean], dim=1)
        z = F.normalize(z, p=2, dim=1)

    return z.cpu().numpy().astype(np.float32, copy=False)


# -----------------------------------------------------------------------------
# Sentinel hybrid reader
# -----------------------------------------------------------------------------


_thread_local = threading.local()


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
    adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    _thread_local.http_session = session
    return session


def _fetch_signed_missing_assets(item_id: str, timeout_seconds: int) -> dict[str, str]:
    url = f"{STAC_API}/collections/{SENTINEL_COLLECTION}/items/{item_id}"
    response = _http_session().get(url, timeout=timeout_seconds)
    response.raise_for_status()
    item = pc.sign(pystac.Item.from_dict(response.json()))

    missing = [key for key in SENTINEL_MISSING_ASSET_KEYS if key not in item.assets]
    if missing:
        raise KeyError(f"Sentinel item {item_id} missing STAC assets {missing}")

    return {key: item.assets[key].href for key in SENTINEL_MISSING_ASSET_KEYS}


def _parse_processing_baseline(value: Any) -> float | None:
    if value is None:
        return None
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _sentinel_dn_to_reflectance(arr: np.ndarray, processing_baseline: Any) -> np.ndarray:
    """
    Convert Sentinel-2 L2A integer DN to BOA reflectance.

    PB >= 04.00 introduced BOA_ADD_OFFSET=-1000 with QUANTIFICATION_VALUE=10000.
    DN=0 remains nodata and is kept at zero.
    """
    x = arr.astype(np.float32, copy=False)
    nodata = x == 0
    baseline = _parse_processing_baseline(processing_baseline)

    if baseline is not None and baseline >= 4.0:
        x = (x - 1000.0) / 10000.0
    else:
        x = x / 10000.0

    x[nodata] = 0.0
    x = np.nan_to_num(x, nan=0.0, posinf=1.5, neginf=-0.2)
    return np.clip(x, -0.2, 1.5)


def _fill_bad_pixels_with_band_median(
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


class _SentinelSceneBundle:
    def __init__(self, row: dict, timeout_seconds: int):
        self.created_monotonic = time.monotonic()
        self.item_id = row["item_id"]
        self.gcs_uri = row["gcs_uri"]
        self.stack = rasterio.open(gcs_uri_to_vsigs(self.gcs_uri), sharing=False)
        self.remote_sources: dict[str, rasterio.io.DatasetReader] = {}
        self.mode = "patched13" if self.stack.count >= 13 else "lazy7"

        if self.mode == "lazy7":
            if self.stack.count < 7:
                self.close()
                raise ValueError(
                    f"Sentinel stack {self.item_id} has {self.stack.count} bands; expected 7 or 13"
                )
            assets = _fetch_signed_missing_assets(self.item_id, timeout_seconds)
            for key, href in assets.items():
                self.remote_sources[key] = rasterio.open(href, sharing=False)

    def close(self) -> None:
        for src in self.remote_sources.values():
            try:
                src.close()
            except Exception:
                pass
        self.remote_sources.clear()
        try:
            self.stack.close()
        except Exception:
            pass

    def _read_remote_band(
        self,
        asset_key: str,
        target_window: Window,
        out_size: int,
    ) -> np.ndarray:
        src = self.remote_sources[asset_key]
        dst_transform = window_transform(target_window, self.stack.transform)
        src_nodata = src.nodata if src.nodata is not None else 0

        # Critical optimization: the VRT grid is ONLY the H3 crop, not the full
        # Sentinel scene. GDAL therefore range-reads only source blocks needed for
        # this crop instead of resampling/re-writing the entire MGRS scene.
        with WarpedVRT(
            src,
            crs=self.stack.crs,
            transform=dst_transform,
            width=int(target_window.width),
            height=int(target_window.height),
            resampling=Resampling.bilinear,
            src_nodata=src_nodata,
            nodata=0,
            dtype=self.stack.dtypes[0],
        ) as vrt:
            arr = vrt.read(
                1,
                out_shape=(out_size, out_size),
                resampling=Resampling.bilinear,
                out_dtype=self.stack.dtypes[0],
            )
        return arr

    def read_olmo_frame(self, row: dict, out_size: int) -> tuple[np.ndarray, str]:
        base_window = _window_from_row(row)
        target_window = _center_window(base_window, out_size)

        scl = self.stack.read(
            SENTINEL_SCL_BAND,
            window=target_window,
            out_shape=(out_size, out_size),
            resampling=Resampling.nearest,
            boundless=True,
            fill_value=0,
            out_dtype="uint8",
        )
        bad_mask = np.isin(scl, tuple(BAD_CLASSES))

        if self.mode == "patched13":
            chw = self.stack.read(
                SENTINEL_PATCHED_OLMO_INDEXES,
                window=target_window,
                out_shape=(12, out_size, out_size),
                resampling=Resampling.bilinear,
                boundless=True,
                fill_value=0,
            )
            spectral = np.moveaxis(chw, 0, -1)
        else:
            bands: dict[str, np.ndarray] = {}
            for name, idx in SENTINEL_EXISTING_LOCAL_INDEXES.items():
                bands[name] = self.stack.read(
                    idx,
                    window=target_window,
                    out_shape=(out_size, out_size),
                    resampling=Resampling.bilinear,
                    boundless=True,
                    fill_value=0,
                )
            for name in SENTINEL_MISSING_ASSET_KEYS:
                bands[name] = self._read_remote_band(name, target_window, out_size)

            spectral = np.stack([bands[name] for name in SENTINEL_OLMO_BAND_ORDER], axis=-1)

        spectral = _sentinel_dn_to_reflectance(
            spectral,
            row.get("s2_processing_baseline"),
        )
        spectral = _fill_bad_pixels_with_band_median(spectral, bad_mask)
        return spectral.astype(np.float32, copy=False), self.mode


class _SentinelSceneCache:
    def __init__(self, max_size: int, timeout_seconds: int, ttl_seconds: int = 2700):
        self.max_size = max(1, max_size)
        self.timeout_seconds = timeout_seconds
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, _SentinelSceneBundle] = OrderedDict()

    def get(self, row: dict) -> _SentinelSceneBundle:
        key = row["item_id"]
        bundle = self._cache.pop(key, None)
        if bundle is not None:
            if time.monotonic() - bundle.created_monotonic <= self.ttl_seconds:
                self._cache[key] = bundle
                return bundle
            # Planetary Computer SAS URLs expire. Refresh long-lived cached scenes
            # before the signed URLs can age out during a multi-hour production run.
            bundle.close()

        bundle = _SentinelSceneBundle(row, self.timeout_seconds)
        self._cache[key] = bundle
        while len(self._cache) > self.max_size:
            _, evicted = self._cache.popitem(last=False)
            evicted.close()
        return bundle

    def close(self) -> None:
        for bundle in self._cache.values():
            bundle.close()
        self._cache.clear()


# -----------------------------------------------------------------------------
# OlmoEarth model
# -----------------------------------------------------------------------------


def _load_olmoearth(device: torch.device):
    try:
        from olmoearth_pretrain_minimal import ModelID, Normalizer, load_model_from_id
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import (
            MaskedOlmoEarthSample,
        )
    except ImportError as exc:
        raise RuntimeError(
            "OlmoEarth requires `olmoearth-pretrain-minimal`. Install with "
            "`pip install -U olmoearth-pretrain-minimal`."
        ) from exc

    full_model = load_model_from_id(ModelID.OLMOEARTH_V1_2_BASE, load_weights=True)
    # Only the encoder is used for durable features. Drop the MAE decoder / target
    # encoder wrapper so they do not occupy precious GPU memory on the 5090.
    encoder = full_model.encoder.eval().to(device)
    del full_model
    normalizer = Normalizer(std_multiplier=2.0)
    return encoder, normalizer, Modality, MaskedOlmoEarthSample


def _olmo_timestamps(rows: list[dict]) -> np.ndarray:
    # OlmoEarth timestamps are [day, month, year].
    result = np.zeros((len(rows), 3), dtype=np.int64)
    for i, row in enumerate(rows):
        dt = row["capture_timestamp_utc"]
        result[i] = [dt.day, dt.month, dt.year]
    return result


def _encode_olmo_batch(
    model,
    normalizer,
    Modality,
    MaskedOlmoEarthSample,
    sequences: list[np.ndarray],  # each [T,H,W,C]
    timestamp_sequences: list[np.ndarray],  # each [T,3]
    device: torch.device,
    precision: str,
) -> np.ndarray:
    if not sequences:
        return np.empty((0, 0), dtype=np.float32)

    t = sequences[0].shape[0]
    if any(x.shape[0] != t for x in sequences):
        raise ValueError("OlmoEarth batch must have a single common sequence length")

    # [B,T,H,W,C] -> [B,H,W,T,C]
    batch = np.stack(sequences, axis=0)
    batch = np.transpose(batch, (0, 2, 3, 1, 4)).astype(np.float32, copy=False)
    normalized = normalizer.normalize(Modality.SENTINEL2_L2A, batch)

    x = torch.from_numpy(np.asarray(normalized, dtype=np.float32)).to(
        device,
        non_blocking=True,
    )
    ts = torch.from_numpy(np.stack(timestamp_sequences, axis=0)).long().to(
        device,
        non_blocking=True,
    )

    # We fill SCL-rejected pixels before normalization, so there are no missing
    # tokens from the encoder's perspective and fast_pass=True remains valid.
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

    with torch.inference_mode(), _autocast_context(device, precision):
        out = model(sample, patch_size=8, input_res=10, fast_pass=True)
        tokens_and_masks = out["tokens_and_masks"]
        tokens = tokens_and_masks.sentinel2_l2a.float()
        # Shape is B x patchH x patchW x T x bandsets x D.
        dims = tuple(range(1, tokens.ndim - 1))
        z = tokens.mean(dim=dims)
        z = F.normalize(z, p=2, dim=1)

    return z.cpu().numpy().astype(np.float32, copy=False)


# -----------------------------------------------------------------------------
# Output row helpers
# -----------------------------------------------------------------------------


def _embedding_output_row(
    row: dict,
    embedding: np.ndarray,
    *,
    encoder_name: str,
    encoder_version: str,
    sequence_length: int,
    sentinel_input_mode: str | None,
) -> dict:
    return {
        "source": row["source"],
        "h3_cell": row["h3_cell"],
        "h3_resolution": row["h3_resolution"],
        "capture_period": row["capture_period"],
        "capture_timestamp_utc": row["capture_timestamp_utc"],
        "valid_from_utc": row["valid_from_utc"],
        "valid_to_utc": row["valid_to_utc"],
        "item_id": row["item_id"],
        "gcs_uri": row["gcs_uri"],
        "coverage_fraction": row["coverage_fraction"],
        "requires_mosaic": row["requires_mosaic"],
        "local_bad_fraction": row["local_bad_fraction"],
        "local_clear_fraction": row["local_clear_fraction"],
        "s2_processing_baseline": row["s2_processing_baseline"],
        "temporal_sequence_length": sequence_length,
        "sentinel_input_mode": sentinel_input_mode,
        "embedding": embedding.tolist(),
        "embedding_dim": int(embedding.shape[0]),
        "encoder_name": encoder_name,
        "encoder_version": encoder_version,
    }


def _flush_rows_to_shards(
    *,
    output_prefix: str,
    source: str,
    shard_rows: list[dict],
    next_shard_idx: int,
    rows_per_shard: int,
    force: bool,
    context: dg.AssetExecutionContext,
) -> tuple[list[dict], int, int]:
    written = 0
    while len(shard_rows) >= rows_per_shard or (force and shard_rows):
        take = rows_per_shard if len(shard_rows) >= rows_per_shard else len(shard_rows)
        chunk = shard_rows[:take]
        shard_rows = shard_rows[take:]
        uri = _join_gs(output_prefix, f"{source}/part-{next_shard_idx:05d}.parquet")
        _write_parquet_shard(uri, chunk)
        context.log.info(
            f"Wrote {source} embedding shard {next_shard_idx:05d}: "
            f"{len(chunk):,} rows -> {uri}"
        )
        next_shard_idx += 1
        written += len(chunk)
    return shard_rows, next_shard_idx, written


# -----------------------------------------------------------------------------
# Source encoders
# -----------------------------------------------------------------------------


def _encode_naip(
    *,
    df: pl.DataFrame,
    config: ImageryEmbeddingConfig,
    device: torch.device,
    context: dg.AssetExecutionContext,
    resume_rows: int = 0,
    start_shard_idx: int = 0,
) -> tuple[int, int, int]:
    if df.is_empty():
        return 0, 0, 0
    if resume_rows == df.height:
        context.log.info("All NAIP rows already exist in completed shards; skipping DINOv3.")
        return resume_rows, start_shard_idx, start_shard_idx

    processor, model = _load_dino(device)
    output_prefix = config.output_prefix

    batch_images: list[np.ndarray] = []
    batch_rows: list[dict] = []
    if resume_rows < 0 or resume_rows > df.height:
        raise ValueError(f"Invalid NAIP resume_rows={resume_rows} for {df.height} rows")

    shard_rows: list[dict] = []
    shard_idx = start_shard_idx
    encoded = resume_rows
    skipped_shards = start_shard_idx

    work_df = df.slice(resume_rows)
    if resume_rows:
        context.log.info(
            f"Resuming NAIP after {resume_rows:,} completed rows / {start_shard_idx:,} shards"
        )

    current_uri = None
    src = None

    def flush_batch() -> None:
        nonlocal batch_images, batch_rows, shard_rows, encoded
        if not batch_images:
            return
        embeddings = _encode_dino_batch(
            processor,
            model,
            batch_images,
            device,
            config.precision,
            config.naip_image_size,
        )
        for row, z in zip(batch_rows, embeddings, strict=True):
            shard_rows.append(
                _embedding_output_row(
                    row,
                    z,
                    encoder_name="dinov3-vitl16-sat493m",
                    encoder_version=DINO_MODEL_ID,
                    sequence_length=1,
                    sentinel_input_mode=None,
                )
            )
        encoded += len(batch_rows)
        batch_images = []
        batch_rows = []

    try:
        with _raster_env():
            for row in work_df.iter_rows(named=True):
                uri = row["gcs_uri"]
                if uri != current_uri:
                    if src is not None:
                        src.close()
                    src = rasterio.open(gcs_uri_to_vsigs(uri), sharing=False)
                    current_uri = uri

                batch_images.append(_naip_rgb_uint8(src, row, config.naip_image_size))
                batch_rows.append(row)

                if len(batch_images) >= config.naip_batch_size:
                    flush_batch()

                if len(shard_rows) >= config.rows_per_shard:
                    shard_rows, shard_idx, _ = _flush_rows_to_shards(
                        output_prefix=output_prefix,
                        source="naip",
                        shard_rows=shard_rows,
                        next_shard_idx=shard_idx,
                        rows_per_shard=config.rows_per_shard,
                        force=False,
                        context=context,
                    )

                if encoded and encoded % 5000 < config.naip_batch_size:
                    context.log.info(f"DINOv3 NAIP encoded: {encoded:,}/{df.height:,}")

            flush_batch()
            shard_rows, shard_idx, _ = _flush_rows_to_shards(
                output_prefix=output_prefix,
                source="naip",
                shard_rows=shard_rows,
                next_shard_idx=shard_idx,
                rows_per_shard=config.rows_per_shard,
                force=True,
                context=context,
            )
    finally:
        if src is not None:
            src.close()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return encoded, shard_idx, skipped_shards


def _iter_h3_groups(df: pl.DataFrame):
    current_h3 = None
    rows: list[dict] = []
    for row in df.iter_rows(named=True):
        h3_cell = row["h3_cell"]
        if current_h3 is None:
            current_h3 = h3_cell
        if h3_cell != current_h3:
            yield current_h3, rows
            current_h3 = h3_cell
            rows = []
        rows.append(row)
    if rows:
        yield current_h3, rows


def _encode_sentinel(
    *,
    df: pl.DataFrame,
    config: ImageryEmbeddingConfig,
    device: torch.device,
    context: dg.AssetExecutionContext,
    resume_rows: int = 0,
    start_shard_idx: int = 0,
) -> tuple[int, int, dict[str, int]]:
    if df.is_empty():
        return 0, 0, {"patched13": 0, "lazy7": 0}
    if resume_rows == df.height:
        context.log.info("All Sentinel rows already exist in completed shards; skipping OlmoEarth.")
        return resume_rows, start_shard_idx, {"patched13": 0, "lazy7": 0}

    model, normalizer, Modality, MaskedOlmoEarthSample = _load_olmoearth(device)
    cache = _SentinelSceneCache(
        max_size=config.sentinel_scene_cache_size,
        timeout_seconds=config.stac_timeout_seconds,
    )

    # Bucket samples by T because OlmoEarth batches require a common sequence length.
    buckets: dict[int, list[tuple[dict, np.ndarray, np.ndarray, str]]] = {
        t: [] for t in range(1, config.sentinel_max_timesteps + 1)
    }

    if resume_rows < 0 or resume_rows > df.height:
        raise ValueError(f"Invalid Sentinel resume_rows={resume_rows} for {df.height} rows")

    shard_rows: list[dict] = []
    shard_idx = start_shard_idx
    encoded = resume_rows
    input_mode_counts = {"patched13": 0, "lazy7": 0}

    if resume_rows:
        context.log.info(
            f"Resuming Sentinel after {resume_rows:,} completed rows / "
            f"{start_shard_idx:,} shards. Entire completed H3 groups are skipped; "
            "only a boundary H3 is replayed to rebuild temporal history."
        )

    def flush_bucket(t: int, force: bool = False) -> None:
        nonlocal shard_rows, encoded, shard_idx
        bucket = buckets[t]
        if not bucket:
            return
        if not force and len(bucket) < config.sentinel_batch_size:
            return

        while len(bucket) >= config.sentinel_batch_size or (force and bucket):
            take = min(config.sentinel_batch_size, len(bucket))
            chunk = bucket[:take]
            del bucket[:take]

            rows = [x[0] for x in chunk]
            seqs = [x[1] for x in chunk]
            timestamps = [x[2] for x in chunk]
            modes = [x[3] for x in chunk]

            embeddings = _encode_olmo_batch(
                model,
                normalizer,
                Modality,
                MaskedOlmoEarthSample,
                seqs,
                timestamps,
                device,
                config.precision,
            )

            for row, z, mode in zip(rows, embeddings, modes, strict=True):
                shard_rows.append(
                    _embedding_output_row(
                        row,
                        z,
                        encoder_name="olmoearth-v1.2-base",
                        encoder_version=OLMOEARTH_MODEL_ID,
                        sequence_length=t,
                        sentinel_input_mode=mode,
                    )
                )
                input_mode_counts[mode] += 1

            encoded += len(chunk)

            if len(shard_rows) >= config.rows_per_shard:
                shard_rows, shard_idx, _ = _flush_rows_to_shards(
                    output_prefix=config.output_prefix,
                    source="sentinel2",
                    shard_rows=shard_rows,
                    next_shard_idx=shard_idx,
                    rows_per_shard=config.rows_per_shard,
                    force=False,
                    context=context,
                )

    try:
        with _raster_env():
            h3_count = 0
            global_row_idx = 0
            for h3_cell, rows in _iter_h3_groups(df):
                h3_count += 1
                group_start = global_row_idx
                group_end = group_start + len(rows)
                global_row_idx = group_end

                # Sentinel temporal history never crosses H3 boundaries. Therefore a
                # fully completed H3 group can be skipped without any I/O on resume.
                if group_end <= resume_rows:
                    continue

                history: deque[tuple[dict, np.ndarray, str]] = deque()

                for local_idx, row in enumerate(rows):
                    absolute_idx = group_start + local_idx
                    bundle = cache.get(row)
                    frame, mode = bundle.read_olmo_frame(row, config.sentinel_image_size)

                    cutoff = row["capture_timestamp_utc"] - timedelta(
                        days=config.sentinel_lookback_days
                    )
                    while history and history[0][0]["capture_timestamp_utc"] < cutoff:
                        history.popleft()

                    history.append((row, frame, mode))
                    while len(history) > config.sentinel_max_timesteps:
                        history.popleft()

                    # If resume lands inside this H3, replay preceding frames solely
                    # to reconstruct the exact trailing temporal context; don't encode
                    # outputs that already exist in completed Parquet shards.
                    if absolute_idx < resume_rows:
                        continue

                    seq_rows = [x[0] for x in history]
                    seq = np.stack([x[1] for x in history], axis=0)
                    timestamps = _olmo_timestamps(seq_rows)
                    t = len(history)

                    # The target row's source mode is useful for auditing. A temporal
                    # sequence can mix patched and lazy source scenes; mark that case.
                    sequence_modes = {x[2] for x in history}
                    sequence_mode = (
                        next(iter(sequence_modes))
                        if len(sequence_modes) == 1
                        else "mixed"
                    )
                    if sequence_mode == "mixed":
                        input_mode_counts.setdefault("mixed", 0)

                    buckets[t].append((row, seq, timestamps, sequence_mode))
                    flush_bucket(t, force=False)

                if h3_count % 250 == 0:
                    context.log.info(
                        f"OlmoEarth progress: H3={h3_count:,}, encoded={encoded:,}/{df.height:,}, "
                        f"scene_cache={len(cache._cache)}"
                    )

            for t in range(1, config.sentinel_max_timesteps + 1):
                flush_bucket(t, force=True)

            shard_rows, shard_idx, _ = _flush_rows_to_shards(
                output_prefix=config.output_prefix,
                source="sentinel2",
                shard_rows=shard_rows,
                next_shard_idx=shard_idx,
                rows_per_shard=config.rows_per_shard,
                force=True,
                context=context,
            )
    finally:
        cache.close()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return encoded, shard_idx, input_mode_counts


# -----------------------------------------------------------------------------
# Dagster asset
# -----------------------------------------------------------------------------


@dg.asset(
    deps=[silver_imagery_h3_temporal_index],
    group_name="imagery",
    compute_kind="pytorch/rasterio",
    description=(
        "Produce durable H3/time foundation-model imagery embeddings. NAIP uses "
        "DINOv3 ViT-L/16 SAT-493M. Sentinel uses OlmoEarth v1.2 Base over trailing "
        "leakage-safe monthly sequences. Sentinel input is hybrid: already-patched "
        "13-band CrimeNet COGs are used directly; original 7-band COGs lazily range-read "
        "only the six missing Planetary Computer bands for each selected H3 crop."
    ),
)
def gold_imagery_embeddings(
    context: dg.AssetExecutionContext,
    config: ImageryEmbeddingConfig,
) -> dg.MaterializeResult:
    _ensure_gdal_gcs_credentials()

    if config.naip_batch_size <= 0 or config.sentinel_batch_size <= 0:
        raise dg.Failure("Batch sizes must be > 0")
    if config.rows_per_shard <= 0:
        raise dg.Failure("rows_per_shard must be > 0")
    if config.naip_image_size <= 0 or config.sentinel_image_size <= 0:
        raise dg.Failure("Image sizes must be > 0")
    if config.sentinel_max_timesteps < 1 or config.sentinel_max_timesteps > 12:
        raise dg.Failure("sentinel_max_timesteps must be in [1, 12]")
    if config.sentinel_lookback_days <= 0:
        raise dg.Failure("sentinel_lookback_days must be > 0")

    context.log.info(f"Reading temporal imagery index: {SILVER_H3_TEMPORAL_INDEX_URI}")
    temporal = _read_temporal_index(SILVER_H3_TEMPORAL_INDEX_URI)
    temporal = (
        temporal
        .filter(pl.col("selected_in_period") == True)
        .filter(pl.col("is_usable") == True)
        .filter(pl.col("error").is_null())
        .filter(pl.col("source").is_in(["naip", "sentinel2"]))
    )

    if temporal.is_empty():
        raise dg.Failure("No selected usable imagery rows were found.")

    duplicate_keys = (
        temporal
        .group_by(["source", "h3_cell", "valid_from_utc"])
        .len()
        .filter(pl.col("len") > 1)
    )
    if duplicate_keys.height:
        raise dg.Failure(
            f"Temporal index has {duplicate_keys.height:,} duplicate "
            "(source,h3_cell,valid_from_utc) keys."
        )

    naip = (
        temporal
        .filter(pl.col("source") == "naip")
        .sort(["gcs_uri", "h3_cell", "capture_timestamp_utc", "item_id"])
    )
    sentinel = (
        temporal
        .filter(pl.col("source") == "sentinel2")
        .sort(["h3_cell", "capture_timestamp_utc", "item_id"])
    )

    if config.limit_naip_rows > 0:
        naip = naip.head(config.limit_naip_rows)
        context.log.warning(f"NAIP smoke limit active: {naip.height:,} rows")

    if config.limit_sentinel_h3_cells > 0 and not sentinel.is_empty():
        keep_h3 = (
            sentinel.select("h3_cell")
            .unique(maintain_order=True)
            .head(config.limit_sentinel_h3_cells)
            .get_column("h3_cell")
        )
        sentinel = sentinel.filter(pl.col("h3_cell").is_in(keep_h3))
        context.log.warning(
            f"Sentinel smoke limit active: {len(keep_h3):,} H3 cells / {sentinel.height:,} rows"
        )

    if config.overwrite:
        context.log.warning(f"Deleting existing output prefix: {config.output_prefix}")
        _delete_prefix(config.output_prefix)
    elif _gcs_exists(_join_gs(config.output_prefix, "_SUCCESS.json")):
        if config.resume:
            context.log.info("_SUCCESS already exists; returning without recomputing.")
            return dg.MaterializeResult(
                metadata={
                    "output_prefix": dg.MetadataValue.path(config.output_prefix),
                    "status": "already_complete",
                }
            )
        raise dg.Failure(
            f"Output is already complete: {config.output_prefix}. "
            "Set overwrite=true to rebuild."
        )

    naip_existing_shards = 0
    sentinel_existing_shards = 0
    if config.resume and not config.overwrite:
        naip_existing_shards = _existing_shard_count(config.output_prefix, "naip")
        sentinel_existing_shards = _existing_shard_count(config.output_prefix, "sentinel2")
    elif not config.overwrite:
        # Refuse to accidentally overwrite partial deterministic shards when resume
        # was explicitly disabled.
        if (
            _existing_shard_count(config.output_prefix, "naip")
            or _existing_shard_count(config.output_prefix, "sentinel2")
        ):
            raise dg.Failure(
                "Partial embedding shards exist and resume=false. "
                "Use resume=true or overwrite=true."
            )

    naip_max_shards = (naip.height + config.rows_per_shard - 1) // config.rows_per_shard
    sentinel_max_shards = (sentinel.height + config.rows_per_shard - 1) // config.rows_per_shard
    if naip_existing_shards > naip_max_shards or sentinel_existing_shards > sentinel_max_shards:
        raise dg.Failure(
            "Existing shard count is incompatible with the current temporal index/config. "
            "Use a fresh output_prefix or overwrite=true."
        )

    # min(...) correctly handles the rare case where the final partial shard was
    # uploaded but the process died before writing _SUCCESS.json.
    naip_resume_rows = min(
        naip_existing_shards * config.rows_per_shard,
        naip.height,
    )
    sentinel_resume_rows = min(
        sentinel_existing_shards * config.rows_per_shard,
        sentinel.height,
    )

    device = _resolve_device(config.device)
    context.log.info(
        f"Foundation embedding run: device={device}, precision={config.precision}, "
        f"NAIP={naip.height:,}, Sentinel={sentinel.height:,}"
    )

    # Record the intended recipe before expensive work starts.
    recipe = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_temporal_index_uri": SILVER_H3_TEMPORAL_INDEX_URI,
        "naip": {
            "model": DINO_MODEL_ID,
            "image_size": config.naip_image_size,
            "pooling": "L2-normalize(concat(CLS, mean(patch_tokens)))",
        },
        "sentinel2": {
            "model": OLMOEARTH_MODEL_ID,
            "band_order": SENTINEL_OLMO_BAND_ORDER,
            "image_size": config.sentinel_image_size,
            "input_res_m": 10,
            "patch_size": 8,
            "max_timesteps": config.sentinel_max_timesteps,
            "lookback_days": config.sentinel_lookback_days,
            "cloud_handling": "SCL BAD_CLASSES -> per-band median fill before OlmoEarth normalization",
            "hybrid_input": {
                "patched13_indexes": SENTINEL_PATCHED_OLMO_INDEXES,
                "lazy7_missing_assets": SENTINEL_MISSING_ASSET_KEYS,
            },
            "radiometry": "PB>=04.00: (DN-1000)/10000; earlier: DN/10000; DN=0 nodata",
            "pooling": "mean all Sentinel token dimensions except batch/embedding; L2 normalize",
        },
        "precision": config.precision,
    }
    recipe_uri = _join_gs(config.output_prefix, "_RECIPE.json")
    existing_recipe = _read_json_if_exists(recipe_uri)
    if existing_recipe is not None and (naip_existing_shards or sentinel_existing_shards):
        comparable_keys = ["source_temporal_index_uri", "naip", "sentinel2", "precision"]
        if any(existing_recipe.get(k) != recipe.get(k) for k in comparable_keys):
            raise dg.Failure(
                "Existing partial shards were produced by a different embedding recipe. "
                "Use overwrite=true or a new output_prefix."
            )
    else:
        _upload_bytes(
            recipe_uri,
            json.dumps(recipe, indent=2).encode("utf-8"),
            "application/json",
        )

    context.log.info("Loading/encoding NAIP with DINOv3-SAT")
    naip_encoded, naip_shards, _ = _encode_naip(
        df=naip,
        config=config,
        device=device,
        context=context,
        resume_rows=naip_resume_rows,
        start_shard_idx=naip_existing_shards,
    )

    context.log.info("Loading/encoding Sentinel with OlmoEarth v1.2 Base")
    sentinel_encoded, sentinel_shards, input_mode_counts = _encode_sentinel(
        df=sentinel,
        config=config,
        device=device,
        context=context,
        resume_rows=sentinel_resume_rows,
        start_shard_idx=sentinel_existing_shards,
    )

    if naip_encoded != naip.height:
        raise dg.Failure(
            f"NAIP row-count invariant failed: encoded={naip_encoded:,}, expected={naip.height:,}"
        )
    if sentinel_encoded != sentinel.height:
        raise dg.Failure(
            f"Sentinel row-count invariant failed: encoded={sentinel_encoded:,}, expected={sentinel.height:,}"
        )

    success = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dagster_run_id": context.run_id,
        "output_prefix": config.output_prefix,
        "source_temporal_index_uri": SILVER_H3_TEMPORAL_INDEX_URI,
        "row_counts": {
            "naip": naip_encoded,
            "sentinel2": sentinel_encoded,
        },
        "shard_counts": {
            "naip": naip_shards,
            "sentinel2": sentinel_shards,
        },
        "sentinel_input_modes_newly_encoded": input_mode_counts,
        "resume": {
            "naip_existing_shards": naip_existing_shards,
            "sentinel_existing_shards": sentinel_existing_shards,
            "naip_resume_rows": naip_resume_rows,
            "sentinel_resume_rows": sentinel_resume_rows,
        },
        "models": {
            "naip": DINO_MODEL_ID,
            "sentinel2": OLMOEARTH_MODEL_ID,
        },
    }
    success_uri = _join_gs(config.output_prefix, "_SUCCESS.json")
    _upload_bytes(
        success_uri,
        json.dumps(success, indent=2).encode("utf-8"),
        "application/json",
    )

    return dg.MaterializeResult(
        metadata={
            "output_prefix": dg.MetadataValue.path(config.output_prefix),
            "success_manifest": dg.MetadataValue.path(success_uri),
            "naip_rows": naip_encoded,
            "sentinel_rows": sentinel_encoded,
            "naip_shards": naip_shards,
            "sentinel_shards": sentinel_shards,
            "sentinel_patched13_rows": int(input_mode_counts.get("patched13", 0)),
            "sentinel_lazy7_rows": int(input_mode_counts.get("lazy7", 0)),
            "sentinel_mixed_rows": int(input_mode_counts.get("mixed", 0)),
            "device": str(device),
        }
    )
