import json
import os
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
import tempfile

import dagster as dg
from google.cloud import storage
import numpy as np
import polars as pl
import rasterio
from rasterio.windows import Window
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from .silver import (
    SILVER_H3_TEMPORAL_INDEX_URI,
    silver_imagery_h3_temporal_index,
)
from .transformations import BAD_CLASSES, gcs_uri_to_vsigs


GOLD_IMAGERY_EMBEDDINGS_PREFIX = "gs://crimenet/gold/imagery/embeddings"
ENCODER_NAME = "resnet18"
ENCODER_VERSION = "imagenet1k_v1_crimenet_imagery_recipe_v1"
EMBEDDING_DIM = 512

# Current Sentinel stack layout produced by the ingestion job:
#   1 B02, 2 B03, 3 B04, 4 B08, 5 B11, 6 B12, 7 SCL
SENTINEL_SPECTRAL_BANDS = [1, 2, 3, 4, 5, 6]
SENTINEL_SCL_BAND = 7

# ImageNet normalization for the pretrained RGB backbone.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

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
    batch_size: int = 64
    image_size: int = 224
    rows_per_shard: int = 5000
    device: str = "auto"  # auto | cpu | cuda | mps
    overwrite: bool = True
    limit_rows: int = 0  # 0 = all rows; useful for smoke tests

    # Sentinel L2A reflectance conversion. This assumes the stacked bands are
    # on a comparable DN scale. If the ingestion path is not harmonized across
    # processing baselines, harmonize that upstream before relying on subtle
    # temporal spectral differences.
    sentinel_dn_scale: float = 10000.0
    sentinel_reflectance_clip: float = 0.50


class ResNet18FeatureEncoder(nn.Module):
    """Stable 512-D feature extractor from an ImageNet-pretrained ResNet-18."""

    def __init__(self) -> None:
        super().__init__()
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        return torch.flatten(z, 1)


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


def _read_temporal_index(uri: str) -> pl.DataFrame:
    return pl.read_parquet(
        uri,
        columns=TEMPORAL_COLUMNS,
        credential_provider=pl.CredentialProviderGCP(),
    )


def _upload_bytes(uri: str, payload: bytes, content_type: str) -> None:
    bucket_name, blob_name = _parse_gs_uri(uri)
    storage.Client().bucket(bucket_name).blob(blob_name).upload_from_string(
        payload,
        content_type=content_type,
    )


def _write_parquet_shard(uri: str, rows: list[dict]) -> None:
    if not rows:
        return

    # Explicitly force embeddings to Float32 rather than Python's default Float64.
    df = pl.DataFrame(rows).with_columns(
        pl.col("embedding").cast(pl.List(pl.Float32)),
        pl.col("embedding_dim").cast(pl.Int32),
        pl.col("h3_resolution").cast(pl.Int8),
    )

    bucket_name, blob_name = _parse_gs_uri(uri)
    client = storage.Client()

    with tempfile.TemporaryDirectory(prefix="crimenet_imagery_embedding_") as tmpdir:
        local = Path(tmpdir) / "part.parquet"
        df.write_parquet(
            local,
            compression="zstd",
            compression_level=6,
            statistics=True,
        )
        client.bucket(bucket_name).blob(blob_name).upload_from_filename(
            str(local),
            content_type="application/octet-stream",
        )


def _existing_output_objects(prefix: str) -> list[str]:
    bucket_name, blob_prefix = _parse_gs_uri(prefix.rstrip("/") + "/_probe")
    # Remove the synthetic suffix used only to make _parse_gs_uri happy.
    blob_prefix = blob_prefix.rsplit("/", 1)[0] + "/"
    client = storage.Client()
    return [blob.name for blob in client.list_blobs(bucket_name, prefix=blob_prefix)]


def _prepare_output_prefix(prefix: str, overwrite: bool) -> None:
    bucket_name, probe = _parse_gs_uri(prefix.rstrip("/") + "/_probe")
    blob_prefix = probe.rsplit("/", 1)[0] + "/"
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    existing = list(client.list_blobs(bucket_name, prefix=blob_prefix))

    if existing and not overwrite:
        raise dg.Failure(
            f"Gold imagery embedding prefix already contains {len(existing):,} objects: "
            f"{prefix}. Set overwrite=true or use a new output prefix."
        )

    if overwrite:
        for blob in existing:
            blob.delete()


# -----------------------------------------------------------------------------
# Raster / device helpers
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

    raise ValueError(f"Unsupported device {requested!r}; use auto/cpu/cuda/mps")


def _raster_env() -> rasterio.Env:
    # COG-friendly settings. The job performs many small window reads against
    # private GCS objects, so avoid directory scans and retry transient HTTP errors.
    return rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_MAX_RETRY="5",
        GDAL_HTTP_RETRY_DELAY="1",
        VSI_CACHE="TRUE",
        VSI_CACHE_SIZE="67108864",
    )


def _window_from_row(row: dict) -> Window:
    return Window(
        col_off=int(row["window_col_off"]),
        row_off=int(row["window_row_off"]),
        width=int(row["window_width"]),
        height=int(row["window_height"]),
    )


# -----------------------------------------------------------------------------
# Source-specific preprocessing
# -----------------------------------------------------------------------------


def _resize_and_normalize_imagenet(x: torch.Tensor, image_size: int) -> torch.Tensor:
    """x: [3,H,W] in [0,1] -> normalized [3,image_size,image_size]."""
    if x.ndim != 3 or x.shape[0] != 3:
        raise ValueError(f"Expected [3,H,W], got {tuple(x.shape)}")

    x = F.interpolate(
        x.unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)

    return (x - IMAGENET_MEAN) / IMAGENET_STD


def _naip_view(
    src: rasterio.io.DatasetReader,
    row: dict,
    image_size: int,
) -> torch.Tensor:
    if src.count < 3:
        raise ValueError(f"NAIP source has {src.count} bands; expected >=3")

    arr = src.read([1, 2, 3], window=_window_from_row(row))
    if arr.size == 0:
        raise ValueError("NAIP crop is empty")

    original_dtype = arr.dtype
    x = arr.astype(np.float32, copy=False)

    if np.issubdtype(original_dtype, np.integer):
        max_value = float(np.iinfo(original_dtype).max)
        if max_value <= 0:
            raise ValueError(f"Invalid NAIP integer dtype: {original_dtype}")
        x /= max_value
    else:
        finite = x[np.isfinite(x)]
        if finite.size == 0:
            raise ValueError("NAIP crop contains no finite pixels")
        # Float imagery may already be [0,1], or may be byte-like values.
        if float(np.nanmax(finite)) > 1.5:
            x /= 255.0

    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    x = np.clip(x, 0.0, 1.0)

    return _resize_and_normalize_imagenet(torch.from_numpy(x), image_size)


def _fill_masked_with_band_median(
    spectral: np.ndarray,
    bad_mask: np.ndarray,
) -> np.ndarray:
    out = spectral.copy()
    good_mask = ~bad_mask

    if not np.any(good_mask):
        raise ValueError("Sentinel crop contains no SCL-valid pixels")

    for band_idx in range(out.shape[0]):
        valid = out[band_idx][good_mask]
        valid = valid[np.isfinite(valid)]
        if valid.size == 0:
            fill = 0.0
        else:
            fill = float(np.median(valid))
        out[band_idx][bad_mask] = fill

    return out


def _sentinel_views(
    src: rasterio.io.DatasetReader,
    row: dict,
    image_size: int,
    dn_scale: float,
    reflectance_clip: float,
) -> torch.Tensor:
    """
    Return two 3-channel views [2,3,H,W]:
      view 0 = true color      [B04, B03, B02]
      view 1 = multispectral   [B08, B11, B12]

    SCL is used only for masking; it is never fed into the encoder.
    """
    if src.count < SENTINEL_SCL_BAND:
        raise ValueError(f"Sentinel source has {src.count} bands; expected >=7")
    if dn_scale <= 0:
        raise ValueError("sentinel_dn_scale must be > 0")
    if reflectance_clip <= 0:
        raise ValueError("sentinel_reflectance_clip must be > 0")

    window = _window_from_row(row)
    spectral = src.read(SENTINEL_SPECTRAL_BANDS, window=window).astype(np.float32)
    scl = src.read(SENTINEL_SCL_BAND, window=window, out_dtype="uint8")

    if spectral.size == 0 or scl.size == 0:
        raise ValueError("Sentinel crop is empty")

    bad_mask = np.isin(scl, tuple(BAD_CLASSES))
    spectral = _fill_masked_with_band_median(spectral, bad_mask)

    # Convert source DN to approximate surface reflectance, then map a fixed
    # reflectance interval to [0,1]. Fixed scaling preserves temporal magnitude
    # information better than independently percentile-normalizing every crop.
    spectral = spectral / float(dn_scale)
    spectral = np.nan_to_num(spectral, nan=0.0, posinf=reflectance_clip, neginf=0.0)
    spectral = np.clip(spectral, 0.0, reflectance_clip) / float(reflectance_clip)

    # Stack order is B02,B03,B04,B08,B11,B12.
    rgb = spectral[[2, 1, 0], :, :]
    nir_swir = spectral[[3, 4, 5], :, :]

    rgb_t = _resize_and_normalize_imagenet(torch.from_numpy(rgb), image_size)
    nir_swir_t = _resize_and_normalize_imagenet(
        torch.from_numpy(nir_swir), image_size
    )

    return torch.stack([rgb_t, nir_swir_t], dim=0)


# -----------------------------------------------------------------------------
# Batched encoding
# -----------------------------------------------------------------------------


def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _encode_naip_batch(
    model: nn.Module,
    batch: list[torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    x = torch.stack(batch, dim=0).to(device, non_blocking=True)
    with torch.inference_mode(), _autocast_context(device):
        z = model(x)
        z = F.normalize(z.float(), p=2, dim=1)
    return z.cpu().numpy().astype(np.float32, copy=False)


def _encode_sentinel_batch(
    model: nn.Module,
    batch: list[torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    # [N,2,3,H,W] -> [2N,3,H,W]
    x = torch.stack(batch, dim=0)
    n = x.shape[0]
    x = x.reshape(n * 2, *x.shape[2:]).to(device, non_blocking=True)

    with torch.inference_mode(), _autocast_context(device):
        z = model(x).float()
        z = F.normalize(z, p=2, dim=1)
        z = z.reshape(n, 2, -1).mean(dim=1)
        z = F.normalize(z, p=2, dim=1)

    return z.cpu().numpy().astype(np.float32, copy=False)


def _embedding_output_row(row: dict, embedding: np.ndarray) -> dict:
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
        "embedding": embedding.tolist(),
        "embedding_dim": EMBEDDING_DIM,
        "encoder_name": ENCODER_NAME,
        "encoder_version": ENCODER_VERSION,
    }


def _flush_embedding_batch(
    *,
    source: str,
    model: nn.Module,
    tensors: list[torch.Tensor],
    metadata_rows: list[dict],
    device: torch.device,
) -> list[dict]:
    if not tensors:
        return []

    if source == "naip":
        embeddings = _encode_naip_batch(model, tensors, device)
    elif source == "sentinel2":
        embeddings = _encode_sentinel_batch(model, tensors, device)
    else:
        raise ValueError(f"Unsupported source: {source}")

    if embeddings.shape != (len(metadata_rows), EMBEDDING_DIM):
        raise RuntimeError(
            f"Unexpected embedding shape {embeddings.shape}; "
            f"expected ({len(metadata_rows)}, {EMBEDDING_DIM})"
        )

    return [
        _embedding_output_row(row, embedding)
        for row, embedding in zip(metadata_rows, embeddings, strict=True)
    ]


# -----------------------------------------------------------------------------
# Dagster asset
# -----------------------------------------------------------------------------


@dg.asset(
    deps=[silver_imagery_h3_temporal_index],
    group_name="imagery",
    compute_kind="pytorch/rasterio",
    description=(
        "Encode each selected H3/time imagery observation once, before the final "
        "model-table as-of join. NAIP uses RGB; Sentinel uses SCL-masked RGB and "
        "NIR/SWIR views whose 512-D embeddings are averaged and L2-normalized."
    ),
)
def gold_imagery_embeddings(
    context: dg.AssetExecutionContext,
    config: ImageryEmbeddingConfig,
) -> dg.MaterializeResult:
    _ensure_gdal_gcs_credentials()

    if config.batch_size <= 0:
        raise dg.Failure("batch_size must be > 0")
    if config.rows_per_shard <= 0:
        raise dg.Failure("rows_per_shard must be > 0")
    if config.image_size <= 0:
        raise dg.Failure("image_size must be > 0")

    context.log.info(f"Reading temporal imagery index: {SILVER_H3_TEMPORAL_INDEX_URI}")
    temporal = _read_temporal_index(SILVER_H3_TEMPORAL_INDEX_URI)

    temporal = (
        temporal
        .filter(pl.col("selected_in_period") == True)
        .filter(pl.col("is_usable") == True)
        .filter(pl.col("error").is_null())
        .filter(pl.col("source").is_in(["naip", "sentinel2"]))
        .sort(
            ["source", "gcs_uri", "h3_cell", "capture_timestamp_utc", "item_id"]
        )
    )

    if config.limit_rows > 0:
        temporal = temporal.head(config.limit_rows)
        context.log.warning(
            f"Smoke-test limit active: encoding only {temporal.height:,} rows"
        )

    if temporal.is_empty():
        raise dg.Failure("No usable selected imagery rows were found to encode.")

    duplicate_keys = (
        temporal
        .group_by(["source", "h3_cell", "valid_from_utc"])
        .len()
        .filter(pl.col("len") > 1)
    )
    if duplicate_keys.height:
        raise dg.Failure(
            f"Temporal index contains {duplicate_keys.height:,} duplicate "
            "(source, h3_cell, valid_from_utc) keys."
        )

    _prepare_output_prefix(GOLD_IMAGERY_EMBEDDINGS_PREFIX, config.overwrite)

    device = _resolve_device(config.device)
    context.log.info(
        f"Loading {ENCODER_NAME}/{ENCODER_VERSION} on device={device}; "
        f"embedding_dim={EMBEDDING_DIM}, image_size={config.image_size}, "
        f"batch_size={config.batch_size}"
    )

    model = ResNet18FeatureEncoder().eval().to(device)

    source_counts = {
        "naip": temporal.filter(pl.col("source") == "naip").height,
        "sentinel2": temporal.filter(pl.col("source") == "sentinel2").height,
    }

    total_encoded = 0
    total_shards = 0
    source_shards = {"naip": 0, "sentinel2": 0}
    source_encoded = {"naip": 0, "sentinel2": 0}

    # Process source types separately so every GPU batch has the same tensor shape:
    # NAIP => [N,3,H,W], Sentinel => [N,2,3,H,W].
    for source in ["naip", "sentinel2"]:
        source_df = temporal.filter(pl.col("source") == source)
        if source_df.is_empty():
            continue

        context.log.info(f"Encoding {source_counts[source]:,} {source} rows")

        batch_tensors: list[torch.Tensor] = []
        batch_metadata: list[dict] = []
        shard_rows: list[dict] = []
        current_uri: str | None = None
        src: rasterio.io.DatasetReader | None = None

        def flush_batch() -> None:
            nonlocal batch_tensors, batch_metadata, shard_rows
            encoded = _flush_embedding_batch(
                source=source,
                model=model,
                tensors=batch_tensors,
                metadata_rows=batch_metadata,
                device=device,
            )
            shard_rows.extend(encoded)
            batch_tensors = []
            batch_metadata = []

        def flush_shard(force: bool = False) -> None:
            nonlocal shard_rows, total_shards
            if not shard_rows:
                return
            if not force and len(shard_rows) < config.rows_per_shard:
                return

            while len(shard_rows) >= config.rows_per_shard or (force and shard_rows):
                take = (
                    config.rows_per_shard
                    if len(shard_rows) >= config.rows_per_shard
                    else len(shard_rows)
                )
                chunk = shard_rows[:take]
                shard_rows = shard_rows[take:]

                shard_idx = source_shards[source]
                shard_uri = _join_gs(
                    GOLD_IMAGERY_EMBEDDINGS_PREFIX,
                    f"{source}/part-{shard_idx:05d}.parquet",
                )
                _write_parquet_shard(shard_uri, chunk)
                source_shards[source] += 1
                total_shards += 1
                context.log.info(
                    f"Wrote {source} embedding shard {shard_idx:05d}: "
                    f"{len(chunk):,} rows -> {shard_uri}"
                )

        try:
            with _raster_env():
                for row in source_df.iter_rows(named=True):
                    uri = row["gcs_uri"]
                    if uri != current_uri:
                        if src is not None:
                            src.close()
                        src = rasterio.open(gcs_uri_to_vsigs(uri), sharing=False)
                        current_uri = uri

                    try:
                        if source == "naip":
                            tensor = _naip_view(src, row, config.image_size)
                        else:
                            tensor = _sentinel_views(
                                src,
                                row,
                                config.image_size,
                                config.sentinel_dn_scale,
                                config.sentinel_reflectance_clip,
                            )
                    except Exception as exc:
                        raise RuntimeError(
                            f"Failed to preprocess {source} embedding input: "
                            f"item_id={row['item_id']}, h3={row['h3_cell']}, "
                            f"capture={row['capture_timestamp_utc']}, uri={uri}: {exc!r}"
                        ) from exc

                    batch_tensors.append(tensor)
                    batch_metadata.append(row)

                    if len(batch_tensors) >= config.batch_size:
                        flush_batch()
                        source_encoded[source] += config.batch_size
                        total_encoded += config.batch_size
                        flush_shard()

                        if total_encoded % 5000 < config.batch_size:
                            context.log.info(
                                f"Encoded {total_encoded:,}/{temporal.height:,} total rows"
                            )

                # Final partial GPU batch.
                remaining = len(batch_tensors)
                if remaining:
                    flush_batch()
                    source_encoded[source] += remaining
                    total_encoded += remaining

                flush_shard(force=True)

        finally:
            if src is not None:
                src.close()

    if total_encoded != temporal.height:
        raise dg.Failure(
            f"Embedding row-count invariant failed: encoded={total_encoded:,}, "
            f"expected={temporal.height:,}"
        )

    success_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dagster_run_id": context.run_id,
        "source_temporal_index_uri": SILVER_H3_TEMPORAL_INDEX_URI,
        "output_prefix": GOLD_IMAGERY_EMBEDDINGS_PREFIX,
        "encoder_name": ENCODER_NAME,
        "encoder_version": ENCODER_VERSION,
        "embedding_dim": EMBEDDING_DIM,
        "image_size": config.image_size,
        "sentinel_recipe": {
            "spectral_stack": ["B02", "B03", "B04", "B08", "B11", "B12"],
            "scl_band": 7,
            "view_1": ["B04", "B03", "B02"],
            "view_2": ["B08", "B11", "B12"],
            "aggregation": "L2-normalize each view -> mean -> L2-normalize",
            "scl_bad_classes": sorted(int(x) for x in BAD_CLASSES),
            "dn_scale": config.sentinel_dn_scale,
            "reflectance_clip": config.sentinel_reflectance_clip,
        },
        "row_counts": source_encoded,
        "shard_counts": source_shards,
    }

    success_uri = _join_gs(GOLD_IMAGERY_EMBEDDINGS_PREFIX, "_SUCCESS.json")
    _upload_bytes(
        success_uri,
        json.dumps(success_manifest, indent=2).encode("utf-8"),
        "application/json",
    )

    return dg.MaterializeResult(
        metadata={
            "output_prefix": dg.MetadataValue.path(GOLD_IMAGERY_EMBEDDINGS_PREFIX),
            "success_manifest": dg.MetadataValue.path(success_uri),
            "rows": total_encoded,
            "naip_rows": source_encoded["naip"],
            "sentinel_rows": source_encoded["sentinel2"],
            "shards": total_shards,
            "embedding_dim": EMBEDDING_DIM,
            "encoder": f"{ENCODER_NAME}/{ENCODER_VERSION}",
            "device": str(device),
        }
    )