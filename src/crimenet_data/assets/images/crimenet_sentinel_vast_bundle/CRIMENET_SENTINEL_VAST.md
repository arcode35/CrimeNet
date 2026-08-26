# CrimeNet Sentinel Vast pipeline

## What this does

This is the disk-first, multi-GPU implementation for extending the existing
`foundation_v1` Sentinel OlmoEarth feature store from the recent B2 native-band
Sentinel-2 L2A archive.

It preserves the old scientific contract:

- H3 resolution 9
- H3 polygon + 600 m candidate context
- usable iff local SCL bad fraction <= 0.20
- one best image per H3/month, ordered by local bad fraction, coverage, scene cloud,
  newest capture, then item id
- bands `B02 B03 B04 B08 B05 B06 B07 B8A B11 B12 B01 B09`
- 128 x 128, 10 m OlmoEarth input grid
- SCL nearest-neighbor mask
- PB >= 04.00 radiometry `(DN - 1000) / 10000`; older `DN / 10000`
- SCL bad pixels filled with per-band median
- trailing <= 12 observations and <= 400 days
- `OLMOEARTH_V1_2_BASE`, bf16, patch size 8, input resolution 10 m
- mean token pooling + L2 normalization
- exact existing Gold output schema

## Storage design

The 2 TB B2 GeoTIFF archive is copied to local NVMe first. The raw GeoTIFF bytes
are **not** stuffed into Parquet because that would throw away efficient tiled
raster access.

After H3/month selection, the CPU phase writes a much smaller temporary prepared
frame cache as ZSTD Parquet. Each row contains lossless:

- `12 x 128 x 128` uint16 spectral DN bytes
- `128 x 128` uint8 SCL bytes
- temporal/selection metadata

The GPU stage therefore performs sequential NVMe Parquet reads and never touches
B2 or Rasterio.

## Historical temporal context

Months already present in the existing Gold store are not replaced. New B2
candidates are anti-joined against existing `(h3_cell, capture_period)` keys.

For new targets, the script selects existing Gold rows within the preceding
400-day window and reacquires only those source Sentinel scenes from Microsoft
Planetary Computer. This lets new embeddings retain up to 12 historical monthly
frames rather than degrading to a 1-3 frame sequence.

## Install

```bash
pip install -r requirements-crimenet-sentinel-vast.txt
```

Install/configure `rclone`, including a remote named `b2`.

## Run the full pipeline

```bash
./run_crimenet_sentinel_vast.sh
```

Or run individual restartable phases:

```bash
python crimenet_sentinel_vast.py --phase stage     --work-dir /workspace/crimenet-sentinel-vast
python crimenet_sentinel_vast.py --phase inventory --work-dir /workspace/crimenet-sentinel-vast
python crimenet_sentinel_vast.py --phase candidates --work-dir /workspace/crimenet-sentinel-vast
python crimenet_sentinel_vast.py --phase select    --work-dir /workspace/crimenet-sentinel-vast
python crimenet_sentinel_vast.py --phase context   --work-dir /workspace/crimenet-sentinel-vast
python crimenet_sentinel_vast.py --phase frames    --work-dir /workspace/crimenet-sentinel-vast
python crimenet_sentinel_vast.py --phase embed     --work-dir /workspace/crimenet-sentinel-vast
python crimenet_sentinel_vast.py --phase finalize  --work-dir /workspace/crimenet-sentinel-vast
python crimenet_sentinel_vast.py --phase publish   --work-dir /workspace/crimenet-sentinel-vast
```

## Output

Default publish target:

```text
b2:crimenet-data/gold/imagery/embeddings/foundation_v1_b2_staging/sentinel2
```

Part numbers begin after the staged existing Gold part index (currently expected
to be 758), so after audit they can be promoted into the canonical
`foundation_v1/sentinel2` directory without collisions.

The final directory includes:

- `part-XXXXX.parquet`
- `_B2_VAST_RECIPE.json`
- `_B2_VAST_SUCCESS.json`
- `_EXISTING_VALIDITY_PATCH.parquet` when old latest intervals need closing

## Safety

The implementation does **not** delete the remote 2 TB B2 Bronze archive. Keep it
until the staging embeddings have passed row-count/schema/coverage checks and you
have decided whether the raw source should be retained.
