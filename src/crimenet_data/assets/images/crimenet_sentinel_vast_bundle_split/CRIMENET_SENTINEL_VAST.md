# CrimeNet Sentinel Vast pipeline — split CPU/GPU execution

## Purpose

This extends `foundation_v1` Sentinel OlmoEarth embeddings while preserving the
existing scientific contract, but it can now be run on two independent hosts:

```text
CPU/RAM/NVMe host
B2 raw TIFFs
  -> inventory / SCL candidate scoring / H3-month selection
  -> historical context hydration
  -> exact 12-band 128x128 prepared frames
  -> ZSTD Parquet cache
  -> B2 temporary prepared cache

GPU host
B2 temporary prepared cache
  -> local sequential Parquet reads
  -> multi-GPU OlmoEarth inference
  -> finalize / schema audit
  -> B2 Gold staging
```

The GPU host never needs the ~2 TB raw TIFF archive, scene-selection work, or the
existing Gold embedding shards.

## Scientific contract retained

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

## Temporary B2 cache

Default:

```text
b2:crimenet-data/tmp/sentinel_prepared_v1
```

Layout:

```text
sentinel_prepared_v1/
  frames/
    bucket=000/part-....parquet
    ...
    bucket=063/part-....parquet
  handoff/
    preprocess_handoff.json
    existing_validity_patch.parquet   # only when needed
```

Each frame row stores losslessly:

- `12 x 128 x 128` uint16 spectral DN bytes
- `128 x 128` uint8 SCL bytes
- H3 / timestamp / selection / temporal-target metadata

`preprocess_handoff.json` stores frame counts and sizes, the existing Gold part
start, existing Gold schema, scientific-contract identifiers, and any validity
patch reference. The GPU host validates this before inference.

The temporary prepared cache is intentionally durable across Vast rentals. Delete
it only after the new Gold staging output has been validated.

## Install

```bash
pip install -r requirements-crimenet-sentinel-vast.txt
```

Configure an rclone remote named `b2`.

## CPU/RAM/NVMe host

Pick a work directory on the large local disk:

```bash
export WORK_DIR=/workspace/crimenet-sentinel-preprocess
export PREPROCESSED_REMOTE=b2:crimenet-data/tmp/sentinel_prepared_v1
./run_crimenet_sentinel_vast.sh preprocess
```

Equivalent direct command:

```bash
python crimenet_sentinel_vast.py \
  --phase preprocess \
  --work-dir /workspace/crimenet-sentinel-preprocess \
  --preprocessed-remote b2:crimenet-data/tmp/sentinel_prepared_v1
```

`preprocess` expands to:

```text
stage -> inventory -> candidates -> select -> context -> frames -> upload-preprocessed
```

After `upload-preprocessed` succeeds and verifies, the CPU instance can be
terminated. The remote raw Bronze imagery is never deleted by this script.

## GPU host

The GPU machine can start with an otherwise empty work directory:

```bash
export WORK_DIR=/workspace/crimenet-sentinel-gpu
export PREPROCESSED_REMOTE=b2:crimenet-data/tmp/sentinel_prepared_v1
./run_crimenet_sentinel_vast.sh gpu
```

Equivalent direct command:

```bash
python crimenet_sentinel_vast.py \
  --phase gpu \
  --work-dir /workspace/crimenet-sentinel-gpu \
  --preprocessed-remote b2:crimenet-data/tmp/sentinel_prepared_v1 \
  --gpus 0 \
  --precision bf16
```

`gpu` expands to:

```text
fetch-preprocessed -> embed -> finalize -> publish
```

`--gpus 0` means all visible GPUs.

## Restartable low-level phases

```text
stage
inventory
candidates
select
context
frames
upload-preprocessed
fetch-preprocessed
embed
finalize
publish
```

All continue to support `--resume`.

## Final output

Default:

```text
b2:crimenet-data/gold/imagery/embeddings/foundation_v1_b2_staging/sentinel2
```

The GPU-side finalize phase uses the Gold schema and existing part offset recorded
by the CPU-side handoff, so it does not need to download the existing embedding
store.

## Cleanup

After final staging is fully audited, remove only the temporary prepared cache if
you no longer want to retain it:

```bash
rclone delete b2:crimenet-data/tmp/sentinel_prepared_v1 --rmdirs
```

Do not delete the raw Bronze Sentinel archive until you have separately decided it
is safe to do so.
