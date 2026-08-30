# CrimeSense hourly risk forecast implementation

This patch preserves the existing live snapshot behavior and adds timestamp-addressable forecast snapshots for the UI time slider.

## Files changed

- `backend/api.py`
  - Adds `GET /api/v1/intensity/timeline`.
  - Adds optional `valid_utc_hour` to `/intensity/point`, `/intensity/cell/{cell}`, `/intensity/viewport`, `/predict/point`, and `/predict/cell/{cell}`.
  - Keeps the old behavior when `valid_utc_hour` is omitted: serve `intensity_current.json`.
  - Uses a small mmap LRU for recently viewed hours.
  - Invalidates a cached future hour when that timestamp is rebuilt from a newer weather forecast.
- `backend/build_environmental_snapshot.py`
  - Adds `--no-publish-current` so future snapshots do not move the live pointer.
  - Automatically requests enough Open-Meteo forecast range to include the requested future hour.
- `backend/build_national_intensity.py`
  - Adds `--environmental-snapshot-path`, `--expected-valid-utc-hour`, and `--no-publish-current`.
  - Future intensity snapshots are built from exact environmental snapshot provenance without touching the live pointer.
- `backend/build_forecast_horizon.py`
  - New orchestration command that materializes +1h ... +Nh snapshots and publishes `intensity_timeline.json` incrementally.
- `backend/refresh_crimenet.py`
  - Automatically ensures the configured forecast horizon after the live hourly snapshot is current.
  - Default horizon is 24h. Configure with `CRIMENET_FORECAST_HOURS`.
- `frontend_example/forecast-slider.tsx`
  - TanStack Query timeline + viewport hooks and a slider component.
  - Prefetches ±2 adjacent hours.
  - Cache key includes the live forecast generation (`as_of_utc_hour`) so a newer weather run replaces an older forecast for the same valid hour.

## Install on the serving machine

Copy the patched backend files over `~/crimenet-serving/` and keep the new `build_forecast_horizon.py` there too.

Then syntax-check:

```bash
cd ~/crimenet-serving
python -m py_compile \
  api.py \
  build_environmental_snapshot.py \
  build_national_intensity.py \
  build_forecast_horizon.py \
  refresh_crimenet.py
```

## First test: build only 3 future hours

Stop the refresh supervisor temporarily if it is already running, then:

```bash
cd ~/crimenet-serving
python build_forecast_horizon.py --hours 3
```

This must **not** change `data/national_feature_store/intensity_current.json`.

Check the manifest:

```bash
jq . data/national_feature_store/intensity_timeline.json
```

You should have one live entry plus +1h, +2h, +3h forecast entries.

## API tests

Restart the FastAPI process after replacing `api.py`.

Timeline:

```powershell
curl.exe "http://192.168.68.63:8000/api/v1/intensity/timeline"
```

Take one `valid_utc_hour` returned from that response, for example `2026-08-30T06:00:00+00:00`.

Viewport at that exact forecast hour:

```powershell
curl.exe "http://192.168.68.63:8000/api/v1/intensity/viewport?west=-118.7&south=34.0&east=-118.3&north=34.4&valid_utc_hour=2026-08-30T06%3A00%3A00%2B00%3A00"
```

Cell at that hour:

```powershell
curl.exe "http://192.168.68.63:8000/api/v1/intensity/cell/8929a115dd7ffff?valid_utc_hour=2026-08-30T06%3A00%3A00%2B00%3A00"
```

Omitting `valid_utc_hour` still serves the current live snapshot exactly as before.

## Automatic horizon

`refresh_crimenet.py` defaults to 24 future hours. For 72 hours:

```bash
export CRIMENET_FORECAST_HOURS=72
python refresh_crimenet.py
```

For systemd, put the environment variable in the service unit/environment file rather than relying on an interactive shell export.

Set `CRIMENET_FORECAST_HOURS=0` to disable automatic forecast materialization while keeping the live refresh behavior.

## Storage warning

The canonical r9 store has ~25.6M rows. Two float32 r9 arrays (`intensity.npy` + `log_intensity.npy`) alone are about 205 MB per hour before LOD arrays and environmental artifacts. Therefore:

- 24 future hours: at least ~4.9 GB for those two arrays alone.
- 72 future hours: at least ~14.7 GB for those two arrays alone.

Start with 3-6 hours to validate the complete flow, then choose 24/48/72 based on disk and rebuild speed.

## Frontend use

The provided `forecast-slider.tsx` expects the existing TanStack Query dependency. Feed the returned `ViewportResponse.cells` into the same deck.gl surface you already use for the live map.

The critical flow is:

```text
GET /api/v1/intensity/timeline
           ↓
slider selects snapshots[i].valid_utc_hour
           ↓
GET /api/v1/intensity/viewport?...&valid_utc_hour=<selected hour>
           ↓
replace deck.gl H3 layer data
```

Do not interpolate the numerical model value between hours. If desired, use deck.gl transitions only for visual cross-fading between genuine hourly model surfaces.
