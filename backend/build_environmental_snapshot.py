from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
import requests


# ============================================================
# Configuration
# ============================================================

OPEN_METEO_URL = "http://127.0.0.1:8080/v1/forecast"
OPEN_METEO_MODEL = "ncep_gfs_seamless"

WEATHER_BATCH_SIZE = 2250
WEATHER_WORKERS = 8
WEATHER_RETRIES = 3
WEATHER_TIMEOUT = (10, 120)

# More than enough for the supervisor's current-hour probe.
PAST_HOURS = 12
FORECAST_HOURS = 6

# GFS should normally give essentially complete national coverage.
# If something is badly wrong, retry next poll instead of publishing
# a broken national snapshot.
MIN_WEATHER_COVERAGE = 0.99

LIGHTING_DEFINITION_VERSION = (
    "solar_elevation_twilight_v1"
)

# Codes consumed by build_national_intensity.py.
SERVING_LIGHTING_LEVELS = [
    "night",                    # 0
    "astronomical_twilight",    # 1
    "nautical_twilight",        # 2
    "civil_twilight",           # 3
    "daylight",                 # 4
]


# ============================================================
# Paths
# ============================================================

HOME = Path.home()

SERVING_ROOT = (
    HOME
    / "crimenet-serving"
)

ROOT = (
    SERVING_ROOT
    / "data"
    / "national_feature_store"
)

ENV_ROOT = (
    ROOT
    / "environmental"
)

R6_KEYS_PATH = (
    ENV_ROOT
    / "r6_keys.npy"
)

R6_LAT_PATH = (
    ENV_ROOT
    / "r6_lat.npy"
)

R6_LON_PATH = (
    ENV_ROOT
    / "r6_lon.npy"
)

SNAPSHOT_ROOT = (
    ENV_ROOT
    / "environmental_snapshots"
)

CURRENT_POINTER = (
    ENV_ROOT
    / "environmental_current.json"
)


# ============================================================
# Helpers
# ============================================================

def atomic_json_write(
    path: Path,
    payload: dict,
) -> None:

    tmp = path.with_name(
        f".{path.name}.tmp-"
        f"{os.getpid()}-"
        f"{uuid.uuid4().hex[:8]}"
    )

    with tmp.open("w") as f:
        json.dump(
            payload,
            f,
            indent=2,
        )

        f.flush()

        os.fsync(
            f.fileno()
        )

    os.replace(
        tmp,
        path,
    )


def publish_directory(
    temporary: Path,
    final: Path,
) -> None:

    backup = None

    if final.exists():

        backup = final.with_name(
            f".{final.name}.old-"
            f"{uuid.uuid4().hex[:8]}"
        )

        os.replace(
            final,
            backup,
        )

    try:

        os.replace(
            temporary,
            final,
        )

    except Exception:

        if (
            backup is not None
            and backup.exists()
        ):

            if final.exists():

                shutil.rmtree(
                    final,
                    ignore_errors=True,
                )

            os.replace(
                backup,
                final,
            )

        raise

    if backup is not None:

        shutil.rmtree(
            backup,
            ignore_errors=True,
        )


def normalize_utc_hour(
    value: str,
) -> pd.Timestamp:

    ts = pd.Timestamp(
        value
    )

    if ts.tzinfo is None:

        ts = ts.tz_localize(
            "UTC"
        )

    else:

        ts = ts.tz_convert(
            "UTC"
        )

    return ts.floor(
        "h"
    )


def finite_float(
    value,
) -> float | None:

    if value is None:
        return None

    try:

        result = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return None

    if not math.isfinite(
        result
    ):

        return None

    return result


# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--expected-valid-utc-hour",
    required=True,
    help=(
        "Exact UTC hour the snapshot must represent. "
        "Example: 2026-08-29T21:00:00+00:00"
    ),
)

parser.add_argument(
    "--no-publish-current",
    action="store_true",
    help=(
        "Build the timestamped environmental snapshot without "
        "advancing environmental_current.json. Used for forecasts."
    ),
)

args = parser.parse_args()

valid_utc = normalize_utc_hour(
    args.expected_valid_utc_hour
)

snapshot_id = (
    valid_utc
    .strftime(
        "%Y%m%dT%H%M"
    )
)

# Open-Meteo returns hourly timestamps without the UTC suffix
# when timezone=GMT.
target_open_meteo_time = (
    valid_utc
    .strftime(
        "%Y-%m-%dT%H:00"
    )
)

# Request enough Open-Meteo range to include the exact target hour.
# The live builder normally needs only a few hours, while forecast
# materialization may ask for +24h, +48h, +72h, etc.
now_utc_hour = (
    datetime.now(timezone.utc)
    .replace(
        minute=0,
        second=0,
        microsecond=0,
    )
)

hours_delta = (
    valid_utc - now_utc_hour
).total_seconds() / 3600.0

request_forecast_hours = max(
    FORECAST_HOURS,
    int(math.ceil(max(0.0, hours_delta))) + 2,
)

request_past_hours = max(
    PAST_HOURS,
    int(math.ceil(max(0.0, -hours_delta))) + 2,
)


# ============================================================
# Permanent R6 index
# ============================================================

print(
    "Loading permanent H3-R6 environmental index..."
)

for required in (
    R6_KEYS_PATH,
    R6_LAT_PATH,
    R6_LON_PATH,
):

    if not required.exists():

        raise FileNotFoundError(
            required
        )


r6_keys = np.load(
    R6_KEYS_PATH,
    mmap_mode="r",
)

r6_lat = np.load(
    R6_LAT_PATH,
    mmap_mode="r",
)

r6_lon = np.load(
    R6_LON_PATH,
    mmap_mode="r",
)


r6_count = len(
    r6_keys
)


if (
    len(r6_lat) != r6_count
    or
    len(r6_lon) != r6_count
):

    raise RuntimeError(
        "r6_keys/r6_lat/r6_lon row counts disagree"
    )


if not (
    np.isfinite(
        r6_lat
    ).all()
    and
    np.isfinite(
        r6_lon
    ).all()
):

    raise RuntimeError(
        "Non-finite R6 centroid coordinates"
    )


print(
    f"R6 cells:          "
    f"{r6_count:,}"
)

print(
    f"Valid UTC hour:    "
    f"{valid_utc.isoformat()}"
)

print(
    f"Open-Meteo model:  "
    f"{OPEN_METEO_MODEL}"
)


# ============================================================
# Output weather arrays
# ============================================================

temperature = np.full(
    r6_count,
    np.nan,
    dtype=np.float32,
)

humidity = np.full(
    r6_count,
    np.nan,
    dtype=np.float32,
)

weather_available = np.zeros(
    r6_count,
    dtype=np.uint8,
)


# ============================================================
# Open-Meteo batch request
# ============================================================

def fetch_weather_batch(
    batch_number: int,
    start: int,
    end: int,
):

    lat_text = ",".join(
        format(
            float(value),
            ".6f",
        )
        for value in r6_lat[
            start:end
        ]
    )

    lon_text = ",".join(
        format(
            float(value),
            ".6f",
        )
        for value in r6_lon[
            start:end
        ]
    )


    params = {

        "latitude":
            lat_text,

        "longitude":
            lon_text,

        "hourly":
            (
                "temperature_2m,"
                "relative_humidity_2m"
            ),

        "models":
            OPEN_METEO_MODEL,

        "timezone":
            "GMT",

        "past_hours":
            request_past_hours,

        "forecast_hours":
            request_forecast_hours,
    }


    last_error = None


    for attempt in range(
        1,
        WEATHER_RETRIES + 1,
    ):

        try:

            response = requests.get(
                OPEN_METEO_URL,
                params=params,
                timeout=WEATHER_TIMEOUT,
            )

            response.raise_for_status()

            payload = response.json()


            # One coordinate -> object.
            # Multiple coordinates -> list.
            if isinstance(
                payload,
                dict,
            ):

                items = [
                    payload
                ]

            elif isinstance(
                payload,
                list,
            ):

                items = payload

            else:

                raise RuntimeError(
                    "Unexpected Open-Meteo response type"
                )


            expected_count = (
                end
                - start
            )


            if (
                len(items)
                != expected_count
            ):

                raise RuntimeError(
                    "Open-Meteo batch length mismatch: "
                    f"expected={expected_count}, "
                    f"actual={len(items)}"
                )


            batch_temperature = np.full(
                expected_count,
                np.nan,
                dtype=np.float32,
            )

            batch_humidity = np.full(
                expected_count,
                np.nan,
                dtype=np.float32,
            )

            batch_available = np.zeros(
                expected_count,
                dtype=np.uint8,
            )


            target_seen = 0


            for i, item in enumerate(
                items
            ):

                hourly = (
                    item.get(
                        "hourly"
                    )
                    or {}
                )

                times = (
                    hourly.get(
                        "time"
                    )
                    or []
                )


                try:

                    hour_index = times.index(
                        target_open_meteo_time
                    )

                except ValueError:

                    continue


                target_seen += 1


                temp_values = (
                    hourly.get(
                        "temperature_2m"
                    )
                    or []
                )

                rh_values = (
                    hourly.get(
                        "relative_humidity_2m"
                    )
                    or []
                )


                if (
                    hour_index
                    >= len(
                        temp_values
                    )
                    or
                    hour_index
                    >= len(
                        rh_values
                    )
                ):

                    continue


                temp = finite_float(
                    temp_values[
                        hour_index
                    ]
                )

                rh = finite_float(
                    rh_values[
                        hour_index
                    ]
                )


                if (
                    temp is None
                    or
                    rh is None
                ):

                    continue


                batch_temperature[
                    i
                ] = np.float32(
                    temp
                )

                batch_humidity[
                    i
                ] = np.float32(
                    rh
                )

                batch_available[
                    i
                ] = 1


            # Important temporal safety check.
            #
            # If the requested hour vanished between the
            # supervisor's readiness probe and this build,
            # do NOT silently publish another hour.
            if target_seen == 0:

                raise RuntimeError(
                    "Expected UTC hour absent from "
                    "Open-Meteo response: "
                    f"{target_open_meteo_time}"
                )


            return (
                batch_number,
                start,
                end,
                batch_temperature,
                batch_humidity,
                batch_available,
            )


        except Exception as exc:

            last_error = exc


            if (
                attempt
                == WEATHER_RETRIES
            ):

                break


            time.sleep(
                2 ** (
                    attempt
                    - 1
                )
            )


    raise RuntimeError(
        f"Weather batch "
        f"{batch_number} failed "
        f"after {WEATHER_RETRIES} attempts"
    ) from last_error


# ============================================================
# Build batch ranges
# ============================================================

weather_ranges = []


for (
    batch_number,
    start,
) in enumerate(
    range(
        0,
        r6_count,
        WEATHER_BATCH_SIZE,
    ),
    start=1,
):

    end = min(
        start
        + WEATHER_BATCH_SIZE,
        r6_count,
    )

    weather_ranges.append(
        (
            batch_number,
            start,
            end,
        )
    )


print()

print(
    "Fetching national weather..."
)

print(
    f"Coordinate batches: "
    f"{len(weather_ranges)}"
)

print(
    f"Workers:            "
    f"{WEATHER_WORKERS}"
)


weather_start = (
    time.perf_counter()
)

completed = 0


with ThreadPoolExecutor(
    max_workers=WEATHER_WORKERS
) as executor:

    futures = [

        executor.submit(
            fetch_weather_batch,
            batch_number,
            start,
            end,
        )

        for (
            batch_number,
            start,
            end,
        )
        in weather_ranges
    ]


    for future in as_completed(
        futures
    ):

        (
            batch_number,
            start,
            end,
            batch_temperature,
            batch_humidity,
            batch_available,
        ) = future.result()


        temperature[
            start:end
        ] = batch_temperature

        humidity[
            start:end
        ] = batch_humidity

        weather_available[
            start:end
        ] = batch_available


        completed += 1


        if (
            completed % 10 == 0
            or
            completed
            == len(
                weather_ranges
            )
        ):

            print(
                f"Weather batches:   "
                f"{completed}/"
                f"{len(weather_ranges)}"
            )


weather_elapsed = (
    time.perf_counter()
    - weather_start
)


weather_available_count = int(
    np.count_nonzero(
        weather_available
    )
)


weather_coverage = (
    weather_available_count
    / r6_count
)


print()

print(
    f"Weather build:     "
    f"{weather_elapsed:.3f}s"
)

print(
    f"Weather coverage:  "
    f"{weather_available_count:,}/"
    f"{r6_count:,} "
    f"({weather_coverage:.4%})"
)


if (
    weather_coverage
    < MIN_WEATHER_COVERAGE
):

    raise RuntimeError(
        "Weather coverage below "
        "publication threshold: "
        f"{weather_coverage:.4%} "
        f"< "
        f"{MIN_WEATHER_COVERAGE:.4%}"
    )


# ============================================================
# Solar position
#
# EXACT model/training contract:
#
# pvlib.solarposition.get_solarposition(
#     ...,
#     method="nrel_numpy",
# )
#
# Use geometric elevation: "elevation"
# NOT apparent_elevation.
# ============================================================

print()

print(
    "Computing national solar position..."
)


solar_start = (
    time.perf_counter()
)


# One timestamp per R6 location.
solar_times = pd.DatetimeIndex(
    [
        valid_utc
    ]
    * r6_count
)


solar = (
    pvlib
    .solarposition
    .get_solarposition(
        solar_times,
        latitude=np.asarray(
            r6_lat,
            dtype=np.float64,
        ),
        longitude=np.asarray(
            r6_lon,
            dtype=np.float64,
        ),
        method="nrel_numpy",
    )
)


solar_elevation = np.asarray(
    solar[
        "elevation"
    ],
    dtype=np.float32,
)


solar_azimuth = np.asarray(
    solar[
        "azimuth"
    ],
    dtype=np.float32,
)


if (
    len(
        solar_elevation
    )
    != r6_count
    or
    len(
        solar_azimuth
    )
    != r6_count
):

    raise RuntimeError(
        "pvlib solar output row count mismatch"
    )


if not (
    np.isfinite(
        solar_elevation
    ).all()
    and
    np.isfinite(
        solar_azimuth
    ).all()
):

    raise RuntimeError(
        "Non-finite pvlib solar output"
    )


solar_elapsed = (
    time.perf_counter()
    - solar_start
)


# ============================================================
# Lighting classification
#
# Exact thresholds:
#
# elevation >=   0 : daylight
# elevation >=  -6 : civil twilight
# elevation >= -12 : nautical twilight
# elevation >= -18 : astronomical twilight
# otherwise         : night
# ============================================================

is_daylight = (
    solar_elevation
    >= 0.0
).astype(
    np.uint8
)


lighting_condition_code = np.zeros(
    r6_count,
    dtype=np.uint8,
)


lighting_condition_code[
    solar_elevation
    >= -18.0
] = 1


lighting_condition_code[
    solar_elevation
    >= -12.0
] = 2


lighting_condition_code[
    solar_elevation
    >= -6.0
] = 3


lighting_condition_code[
    solar_elevation
    >= 0.0
] = 4


lighting_counts = {

    SERVING_LIGHTING_LEVELS[
        code
    ]:
        int(
            np.count_nonzero(
                lighting_condition_code
                == code
            )
        )

    for code in range(
        len(
            SERVING_LIGHTING_LEVELS
        )
    )
}


print(
    f"Solar build:       "
    f"{solar_elapsed:.3f}s"
)

print(
    f"Elevation min:     "
    f"{float(np.min(solar_elevation)):.3f}"
)

print(
    f"Elevation mean:    "
    f"{float(np.mean(solar_elevation)):.3f}"
)

print(
    f"Elevation max:     "
    f"{float(np.max(solar_elevation)):.3f}"
)

print(
    f"Daylight cells:    "
    f"{int(np.count_nonzero(is_daylight)):,}/"
    f"{r6_count:,}"
)


# ============================================================
# Atomic environmental snapshot
# ============================================================

SNAPSHOT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


final_dir = (
    SNAPSHOT_ROOT
    / snapshot_id
)


tmp_dir = (
    SNAPSHOT_ROOT
    / (
        f".{snapshot_id}.tmp-"
        f"{os.getpid()}-"
        f"{uuid.uuid4().hex[:8]}"
    )
)


if tmp_dir.exists():

    shutil.rmtree(
        tmp_dir
    )


tmp_dir.mkdir(
    parents=True
)


try:

    np.save(
        tmp_dir
        / "temperature_2m.npy",
        temperature,
        allow_pickle=False,
    )

    np.save(
        tmp_dir
        / "relative_humidity_2m.npy",
        humidity,
        allow_pickle=False,
    )

    np.save(
        tmp_dir
        / "weather_available.npy",
        weather_available,
        allow_pickle=False,
    )

    np.save(
        tmp_dir
        / "solar_elevation_deg.npy",
        solar_elevation,
        allow_pickle=False,
    )

    np.save(
        tmp_dir
        / "solar_azimuth_deg.npy",
        solar_azimuth,
        allow_pickle=False,
    )

    np.save(
        tmp_dir
        / "is_daylight.npy",
        is_daylight,
        allow_pickle=False,
    )

    np.save(
        tmp_dir
        / "lighting_condition_code.npy",
        lighting_condition_code,
        allow_pickle=False,
    )


    metadata = {

        "schema":
            "crimenet_environmental_snapshot_v2",

        "snapshot_id":
            snapshot_id,

        "valid_utc_hour":
            valid_utc.isoformat(),

        "rows":
            int(
                r6_count
            ),

        "h3_resolution":
            6,

        "open_meteo_url":
            OPEN_METEO_URL,

        "open_meteo_model":
            OPEN_METEO_MODEL,

        "weather_variables": [
            "temperature_2m",
            "relative_humidity_2m",
        ],

        "weather_batch_size":
            WEATHER_BATCH_SIZE,

        "weather_workers":
            WEATHER_WORKERS,

        "open_meteo_past_hours_requested":
            request_past_hours,

        "open_meteo_forecast_hours_requested":
            request_forecast_hours,

        "publication_mode":
            (
                "timestamp_only"
                if args.no_publish_current
                else "current"
            ),

        "weather_available_rows":
            weather_available_count,

        "weather_coverage":
            weather_coverage,

        "minimum_weather_coverage":
            MIN_WEATHER_COVERAGE,

        "temperature_2m_c": {

            "min":
                float(
                    np.nanmin(
                        temperature
                    )
                ),

            "mean":
                float(
                    np.nanmean(
                        temperature
                    )
                ),

            "max":
                float(
                    np.nanmax(
                        temperature
                    )
                ),
        },

        "relative_humidity_2m_pct": {

            "min":
                float(
                    np.nanmin(
                        humidity
                    )
                ),

            "mean":
                float(
                    np.nanmean(
                        humidity
                    )
                ),

            "max":
                float(
                    np.nanmax(
                        humidity
                    )
                ),
        },

        "solar_method":
            (
                "pvlib.solarposition."
                "get_solarposition:nrel_numpy"
            ),

        "solar_elevation_field":
            "elevation",

        "lighting_definition_version":
            LIGHTING_DEFINITION_VERSION,

        "lighting_thresholds_deg": {

            "daylight":
                0.0,

            "civil_twilight":
                -6.0,

            "nautical_twilight":
                -12.0,

            "astronomical_twilight":
                -18.0,
        },

        "lighting_code_levels": {

            str(i):
                value

            for i, value
            in enumerate(
                SERVING_LIGHTING_LEVELS
            )
        },

        "lighting_counts":
            lighting_counts,

        "weather_build_seconds":
            weather_elapsed,

        "solar_build_seconds":
            solar_elapsed,

        "published_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }


    atomic_json_write(
        tmp_dir
        / "metadata.json",
        metadata,
    )


    publish_directory(
        tmp_dir,
        final_dir,
    )


    pointer = {

        "schema":
            "crimenet_environmental_pointer_v1",

        "snapshot_id":
            snapshot_id,

        "snapshot_path":
            str(
                final_dir
            ),

        "valid_utc_hour":
            valid_utc.isoformat(),

        "weather_coverage":
            weather_coverage,

        "published_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }


    if not args.no_publish_current:
        atomic_json_write(
            CURRENT_POINTER,
            pointer,
        )


except Exception:

    if tmp_dir.exists():

        shutil.rmtree(
            tmp_dir,
            ignore_errors=True,
        )

    raise


# ============================================================
# Final report
# ============================================================

print()

print(
    "========================================"
)

print(
    "NATIONAL ENVIRONMENTAL SNAPSHOT"
)

print(
    "========================================"
)


print(
    f"UTC hour:        "
    f"{valid_utc.isoformat()}"
)

print(
    f"R6 cells:        "
    f"{r6_count:,}"
)

print(
    f"Weather:         "
    f"{weather_available_count:,}/"
    f"{r6_count:,} "
    f"({weather_coverage:.4%})"
)

print(
    f"Temperature C:   "
    f"min={float(np.nanmin(temperature)):.2f} "
    f"mean={float(np.nanmean(temperature)):.2f} "
    f"max={float(np.nanmax(temperature)):.2f}"
)

print(
    f"Humidity %:      "
    f"min={float(np.nanmin(humidity)):.2f} "
    f"mean={float(np.nanmean(humidity)):.2f} "
    f"max={float(np.nanmax(humidity)):.2f}"
)

print(
    f"Solar elevation: "
    f"min={float(np.min(solar_elevation)):.2f} "
    f"mean={float(np.mean(solar_elevation)):.2f} "
    f"max={float(np.max(solar_elevation)):.2f}"
)

print(
    f"Lighting counts: "
    f"{lighting_counts}"
)

print(
    f"Snapshot:        "
    f"{final_dir}"
)

print(
    "Publication:     "
    + (
        f"current pointer -> {CURRENT_POINTER}"
        if not args.no_publish_current
        else "timestamp-only forecast snapshot"
    )
)

print()

print(
    "PUBLISHED"
)
