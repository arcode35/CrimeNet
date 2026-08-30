from __future__ import annotations

import fcntl
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


# ============================================================
# Paths
# ============================================================

SERVING_ROOT = (
    Path.home()
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

ENV_POINTER = (
    ENV_ROOT
    / "environmental_current.json"
)

INTENSITY_POINTER = (
    ROOT
    / "intensity_current.json"
)


# Change ONLY this line if your environmental builder
# currently has a different filename.
ENV_BUILDER = (
    SERVING_ROOT
    / "build_environmental_snapshot.py"
)

INTENSITY_BUILDER = (
    SERVING_ROOT
    / "build_national_intensity.py"
)

FORECAST_BUILDER = (
    SERVING_ROOT
    / "build_forecast_horizon.py"
)

FORECAST_TIMELINE = (
    ROOT
    / "intensity_timeline.json"
)

# Set CRIMENET_FORECAST_HOURS=0 to disable forecast materialization.
FORECAST_HOURS = int(
    os.environ.get(
        "CRIMENET_FORECAST_HOURS",
        "24",
    )
)


# ============================================================
# Polling
# ============================================================

OPEN_METEO_URL = (
    "http://127.0.0.1:8080/v1/forecast"
)

POLL_SECONDS = 300

LOOKBACK_HOURS = 3


# Cheap probe locations.
#
# We are NOT using these to build the national weather store.
# They only tell us whether the local Open-Meteo database
# has a usable current hourly timestep yet.
SENTINELS = [
    (29.7604, -95.3698),
    (40.7128, -74.0060),
    (47.6062, -122.3321),
]


LOCK_PATH = (
    SERVING_ROOT
    / ".refresh_crimenet.lock"
)


# ============================================================
# Helpers
# ============================================================

def log(message: str) -> None:
    now = (
        datetime.now(timezone.utc)
        .isoformat(
            timespec="seconds"
        )
    )

    print(
        f"[{now}] {message}",
        flush=True,
    )


def load_json(
    path: Path,
) -> dict:

    with path.open("r") as f:
        return json.load(f)


def parse_hour(
    value: str,
) -> datetime:

    dt = datetime.fromisoformat(
        value.replace(
            "Z",
            "+00:00",
        )
    )

    if dt.tzinfo is None:
        dt = dt.replace(
            tzinfo=timezone.utc
        )

    dt = dt.astimezone(
        timezone.utc
    )

    return dt.replace(
        minute=0,
        second=0,
        microsecond=0,
    )


def hour_text(
    dt: datetime,
) -> str:

    return (
        dt.astimezone(
            timezone.utc
        )
        .isoformat(
            timespec="seconds"
        )
    )


# ============================================================
# Pointer readers
# ============================================================

def read_environmental_hour() -> datetime | None:

    if not ENV_POINTER.exists():
        return None

    pointer = load_json(
        ENV_POINTER
    )

    if pointer.get(
        "valid_utc_hour"
    ):
        return parse_hour(
            pointer[
                "valid_utc_hour"
            ]
        )

    snapshot_path = pointer.get(
        "snapshot_path"
    )

    if not snapshot_path:
        raise RuntimeError(
            f"{ENV_POINTER} has neither "
            "valid_utc_hour nor snapshot_path"
        )

    metadata = load_json(
        Path(snapshot_path)
        / "metadata.json"
    )

    return parse_hour(
        metadata[
            "valid_utc_hour"
        ]
    )


def read_intensity_hour() -> datetime | None:

    if not INTENSITY_POINTER.exists():
        return None

    pointer = load_json(
        INTENSITY_POINTER
    )

    if pointer.get(
        "valid_utc_hour"
    ):
        return parse_hour(
            pointer[
                "valid_utc_hour"
            ]
        )

    snapshot_path = pointer.get(
        "snapshot_path"
    )

    if not snapshot_path:
        raise RuntimeError(
            f"{INTENSITY_POINTER} has neither "
            "valid_utc_hour nor snapshot_path"
        )

    metadata = load_json(
        Path(snapshot_path)
        / "metadata.json"
    )

    return parse_hour(
        metadata[
            "valid_utc_hour"
        ]
    )


# ============================================================
# Subprocess
# ============================================================

def run_checked(
    command: list[str],
) -> None:

    log(
        "RUN "
        + " ".join(command)
    )

    subprocess.run(
        command,
        cwd=SERVING_ROOT,
        check=True,
    )


# ============================================================
# Cheap Open-Meteo readiness probe
# ============================================================

def weather_value_ready(
    latitude: float,
    longitude: float,
    candidate: datetime,
) -> bool:

    params = {
        "latitude":
            latitude,

        "longitude":
            longitude,

        "hourly":
            (
                "temperature_2m,"
                "relative_humidity_2m"
            ),

        "models":
            "ncep_gfs_seamless",

        "timezone":
            "GMT",

        "past_hours":
            4,

        "forecast_hours":
            2,
    }

    url = (
        OPEN_METEO_URL
        + "?"
        + urlencode(
            params
        )
    )

    with urlopen(
        url,
        timeout=20,
    ) as response:

        payload = json.load(
            response
        )

    hourly = (
        payload.get("hourly")
        or {}
    )

    times = (
        hourly.get("time")
        or []
    )

    temperature = (
        hourly.get(
            "temperature_2m"
        )
        or []
    )

    humidity = (
        hourly.get(
            "relative_humidity_2m"
        )
        or []
    )

    target = candidate.strftime(
        "%Y-%m-%dT%H:00"
    )

    try:
        index = times.index(
            target
        )

    except ValueError:
        return False

    if (
        index >= len(temperature)
        or
        index >= len(humidity)
    ):
        return False

    temp = temperature[
        index
    ]

    rh = humidity[
        index
    ]

    if (
        temp is None
        or rh is None
    ):
        return False

    return (
        math.isfinite(
            float(temp)
        )
        and
        math.isfinite(
            float(rh)
        )
    )


def hour_ready(
    candidate: datetime,
) -> bool:

    for (
        latitude,
        longitude,
    ) in SENTINELS:

        if not weather_value_ready(
            latitude,
            longitude,
            candidate,
        ):
            return False

    return True


def newest_ready_hour() -> datetime | None:

    now_hour = (
        datetime.now(
            timezone.utc
        )
        .replace(
            minute=0,
            second=0,
            microsecond=0,
        )
    )

    for offset in range(
        LOOKBACK_HOURS + 1
    ):

        candidate = (
            now_hour
            - timedelta(
                hours=offset
            )
        )

        try:

            if hour_ready(
                candidate
            ):
                return candidate

        except Exception as exc:

            log(
                "Open-Meteo probe failed: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            return None

    return None


# ============================================================
# Builders
# ============================================================

def build_environmental(
    candidate: datetime,
) -> None:

    run_checked(
        [
            sys.executable,
            str(
                ENV_BUILDER
            ),
            "--expected-valid-utc-hour",
            hour_text(
                candidate
            ),
        ]
    )

    published = (
        read_environmental_hour()
    )

    if published != candidate:

        raise RuntimeError(
            "Environmental builder "
            "published wrong hour: "
            f"expected="
            f"{hour_text(candidate)}, "
            f"actual={published}"
        )


def build_intensity(
    expected_hour: datetime,
) -> None:

    run_checked(
        [
            sys.executable,
            str(
                INTENSITY_BUILDER
            ),
        ]
    )

    published = (
        read_intensity_hour()
    )

    if published != expected_hour:

        raise RuntimeError(
            "Intensity builder "
            "published wrong hour: "
            f"expected="
            f"{hour_text(expected_hour)}, "
            f"actual={published}"
        )


def forecast_is_current(
    current_hour: datetime,
) -> bool:

    if FORECAST_HOURS <= 0:
        return True

    if not FORECAST_TIMELINE.exists():
        return False

    try:
        payload = load_json(
            FORECAST_TIMELINE
        )

        as_of = parse_hour(
            payload[
                "as_of_utc_hour"
            ]
        )

        available = int(
            payload.get(
                "hours_available",
                0,
            )
        )

        return (
            as_of == current_hour
            and available >= FORECAST_HOURS
        )

    except Exception:
        return False


def ensure_forecast(
    current_hour: datetime,
) -> None:

    if FORECAST_HOURS <= 0:
        return

    if forecast_is_current(
        current_hour
    ):
        log(
            "Forecast horizon already current "
            f"(+{FORECAST_HOURS}h)"
        )
        return

    if not FORECAST_BUILDER.exists():
        raise FileNotFoundError(
            FORECAST_BUILDER
        )

    log(
        "Refreshing forecast horizon "
        f"(+{FORECAST_HOURS}h)"
    )

    run_checked(
        [
            sys.executable,
            str(FORECAST_BUILDER),
            "--hours",
            str(FORECAST_HOURS),
        ]
    )

    if not forecast_is_current(
        current_hour
    ):
        raise RuntimeError(
            "Forecast builder completed but timeline "
            "does not match current live hour"
        )

    log(
        "Forecast horizon publish complete"
    )


# ============================================================
# State machine
# ============================================================

def reconcile_once() -> None:

    env_hour = (
        read_environmental_hour()
    )

    intensity_hour = (
        read_intensity_hour()
    )

    log(
        "STATE "
        f"environmental={env_hour} "
        f"intensity={intensity_hour}"
    )


    # Impossible state.
    if (
        env_hour is not None
        and
        intensity_hour is not None
        and
        intensity_hour > env_hour
    ):

        raise RuntimeError(
            "Invalid state: intensity "
            "is newer than environmental"
        )


    # ========================================================
    # Recovery
    #
    # Environmental build succeeded previously, but intensity
    # failed. Do NOT rebuild weather. Just catch intensity up.
    # ========================================================

    if (
        env_hour is not None
        and
        (
            intensity_hour is None
            or
            intensity_hour < env_hour
        )
    ):

        log(
            "Intensity behind environmental; "
            f"catching up to "
            f"{hour_text(env_hour)}"
        )

        build_intensity(
            env_hour
        )

        log(
            "Catch-up intensity "
            "publish complete"
        )

        ensure_forecast(
            env_hour
        )

        return


    # ========================================================
    # Probe local Open-Meteo
    # ========================================================

    candidate = (
        newest_ready_hour()
    )

    if candidate is None:

        log(
            "No usable weather hour "
            "available yet"
        )

        return


    # Already current.
    if (
        env_hour is not None
        and
        candidate <= env_hour
    ):

        log(
            "No new environmental hour "
            f"(candidate="
            f"{hour_text(candidate)})"
        )

        ensure_forecast(
            env_hour
        )

        return


    # ========================================================
    # New hour
    # ========================================================

    log(
        "New weather hour ready: "
        f"{hour_text(candidate)}"
    )


    # Weather + solar + lighting + calendar.
    build_environmental(
        candidate
    )

    log(
        "Environmental publish complete"
    )


    # National model inference.
    build_intensity(
        candidate
    )

    log(
        "Intensity publish complete"
    )


    # ========================================================
    # Final consistency invariant
    # ========================================================

    final_env = (
        read_environmental_hour()
    )

    final_intensity = (
        read_intensity_hour()
    )

    if (
        final_env != candidate
        or
        final_intensity != candidate
    ):

        raise RuntimeError(
            "Post-publish invariant failed: "
            f"environmental={final_env}, "
            f"intensity={final_intensity}, "
            f"candidate={candidate}"
        )

    log(
        "PUBLISHED coherent snapshot pair "
        f"{hour_text(candidate)}"
    )

    ensure_forecast(
        candidate
    )


# ============================================================
# Main loop
# ============================================================

def main() -> None:

    if not ENV_BUILDER.exists():
        raise FileNotFoundError(
            ENV_BUILDER
        )

    if not INTENSITY_BUILDER.exists():
        raise FileNotFoundError(
            INTENSITY_BUILDER
        )

    if (
        FORECAST_HOURS > 0
        and not FORECAST_BUILDER.exists()
    ):
        raise FileNotFoundError(
            FORECAST_BUILDER
        )

    LOCK_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with LOCK_PATH.open(
        "w"
    ) as lock_file:

        # Ensures only one refresh supervisor can exist.
        fcntl.flock(
            lock_file.fileno(),
            fcntl.LOCK_EX
            | fcntl.LOCK_NB,
        )

        log(
            "CrimeNet refresh supervisor "
            f"started; poll="
            f"{POLL_SECONDS}s"
        )

        while True:

            started = (
                time.monotonic()
            )

            try:

                reconcile_once()

            except Exception as exc:

                # Do not kill the daemon.
                # The next cycle retries safely.
                log(
                    "REFRESH FAILED: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

            elapsed = (
                time.monotonic()
                - started
            )

            sleep_for = max(
                1.0,
                POLL_SECONDS
                - elapsed,
            )

            time.sleep(
                sleep_for
            )


if __name__ == "__main__":
    main()
