from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from intensity_lod import write_lod_intensities
import cupy as cp
import numpy as np
import pandas as pd
import xgboost as xgb


# ============================================================
# CrimeNet national intensity serving contract
# ============================================================

MODEL_RUN_ID = "ff94186b570641e68eb8c9d5cf64c567"

INITIAL_LAMBDA = 1.9027141790906403e-07
INITIAL_LOG_LAMBDA = -15.474814268832395

MIN_LOG_INTENSITY = -30.0
MAX_LOG_INTENSITY = 15.0

EXPECTED_FEATURE_COUNT = 38
EXPECTED_BOOSTING_ROUNDS = 91

CHUNK_SIZE = 1_048_576


# ============================================================
# Lighting categorical contract
# ============================================================

# Codes emitted by the environmental snapshot builder.
SERVING_LIGHTING_LEVELS = [
    "night",
    "astronomical_twilight",
    "nautical_twilight",
    "civil_twilight",
    "daylight",
]

# Exact model-side categorical code order.
MODEL_LIGHTING_LEVELS = [
    "astronomical_twilight",
    "civil_twilight",
    "day",
    "nautical_twilight",
    "night",
]

# Serving uses "daylight"; model categorical dictionary uses "day".
LIGHTING_ALIASES = {
    "daylight": "day",
}


# ============================================================
# Paths
# ============================================================

HOME = Path.home()

SERVING_ROOT = HOME / "crimenet-serving"

ROOT = (
    SERVING_ROOT
    / "data"
    / "national_feature_store"
)

STATIC_ROOT = ROOT / "mmap"
ENV_ROOT = ROOT / "environmental"

MODEL_PATH = (
    SERVING_ROOT
    / "models"
    / "intensity"
    / "model.ubj"
)

SNAPSHOT_ROOT = (
    ROOT
    / "intensity_snapshots"
)

CURRENT_POINTER = (
    ROOT
    / "intensity_current.json"
)


# ============================================================
# Helpers
# ============================================================

def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def atomic_json_write(
    path: Path,
    payload: dict,
) -> None:
    """
    Write JSON to a temporary file and atomically replace the target.
    """

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
    """
    Atomically replace an existing snapshot directory.

    The old snapshot is retained until the new snapshot has been
    successfully moved into place.
    """

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


# ============================================================
# Load national static store
# ============================================================

print(
    "Loading national static store..."
)

h3_keys = np.load(
    STATIC_ROOT
    / "h3_keys.npy",
    mmap_mode="r",
)

static_features = np.load(
    STATIC_ROOT
    / "features.npy",
    mmap_mode="r",
)

static_meta = load_json(
    STATIC_ROOT
    / "metadata.json"
)

static_names = list(
    static_meta["features"]
)

n = len(
    h3_keys
)

if static_features.shape[0] != n:
    raise RuntimeError(
        "h3_keys.npy and features.npy "
        "row counts disagree"
    )

if (
    static_features.shape[1]
    != len(static_names)
):
    raise RuntimeError(
        "Static feature matrix width "
        "disagrees with metadata"
    )

if len(static_names) != 25:
    raise RuntimeError(
        f"Expected 25 static features, "
        f"got {len(static_names)}"
    )

if (
    len(set(static_names))
    != len(static_names)
):
    raise RuntimeError(
        "Duplicate static feature names"
    )

print(
    f"R9 cells:          "
    f"{n:,}"
)

print(
    f"Static features:   "
    f"{len(static_names)}"
)


# ============================================================
# Load permanent R9 -> R6 mapping
# ============================================================

r9_to_r6 = np.load(
    ENV_ROOT
    / "r9_to_r6_idx.npy",
    mmap_mode="r",
)

r6_timezone_id = np.load(
    ENV_ROOT
    / "r6_timezone_id.npy",
    mmap_mode="r",
)

timezone_meta = load_json(
    ENV_ROOT
    / "r6_timezones.json"
)

timezone_names = {
    int(k): v
    for k, v
    in timezone_meta["timezones"].items()
}

if len(r9_to_r6) != n:
    raise RuntimeError(
        "r9_to_r6_idx.npy does not align "
        "with national R9 rows"
    )

r6_count = len(
    r6_timezone_id
)

if (
    int(np.max(r9_to_r6))
    >= r6_count
):
    raise RuntimeError(
        "r9_to_r6_idx contains an "
        "out-of-range R6 row"
    )

present_timezone_ids = set(
    np.unique(
        r6_timezone_id
    ).astype(int).tolist()
)

missing_timezone_ids = sorted(
    present_timezone_ids
    - set(timezone_names)
)

if missing_timezone_ids:
    raise RuntimeError(
        "R6 timezone IDs missing from "
        f"r6_timezones.json: {missing_timezone_ids}"
    )

print(
    f"R6 cells:          "
    f"{r6_count:,}"
)

print(
    f"IANA timezones:    "
    f"{len(timezone_names)}"
)


# ============================================================
# Load current environmental snapshot
# ============================================================

environment_pointer = load_json(
    ENV_ROOT
    / "environmental_current.json"
)

env_dir = Path(
    environment_pointer[
        "snapshot_path"
    ]
)

if not env_dir.exists():
    raise RuntimeError(
        "Environmental snapshot does "
        f"not exist: {env_dir}"
    )

env_meta = load_json(
    env_dir
    / "metadata.json"
)

valid_utc = pd.Timestamp(
    env_meta[
        "valid_utc_hour"
    ]
)

if valid_utc.tzinfo is None:
    valid_utc = (
        valid_utc
        .tz_localize("UTC")
    )
else:
    valid_utc = (
        valid_utc
        .tz_convert("UTC")
    )

snapshot_id = (
    valid_utc
    .strftime(
        "%Y%m%dT%H%M"
    )
)

print(
    f"Environmental UTC: "
    f"{valid_utc.isoformat()}"
)

print(
    f"Snapshot ID:       "
    f"{snapshot_id}"
)


def load_r6_array(
    filename: str,
) -> np.ndarray:

    array = np.load(
        env_dir
        / filename,
        mmap_mode="r",
    )

    if len(array) != r6_count:
        raise RuntimeError(
            f"{filename}: "
            f"expected {r6_count:,} rows, "
            f"got {len(array):,}"
        )

    return array


temperature = load_r6_array(
    "temperature_2m.npy"
)

humidity = load_r6_array(
    "relative_humidity_2m.npy"
)

weather_available = load_r6_array(
    "weather_available.npy"
)

solar_elevation = load_r6_array(
    "solar_elevation_deg.npy"
)

solar_azimuth = load_r6_array(
    "solar_azimuth_deg.npy"
)

is_daylight = load_r6_array(
    "is_daylight.npy"
)

lighting_serving_code = load_r6_array(
    "lighting_condition_code.npy"
)


# ============================================================
# Calendar features
#
# Exact training semantics:
#
# UTC timestamp
#     -> row IANA timezone
#     -> local hour
#     -> Monday=0 weekday
#     -> cyclic encodings
# ============================================================

print(
    "Building calendar features..."
)

timezone_id_max = int(
    np.max(
        r6_timezone_id
    )
)

hour_by_timezone = np.empty(
    timezone_id_max + 1,
    dtype=np.float32,
)

dow_by_timezone = np.empty(
    timezone_id_max + 1,
    dtype=np.float32,
)

utc_python = (
    valid_utc
    .to_pydatetime()
)

for (
    tz_id,
    tz_name,
) in timezone_names.items():

    local_dt = (
        utc_python
        .astimezone(
            ZoneInfo(
                tz_name
            )
        )
    )

    hour_by_timezone[
        tz_id
    ] = float(
        local_dt.hour
    )

    # Python weekday:
    # Monday=0 ... Sunday=6.
    dow_by_timezone[
        tz_id
    ] = float(
        local_dt.weekday()
    )


local_hour = (
    hour_by_timezone[
        r6_timezone_id
    ]
)

local_dow = (
    dow_by_timezone[
        r6_timezone_id
    ]
)

local_hour_sin = np.sin(
    local_hour
    * (
        2.0
        * np.pi
        / 24.0
    )
).astype(
    np.float32,
    copy=False,
)

local_hour_cos = np.cos(
    local_hour
    * (
        2.0
        * np.pi
        / 24.0
    )
).astype(
    np.float32,
    copy=False,
)

local_dow_sin = np.sin(
    local_dow
    * (
        2.0
        * np.pi
        / 7.0
    )
).astype(
    np.float32,
    copy=False,
)

local_dow_cos = np.cos(
    local_dow
    * (
        2.0
        * np.pi
        / 7.0
    )
).astype(
    np.float32,
    copy=False,
)


# ============================================================
# Lighting categorical translation
# ============================================================

translation = []

for serving_name in (
    SERVING_LIGHTING_LEVELS
):

    model_name = (
        LIGHTING_ALIASES.get(
            serving_name,
            serving_name,
        )
    )

    try:
        model_code = (
            MODEL_LIGHTING_LEVELS
            .index(
                model_name
            )
        )

    except ValueError as exc:
        raise RuntimeError(
            "No model categorical "
            "mapping for lighting value "
            f"{serving_name!r}"
        ) from exc

    translation.append(
        model_code
    )


lighting_translation = np.asarray(
    translation,
    dtype=np.float32,
)

lighting_codes_int = np.asarray(
    lighting_serving_code,
    dtype=np.int64,
)

if (
    np.min(
        lighting_codes_int
    ) < 0
    or
    np.max(
        lighting_codes_int
    )
    >= len(
        SERVING_LIGHTING_LEVELS
    )
):
    raise RuntimeError(
        "Environmental snapshot contains "
        "invalid lighting codes"
    )


lighting_model_code = (
    lighting_translation[
        lighting_codes_int
    ]
)


# ============================================================
# Load canonical intensity booster
# ============================================================

print(
    "Loading intensity model..."
)

if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model missing: {MODEL_PATH}"
    )

bst = xgb.Booster()

bst.load_model(
    MODEL_PATH
)

bst.set_param(
    {
        "device":
            "cuda:0",
    }
)

model_names = list(
    bst.feature_names
    or []
)

model_types = list(
    bst.feature_types
    or []
)

if (
    len(model_names)
    != EXPECTED_FEATURE_COUNT
):
    raise RuntimeError(
        f"Expected "
        f"{EXPECTED_FEATURE_COUNT} "
        f"model features, got "
        f"{len(model_names)}"
    )

if (
    len(set(model_names))
    != len(model_names)
):
    raise RuntimeError(
        "Booster contains duplicate "
        "feature names"
    )

rounds = (
    bst
    .num_boosted_rounds()
)

if (
    rounds
    != EXPECTED_BOOSTING_ROUNDS
):
    raise RuntimeError(
        f"Expected "
        f"{EXPECTED_BOOSTING_ROUNDS} "
        f"boosting rounds for run "
        f"{MODEL_RUN_ID}, got "
        f"{rounds}"
    )

if (
    "lighting_condition"
    not in model_names
):
    raise RuntimeError(
        "Model missing "
        "lighting_condition"
    )

cat_idx = (
    model_names
    .index(
        "lighting_condition"
    )
)

if model_types:
    print(
        "lighting_condition type: "
        f"{model_types[cat_idx]}"
    )

print(
    f"Model features:    "
    f"{len(model_names)}"
)

print(
    f"Boosting rounds:   "
    f"{rounds}"
)

print(
    f"Initial lambda:    "
    f"{INITIAL_LAMBDA:.12e}"
)

print(
    f"Initial log lambda:"
    f"{INITIAL_LOG_LAMBDA:.12f}"
)


# ============================================================
# Feature contract
# ============================================================

static_col = {
    name: index
    for (
        index,
        name,
    )
    in enumerate(
        static_names
    )
}


dynamic_r6 = {

    # Calendar
    "local_hour":
        local_hour,

    "local_day_of_week":
        local_dow,

    "local_hour_sin":
        local_hour_sin,

    "local_hour_cos":
        local_hour_cos,

    "local_day_of_week_sin":
        local_dow_sin,

    "local_day_of_week_cos":
        local_dow_cos,

    # Weather
    "weather_temperature_2m_c":
        temperature,

    "weather_relative_humidity_2m_pct":
        humidity,

    "weather_available":
        weather_available,

    # Solar / lighting
    "solar_elevation_deg":
        solar_elevation,

    "solar_azimuth_deg":
        solar_azimuth,

    "is_daylight":
        is_daylight,

    "lighting_condition":
        lighting_model_code,
}


provided_names = (
    set(static_col)
    | set(dynamic_r6)
)

required_names = set(
    model_names
)

missing_features = sorted(
    required_names
    - provided_names
)

if missing_features:
    raise RuntimeError(
        "Serving pipeline is missing "
        "model features: "
        f"{missing_features}"
    )

unused_features = sorted(
    provided_names
    - required_names
)

if unused_features:
    print(
        "Unused serving features: "
        f"{unused_features}"
    )

print(
    "Feature contract:   PASS"
)


# ============================================================
# Missing-value counters
#
# Numeric NaN is valid and is passed directly to XGBoost.
# ============================================================

missing_counts = np.zeros(
    EXPECTED_FEATURE_COUNT,
    dtype=np.uint64,
)


# ============================================================
# Prepare temporary snapshot
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

    # ========================================================
    # Output arrays
    # ========================================================

    log_intensity = (
        np.lib.format.open_memmap(
            tmp_dir
            / "log_intensity.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n,),
        )
    )

    intensity = (
        np.lib.format.open_memmap(
            tmp_dir
            / "intensity.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n,),
        )
    )


    # ========================================================
    # National inference
    # ========================================================

    print()

    print(
        "Running national intensity inference..."
    )

    total_start = (
        time.perf_counter()
    )


    for (
        batch_number,
        start,
    ) in enumerate(
        range(
            0,
            n,
            CHUNK_SIZE,
        ),
        start=1,
    ):

        end = min(
            start
            + CHUNK_SIZE,
            n,
        )

        count = (
            end
            - start
        )

        batch_start = (
            time.perf_counter()
        )


        # ----------------------------------------------------
        # R9 -> R6 gather
        # ----------------------------------------------------

        env_idx = np.asarray(
            r9_to_r6[
                start:end
            ],
            dtype=np.uint32,
        )


        # ----------------------------------------------------
        # Exact 38-column model matrix
        # ----------------------------------------------------

        X_cpu = np.empty(
            (
                count,
                EXPECTED_FEATURE_COUNT,
            ),
            dtype=np.float32,
        )


        for (
            column_index,
            name,
        ) in enumerate(
            model_names
        ):

            if name in static_col:

                X_cpu[
                    :,
                    column_index,
                ] = static_features[
                    start:end,
                    static_col[
                        name
                    ],
                ]

            else:

                X_cpu[
                    :,
                    column_index,
                ] = dynamic_r6[
                    name
                ][
                    env_idx
                ]


        # ----------------------------------------------------
        # Missing-value accounting
        #
        # NaN is valid for numeric features.
        # ----------------------------------------------------

        missing_counts += (
            np.isnan(
                X_cpu
            )
            .sum(
                axis=0,
                dtype=np.uint64,
            )
        )


        # ----------------------------------------------------
        # Input validation
        #
        # NaN:
        #   VALID for numeric XGBoost features.
        #
        # +/- infinity:
        #   NEVER valid.
        #
        # lighting_condition:
        #   MUST be present and in categorical range.
        # ----------------------------------------------------

        inf_mask = np.isinf(
            X_cpu
        )

        if inf_mask.any():

            bad = (
                np.argwhere(
                    inf_mask
                )[0]
            )

            bad_row = int(
                bad[0]
            )

            bad_col = int(
                bad[1]
            )

            raise RuntimeError(
                "Infinite serving feature: "
                f"national_row="
                f"{start + bad_row:,}, "
                f"feature="
                f"{model_names[bad_col]}, "
                f"value="
                f"{X_cpu[bad_row, bad_col]}"
            )


        category_values = (
            X_cpu[
                :,
                cat_idx,
            ]
        )


        category_nan = np.isnan(
            category_values
        )

        if category_nan.any():

            bad_row = int(
                np.flatnonzero(
                    category_nan
                )[0]
            )

            raise RuntimeError(
                "Missing "
                "lighting_condition: "
                f"national_row="
                f"{start + bad_row:,}"
            )


        if (
            np.any(
                category_values
                < 0
            )
            or
            np.any(
                category_values
                >= len(
                    MODEL_LIGHTING_LEVELS
                )
            )
        ):
            raise RuntimeError(
                "Invalid "
                "lighting_condition "
                "model code"
            )

        # ============================================================
        # TEMP DEBUG: capture the exact feature vector used for
        # the known extreme cell.
        # ============================================================

        DEBUG_ROW = 13_984_892

        if start <= DEBUG_ROW < end:
            debug_offset = DEBUG_ROW - start

            debug_payload = {
                "national_row": DEBUG_ROW,
                "h3": int(h3_keys[DEBUG_ROW]),
                "r6_idx": int(env_idx[debug_offset]),
                "features": {
                    name: (
                        None
                        if np.isnan(X_cpu[debug_offset, i])
                        else float(X_cpu[debug_offset, i])
                    )
                    for i, name in enumerate(model_names)
                },
            }

            debug_path = (
                ROOT
                / "debug_exact_max_cell_features.json"
            )

            with debug_path.open("w") as f:
                json.dump(
                    debug_payload,
                    f,
                    indent=2,
                )

            print(
                f"DEBUG exact feature row saved: {debug_path}"
            )
        # ----------------------------------------------------
        # CPU -> GPU
        # ----------------------------------------------------

        X_gpu = cp.asarray(
            X_cpu
        )


        # ----------------------------------------------------
        # Critical training-time base margin
        # ----------------------------------------------------

        base_margin_gpu = cp.full(
            count,
            INITIAL_LOG_LAMBDA,
            dtype=cp.float32,
        )


        # ----------------------------------------------------
        # XGBoost margin = log intensity
        #
        # Reproduces training-time base_margin semantics.
        # ----------------------------------------------------

        margin_gpu = (
            bst.inplace_predict(
                X_gpu,
                predict_type="margin",
                validate_features=False,
                base_margin=base_margin_gpu,
            )
        )


        # ----------------------------------------------------
        # Exact numerical contract from training/evaluation
        # ----------------------------------------------------

        effective_log_lambda_gpu = (
            cp.clip(
                margin_gpu,
                MIN_LOG_INTENSITY,
                MAX_LOG_INTENSITY,
            )
        )

        lambda_gpu = cp.exp(
            effective_log_lambda_gpu
        )


        cp.cuda.runtime.deviceSynchronize()


        # ----------------------------------------------------
        # GPU -> CPU
        # ----------------------------------------------------

        batch_log_lambda = (
            cp.asnumpy(
                effective_log_lambda_gpu
            )
            .astype(
                np.float32,
                copy=False,
            )
        )

        batch_lambda = (
            cp.asnumpy(
                lambda_gpu
            )
            .astype(
                np.float32,
                copy=False,
            )
        )


        # ----------------------------------------------------
        # Output validation
        # ----------------------------------------------------

        if not np.isfinite(
            batch_log_lambda
        ).all():
            raise RuntimeError(
                "Non-finite log intensity "
                f"in batch {batch_number}"
            )

        if not np.isfinite(
            batch_lambda
        ).all():
            raise RuntimeError(
                "Non-finite intensity "
                f"in batch {batch_number}"
            )

        if np.any(
            batch_lambda
            < 0
        ):
            raise RuntimeError(
                "Negative intensity "
                f"in batch {batch_number}"
            )


        # ----------------------------------------------------
        # Persist batch
        # ----------------------------------------------------

        log_intensity[
            start:end
        ] = batch_log_lambda

        intensity[
            start:end
        ] = batch_lambda


        elapsed = (
            time.perf_counter()
            - batch_start
        )


        print(
            f"Batch {batch_number:02d} | "
            f"{start:,}:{end:,} | "
            f"{elapsed:.3f}s | "
            f"{count / elapsed:,.0f} cells/s"
        )


        # ----------------------------------------------------
        # Release batch memory
        # ----------------------------------------------------

        del (
            env_idx,
            X_cpu,
            inf_mask,
            category_values,
            category_nan,
            X_gpu,
            base_margin_gpu,
            margin_gpu,
            effective_log_lambda_gpu,
            lambda_gpu,
            batch_log_lambda,
            batch_lambda,
        )


    log_intensity.flush()
    intensity.flush()


    total_elapsed = (
        time.perf_counter()
        - total_start
    )


    # ========================================================
    # H3 level-of-detail intensity pyramid
    #
    # The model is inferred exactly once at canonical H3-r9.
    # Coarser r4-r8 arrays are deterministic SUM reductions of
    # the already-produced r9 intensity field.
    #
    # This MUST finish before metadata/publication so the
    # current pointer can never reference an incomplete LOD
    # snapshot.
    # ========================================================

    print()
    print(
        "Building H3 LOD intensity pyramid..."
    )

    lod_start = (
        time.perf_counter()
    )

    write_lod_intensities(
        tmp_dir,
        intensity,
    )

    lod_elapsed = (
        time.perf_counter()
        - lod_start
    )

    print(
        "H3 LOD pyramid complete | "
        f"{lod_elapsed:.3f}s"
    )


    # ========================================================
    # Distribution diagnostics
    # ========================================================

    intensity_min = float(
        np.min(
            intensity
        )
    )

    intensity_mean = float(
        np.mean(
            intensity
        )
    )

    intensity_median = float(
        np.median(
            intensity
        )
    )

    intensity_max = float(
        np.max(
            intensity
        )
    )


    max_row = int(
        np.argmax(
            intensity
        )
    )

    max_h3 = int(
        h3_keys[
            max_row
        ]
    )


    quantile_levels = [
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
        0.999,
        0.9999,
        0.99999,
    ]

    quantile_values = (
        np.quantile(
            intensity,
            quantile_levels,
        )
    )

    quantiles = {
        str(q): float(v)
        for (
            q,
            v,
        )
        in zip(
            quantile_levels,
            quantile_values,
            strict=True,
        )
    }


    # ========================================================
    # Missing-feature metadata
    # ========================================================

    missing_feature_counts = {
        name: int(count)
        for (
            name,
            count,
        )
        in zip(
            model_names,
            missing_counts,
            strict=True,
        )
        if count
    }

    missing_feature_rates = {
        name: (
            int(count)
            / n
        )
        for (
            name,
            count,
        )
        in zip(
            model_names,
            missing_counts,
            strict=True,
        )
        if count
    }


    # ========================================================
    # Metadata
    # ========================================================

    metadata = {

        "schema":
            "crimenet_national_intensity_snapshot_v2",

        "snapshot_id":
            snapshot_id,

        "valid_utc_hour":
            valid_utc.isoformat(),

        "rows":
            int(n),

        "h3_resolution":
            9,

        "model_run_id":
            MODEL_RUN_ID,

        "model_path":
            str(
                MODEL_PATH
            ),

        "model_feature_count":
            EXPECTED_FEATURE_COUNT,

        "model_features":
            model_names,

        "model_feature_types":
            model_types,

        "boosting_rounds":
            rounds,

        "initial_lambda":
            INITIAL_LAMBDA,

        "initial_log_lambda":
            INITIAL_LOG_LAMBDA,

        "prediction_base_margin":
            "initial_log_lambda",

        "min_log_intensity":
            MIN_LOG_INTENSITY,

        "max_log_intensity":
            MAX_LOG_INTENSITY,

        "log_intensity_definition":
            (
                "clip("
                "xgboost_margin_with_initial_log_lambda_base_margin,"
                "-30,15)"
            ),

        "intensity_definition":
            "exp(log_intensity)",

        "intensity_units":
            "events_per_second",

        "numeric_missing_value_policy":
            (
                "preserve_nan_and_delegate_to_"
                "xgboost_missing_value_routing"
            ),

        "categorical_missing_value_policy":
            (
                "lighting_condition_must_be_present"
            ),

        "missing_feature_counts":
            missing_feature_counts,

        "missing_feature_rates":
            missing_feature_rates,

        "environmental_snapshot_path":
            str(
                env_dir
            ),

        "environmental_valid_utc_hour":
            valid_utc.isoformat(),

        "chunk_size":
            CHUNK_SIZE,

        "total_seconds":
            total_elapsed,

        "throughput_cells_per_second":
            n
            / total_elapsed,

        "lod": {

            "schema":
                "crimenet_intensity_lod_v1",

            "source_resolution":
                9,

            "resolutions":
                [4, 5, 6, 7, 8, 9],

            "coarse_aggregation":
                "sum_r9_child_intensity",

            "visualization_metric":
                "mean_r9_child_intensity",

            "intensity_units":
                "events_per_second",

            "generation_seconds":
                lod_elapsed,

            "metadata_path":
                "lod/metadata.json",
        },

        "distribution": {

            "min":
                intensity_min,

            "mean":
                intensity_mean,

            "median":
                intensity_median,

            "max":
                intensity_max,

            "quantiles":
                quantiles,

            "max_row":
                max_row,

            "max_h3":
                max_h3,

            "max_events_per_hour":
                intensity_max
                * 3600.0,
        },

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


    # ========================================================
    # Publish under canonical timestamp.
    #
    # Existing snapshot with the same timestamp is replaced only
    # after the new one has completed successfully.
    # ========================================================

    publish_directory(
        tmp_dir,
        final_dir,
    )


    # ========================================================
    # Update current pointer only after successful publication.
    # ========================================================

    pointer = {

        "schema":
            "crimenet_national_intensity_pointer_v1",

        "snapshot_id":
            snapshot_id,

        "snapshot_path":
            str(
                final_dir
            ),

        "valid_utc_hour":
            valid_utc.isoformat(),

        "model_run_id":
            MODEL_RUN_ID,

        "published_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }


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
    "NATIONAL INTENSITY SNAPSHOT"
)

print(
    "========================================"
)

print(
    f"UTC hour:       "
    f"{valid_utc.isoformat()}"
)

print(
    f"Cells:          "
    f"{n:,}"
)

print(
    f"Total time:     "
    f"{total_elapsed:.3f}s"
)

print(
    f"Throughput:     "
    f"{n / total_elapsed:,.0f} cells/s"
)

print(
    f"LOD build time: "
    f"{lod_elapsed:.3f}s"
)


print()

print(
    "Intensity — events/second"
)

print(
    f"  min:          "
    f"{intensity_min:.12g}"
)

print(
    f"  mean:         "
    f"{intensity_mean:.12g}"
)

print(
    f"  median:       "
    f"{intensity_median:.12g}"
)

print(
    f"  max:          "
    f"{intensity_max:.12g}"
)


print()

print(
    "Intensity — events/hour"
)

print(
    f"  mean:         "
    f"{intensity_mean * 3600:.12g}"
)

print(
    f"  median:       "
    f"{intensity_median * 3600:.12g}"
)

print(
    f"  max:          "
    f"{intensity_max * 3600:.12g}"
)


print()

print(
    "Quantiles"
)

for (
    q,
    value,
) in zip(
    quantile_levels,
    quantile_values,
    strict=True,
):

    print(
        f"  {q:8.5f}: "
        f"{value:.12g}"
    )


print()

print(
    "Missing feature coverage"
)

if not missing_feature_counts:

    print(
        "  none"
    )

else:

    for (
        name,
        count,
    ) in zip(
        model_names,
        missing_counts,
        strict=True,
    ):

        if not count:
            continue

        print(
            f"  {name:42s} "
            f"{int(count):10,d} "
            f"{int(count) / n:10.4%}"
        )


print()

print(
    "Maximum cell"
)

print(
    f"  row:          "
    f"{max_row:,}"
)

print(
    f"  h3:           "
    f"{max_h3}"
)

print(
    f"  lambda/sec:   "
    f"{intensity_max:.12g}"
)

print(
    f"  lambda/hour:  "
    f"{intensity_max * 3600:.12g}"
)


print()

print(
    f"Snapshot:       "
    f"{final_dir}"
)

print(
    f"Current pointer:"
    f" {CURRENT_POINTER}"
)

print()

print(
    "PUBLISHED"
)