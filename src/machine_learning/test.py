from __future__ import annotations

import json
from pathlib import Path
from zoneinfo import ZoneInfo

import cupy as cp
import h3
import numpy as np
import pandas as pd
import xgboost as xgb


# ============================================================
# Configuration
# ============================================================

ROOT = (
    Path.home()
    / "crimenet-serving"
    / "data"
    / "national_feature_store"
)

STATIC = ROOT / "mmap"
ENV = ROOT / "environmental"

MODEL_PATH = (
    Path.home()
    / "crimenet-serving"
    / "models"
    / "intensity"
    / "model.ubj"
)

SNAPSHOT = (
    ROOT
    / "intensity_snapshots"
    / "20260829T2000"
)

DEBUG_FEATURE_PATH = (
    ROOT
    / "debug_exact_max_cell_features.json"
)

INITIAL_LOG_LAMBDA = -15.474814268832395
MIN_LOG_INTENSITY = -30.0
MAX_LOG_INTENSITY = 15.0

CELLS = [
    "8929a115dd7ffff",  # extreme
    "8929a115dc3ffff",  # same-tract neighbor
    "8929a115dc7ffff",  # same-tract neighbor
]

MODEL_LIGHTING_LEVELS = [
    "astronomical_twilight",
    "civil_twilight",
    "day",
    "nautical_twilight",
    "night",
]

# environmental snapshot code:
#
# 0 night
# 1 astronomical_twilight
# 2 nautical_twilight
# 3 civil_twilight
# 4 daylight
#
# model categorical code:
#
# 0 astronomical_twilight
# 1 civil_twilight
# 2 day
# 3 nautical_twilight
# 4 night
LIGHTING_TRANSLATION = np.asarray(
    [4, 0, 3, 1, 2],
    dtype=np.float32,
)


# ============================================================
# Helpers
# ============================================================

def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def load_array(path: Path) -> np.ndarray:
    return np.load(
        path,
        mmap_mode="r",
    )


def hourly(log_lambda: float) -> float:
    return float(
        np.exp(
            np.clip(
                log_lambda,
                MIN_LOG_INTENSITY,
                MAX_LOG_INTENSITY,
            )
        )
        * 3600.0
    )


# ============================================================
# Snapshot provenance
# ============================================================

intensity_meta = load_json(
    SNAPSHOT / "metadata.json"
)

env_dir = Path(
    intensity_meta[
        "environmental_snapshot_path"
    ]
)

valid_utc = pd.Timestamp(
    intensity_meta[
        "environmental_valid_utc_hour"
    ]
)

if valid_utc.tzinfo is None:
    valid_utc = valid_utc.tz_localize(
        "UTC"
    )
else:
    valid_utc = valid_utc.tz_convert(
        "UTC"
    )

print(
    f"Intensity snapshot: "
    f"{SNAPSHOT}"
)

print(
    f"Environmental snapshot: "
    f"{env_dir}"
)

print(
    f"Valid UTC: "
    f"{valid_utc.isoformat()}"
)


# ============================================================
# Static feature store
# ============================================================

keys = load_array(
    STATIC / "h3_keys.npy"
)

static_features = load_array(
    STATIC / "features.npy"
)

static_meta = load_json(
    STATIC / "metadata.json"
)

static_names = list(
    static_meta["features"]
)

static_col = {
    name: i
    for i, name
    in enumerate(static_names)
}


# ============================================================
# Locate target cells
# ============================================================

cell_ints = np.asarray(
    [
        h3.str_to_int(cell)
        for cell in CELLS
    ],
    dtype=np.uint64,
)

rows = np.searchsorted(
    keys,
    cell_ints,
)

for cell, key, row in zip(
    CELLS,
    cell_ints,
    rows,
    strict=True,
):
    if (
        row >= len(keys)
        or keys[row] != key
    ):
        raise RuntimeError(
            f"Missing H3 cell: {cell}"
        )


# ============================================================
# Permanent R9 -> R6 mapping
# ============================================================

r9_to_r6 = load_array(
    ENV / "r9_to_r6_idx.npy"
)

r6_idx = np.asarray(
    r9_to_r6[rows],
    dtype=np.uint32,
)

print()

print("R6 INDICES")

for cell, idx in zip(
    CELLS,
    r6_idx,
    strict=True,
):
    print(
        f"{cell} -> {int(idx)}"
    )


# ============================================================
# Environmental arrays
# ============================================================

def env_array(filename: str) -> np.ndarray:
    return load_array(
        env_dir / filename
    )


temperature = env_array(
    "temperature_2m.npy"
)

humidity = env_array(
    "relative_humidity_2m.npy"
)

weather_available = env_array(
    "weather_available.npy"
)

solar_elevation = env_array(
    "solar_elevation_deg.npy"
)

solar_azimuth = env_array(
    "solar_azimuth_deg.npy"
)

is_daylight = env_array(
    "is_daylight.npy"
)

lighting_serving_code = env_array(
    "lighting_condition_code.npy"
)


# ============================================================
# Timezone + calendar
#
# IMPORTANT:
# These expressions intentionally match
# build_national_intensity.py exactly.
# ============================================================

r6_timezone_id = load_array(
    ENV / "r6_timezone_id.npy"
)

timezone_meta = load_json(
    ENV / "r6_timezones.json"
)

timezone_names = {
    int(k): v
    for k, v
    in timezone_meta["timezones"].items()
}

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

for tz_id, tz_name in (
    timezone_names.items()
):
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


# EXACT publisher expressions.
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
# Lighting categorical encoding
# ============================================================

lighting_codes_int = np.asarray(
    lighting_serving_code,
    dtype=np.int64,
)

if (
    np.min(lighting_codes_int) < 0
    or
    np.max(lighting_codes_int)
    >= len(LIGHTING_TRANSLATION)
):
    raise RuntimeError(
        "Invalid environmental "
        "lighting code"
    )

lighting_model_code = (
    LIGHTING_TRANSLATION[
        lighting_codes_int
    ]
)


# ============================================================
# Model
# ============================================================

bst = xgb.Booster()

bst.load_model(
    MODEL_PATH
)

bst.set_param(
    {
        "device": "cuda:0",
    }
)

model_names = list(
    bst.feature_names
)

model_types = list(
    bst.feature_types
)

if len(model_names) != 38:
    raise RuntimeError(
        f"Expected 38 features, "
        f"got {len(model_names)}"
    )

cat_idx = model_names.index(
    "lighting_condition"
)


# ============================================================
# Dynamic R6 feature map
# ============================================================

dynamic_r6 = {
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

    "weather_temperature_2m_c":
        temperature,

    "weather_relative_humidity_2m_pct":
        humidity,

    "weather_available":
        weather_available,

    "solar_elevation_deg":
        solar_elevation,

    "solar_azimuth_deg":
        solar_azimuth,

    "is_daylight":
        is_daylight,

    "lighting_condition":
        lighting_model_code,
}


# ============================================================
# Build exact three feature rows
# ============================================================

X = np.empty(
    (
        len(CELLS),
        len(model_names),
    ),
    dtype=np.float32,
)

for i, (
    row,
    env_i,
) in enumerate(
    zip(
        rows,
        r6_idx,
        strict=True,
    )
):
    env_i = int(
        env_i
    )

    for j, name in enumerate(
        model_names
    ):
        if name in static_col:
            X[
                i,
                j,
            ] = static_features[
                row,
                static_col[name],
            ]

        else:
            X[
                i,
                j,
            ] = dynamic_r6[
                name
            ][
                env_i
            ]


# ============================================================
# Exact publisher-row verification
# ============================================================

print()

print(
    "============================================================"
)

print(
    "PUBLISHER FEATURE VECTOR PARITY"
)

print(
    "============================================================"
)


if DEBUG_FEATURE_PATH.exists():

    debug = load_json(
        DEBUG_FEATURE_PATH
    )

    debug_row = int(
        debug["national_row"]
    )

    if debug_row != int(rows[0]):
        raise RuntimeError(
            "Debug feature vector is not "
            "for the expected extreme cell"
        )

    feature_differences = []

    max_diff = 0.0

    for j, name in enumerate(
        model_names
    ):
        expected = (
            debug[
                "features"
            ][
                name
            ]
        )

        actual = float(
            X[
                0,
                j,
            ]
        )

        if expected is None:
            matches = np.isnan(
                actual
            )

            diff = (
                0.0
                if matches
                else np.inf
            )

        elif np.isnan(actual):
            diff = np.inf

        else:
            diff = abs(
                actual
                - float(expected)
            )

        max_diff = max(
            max_diff,
            diff,
        )

        if diff != 0.0:
            feature_differences.append(
                (
                    name,
                    expected,
                    actual,
                    diff,
                )
            )


    print(
        "max absolute feature difference:",
        max_diff,
    )

    if feature_differences:
        print()

        print(
            "Differing features:"
        )

        for (
            name,
            expected,
            actual,
            diff,
        ) in feature_differences:

            print(
                f"{name:42s} "
                f"publisher={expected!r} "
                f"reconstructed={actual!r} "
                f"diff={diff:.12g}"
            )

    else:
        print(
            "Exact feature vector: PASS"
        )

else:
    print(
        "Debug feature dump not found; "
        "skipping publisher-vector check."
    )


# ============================================================
# A. Exact production inference path
# ============================================================

base_margin = np.full(
    len(CELLS),
    INITIAL_LOG_LAMBDA,
    dtype=np.float32,
)

X_gpu = cp.asarray(
    X,
    dtype=cp.float32,
)

base_margin_gpu = cp.full(
    len(CELLS),
    INITIAL_LOG_LAMBDA,
    dtype=cp.float32,
)

margin_inplace = cp.asnumpy(
    bst.inplace_predict(
        X_gpu,
        predict_type="margin",
        validate_features=False,
        base_margin=base_margin_gpu,
    )
)


# ============================================================
# B. Raw DMatrix
# ============================================================

dmat_raw = xgb.DMatrix(
    X,
    feature_names=model_names,
    feature_types=model_types,
    base_margin=base_margin,
    enable_categorical=True,
)

margin_raw = bst.predict(
    dmat_raw,
    output_margin=True,
)


# ============================================================
# C. Recoded categorical DMatrix
# ============================================================

X_df = pd.DataFrame(
    X,
    columns=model_names,
)

lighting_codes = (
    X[
        :,
        cat_idx,
    ]
    .astype(
        np.int32
    )
)

lighting_strings = [
    MODEL_LIGHTING_LEVELS[
        code
    ]
    for code in lighting_codes
]

X_df[
    "lighting_condition"
] = pd.Categorical(
    lighting_strings,
    categories=MODEL_LIGHTING_LEVELS,
)

model_categories = (
    bst.get_categories()
)

dmat_recoded = xgb.DMatrix(
    X_df,
    feature_types=model_categories,
    enable_categorical=True,
    base_margin=base_margin,
)

margin_recoded = bst.predict(
    dmat_recoded,
    output_margin=True,
)


# ============================================================
# Stored snapshot
# ============================================================

snapshot_log_all = load_array(
    SNAPSHOT
    / "log_intensity.npy"
)

snapshot_intensity_all = load_array(
    SNAPSHOT
    / "intensity.npy"
)

snapshot_log = np.asarray(
    snapshot_log_all[
        rows
    ],
    dtype=np.float32,
)


# ============================================================
# Prediction parity
# ============================================================

print()

print(
    "============================================================"
)

print(
    "PREDICTION PATH PARITY"
)

print(
    "============================================================"
)

for i, cell in enumerate(
    CELLS
):

    print()

    print(
        cell
    )

    print(
        f"  snapshot        "
        f"log={snapshot_log[i]: .9f} "
        f"hr={float(snapshot_intensity_all[rows[i]]) * 3600:.9f}"
    )

    print(
        f"  inplace/cupy    "
        f"log={margin_inplace[i]: .9f} "
        f"hr={hourly(margin_inplace[i]):.9f}"
    )

    print(
        f"  raw DMatrix     "
        f"log={margin_raw[i]: .9f} "
        f"hr={hourly(margin_raw[i]):.9f}"
    )

    print(
        f"  recoded DMatrix "
        f"log={margin_recoded[i]: .9f} "
        f"hr={hourly(margin_recoded[i]):.9f}"
    )


snapshot_vs_inplace = float(
    np.max(
        np.abs(
            snapshot_log
            - margin_inplace
        )
    )
)

inplace_vs_raw = float(
    np.max(
        np.abs(
            margin_inplace
            - margin_raw
        )
    )
)

inplace_vs_recoded = float(
    np.max(
        np.abs(
            margin_inplace
            - margin_recoded
        )
    )
)


print()

print(
    "MAX ABS DIFFERENCES"
)

print(
    "snapshot vs inplace:       ",
    snapshot_vs_inplace,
)

print(
    "inplace vs raw DMatrix:    ",
    inplace_vs_raw,
)

print(
    "inplace vs recoded DMatrix:",
    inplace_vs_recoded,
)


if not np.allclose(
    snapshot_log,
    margin_inplace,
    atol=1e-6,
    rtol=0.0,
):
    raise RuntimeError(
        "Feature reconstruction does not "
        "reproduce the published snapshot."
    )


print()

print(
    "Snapshot reconstruction: PASS"
)


# ============================================================
# Choose parity-safe SHAP DMatrix
# ============================================================

if np.allclose(
    margin_inplace,
    margin_recoded,
    atol=1e-6,
    rtol=0.0,
):

    shap_dmatrix = (
        dmat_recoded
    )

    shap_path = (
        "recoded DMatrix"
    )

elif np.allclose(
    margin_inplace,
    margin_raw,
    atol=1e-6,
    rtol=0.0,
):

    shap_dmatrix = (
        dmat_raw
    )

    shap_path = (
        "raw DMatrix"
    )

else:
    raise RuntimeError(
        "No DMatrix prediction path "
        "matches production inference."
    )


print(
    f"SHAP path: {shap_path}"
)


# ============================================================
# SHAP contributions
# ============================================================

contrib = bst.predict(
    shap_dmatrix,
    pred_contribs=True,
)

shap_sum = (
    contrib.sum(
        axis=1
    )
)

shap_error = float(
    np.max(
        np.abs(
            shap_sum
            - margin_inplace
        )
    )
)


print()

print(
    "SHAP reconstruction max error:",
    shap_error,
)

if shap_error > 1e-4:
    raise RuntimeError(
        "SHAP contributions do not "
        "reconstruct production margins."
    )


# ============================================================
# Extreme vs same-tract neighbors
# ============================================================

extreme = (
    contrib[
        0,
        :-1,
    ]
)

neighbor_mean = (
    contrib[
        1:,
        :-1,
    ]
    .mean(
        axis=0
    )
)

delta = (
    extreme
    - neighbor_mean
)

order = np.argsort(
    np.abs(
        delta
    )
)[::-1]


print()

print(
    "============================================================"
)

print(
    "FEATURE CONTRIBUTION DIFFERENCES"
)

print(
    "extreme SHAP - mean neighbor SHAP"
)

print(
    "============================================================"
)

for idx in order:

    if abs(
        delta[idx]
    ) < 1e-5:
        continue

    print(
        f"{model_names[idx]:42s} "
        f"extreme={extreme[idx]: 10.6f} "
        f"neighbor={neighbor_mean[idx]: 10.6f} "
        f"delta={delta[idx]: 10.6f}"
    )


# ============================================================
# Extreme cell contributions
# ============================================================

print()

print(
    "============================================================"
)

print(
    "EXTREME CELL SHAP CONTRIBUTIONS"
)

print(
    "============================================================"
)

extreme_order = np.argsort(
    np.abs(
        extreme
    )
)[::-1]

for idx in extreme_order:

    if abs(
        extreme[idx]
    ) < 1e-5:
        continue

    print(
        f"{model_names[idx]:42s} "
        f"{extreme[idx]: 10.6f}"
    )


print()

print(
    f"Bias contribution: "
    f"{contrib[0, -1]:.8f}"
)


# ============================================================
# Leaf IDs
# ============================================================

leaves = bst.predict(
    shap_dmatrix,
    pred_leaf=True,
)

if leaves.ndim == 1:
    leaves = leaves.reshape(
        len(CELLS),
        -1,
    )


different_trees = []

for tree_idx in range(
    leaves.shape[1]
):

    values = (
        leaves[
            :,
            tree_idx,
        ]
    )

    if not (
        values[0] == values[1]
        and
        values[0] == values[2]
    ):
        different_trees.append(
            (
                tree_idx,
                int(values[0]),
                int(values[1]),
                int(values[2]),
            )
        )


print()

print(
    "============================================================"
)

print(
    "TREE LEAF DIFFERENCES"
)

print(
    "============================================================"
)

print(
    f"Trees with differing leaves: "
    f"{len(different_trees)} / "
    f"{leaves.shape[1]}"
)

for (
    tree_idx,
    extreme_leaf,
    neighbor1_leaf,
    neighbor2_leaf,
) in different_trees:

    print(
        f"tree={tree_idx:02d} "
        f"extreme={extreme_leaf:5d} "
        f"neighbor1={neighbor1_leaf:5d} "
        f"neighbor2={neighbor2_leaf:5d}"
    )


print()

print("PASS")