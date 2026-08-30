from pathlib import Path
from zoneinfo import ZoneInfo
import json

import cupy as cp
import h3
import numpy as np
import pandas as pd
import xgboost as xgb


ROOT = Path.home() / "crimenet-serving/data/national_feature_store"
STATIC = ROOT / "mmap"
ENV = ROOT / "environmental"

MODEL_PATH = (
    Path.home()
    / "crimenet-serving/models/intensity/model.ubj"
)

INITIAL_LOG_LAMBDA = -15.474814268832395

CELLS = [
    "8929a115dd7ffff",  # extreme
    "8929a115dc3ffff",  # same tract neighbor
    "8929a115dc7ffff",  # same tract neighbor
]


# ============================================================
# Load static store
# ============================================================

keys = np.load(
    STATIC / "h3_keys.npy",
    mmap_mode="r",
)

features = np.load(
    STATIC / "features.npy",
    mmap_mode="r",
)

with open(STATIC / "metadata.json") as f:
    static_meta = json.load(f)

static_names = static_meta["features"]
static_col = {
    name: i
    for i, name in enumerate(static_names)
}

cell_ints = np.asarray(
    [h3.str_to_int(x) for x in CELLS],
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
    if row >= len(keys) or keys[row] != key:
        raise RuntimeError(
            f"Missing H3 cell: {cell}"
        )


# ============================================================
# R9 -> R6
# ============================================================

r9_to_r6 = np.load(
    ENV / "r9_to_r6_idx.npy",
    mmap_mode="r",
)

r6_idx = np.asarray(
    r9_to_r6[rows],
    dtype=np.uint32,
)


# ============================================================
# Environmental snapshot
# ============================================================
# Use the EXACT environmental snapshot that produced the
# published intensity snapshot.
# ============================================================

SNAPSHOT = (
    ROOT
    / "intensity_snapshots"
    / "20260829T2000"
)

with open(
    SNAPSHOT / "metadata.json"
) as f:
    intensity_meta = json.load(f)

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

print("Intensity snapshot:", SNAPSHOT)
print("Recorded environmental snapshot:", env_dir)
print("Recorded valid UTC:", valid_utc)

if valid_utc.tzinfo is None:
    valid_utc = valid_utc.tz_localize("UTC")
else:
    valid_utc = valid_utc.tz_convert("UTC")


def arr(name):
    return np.load(
        env_dir / name,
        mmap_mode="r",
    )


temperature = arr("temperature_2m.npy")
humidity = arr("relative_humidity_2m.npy")
weather_available = arr("weather_available.npy")
solar_elevation = arr("solar_elevation_deg.npy")
solar_azimuth = arr("solar_azimuth_deg.npy")
is_daylight = arr("is_daylight.npy")
lighting = arr("lighting_condition_code.npy")


# ============================================================
# Calendar
# ============================================================

timezone_ids = np.load(
    ENV / "r6_timezone_id.npy",
    mmap_mode="r",
)

with open(ENV / "r6_timezones.json") as f:
    timezone_meta = json.load(f)

timezone_names = {
    int(k): v
    for k, v in timezone_meta["timezones"].items()
}

calendar = {}

for idx in np.unique(r6_idx):
    idx = int(idx)

    timezone_id = int(
        timezone_ids[idx]
    )

    timezone_name = (
        timezone_names[
            timezone_id
        ]
    )

    local = (
        valid_utc
        .to_pydatetime()
        .astimezone(
            ZoneInfo(
                timezone_name
            )
        )
    )

    hour = float(
        local.hour
    )

    dow = float(
        local.weekday()
    )

    calendar[idx] = {
        "local_hour":
            hour,

        "local_day_of_week":
            dow,

        "local_hour_sin":
            np.sin(
                2 * np.pi
                * hour / 24
            ),

        "local_hour_cos":
            np.cos(
                2 * np.pi
                * hour / 24
            ),

        "local_day_of_week_sin":
            np.sin(
                2 * np.pi
                * dow / 7
            ),

        "local_day_of_week_cos":
            np.cos(
                2 * np.pi
                * dow / 7
            ),
    }


# Exact serving -> model lighting codes.
lighting_translation = np.asarray(
    [4, 0, 3, 1, 2],
    dtype=np.float32,
)


# ============================================================
# Model
# ============================================================

bst = xgb.Booster()
bst.load_model(
    MODEL_PATH
)

model_names = list(
    bst.feature_names
)

model_types = list(
    bst.feature_types
)


# ============================================================
# Exact three feature rows
# ============================================================

X = np.empty(
    (
        len(CELLS),
        len(model_names),
    ),
    dtype=np.float32,
)

for i, (row, env_i) in enumerate(
    zip(
        rows,
        r6_idx,
        strict=True,
    )
):
    env_i = int(env_i)

    dynamic = {
        **calendar[env_i],

        "weather_temperature_2m_c":
            float(
                temperature[env_i]
            ),

        "weather_relative_humidity_2m_pct":
            float(
                humidity[env_i]
            ),

        "weather_available":
            float(
                weather_available[env_i]
            ),

        "solar_elevation_deg":
            float(
                solar_elevation[env_i]
            ),

        "solar_azimuth_deg":
            float(
                solar_azimuth[env_i]
            ),

        "is_daylight":
            float(
                is_daylight[env_i]
            ),

        "lighting_condition":
            float(
                lighting_translation[
                    int(
                        lighting[env_i]
                    )
                ]
            ),
    }

    for j, name in enumerate(
        model_names
    ):
        if name in static_col:
            X[i, j] = features[
                row,
                static_col[name],
            ]
        else:
            X[i, j] = dynamic[name]


# ============================================================
# Prediction-path parity test
# ============================================================

cat_idx = model_names.index("lighting_condition")

MODEL_LIGHTING_LEVELS = [
    "astronomical_twilight",
    "civil_twilight",
    "day",
    "nautical_twilight",
    "night",
]

base_margin = np.full(
    len(CELLS),
    INITIAL_LOG_LAMBDA,
    dtype=np.float32,
)


# ============================================================
# A. Exact production serving path
# ============================================================

bst.set_param(
    {
        "device": "cuda:0",
    }
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

margin_raw_dmatrix = bst.predict(
    dmat_raw,
    output_margin=True,
)


# ============================================================
# C. Categorically recoded DMatrix
# ============================================================

X_df = pd.DataFrame(
    X,
    columns=model_names,
)

lighting_codes = (
    X[:, cat_idx]
    .astype(np.int32)
)

if (
    np.any(lighting_codes < 0)
    or
    np.any(
        lighting_codes
        >= len(MODEL_LIGHTING_LEVELS)
    )
):
    raise RuntimeError(
        f"Invalid lighting codes: "
        f"{lighting_codes}"
    )

lighting_strings = [
    MODEL_LIGHTING_LEVELS[code]
    for code in lighting_codes
]

X_df["lighting_condition"] = (
    pd.Categorical(
        lighting_strings,
        categories=MODEL_LIGHTING_LEVELS,
    )
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
# D. Actual published national snapshot
# ============================================================

snapshot_intensity = np.load(
    SNAPSHOT
    / "intensity.npy",
    mmap_mode="r",
)

snapshot_log_all = np.load(
    SNAPSHOT
    / "log_intensity.npy",
    mmap_mode="r",
)

snapshot_log = (
    snapshot_log_all[rows]
    .astype(
        np.float32,
        copy=False,
    )
)


# ============================================================
# Prediction parity report
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
        f"hr="
        f"{snapshot_intensity[rows[i]] * 3600:.9f}"
    )

    print(
        f"  inplace/cupy    "
        f"log={margin_inplace[i]: .9f} "
        f"hr="
        f"{np.exp(np.clip(margin_inplace[i], -30, 15)) * 3600:.9f}"
    )

    print(
        f"  raw DMatrix     "
        f"log={margin_raw_dmatrix[i]: .9f} "
        f"hr="
        f"{np.exp(np.clip(margin_raw_dmatrix[i], -30, 15)) * 3600:.9f}"
    )

    print(
        f"  recoded DMatrix "
        f"log={margin_recoded[i]: .9f} "
        f"hr="
        f"{np.exp(np.clip(margin_recoded[i], -30, 15)) * 3600:.9f}"
    )


# ============================================================
# Differences
# ============================================================

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
            - margin_raw_dmatrix
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


# ============================================================
# The reconstructed feature rows MUST reproduce the actual
# national production serving result.
# ============================================================

if not np.allclose(
    snapshot_log,
    margin_inplace,
    atol=1e-6,
    rtol=0.0,
):

    raise RuntimeError(
        "Reconstructed feature rows do NOT "
        "reproduce the published serving snapshot. "
        "The feature reconstruction differs from "
        "build_national_intensity.py."
    )

print()

print(
    "snapshot vs inplace: PASS"
)


# ============================================================
# Pick a DMatrix path that exactly reproduces production.
# SHAP and pred_leaf require DMatrix prediction.
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
    margin_raw_dmatrix,
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
        "Production serving matches the snapshot, "
        "but neither DMatrix representation matches "
        "production inference. Do not interpret SHAP."
    )


print(
    f"SHAP prediction path: "
    f"{shap_path}"
)


# ============================================================
# SHAP
# ============================================================

contrib = bst.predict(
    shap_dmatrix,
    pred_contribs=True,
)

reconstruction = (
    contrib.sum(
        axis=1
    )
)

shap_error = float(
    np.max(
        np.abs(
            reconstruction
            - margin_inplace
        )
    )
)

print()

print(
    "SHAP vs serving max error:",
    shap_error,
)

if shap_error > 1e-4:

    raise RuntimeError(
        "SHAP contributions do not "
        "reconstruct production margins."
    )


# ============================================================
# Extreme vs mean of same-tract neighbors
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
    "========================================"
)

print(
    "FEATURE CONTRIBUTION DIFFERENCES"
)

print(
    "extreme SHAP - mean neighbor SHAP"
)

print(
    "========================================"
)


for idx in order:

    if (
        abs(delta[idx])
        < 1e-5
    ):
        continue

    print(
        f"{model_names[idx]:42s} "
        f"extreme="
        f"{extreme[idx]: 10.6f} "
        f"neighbor="
        f"{neighbor_mean[idx]: 10.6f} "
        f"delta="
        f"{delta[idx]: 10.6f}"
    )


# ============================================================
# Extreme-cell SHAP
# ============================================================

print()

print(
    "========================================"
)

print(
    "EXTREME CELL SHAP CONTRIBUTIONS"
)

print(
    "========================================"
)


extreme_order = np.argsort(
    np.abs(
        extreme
    )
)[::-1]


for idx in extreme_order:

    if (
        abs(extreme[idx])
        < 1e-5
    ):
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

    leaves = (
        leaves.reshape(
            len(CELLS),
            -1,
        )
    )


print()

print(
    "========================================"
)

print(
    "TREE LEAF DIFFERENCES"
)

print(
    "========================================"
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


print(
    f"Trees with differing leaves: "
    f"{len(different_trees)} "
    f"/ {leaves.shape[1]}"
)


for (
    tree,
    extreme_leaf,
    neighbor1_leaf,
    neighbor2_leaf,
) in different_trees:

    print(
        f"tree={tree:02d} "
        f"extreme={extreme_leaf:4d} "
        f"neighbor1={neighbor1_leaf:4d} "
        f"neighbor2={neighbor2_leaf:4d}"
    )


print()

print(
    "PASS"
)
