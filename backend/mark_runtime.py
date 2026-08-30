from __future__ import annotations

import json
import threading
from collections import OrderedDict
from pathlib import Path
from zoneinfo import ZoneInfo

import cupy as cp
import h3
import numpy as np
import pandas as pd
import xgboost as xgb


# ============================================================
# Paths / model contract
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

STATIC_ROOT = (
    ROOT
    / "mmap"
)

ENV_ROOT = (
    ROOT
    / "environmental"
)

MARK_MODEL_PATH = (
    SERVING_ROOT
    / "models"
    / "mark"
    / "model.ubj"
)

MARK_LABELS_PATH = (
    SERVING_ROOT
    / "models"
    / "mark"
    / "class_labels.json"
)

MARK_MODEL_RUN_ID = (
    "7efda77cdaec4a66a30321ea50b12ec8"
)

EXPECTED_FEATURE_COUNT = 38
EXPECTED_NUM_CLASSES = 87

CACHE_SIZE = 10_000


# Environmental serving code:
#
# 0 night
# 1 astronomical_twilight
# 2 nautical_twilight
# 3 civil_twilight
# 4 daylight
#
# Exact model categorical order:
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

def load_json(
    path: Path,
) -> dict | list:

    with path.open("r") as f:
        return json.load(f)


def normalize_utc(
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


# ============================================================
# Runtime
# ============================================================

class MarkRuntime:

    def __init__(self) -> None:

        # ----------------------------------------------------
        # Static national store
        # ----------------------------------------------------

        self.h3_keys = np.load(
            STATIC_ROOT
            / "h3_keys.npy",
            mmap_mode="r",
        )

        self.static_features = np.load(
            STATIC_ROOT
            / "features.npy",
            mmap_mode="r",
        )

        static_meta = load_json(
            STATIC_ROOT
            / "metadata.json"
        )

        self.static_names = list(
            static_meta[
                "features"
            ]
        )

        self.static_col = {
            name: i
            for i, name
            in enumerate(
                self.static_names
            )
        }


        # ----------------------------------------------------
        # Permanent R9 -> R6 mapping
        # ----------------------------------------------------

        self.r9_to_r6 = np.load(
            ENV_ROOT
            / "r9_to_r6_idx.npy",
            mmap_mode="r",
        )

        self.r6_timezone_id = np.load(
            ENV_ROOT
            / "r6_timezone_id.npy",
            mmap_mode="r",
        )

        timezone_meta = load_json(
            ENV_ROOT
            / "r6_timezones.json"
        )

        self.timezone_names = {
            int(k): v
            for k, v
            in timezone_meta[
                "timezones"
            ].items()
        }


        # ----------------------------------------------------
        # Mark booster
        # ----------------------------------------------------

        if not MARK_MODEL_PATH.exists():

            raise FileNotFoundError(
                MARK_MODEL_PATH
            )


        print(
            "Loading mark model...",
            flush=True,
        )

        self.bst = xgb.Booster()

        self.bst.load_model(
            MARK_MODEL_PATH
        )

        self.bst.set_param(
            {
                "device":
                    "cuda:0",
            }
        )


        self.model_names = list(
            self.bst.feature_names
            or []
        )

        self.model_types = list(
            self.bst.feature_types
            or []
        )


        if (
            len(
                self.model_names
            )
            != EXPECTED_FEATURE_COUNT
        ):

            raise RuntimeError(
                "Unexpected mark feature count: "
                f"{len(self.model_names)}"
            )


        config = json.loads(
            self.bst.save_config()
        )

        self.num_classes = int(
            config[
                "learner"
            ][
                "learner_model_param"
            ][
                "num_class"
            ]
        )


        if (
            self.num_classes
            != EXPECTED_NUM_CLASSES
        ):

            raise RuntimeError(
                "Unexpected mark class count: "
                f"{self.num_classes}"
            )


        if (
            "lighting_condition"
            not in self.model_names
        ):

            raise RuntimeError(
                "Mark model missing "
                "lighting_condition"
            )


        self.cat_idx = (
            self.model_names
            .index(
                "lighting_condition"
            )
        )


        # ----------------------------------------------------
        # Target labels
        # ----------------------------------------------------

        (
            self.class_labels,
            self.labels_available,
        ) = self._load_labels()


        # ----------------------------------------------------
        # Dynamic snapshot state
        # ----------------------------------------------------

        self.snapshot_id = None
        self.env_dir = None
        self.valid_utc = None

        self.temperature = None
        self.humidity = None
        self.weather_available = None
        self.solar_elevation = None
        self.solar_azimuth = None
        self.is_daylight = None
        self.lighting_code = None


        # State protects snapshot changes.
        self.state_lock = (
            threading.RLock()
        )

        # One GPU traversal at a time.
        self.gpu_lock = (
            threading.Lock()
        )

        # Cache:
        #
        # (snapshot_id, h3)
        #     -> full 87-probability vector
        self.cache_lock = (
            threading.Lock()
        )

        self.cache = (
            OrderedDict()
        )


        print(
            "Mark model ready | "
            f"features={len(self.model_names)} "
            f"classes={self.num_classes} "
            f"labels={self.labels_available}",
            flush=True,
        )


    # ========================================================
    # Labels
    # ========================================================

    def _load_labels(
        self,
    ) -> tuple[list[str], bool]:

        if not MARK_LABELS_PATH.exists():

            return (
                [
                    f"class_{i}"
                    for i in range(
                        self.num_classes
                    )
                ],
                False,
            )


        payload = load_json(
            MARK_LABELS_PATH
        )


        if isinstance(
            payload,
            list,
        ):

            labels = payload

        elif isinstance(
            payload,
            dict,
        ):

            labels = (
                payload.get(
                    "class_labels"
                )
                or
                payload.get(
                    "labels"
                )
                or
                payload.get(
                    "classes"
                )
            )

        else:

            labels = None


        if (
            not isinstance(
                labels,
                list,
            )
            or
            len(labels)
            != self.num_classes
        ):

            raise RuntimeError(
                "class_labels.json must contain "
                f"exactly {self.num_classes} labels "
                "in training class-ID order"
            )


        return (
            [
                str(label)
                for label in labels
            ],
            True,
        )


    # ========================================================
    # Synchronize to EXACT intensity snapshot provenance
    # ========================================================

    def sync_snapshot(
        self,
        snapshot_id: str,
        intensity_snapshot_path: Path,
    ) -> None:

        with self.state_lock:

            if (
                self.snapshot_id
                == snapshot_id
            ):
                return


            intensity_meta = load_json(
                intensity_snapshot_path
                / "metadata.json"
            )


            env_dir = Path(
                intensity_meta[
                    "environmental_snapshot_path"
                ]
            )


            env_hour = normalize_utc(
                intensity_meta[
                    "environmental_valid_utc_hour"
                ]
            )


            intensity_hour = normalize_utc(
                intensity_meta[
                    "valid_utc_hour"
                ]
            )


            if (
                env_hour
                != intensity_hour
            ):

                raise RuntimeError(
                    "Intensity/environmental "
                    "provenance mismatch"
                )


            if not env_dir.exists():

                raise RuntimeError(
                    "Recorded environmental "
                    f"snapshot missing: {env_dir}"
                )


            def env_array(
                filename: str,
            ):

                array = np.load(
                    env_dir
                    / filename,
                    mmap_mode="r",
                )

                if (
                    len(array)
                    != len(
                        self.r6_timezone_id
                    )
                ):

                    raise RuntimeError(
                        f"{filename}: "
                        "R6 length mismatch"
                    )

                return array


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

            lighting_code = env_array(
                "lighting_condition_code.npy"
            )


            # Only switch after everything validates.
            self.env_dir = env_dir
            self.valid_utc = env_hour

            self.temperature = temperature
            self.humidity = humidity
            self.weather_available = (
                weather_available
            )

            self.solar_elevation = (
                solar_elevation
            )

            self.solar_azimuth = (
                solar_azimuth
            )

            self.is_daylight = (
                is_daylight
            )

            self.lighting_code = (
                lighting_code
            )

            self.snapshot_id = (
                snapshot_id
            )


            # Old-hour cache is useless.
            with self.cache_lock:
                self.cache.clear()


            print(
                "Mark runtime synchronized | "
                f"snapshot={snapshot_id} "
                f"utc={env_hour.isoformat()}",
                flush=True,
            )


    # ========================================================
    # H3 lookup
    # ========================================================

    def _row_for_cell(
        self,
        cell: str,
    ) -> int:

        key = np.uint64(
            h3.str_to_int(
                cell
            )
        )

        row = int(
            np.searchsorted(
                self.h3_keys,
                key,
            )
        )


        if (
            row
            >= len(
                self.h3_keys
            )
            or
            self.h3_keys[
                row
            ] != key
        ):

            raise KeyError(
                cell
            )


        return row


    # ========================================================
    # EXACT per-cell calendar math
    #
    # Operation order intentionally mirrors
    # build_national_intensity.py exactly.
    # ========================================================

    def _calendar_for_r6(
        self,
        r6_idx: int,
    ) -> dict[str, np.float32]:

        timezone_id = int(
            self.r6_timezone_id[
                r6_idx
            ]
        )

        timezone_name = (
            self.timezone_names[
                timezone_id
            ]
        )


        local_dt = (
            self.valid_utc
            .to_pydatetime()
            .astimezone(
                ZoneInfo(
                    timezone_name
                )
            )
        )


        local_hour = np.asarray(
            [
                float(
                    local_dt.hour
                )
            ],
            dtype=np.float32,
        )

        local_dow = np.asarray(
            [
                float(
                    local_dt.weekday()
                )
            ],
            dtype=np.float32,
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


        return {

            "local_hour":
                local_hour[0],

            "local_day_of_week":
                local_dow[0],

            "local_hour_sin":
                local_hour_sin[0],

            "local_hour_cos":
                local_hour_cos[0],

            "local_day_of_week_sin":
                local_dow_sin[0],

            "local_day_of_week_cos":
                local_dow_cos[0],
        }


    # ========================================================
    # Build exact 38-feature row
    # ========================================================

    def build_feature_row(
        self,
        cell: str,
    ) -> tuple[int, np.ndarray]:

        with self.state_lock:

            if (
                self.snapshot_id
                is None
            ):

                raise RuntimeError(
                    "Mark runtime has no "
                    "active snapshot"
                )


            row = self._row_for_cell(
                cell
            )

            r6_idx = int(
                self.r9_to_r6[
                    row
                ]
            )


            calendar = (
                self._calendar_for_r6(
                    r6_idx
                )
            )


            serving_lighting = int(
                self.lighting_code[
                    r6_idx
                ]
            )


            if not (
                0
                <= serving_lighting
                < len(
                    LIGHTING_TRANSLATION
                )
            ):

                raise RuntimeError(
                    "Invalid lighting code"
                )


            model_lighting = (
                LIGHTING_TRANSLATION[
                    serving_lighting
                ]
            )


            dynamic = {

                **calendar,

                "weather_temperature_2m_c":
                    np.float32(
                        self.temperature[
                            r6_idx
                        ]
                    ),

                "weather_relative_humidity_2m_pct":
                    np.float32(
                        self.humidity[
                            r6_idx
                        ]
                    ),

                "weather_available":
                    np.float32(
                        self.weather_available[
                            r6_idx
                        ]
                    ),

                "solar_elevation_deg":
                    np.float32(
                        self.solar_elevation[
                            r6_idx
                        ]
                    ),

                "solar_azimuth_deg":
                    np.float32(
                        self.solar_azimuth[
                            r6_idx
                        ]
                    ),

                "is_daylight":
                    np.float32(
                        self.is_daylight[
                            r6_idx
                        ]
                    ),

                "lighting_condition":
                    np.float32(
                        model_lighting
                    ),
            }


            X = np.empty(
                (
                    1,
                    EXPECTED_FEATURE_COUNT,
                ),
                dtype=np.float32,
            )


            for (
                column,
                name,
            ) in enumerate(
                self.model_names
            ):

                if name in self.static_col:

                    X[
                        0,
                        column,
                    ] = self.static_features[
                        row,
                        self.static_col[
                            name
                        ],
                    ]

                else:

                    if name not in dynamic:

                        raise RuntimeError(
                            "Missing mark feature: "
                            f"{name}"
                        )

                    X[
                        0,
                        column,
                    ] = dynamic[
                        name
                    ]


            if np.isinf(
                X
            ).any():

                raise RuntimeError(
                    "Infinite mark feature"
                )


            category = float(
                X[
                    0,
                    self.cat_idx,
                ]
            )


            if (
                not np.isfinite(
                    category
                )
                or
                category < 0
                or
                category >= 5
            ):

                raise RuntimeError(
                    "Invalid mark "
                    "lighting category"
                )


            return (
                row,
                X,
            )


    # ========================================================
    # Cache
    # ========================================================

    def _cache_get(
        self,
        key,
    ):

        with self.cache_lock:

            value = self.cache.get(
                key
            )

            if value is not None:

                self.cache.move_to_end(
                    key
                )

            return value


    def _cache_put(
        self,
        key,
        value,
    ) -> None:

        with self.cache_lock:

            self.cache[
                key
            ] = value

            self.cache.move_to_end(
                key
            )


            while (
                len(
                    self.cache
                )
                > CACHE_SIZE
            ):

                self.cache.popitem(
                    last=False
                )


    # ========================================================
    # Predict P(mark=k | event, x, t)
    # ========================================================

    def predict(
        self,
        *,
        cell: str,
        snapshot_id: str,
        intensity_snapshot_path: Path,
        top_k: int,
    ) -> dict:

        self.sync_snapshot(
            snapshot_id,
            intensity_snapshot_path,
        )


        key = (
            snapshot_id,
            cell,
        )


        probabilities = (
            self._cache_get(
                key
            )
        )


        if probabilities is None:

            (
                row,
                X,
            ) = self.build_feature_row(
                cell
            )


            # Serialize traversal of the large GPU forest.
            with self.gpu_lock:

                # Another request may have populated
                # the cache while this request waited.
                probabilities = (
                    self._cache_get(
                        key
                    )
                )


                if probabilities is None:

                    X_gpu = cp.asarray(
                        X,
                        dtype=cp.float32,
                    )


                    # Use raw multiclass margins and
                    # explicitly softmax them.
                    margins_gpu = (
                        self.bst
                        .inplace_predict(
                            X_gpu,
                            predict_type="margin",
                            validate_features=False,
                        )
                    )


                    margins = cp.asnumpy(
                        margins_gpu
                    )


                    margins = np.asarray(
                        margins,
                        dtype=np.float32,
                    )


                    if margins.ndim == 2:

                        margins = (
                            margins[
                                0
                            ]
                        )


                    if (
                        margins.ndim != 1
                        or
                        len(
                            margins
                        )
                        != self.num_classes
                    ):

                        raise RuntimeError(
                            "Unexpected mark "
                            "margin shape: "
                            f"{margins.shape}"
                        )


                    # Numerically stable softmax.
                    shifted = (
                        margins
                        - np.max(
                            margins
                        )
                    )


                    exp_values = np.exp(
                        shifted.astype(
                            np.float64
                        )
                    )


                    probabilities = (
                        exp_values
                        / np.sum(
                            exp_values
                        )
                    ).astype(
                        np.float32
                    )


                    if not np.isfinite(
                        probabilities
                    ).all():

                        raise RuntimeError(
                            "Non-finite mark "
                            "probabilities"
                        )


                    probability_sum = float(
                        np.sum(
                            probabilities,
                            dtype=np.float64,
                        )
                    )


                    if not np.isclose(
                        probability_sum,
                        1.0,
                        atol=1e-5,
                        rtol=0.0,
                    ):

                        raise RuntimeError(
                            "Mark probabilities "
                            "do not sum to 1: "
                            f"{probability_sum}"
                        )


                    self._cache_put(
                        key,
                        probabilities,
                    )


        k = min(
            max(
                int(
                    top_k
                ),
                1,
            ),
            self.num_classes,
        )


        if (
            k
            == self.num_classes
        ):

            order = np.argsort(
                -probabilities
            )

        else:

            candidates = np.argpartition(
                -probabilities,
                k - 1,
            )[
                :k
            ]

            order = candidates[
                np.argsort(
                    -probabilities[
                        candidates
                    ]
                )
            ]


        distribution = [

            {
                "class_id":
                    int(
                        class_id
                    ),

                "subtype":
                    self.class_labels[
                        int(
                            class_id
                        )
                    ],

                "probability":
                    float(
                        probabilities[
                            class_id
                        ]
                    ),
            }

            for class_id
            in order
        ]


        return {

            "model_run_id":
                MARK_MODEL_RUN_ID,

            "num_classes":
                self.num_classes,

            "labels_available":
                self.labels_available,

            "distribution":
                distribution,
        }
