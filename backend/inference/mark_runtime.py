from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

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
MARK_BENCHMARK_ROWS = max(200, int(os.getenv("CRIMENET_MARK_BENCHMARK_ROWS", "300")))
MARK_CPU_THREADS = max(1, int(os.getenv("CRIMENET_MARK_CPU_THREADS", "1")))
GPU_MICROBATCH_WINDOW_SECONDS = 0.003
GPU_MICROBATCH_MAX_ROWS = 64
GPU_MICROBATCH_QUEUE_SIZE = 128
MARK_INFERENCE_MODES = frozenset({"cpu", "gpu_batch", "auto"})

performance_logger = logging.getLogger("uvicorn.error")


class MarkCapacityExceeded(RuntimeError):
    pass


def resolve_mark_inference_mode(configured: str | None = None) -> str:
    raw_mode = (
        configured
        if configured is not None
        else os.getenv("CRIMENET_MARK_INFERENCE", "cpu")
    )
    mode = raw_mode.strip().lower()
    if mode not in MARK_INFERENCE_MODES:
        raise RuntimeError(
            f"Invalid CRIMENET_MARK_INFERENCE={raw_mode!r}. "
            "Expected one of: cpu, gpu_batch, auto."
        )
    return mode


def load_cupy():
    try:
        import cupy
    except ImportError as exc:
        raise RuntimeError(
            "GPU mark inference requires the CuPy CUDA runtime."
        ) from exc
    return cupy


@dataclass
class _PredictionFlight:
    event: threading.Event = field(default_factory=threading.Event)
    probabilities: np.ndarray | None = None
    error: BaseException | None = None


@dataclass
class _GpuBatchItem:
    features: np.ndarray
    event: threading.Event = field(default_factory=threading.Event)
    margins: np.ndarray | None = None
    error: BaseException | None = None


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

    def __init__(self, *, inference_mode: str | None = None) -> None:

        self.configured_inference_mode = resolve_mark_inference_mode(
            inference_mode
        )
        performance_logger.info(
            "mark_inference mode=%s benchmark=%s",
            self.configured_inference_mode,
            "true" if self.configured_inference_mode == "auto" else "false",
        )

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

        self.cpu_bst: xgb.Booster | None = None
        self.gpu_bst: xgb.Booster | None = None
        self.cp = None
        self.gpu_queue: queue.Queue[_GpuBatchItem] | None = None
        self.inference_device = "cpu"
        self.inference_benchmark: dict | None = None
        self._initialize_model_for_mode(self.configured_inference_mode)


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

        # Concurrent duplicate misses share one prediction.
        self.inflight_lock = threading.Lock()
        self.inflight: dict[tuple[str, str], _PredictionFlight] = {}

        self._activate_inference_mode(self.configured_inference_mode)


        print(
            "Mark model ready | "
            f"features={len(self.model_names)} "
            f"classes={self.num_classes} "
            f"labels={self.labels_available}",
            flush=True,
        )


    def _load_booster(self, device: str) -> xgb.Booster:
        booster = xgb.Booster()
        booster.load_model(MARK_MODEL_PATH)
        parameters: dict[str, str | int] = {"device": device}
        if device == "cpu":
            parameters["nthread"] = MARK_CPU_THREADS
        booster.set_param(parameters)
        return booster


    def _initialize_model_for_mode(self, mode: str) -> None:
        if mode == "gpu_batch":
            self.gpu_bst = self._load_booster("cuda:0")
            self.bst = self.gpu_bst
            return

        self.cpu_bst = self._load_booster("cpu")
        self.bst = self.cpu_bst


    def _initialize_gpu_runtime(self) -> None:
        cupy = load_cupy()
        try:
            device_count = cupy.cuda.runtime.getDeviceCount()
        except Exception as exc:
            raise RuntimeError(
                "GPU mark inference could not initialize CUDA."
            ) from exc
        if device_count < 1:
            raise RuntimeError("GPU mark inference found no CUDA devices.")
        self.cp = cupy


    def _start_gpu_batch_worker(self) -> None:
        self.gpu_queue = queue.Queue(maxsize=GPU_MICROBATCH_QUEUE_SIZE)
        threading.Thread(
            target=self._gpu_batch_worker,
            name="crimenet-mark-gpu-batcher",
            daemon=True,
        ).start()


    def _activate_inference_mode(self, mode: str) -> None:
        if mode == "cpu":
            self.inference_device = "cpu"
            self.inference_benchmark = None
            return

        if mode == "gpu_batch":
            self._initialize_gpu_runtime()
            self.inference_device = "gpu_batch"
            self.inference_benchmark = None
            self._start_gpu_batch_worker()
            return

        self.inference_benchmark = self._select_inference_device()


    # ========================================================
    # Interactive inference device selection
    # ========================================================

    def _benchmark_feature_rows(self) -> np.ndarray:
        positions = np.linspace(
            0,
            len(self.static_features) - 1,
            num=MARK_BENCHMARK_ROWS,
            dtype=np.int64,
        )
        features = np.zeros(
            (MARK_BENCHMARK_ROWS, EXPECTED_FEATURE_COUNT),
            dtype=np.float32,
        )
        for column, name in enumerate(self.model_names):
            static_column = self.static_col.get(name)
            if static_column is not None:
                features[:, column] = self.static_features[positions, static_column]
        features[:, self.cat_idx] = 0.0
        return features


    @staticmethod
    def _latency_summary(milliseconds: list[float]) -> dict[str, float]:
        values = np.asarray(milliseconds, dtype=np.float64)
        return {
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "p99_ms": float(np.percentile(values, 99)),
            "throughput_rps": float(1000.0 / np.mean(values)),
        }


    def _predict_cpu_margins(self, features: np.ndarray) -> np.ndarray:
        if self.cpu_bst is None:
            raise RuntimeError("CPU mark inference is unavailable")
        return np.asarray(
            self.cpu_bst.inplace_predict(
                features,
                predict_type="margin",
                validate_features=False,
            ),
            dtype=np.float32,
        )


    def _predict_gpu_direct(self, features: np.ndarray) -> np.ndarray:
        if self.cp is None or self.gpu_bst is None:
            raise RuntimeError("GPU mark inference is unavailable")
        features_gpu = self.cp.asarray(features, dtype=self.cp.float32)
        margins_gpu = self.gpu_bst.inplace_predict(
            features_gpu,
            predict_type="margin",
            validate_features=False,
        )
        return np.asarray(self.cp.asnumpy(margins_gpu), dtype=np.float32)


    def _select_inference_device(self) -> dict:
        rows = self._benchmark_feature_rows()
        for row in rows[:10]:
            self._predict_cpu_margins(row.reshape(1, -1))
        cpu_latencies = []
        cpu_results = []
        for row in rows:
            started = time.perf_counter()
            cpu_results.append(self._predict_cpu_margins(row.reshape(1, -1)))
            cpu_latencies.append((time.perf_counter() - started) * 1000.0)

        benchmark: dict[str, object] = {
            "rows": MARK_BENCHMARK_ROWS,
            "cpu": self._latency_summary(cpu_latencies),
            "gpu": None,
        }
        gpu_numerically_equivalent = True

        gpu_available = False
        try:
            self._initialize_gpu_runtime()
            gpu_available = True
        except RuntimeError as exc:
            performance_logger.warning("mark_gpu_benchmark_unavailable error=%r", exc)

        if gpu_available:
            try:
                self.gpu_bst = self._load_booster("cuda:0")
                for row in rows[:10]:
                    self._predict_gpu_direct(row.reshape(1, -1))
                gpu_latencies = []
                max_margin_difference = 0.0
                max_probability_difference = 0.0
                for index, row in enumerate(rows):
                    started = time.perf_counter()
                    gpu_result = self._predict_gpu_direct(row.reshape(1, -1))
                    gpu_latencies.append((time.perf_counter() - started) * 1000.0)
                    cpu_result = cpu_results[index]
                    cpu_probabilities = self._probabilities_from_margins(cpu_result)
                    gpu_probabilities = self._probabilities_from_margins(gpu_result)
                    if not np.allclose(
                        cpu_probabilities,
                        gpu_probabilities,
                        rtol=1e-5,
                        atol=1e-6,
                    ):
                        gpu_numerically_equivalent = False
                    max_margin_difference = max(
                        max_margin_difference,
                        float(np.max(np.abs(cpu_result - gpu_result))),
                    )
                    max_probability_difference = max(
                        max_probability_difference,
                        float(np.max(np.abs(cpu_probabilities - gpu_probabilities))),
                    )
                benchmark["gpu"] = self._latency_summary(gpu_latencies)
                benchmark["max_cpu_gpu_margin_difference"] = max_margin_difference
                benchmark["max_cpu_gpu_probability_difference"] = max_probability_difference
                benchmark["cpu_gpu_probabilities_equivalent"] = gpu_numerically_equivalent
            except Exception as exc:
                performance_logger.warning("mark_gpu_benchmark_failed error=%r", exc)
                self.gpu_bst = None

        cpu_p95 = float(benchmark["cpu"]["p95_ms"])
        gpu_result = benchmark["gpu"]
        use_gpu = bool(
            self.gpu_bst is not None
            and gpu_result is not None
            and (
                not gpu_numerically_equivalent
                or float(gpu_result["p95_ms"]) < cpu_p95 / 1.10
            )
        )
        if use_gpu:
            self.inference_device = "gpu_batch"
            self.bst = self.gpu_bst
            self.cpu_bst = None
            self._start_gpu_batch_worker()
        else:
            self.inference_device = "cpu"
            self.gpu_bst = None
            self.bst = self.cpu_bst

        benchmark["selected"] = self.inference_device
        benchmark["recommended_serving_mode"] = self.inference_device
        performance_logger.info(
            "mark_inference_benchmark %s",
            json.dumps(benchmark, sort_keys=True, separators=(",", ":")),
        )
        return benchmark


    def _gpu_batch_worker(self) -> None:
        assert self.gpu_queue is not None
        while True:
            first = self.gpu_queue.get()
            items = [first]
            deadline = time.perf_counter() + GPU_MICROBATCH_WINDOW_SECONDS
            while len(items) < GPU_MICROBATCH_MAX_ROWS:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    items.append(self.gpu_queue.get(timeout=remaining))
                except queue.Empty:
                    break
            try:
                batch = np.concatenate([item.features for item in items], axis=0)
                margins = self._predict_gpu_direct(batch)
                if margins.ndim == 1:
                    margins = margins.reshape(1, -1)
                for index, item in enumerate(items):
                    item.margins = margins[index]
            except BaseException as exc:
                for item in items:
                    item.error = exc
            finally:
                for item in items:
                    item.event.set()


    def _predict_gpu_microbatch(self, features: np.ndarray) -> np.ndarray:
        if self.gpu_queue is None:
            raise RuntimeError("GPU microbatch queue is unavailable")
        item = _GpuBatchItem(features=features)
        try:
            self.gpu_queue.put_nowait(item)
        except queue.Full as exc:
            raise MarkCapacityExceeded("mark inference queue is full") from exc
        item.event.wait()
        if item.error is not None:
            raise item.error
        if item.margins is None:
            raise RuntimeError("GPU microbatch produced no mark result")
        return item.margins


    def _predict_margins(self, features: np.ndarray) -> np.ndarray:
        if self.inference_device == "gpu_batch":
            return self._predict_gpu_microbatch(features)
        return self._predict_cpu_margins(features)


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


    def build_feature_row_for_snapshot(
        self,
        *,
        cell: str,
        snapshot_id: str,
        intensity_snapshot_path: Path,
    ) -> tuple[int, np.ndarray]:
        # Keep only snapshot transition + feature capture atomic. Inference and
        # response formatting happen after this lock is released.
        with self.state_lock:
            self.sync_snapshot(snapshot_id, intensity_snapshot_path)
            return self.build_feature_row(cell)


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

        # A request for the previous hour may finish after a snapshot switch.
        # Return its result to its caller, but never repopulate stale cache state.
        with self.state_lock:
            if key[0] != self.snapshot_id:
                return

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

    def _probabilities_from_margins(self, margins: np.ndarray) -> np.ndarray:
        margins = np.asarray(margins, dtype=np.float32)
        if margins.ndim == 2:
            margins = margins[0]
        if margins.ndim != 1 or len(margins) != self.num_classes:
            raise RuntimeError(f"Unexpected mark margin shape: {margins.shape}")

        shifted = margins - np.max(margins)
        exp_values = np.exp(shifted.astype(np.float64))
        probabilities = (exp_values / np.sum(exp_values)).astype(np.float32)
        if not np.isfinite(probabilities).all():
            raise RuntimeError("Non-finite mark probabilities")
        probability_sum = float(np.sum(probabilities, dtype=np.float64))
        if not np.isclose(probability_sum, 1.0, atol=1e-5, rtol=0.0):
            raise RuntimeError(f"Mark probabilities do not sum to 1: {probability_sum}")
        return probabilities

    def predict(
        self,
        *,
        cell: str,
        snapshot_id: str,
        intensity_snapshot_path: Path,
        top_k: int,
        timings: dict[str, float] | None = None,
    ) -> dict:
        prediction_started = time.perf_counter()
        key = (
            snapshot_id,
            cell,
        )
        cache_started = time.perf_counter()
        probabilities = self._cache_get(key)
        if timings is not None:
            timings["cache_lookup_ms"] = (
                time.perf_counter() - cache_started
            ) * 1000.0
            timings["cache_hit"] = 1.0 if probabilities is not None else 0.0
            if probabilities is not None:
                timings["feature_row_ms"] = 0.0
                timings["inference_ms"] = 0.0
                timings["coalesced_wait_ms"] = 0.0

        if probabilities is None:
            with self.inflight_lock:
                flight = self.inflight.get(key)
                leader = flight is None
                if flight is None:
                    flight = _PredictionFlight()
                    self.inflight[key] = flight

            if leader:
                try:
                    feature_started = time.perf_counter()
                    _, features = self.build_feature_row_for_snapshot(
                        cell=cell,
                        snapshot_id=snapshot_id,
                        intensity_snapshot_path=intensity_snapshot_path,
                    )
                    if timings is not None:
                        timings["feature_row_ms"] = (
                            time.perf_counter() - feature_started
                        ) * 1000.0

                    inference_started = time.perf_counter()
                    margins = self._predict_margins(features)
                    probabilities = self._probabilities_from_margins(margins)
                    if timings is not None:
                        timings["inference_ms"] = (
                            time.perf_counter() - inference_started
                        ) * 1000.0
                        timings["coalesced_wait_ms"] = 0.0
                    self._cache_put(key, probabilities)
                    flight.probabilities = probabilities
                except BaseException as exc:
                    flight.error = exc
                    raise
                finally:
                    flight.event.set()
                    with self.inflight_lock:
                        if self.inflight.get(key) is flight:
                            self.inflight.pop(key, None)
            else:
                wait_started = time.perf_counter()
                flight.event.wait()
                if timings is not None:
                    timings["coalesced_wait_ms"] = (
                        time.perf_counter() - wait_started
                    ) * 1000.0
                    timings["feature_row_ms"] = 0.0
                    timings["inference_ms"] = 0.0
                if flight.error is not None:
                    raise flight.error
                probabilities = flight.probabilities
                if probabilities is None:
                    raise RuntimeError("Coalesced mark request produced no result")

        result_started = time.perf_counter()
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

        if timings is not None:
            timings["result_construction_ms"] = (
                time.perf_counter() - result_started
            ) * 1000.0
            timings["mark_runtime_total_ms"] = (
                time.perf_counter() - prediction_started
            ) * 1000.0
            timings["inference_device_gpu"] = (
                1.0 if self.inference_device == "gpu_batch" else 0.0
            )


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
