from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib
import json
import multiprocessing as mp
import os
import queue as queue_module
import re
import shutil
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import optuna
import yaml


# ---------------------------------------------------------------------------
# CrimeNet XGBoost HPO
#
# Purpose:
#   Aggressively optimize the final XGBoost baseline for:
#     1) point-process intensity: validation NLL/event
#     2) conditional 87-way mark classifier: validation multiclass log loss
#
# Protocol:
#   Stage 1 (explore): broad TPE search on deterministic fractions.
#   Stage 2 (refine):  full training data, larger validation fraction.
#   Stage 3 (tournament): top K configs, full train + full validation,
#                         repeated over multiple seeds.
#
# Parallel execution:
#   One OS process per GPU. Each process sees exactly one GPU through
#   CUDA_VISIBLE_DEVICES and continuously claims trials from a shared
#   Optuna JournalStorage study. This is distributed HPO, not multi-GPU
#   training of a single XGBoost model.
#
# The 2025+ test split is never touched.
#
# HPO trials call the model-specific train() function directly, intentionally
# bypassing MLflow. This prevents hundreds of tuning trials from polluting the
# experiment tracker. The script exports one final YAML; run that YAML through
# the normal experiment orchestrator to create the official baseline run.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage:
    name: str
    train_fraction: float
    validation_fraction: float
    num_boost_round: int
    early_stopping_rounds: int


INTENSITY_METRIC = "sample_validation_nll_per_event"
MARK_METRIC = "sample_validation_log_loss"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggressive multi-GPU Optuna tuning for CrimeNet XGBoost baselines. "
            "One independent trial is trained per GPU."
        )
    )

    parser.add_argument(
        "--family",
        choices=("intensity", "mark"),
        required=True,
        help="Model family to tune.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Base YAML config. Existing best-depth config is ideal.",
    )
    parser.add_argument(
        "--study-name",
        required=True,
        help="Stable study name. Reusing it resumes JournalStorage studies.",
    )
    parser.add_argument(
        "--module",
        default=None,
        help=(
            "Optional explicit Python training module containing train(). "
            "If omitted, the script checks model.training_module/model.module "
            "and then common CrimeNet module names."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hpo"),
        help="Root for Optuna journals, summaries, and exported YAML.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="XGBoost device written into architecture.device inside each worker.",
    )

    # GPU/process topology. These defaults target the selected 8x RTX 5090 host.
    parser.add_argument(
        "--gpus",
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated physical GPU indices exposed to HPO workers.",
    )
    parser.add_argument(
        "--explore-workers",
        type=int,
        default=8,
        help="Concurrent GPU workers during exploration.",
    )
    parser.add_argument(
        "--refine-workers",
        type=int,
        default=8,
        help="Concurrent GPU workers during refinement.",
    )
    parser.add_argument(
        "--tournament-workers",
        type=int,
        default=8,
        help="Concurrent GPU workers during the finalist tournament.",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=0,
        help=(
            "CPU thread cap per worker. 0 = automatically partition the host CPU "
            "set across active workers (about 24 threads each on a 192-thread/8-GPU host)."
        ),
    )
    parser.add_argument(
        "--monitor-seconds",
        type=float,
        default=60.0,
        help="Coordinator progress-print interval.",
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=8,
        help="Abort one worker after this many consecutive failed trials.",
    )
    parser.add_argument(
        "--hpo-model-table-root",
        default=None,
        help=(
            "Optional model-table override used only by HPO workers. Stage the Delta "
            "table to local NVMe and pass its local path to avoid 8 workers repeatedly "
            "scanning GCS. The exported final YAML keeps the original table root."
        ),
    )

    parser.add_argument(
        "--stage-cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for immutable sampled Arrow IPC caches. Default: "
            "<study-output>/stage_cache. Put this on local NVMe for best throughput."
        ),
    )
    parser.add_argument(
        "--disable-stage-cache",
        action="store_true",
        help=(
            "Disable the sampled-data cache and use the model trainer's normal "
            "Delta scan/sampling path for every trial."
        ),
    )
    parser.add_argument(
        "--rebuild-stage-cache",
        action="store_true",
        help="Rebuild sampled Arrow IPC caches even when matching cache files exist.",
    )
    parser.add_argument(
        "--no-prepared-xy-cache",
        action="store_true",
        help=(
            "Keep the shared sampled-frame cache but repeat Polars->Pandas/NumPy "
            "conversion for each trial. Useful only if host RAM is constrained."
        ),
    )

    # Aggressive defaults: exploration first, then expensive confirmation.
    parser.add_argument("--explore-trials", type=int, default=120)
    parser.add_argument("--refine-trials", type=int, default=60)
    parser.add_argument("--seed-top-k", type=int, default=20)
    parser.add_argument("--finalists", type=int, default=6)
    parser.add_argument(
        "--tournament-seeds",
        default="42,1337,2026",
        help="Comma-separated seeds used for full-data finalist tournament.",
    )

    parser.add_argument("--explore-train-fraction", type=float, default=0.25)
    parser.add_argument("--explore-validation-fraction", type=float, default=0.25)
    parser.add_argument("--refine-train-fraction", type=float, default=1.0)
    parser.add_argument("--refine-validation-fraction", type=float, default=0.25)

    parser.add_argument("--explore-rounds", type=int, default=2500)
    parser.add_argument("--refine-rounds", type=int, default=5000)
    parser.add_argument("--tournament-rounds", type=int, default=6000)

    parser.add_argument("--explore-early-stop", type=int, default=100)
    parser.add_argument("--refine-early-stop", type=int, default=150)
    parser.add_argument("--tournament-early-stop", type=int, default=200)

    parser.add_argument(
        "--keep-trial-artifacts",
        action="store_true",
        help="Keep every native trial artifact. Default deletes them after scoring.",
    )
    parser.add_argument(
        "--final-model-name",
        default=None,
        help="Override model.name in the exported final YAML.",
    )

    return parser.parse_args()


def parse_gpus(raw: str) -> list[str]:
    gpus = [value.strip() for value in raw.split(",") if value.strip()]
    if not gpus:
        raise ValueError("At least one GPU must be supplied through --gpus.")
    if len(set(gpus)) != len(gpus):
        raise ValueError(f"Duplicate GPU identifiers in --gpus: {gpus}")
    return gpus


def resolved_worker_count(requested: int, gpus: list[str]) -> int:
    if requested <= 0:
        raise ValueError("Worker counts must be positive.")
    return min(int(requested), len(gpus))


def _cpu_slice(worker_index: int, worker_count: int) -> list[int]:
    try:
        available = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = os.cpu_count() or 1
        available = list(range(count))

    start = (len(available) * worker_index) // worker_count
    end = (len(available) * (worker_index + 1)) // worker_count
    cpus = available[start:end]
    return cpus or available[:1]


def configure_worker_environment(
    *,
    gpu_id: str,
    worker_index: int,
    worker_count: int,
    threads_per_worker: int,
) -> int:
    # Must run before importing the training module (and therefore xgboost/polars).
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cpus = _cpu_slice(worker_index, worker_count)
    try:
        os.sched_setaffinity(0, set(cpus))
    except (AttributeError, OSError):
        pass

    threads = int(threads_per_worker) if threads_per_worker > 0 else len(cpus)
    threads = max(1, threads)

    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "POLARS_MAX_THREADS",
    ):
        os.environ[key] = str(threads)

    # Keep OpenMP workers inside the CPU partition assigned to this GPU worker.
    os.environ["OMP_PROC_BIND"] = "close"
    os.environ["OMP_PLACES"] = "cores"

    print(
        f"[worker {worker_index}] physical GPU={gpu_id} -> CUDA logical GPU 0; "
        f"CPU threads={threads}; affinity={cpus[0]}..{cpus[-1]}"
    )
    return threads


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = yaml.safe_load(f)
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping at YAML root: {path}")
    return value


def stable_config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not seeds:
        raise ValueError("At least one tournament seed is required.")
    return seeds


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "study"


def final_model_name(base_name: str, override: str | None) -> str:
    if override:
        return override
    # Avoid calling a tuned model "depth12" if HPO discovers depth11/13/etc.
    stripped = re.sub(r"_depth\d+$", "", base_name)
    return f"{stripped}_hpo_tuned"


def resolve_training_module(
    config: dict[str, Any],
    *,
    family: str,
    explicit: str | None,
):
    model_cfg = config.get("model", {})

    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)

    for key in ("training_module", "module"):
        value = model_cfg.get(key)
        if isinstance(value, str) and value:
            candidates.append(value)

    if family == "intensity":
        candidates.extend(
            [
                "machine_learning.models.xgboost.model",
                "machine_learning.models.xgboost.intensity_model",
                "machine_learning.models.xgboost.intensity",
            ]
        )
    else:
        candidates.extend(
            [
                "machine_learning.models.xgboost.mark_model",
                "machine_learning.models.xgboost.mark",
                "machine_learning.models.xgboost.mark_subtype",
                "machine_learning.models.xgboost.subtype_model",
            ]
        )

    errors: list[str] = []
    seen: set[str] = set()

    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            errors.append(f"{name}: import failed: {exc}")
            continue

        train_fn = getattr(module, "train", None)
        if callable(train_fn):
            print(f"Training module: {name}")
            return module

        errors.append(f"{name}: no callable train()")

    joined = "\n  ".join(errors)
    raise ImportError(
        "Could not resolve model training module.\n"
        "Pass --module <python.module.path> explicitly.\n"
        f"Attempts:\n  {joined}"
    )


def objective_metric(family: str) -> str:
    return INTENSITY_METRIC if family == "intensity" else MARK_METRIC


def categorical_choices_with_base(
    base: int,
    defaults: Iterable[int],
) -> list[int]:
    return sorted(set([int(base), *[int(x) for x in defaults]]))


def build_space(config: dict[str, Any], family: str) -> dict[str, Any]:
    arch = config["architecture"]

    max_bin_choices = categorical_choices_with_base(
        int(arch["max_bin"]),
        (128, 256, 512, 1024),
    )
    cat_onehot_choices = categorical_choices_with_base(
        int(arch["max_cat_to_onehot"]),
        (1, 4, 8, 16, 32, 64),
    )

    if family == "intensity":
        # D18 won the depth sweep at the upper boundary, so deliberately search
        # beyond it. Cap 24 to avoid absurd exponential tree growth.
        depth_low, depth_high = 8, 24
    else:
        # D12 won while D14 regressed, but joint regularization can shift the
        # optimum, so search a broad neighborhood.
        depth_low, depth_high = 6, 18

    return {
        "depth_low": depth_low,
        "depth_high": depth_high,
        "max_bin_choices": max_bin_choices,
        "max_cat_to_onehot_choices": cat_onehot_choices,
    }


def suggest_params(
    trial: optuna.Trial,
    *,
    family: str,
    space: dict[str, Any],
) -> dict[str, Any]:
    # Broad, intentionally non-conservative ranges.
    params: dict[str, Any] = {
        "max_depth": trial.suggest_int(
            "max_depth",
            int(space["depth_low"]),
            int(space["depth_high"]),
        ),
        "max_bin": trial.suggest_categorical(
            "max_bin",
            space["max_bin_choices"],
        ),
        "max_cat_to_onehot": trial.suggest_categorical(
            "max_cat_to_onehot",
            space["max_cat_to_onehot_choices"],
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.003,
            0.20,
            log=True,
        ),
        "subsample": trial.suggest_float(
            "subsample",
            0.45,
            1.0,
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.40,
            1.0,
        ),
        "min_child_weight": trial.suggest_float(
            "min_child_weight",
            1e-3 if family == "intensity" else 1e-2,
            1e5 if family == "intensity" else 1e4,
            log=True,
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            1e-6,
            1e4,
            log=True,
        ),
    }

    # L1 regularization has a meaningful exact-zero state, which cannot be
    # represented by an Optuna log distribution. Model it as off vs log-scale.
    use_reg_alpha = trial.suggest_categorical(
        "use_reg_alpha",
        [False, True],
    )
    params["reg_alpha"] = (
        trial.suggest_float(
            "reg_alpha_nonzero",
            1e-8,
            1e3,
            log=True,
        )
        if use_reg_alpha
        else 0.0
    )

    if family == "intensity":
        params["max_delta_step"] = trial.suggest_float(
            "max_delta_step",
            0.0,
            12.0,
        )
    else:
        # Include exact gamma=0 as a real branch; otherwise explore log-scale.
        use_gamma = trial.suggest_categorical("use_gamma", [False, True])
        params["gamma"] = (
            trial.suggest_float("gamma_nonzero", 1e-8, 100.0, log=True)
            if use_gamma
            else 0.0
        )

    return params


def params_for_enqueue(
    config: dict[str, Any],
    *,
    family: str,
    space: dict[str, Any],
) -> dict[str, Any] | None:
    """Translate base config into Optuna parameter names for a seeded trial."""
    arch = config["architecture"]
    opt = config["optimization"]

    max_bin = int(arch["max_bin"])
    max_cat = int(arch["max_cat_to_onehot"])

    if max_bin not in space["max_bin_choices"]:
        return None
    if max_cat not in space["max_cat_to_onehot_choices"]:
        return None

    p: dict[str, Any] = {
        "max_depth": int(arch["max_depth"]),
        "max_bin": max_bin,
        "max_cat_to_onehot": max_cat,
        "learning_rate": float(opt["learning_rate"]),
        "subsample": float(opt["subsample"]),
        "colsample_bytree": float(opt["colsample_bytree"]),
        "min_child_weight": float(opt["min_child_weight"]),
        "reg_lambda": float(opt["reg_lambda"]),
    }

    reg_alpha = float(opt.get("reg_alpha", 0.0))
    p["use_reg_alpha"] = reg_alpha > 0.0
    if reg_alpha > 0.0:
        p["reg_alpha_nonzero"] = reg_alpha

    if family == "intensity":
        p["max_delta_step"] = float(opt["max_delta_step"])
    else:
        gamma = float(opt.get("gamma", 0.0))
        p["use_gamma"] = gamma > 0.0
        if gamma > 0.0:
            p["gamma_nonzero"] = gamma

    return p


def normalized_params_from_trial(
    params: dict[str, Any],
    *,
    family: str,
) -> dict[str, Any]:
    """Convert Optuna's conditional parameter dictionary to actual config values."""
    out = dict(params)

    use_reg_alpha = bool(out.pop("use_reg_alpha", False))
    reg_alpha_nonzero = out.pop("reg_alpha_nonzero", None)
    out["reg_alpha"] = (
        float(reg_alpha_nonzero)
        if use_reg_alpha
        else 0.0
    )

    if family == "mark":
        use_gamma = bool(out.pop("use_gamma", False))
        gamma_nonzero = out.pop("gamma_nonzero", None)
        out["gamma"] = float(gamma_nonzero) if use_gamma else 0.0

    return out


def apply_params(
    config: dict[str, Any],
    params: dict[str, Any],
    *,
    family: str,
) -> None:
    arch = config["architecture"]
    opt = config["optimization"]

    arch["max_depth"] = int(params["max_depth"])
    arch["max_bin"] = int(params["max_bin"])
    arch["max_cat_to_onehot"] = int(params["max_cat_to_onehot"])

    opt["learning_rate"] = float(params["learning_rate"])
    opt["subsample"] = float(params["subsample"])
    opt["colsample_bytree"] = float(params["colsample_bytree"])
    opt["min_child_weight"] = float(params["min_child_weight"])
    opt["reg_lambda"] = float(params["reg_lambda"])
    opt["reg_alpha"] = float(params["reg_alpha"])

    if family == "intensity":
        opt["max_delta_step"] = float(params["max_delta_step"])
    else:
        opt["gamma"] = float(params["gamma"])


def configure_stage(
    config: dict[str, Any],
    *,
    stage: Stage,
    device: str,
    seed: int | None = None,
    model_table_root: str | None = None,
) -> None:
    config["architecture"]["device"] = device
    config["data"]["train_fraction"] = float(stage.train_fraction)
    config["data"]["validation_fraction"] = float(stage.validation_fraction)
    config["training"]["num_boost_round"] = int(stage.num_boost_round)
    config["training"]["early_stopping_rounds"] = int(stage.early_stopping_rounds)

    # Silence per-round logs during hundreds of trials.
    config["training"]["verbose_eval"] = False

    if seed is not None:
        config["data"]["seed"] = int(seed)

    if model_table_root is not None:
        config["data"]["model_table_root"] = str(model_table_root)



# ---------------------------------------------------------------------------
# Stage data cache
# ---------------------------------------------------------------------------
# The model trainers were written for standalone runs and therefore scan/filter
# the Delta table and convert Polars -> Pandas/NumPy inside every train() call.
# HPO reuses the same deterministic sample for many trials, so doing that work
# hundreds of times wastes CPU/I/O and leaves fast GPUs idle.
#
# The coordinator materializes each distinct sample specification once as Arrow
# IPC. Workers then intercept the trainer's deterministic sampling helper and
# return the cached frame. Each long-lived worker keeps that frame in memory and
# can also memoize _prepare_xy() and a few pure validation/summary helpers.
# XGBoost QuantileDMatrix is intentionally NOT cached because max_bin is tuned
# and its quantization belongs to the trial-specific representation.


def _fraction_token(value: float) -> str:
    return f"{float(value):.12g}"


def _canonical_sampling_seed(fraction: float, seed: int) -> int:
    # At fraction=1.0 every hash bucket passes, so seed changes model RNG only;
    # it does not change the sampled rows. Canonicalize to maximize cache reuse.
    return 0 if abs(float(fraction) - 1.0) <= 1e-12 else int(seed)


def _sample_lookup_key(*, split: str, fraction: float, seed: int) -> str:
    canonical_seed = _canonical_sampling_seed(fraction, seed)
    return f"{split}|{_fraction_token(fraction)}|{canonical_seed}"


def _hpo_table_root(base_config: dict[str, Any], override: str | None) -> str:
    if override is not None:
        return str(override)
    return str(base_config["data"]["model_table_root"])


def _scan_delta_for_cache(model_table_root: str):
    # Import Polars lazily so the coordinator remains lightweight until a cache
    # actually has to be built.
    import polars as pl

    if str(model_table_root).startswith("gs://"):
        return pl.scan_delta(
            model_table_root,
            credential_provider=pl.CredentialProviderGCP(),
        )

    # Local/NVMe Delta tables must not request GCP ADC.
    return pl.scan_delta(model_table_root)


def _mark_target_column(module, config: dict[str, Any]) -> str:
    target_cfg = config.get("target", {})
    default = getattr(module, "DEFAULT_TARGET_COLUMN", "canonical_subtype_code")
    return str(target_cfg.get("column", default))


def _cache_identity(
    *,
    module_name: str,
    base_config: dict[str, Any],
    family: str,
    model_table_root: str,
    split: str,
    fraction: float,
    seed: int,
    feature_columns: list[str],
    target_column: str | None,
) -> str:
    payload = {
        "version": 2,
        "module": module_name,
        "family": family,
        "model_table_root": str(model_table_root),
        "split": str(split),
        "fraction": float(fraction),
        "sampling_seed": _canonical_sampling_seed(fraction, seed),
        "feature_columns": list(feature_columns),
        "target_column": target_column,
        # Feature config changes can alter resolved feature semantics even if a
        # future resolver happens to emit columns in the same order.
        "features_config": base_config.get("features", {}),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(value, encoding="utf-8")
    temp.replace(path)


def _build_sample_cache_entry(
    *,
    module,
    module_name: str,
    base_config: dict[str, Any],
    family: str,
    model_table_root: str,
    cache_dir: Path,
    split: str,
    fraction: float,
    seed: int,
    rebuild: bool,
) -> tuple[str, str]:
    feature_columns, _ = module._resolve_feature_columns(base_config)
    target_column = _mark_target_column(module, base_config) if family == "mark" else None
    canonical_seed = _canonical_sampling_seed(fraction, seed)

    identity = _cache_identity(
        module_name=module_name,
        base_config=base_config,
        family=family,
        model_table_root=model_table_root,
        split=split,
        fraction=fraction,
        seed=canonical_seed,
        feature_columns=feature_columns,
        target_column=target_column,
    )

    safe_split = sanitize_name(split)
    stem = f"{family}_{safe_split}_f{_fraction_token(fraction)}_{identity}"
    ipc_path = (cache_dir / f"{stem}.arrow").resolve()
    manifest_path = ipc_path.with_suffix(".json")
    lookup_key = _sample_lookup_key(split=split, fraction=fraction, seed=canonical_seed)

    if not rebuild and ipc_path.exists() and manifest_path.exists():
        print(
            f"[cache] reuse {split} fraction={fraction:g}: {ipc_path} "
            f"({ipc_path.stat().st_size / (1024 ** 3):.2f} GiB)"
        )
        return lookup_key, str(ipc_path)

    print(
        f"[cache] BUILD {family} split={split} fraction={fraction:g} "
        f"seed={canonical_seed} from {model_table_root}"
    )

    table = _scan_delta_for_cache(model_table_root)
    if family == "intensity":
        sample_fn = getattr(module, "_deterministic_split_sample")
        frame = sample_fn(
            table=table,
            split=split,
            fraction=float(fraction),
            seed=int(canonical_seed),
            feature_columns=feature_columns,
        )
    else:
        sample_fn = getattr(module, "_deterministic_observed_sample")
        frame = sample_fn(
            table=table,
            split=split,
            fraction=float(fraction),
            seed=int(canonical_seed),
            feature_columns=feature_columns,
            target_column=str(target_column),
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    temp_path = ipc_path.with_suffix(ipc_path.suffix + f".{os.getpid()}.tmp")
    if temp_path.exists():
        temp_path.unlink()

    # Uncompressed Arrow IPC favors read speed and memory mapping on local NVMe.
    frame.write_ipc(temp_path, compression="uncompressed")
    temp_path.replace(ipc_path)

    manifest = {
        "version": 2,
        "family": family,
        "module": module_name,
        "model_table_root": model_table_root,
        "split": split,
        "fraction": float(fraction),
        "sampling_seed": int(canonical_seed),
        "rows": int(frame.height),
        "columns": list(frame.columns),
        "bytes": int(ipc_path.stat().st_size),
        "ipc_path": str(ipc_path),
        "target_column": target_column,
        "feature_count": len(feature_columns),
    }
    _atomic_write_text(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True),
    )

    print(
        f"[cache] ready {split}: rows={frame.height:,}, "
        f"size={ipc_path.stat().st_size / (1024 ** 3):.2f} GiB"
    )
    del frame
    gc.collect()
    return lookup_key, str(ipc_path)


def prepare_stage_sample_cache(
    *,
    module_name: str,
    base_config: dict[str, Any],
    family: str,
    stage: Stage,
    model_table_root: str,
    cache_dir: Path,
    rebuild: bool,
) -> dict[str, str]:
    module = importlib.import_module(module_name)
    data_cfg = base_config["data"]
    base_seed = int(data_cfg["seed"])

    specs = [
        (
            str(data_cfg["train_split"]),
            float(stage.train_fraction),
        ),
        (
            str(data_cfg["validation_split"]),
            float(stage.validation_fraction),
        ),
    ]

    entries: dict[str, str] = {}
    for split, fraction in specs:
        key, path = _build_sample_cache_entry(
            module=module,
            module_name=module_name,
            base_config=base_config,
            family=family,
            model_table_root=model_table_root,
            cache_dir=cache_dir,
            split=split,
            fraction=fraction,
            seed=base_seed,
            rebuild=rebuild,
        )
        entries[key] = path

    return entries


class _NoDeltaScanPolarsProxy:
    """Forward Polars APIs except source access already replaced by stage cache."""

    def __init__(self, real_polars):
        self._real = real_polars

    def __getattr__(self, name: str):
        return getattr(self._real, name)

    def CredentialProviderGCP(self, *args, **kwargs):  # noqa: N802
        return None

    def scan_delta(self, *args, **kwargs):
        # The patched deterministic sample helper ignores the table argument.
        return None


def _memo_token(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_memo_token(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((str(k), _memo_token(v)) for k, v in value.items()))
    # Large DataFrames/arrays should be identified, not serialized.
    return (type(value).__name__, id(value))


def install_worker_stage_cache(
    *,
    module,
    family: str,
    sample_cache_entries: dict[str, str],
    cache_prepared_xy: bool,
    worker_index: int,
) -> None:
    if not sample_cache_entries:
        return

    import polars as real_pl

    sample_fn_name = (
        "_deterministic_split_sample"
        if family == "intensity"
        else "_deterministic_observed_sample"
    )
    original_sample_fn = getattr(module, sample_fn_name)
    loaded_frames: dict[str, Any] = {}

    def cached_sample(*args, **kwargs):
        split = str(kwargs["split"])
        fraction = float(kwargs["fraction"])
        seed = int(kwargs["seed"])
        key = _sample_lookup_key(split=split, fraction=fraction, seed=seed)
        path = sample_cache_entries.get(key)
        if path is None:
            # Defensive fallback for a future trainer/sample shape not prepared
            # by the coordinator.
            return original_sample_fn(*args, **kwargs)

        if path not in loaded_frames:
            print(f"[worker {worker_index}] loading cached sample: {path}")
            loaded_frames[path] = real_pl.read_ipc(path, memory_map=True)
            print(
                f"[worker {worker_index}] cached sample resident: "
                f"rows={loaded_frames[path].height:,}"
            )
        return loaded_frames[path]

    setattr(module, sample_fn_name, cached_sample)

    # train() still constructs a credential provider and a LazyFrame before it
    # calls the deterministic sampling helper. Replace only those source APIs in
    # this worker process: after coordinator cache preparation, train() should
    # not touch Delta/GCS/local source data at all.
    if hasattr(module, "pl"):
        module.pl = _NoDeltaScanPolarsProxy(real_pl)

    if not cache_prepared_xy:
        return

    # Memoize the expensive Polars -> Pandas/NumPy conversion. XGBoost only
    # reads these objects while building QuantileDMatrix, so sequential trials
    # in one GPU worker can safely reuse them.
    original_prepare_xy = getattr(module, "_prepare_xy", None)
    if callable(original_prepare_xy):
        prepared_xy: dict[Any, Any] = {}

        def cached_prepare_xy(frame, *args, **kwargs):
            key = (
                id(frame),
                _memo_token(args),
                _memo_token(kwargs),
            )
            if key not in prepared_xy:
                print(
                    f"[worker {worker_index}] preparing Pandas/NumPy tensors once "
                    f"for cached frame rows={getattr(frame, 'height', 'n/a')}"
                )
                prepared_xy[key] = original_prepare_xy(frame, *args, **kwargs)
            return prepared_xy[key]

        setattr(module, "_prepare_xy", cached_prepare_xy)

    # Cache pure O(N) summaries/vocabulary/validation checks that depend only on
    # the immutable cached frame/arrays. This removes additional repeated CPU
    # passes while preserving model-dependent evaluation.
    original_summary = getattr(module, "_sample_summary", None)
    if callable(original_summary):
        summary_cache: dict[Any, Any] = {}

        def cached_summary(frame, *args, **kwargs):
            key = (id(frame), _memo_token(args), _memo_token(kwargs))
            if key not in summary_cache:
                summary_cache[key] = original_summary(frame, *args, **kwargs)
            return summary_cache[key]

        setattr(module, "_sample_summary", cached_summary)

    if family == "mark":
        original_mapping = getattr(module, "_build_label_mapping", None)
        if callable(original_mapping):
            mapping_cache: dict[Any, Any] = {}

            def cached_mapping(frame, *args, **kwargs):
                key = (id(frame), _memo_token(args), _memo_token(kwargs))
                if key not in mapping_cache:
                    mapping_cache[key] = original_mapping(frame, *args, **kwargs)
                return mapping_cache[key]

            setattr(module, "_build_label_mapping", cached_mapping)

        original_validate = getattr(module, "_validate_labels", None)
        if callable(original_validate):
            validated: set[Any] = set()

            def cached_validate_labels(*args, **kwargs):
                y = kwargs.get("y", args[1] if len(args) > 1 else None)
                num_classes = kwargs.get("num_classes")
                key = (id(y), int(num_classes) if num_classes is not None else None)
                if key not in validated:
                    original_validate(*args, **kwargs)
                    validated.add(key)
                return None

            setattr(module, "_validate_labels", cached_validate_labels)

        original_priors = getattr(module, "_class_priors", None)
        if callable(original_priors):
            priors_cache: dict[Any, Any] = {}

            def cached_class_priors(y, *args, **kwargs):
                key = (id(y), _memo_token(args), _memo_token(kwargs))
                if key not in priors_cache:
                    priors_cache[key] = original_priors(y, *args, **kwargs)
                return priors_cache[key]

            setattr(module, "_class_priors", cached_class_priors)

        original_prior_loss = getattr(module, "_prior_log_loss", None)
        if callable(original_prior_loss):
            loss_cache: dict[Any, Any] = {}

            def cached_prior_loss(y, *args, **kwargs):
                key = (id(y), _memo_token(args), _memo_token(kwargs))
                if key not in loss_cache:
                    loss_cache[key] = original_prior_loss(y, *args, **kwargs)
                return loss_cache[key]

            setattr(module, "_prior_log_loss", cached_prior_loss)

    else:
        original_validate = getattr(module, "_validate_point_process_rows", None)
        if callable(original_validate):
            validated: set[Any] = set()

            def cached_validate_pp(*args, **kwargs):
                y = kwargs.get("y")
                exposure = kwargs.get("exposure")
                tolerance = kwargs.get("event_exposure_tolerance")
                key = (id(y), id(exposure), float(tolerance))
                if key not in validated:
                    original_validate(*args, **kwargs)
                    validated.add(key)
                return None

            setattr(module, "_validate_point_process_rows", cached_validate_pp)

    print(
        f"[worker {worker_index}] stage cache installed: "
        f"samples={len(sample_cache_entries)}, prepared_xy={cache_prepared_xy}"
    )


def cleanup_result_artifacts(result: dict[str, Any]) -> None:
    artifacts = result.get("artifacts", [])
    if not artifacts:
        return

    try:
        artifact_dir = Path(artifacts[0]).resolve().parent
    except Exception:
        return

    if artifact_dir.exists():
        shutil.rmtree(artifact_dir, ignore_errors=True)

    # Each HPO trial uses a unique model.name. Remove that now-empty directory.
    parent = artifact_dir.parent
    try:
        parent.rmdir()
    except OSError:
        pass


def run_train_once(
    *,
    module,
    base_config: dict[str, Any],
    family: str,
    params: dict[str, Any],
    stage: Stage,
    device: str,
    run_label: str,
    seed: int | None,
    keep_artifacts: bool,
    model_table_root: str | None = None,
) -> tuple[float, dict[str, float], int]:
    cfg = copy.deepcopy(base_config)
    apply_params(cfg, params, family=family)
    configure_stage(
        cfg,
        stage=stage,
        device=device,
        seed=seed,
        model_table_root=model_table_root,
    )

    cfg["model"]["name"] = f"hpo_{sanitize_name(run_label)}"

    run_id = uuid.uuid4().hex
    cfg_hash = stable_config_hash(cfg)

    result: dict[str, Any] | None = None
    try:
        result = module.train(
            cfg,
            run_id=run_id,
            config_hash=cfg_hash,
        )

        metrics = result["metrics"]
        metric_key = objective_metric(family)
        score = float(metrics[metric_key])

        if not (score == score and abs(score) != float("inf")):
            raise ValueError(f"Non-finite objective {metric_key}={score}")

        best_iteration = int(float(metrics.get("best_iteration", -1)))

        compact_metrics: dict[str, float] = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }

        return score, compact_metrics, best_iteration

    finally:
        if result is not None and not keep_artifacts:
            cleanup_result_artifacts(result)

        gc.collect()


def trial_objective(
    *,
    module,
    base_config: dict[str, Any],
    family: str,
    space: dict[str, Any],
    stage: Stage,
    device: str,
    study_tag: str,
    keep_artifacts: bool,
    model_table_root: str | None = None,
):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(
            trial,
            family=family,
            space=space,
        )

        label = f"{study_tag}_{stage.name}_trial_{trial.number:05d}"

        try:
            score, metrics, best_iteration = run_train_once(
                module=module,
                base_config=base_config,
                family=family,
                params=params,
                stage=stage,
                device=device,
                run_label=label,
                seed=None,
                keep_artifacts=keep_artifacts,
                model_table_root=model_table_root,
            )
        except Exception:
            trial.set_user_attr("traceback", traceback.format_exc()[-12000:])
            raise

        trial.set_user_attr("best_iteration", best_iteration)

        if family == "intensity":
            for key in (
                "sample_validation_expected_observed",
                "sample_validation_calibration_error_pct",
                "sample_validation_nll_gain_per_event",
                "sample_validation_bits_per_event",
            ):
                if key in metrics:
                    trial.set_user_attr(key, metrics[key])
        else:
            for key in (
                "sample_validation_accuracy",
                "sample_validation_macro_f1",
                "sample_validation_weighted_f1",
                "sample_validation_log_loss_gain",
                "sample_validation_bits_gain",
            ):
                if key in metrics:
                    trial.set_user_attr(key, metrics[key])

        return score

    return objective


def make_sampler(seed: int) -> optuna.samplers.BaseSampler:
    # Multivariate/group TPE learns parameter interactions rather than treating
    # depth, regularization, and sampling as independent knobs.
    return optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=32,
        n_ei_candidates=128,
        multivariate=True,
        group=True,
        constant_liar=True,
    )


def complete_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return sorted(
        (
            t
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            and t.value is not None
        ),
        key=lambda t: float(t.value),
    )


def unique_top_params(
    trials: Iterable[optuna.trial.FrozenTrial],
    *,
    family: str,
    limit: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()

    for trial in trials:
        params = normalized_params_from_trial(trial.params, family=family)
        key = json.dumps(params, sort_keys=True, separators=(",", ":"))
        if key in seen:
            continue
        seen.add(key)
        output.append(params)
        if len(output) >= limit:
            break

    return output


def enqueue_trial_params(
    study: optuna.Study,
    params: dict[str, Any],
    *,
    family: str,
) -> None:
    enqueue = dict(params)

    reg_alpha = float(enqueue.pop("reg_alpha", 0.0))
    enqueue["use_reg_alpha"] = reg_alpha > 0.0
    if reg_alpha > 0.0:
        enqueue["reg_alpha_nonzero"] = reg_alpha

    if family == "mark":
        gamma = float(enqueue.pop("gamma", 0.0))
        enqueue["use_gamma"] = gamma > 0.0
        if gamma > 0.0:
            enqueue["gamma_nonzero"] = gamma

    study.enqueue_trial(enqueue)


def journal_storage(path: Path):
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    path.parent.mkdir(parents=True, exist_ok=True)
    return JournalStorage(
        JournalFileBackend(str(path.resolve()))
    )


def load_journal_study(
    *,
    name: str,
    journal_path: Path,
    sampler_seed: int,
) -> optuna.Study:
    return optuna.create_study(
        study_name=name,
        storage=journal_storage(journal_path),
        direction="minimize",
        sampler=make_sampler(sampler_seed),
        load_if_exists=True,
    )


def _study_counts(study: optuna.Study) -> dict[str, int]:
    counts: dict[str, int] = {}
    for trial in study.get_trials(deepcopy=False):
        name = trial.state.name
        counts[name] = counts.get(name, 0) + 1
    return counts


def _complete_count(study: optuna.Study) -> int:
    return sum(
        1
        for trial in study.get_trials(deepcopy=False)
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
    )


def _fail_stale_running_trials(study: optuna.Study) -> int:
    # This runner intentionally supports one coordinator per study. Therefore,
    # RUNNING trials that exist before new workers are launched are leftovers
    # from an interrupted VM/process and should not remain permanent
    # constant-liar observations. Mark them FAIL before resuming.
    stale = [
        trial
        for trial in study.get_trials(deepcopy=False)
        if trial.state == optuna.trial.TrialState.RUNNING
    ]
    for trial in stale:
        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
    if stale:
        print(
            f"[coordinator] recovered {len(stale)} stale RUNNING trial(s) "
            "from a previous interrupted run -> FAIL."
        )
    return len(stale)


def _optuna_worker(
    *,
    worker_index: int,
    worker_count: int,
    gpu_id: str,
    threads_per_worker: int,
    module_name: str,
    base_config: dict[str, Any],
    family: str,
    space: dict[str, Any],
    stage: Stage,
    device: str,
    study_tag: str,
    keep_artifacts: bool,
    model_table_root: str | None,
    study_name: str,
    journal_path: str,
    target_complete_trials: int,
    sampler_seed: int,
    max_consecutive_failures: int,
    sample_cache_entries: dict[str, str] | None,
    cache_prepared_xy: bool,
) -> None:
    configure_worker_environment(
        gpu_id=gpu_id,
        worker_index=worker_index,
        worker_count=worker_count,
        threads_per_worker=threads_per_worker,
    )

    # Import only after CUDA_VISIBLE_DEVICES/thread limits are configured.
    module = importlib.import_module(module_name)
    if sample_cache_entries:
        install_worker_stage_cache(
            module=module,
            family=family,
            sample_cache_entries=sample_cache_entries,
            cache_prepared_xy=cache_prepared_xy,
            worker_index=worker_index,
        )

    study = load_journal_study(
        name=study_name,
        journal_path=Path(journal_path),
        sampler_seed=sampler_seed + worker_index * 1009,
    )

    objective = trial_objective(
        module=module,
        base_config=base_config,
        family=family,
        space=space,
        stage=stage,
        device=device,
        study_tag=study_tag,
        keep_artifacts=keep_artifacts,
        model_table_root=model_table_root,
    )

    consecutive_failures = 0

    while _complete_count(study) < target_complete_trials:
        captured: list[optuna.trial.FrozenTrial] = []

        def capture(_study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            captured.append(trial)

        study.optimize(
            objective,
            n_trials=1,
            gc_after_trial=True,
            callbacks=[capture],
            catch=(Exception,),
        )

        if captured and captured[-1].state == optuna.trial.TrialState.COMPLETE:
            consecutive_failures = 0
        else:
            consecutive_failures += 1

        if consecutive_failures >= max_consecutive_failures:
            raise RuntimeError(
                f"Worker {worker_index} on GPU {gpu_id} hit "
                f"{consecutive_failures} consecutive failed trials. "
                "This usually indicates a systemic configuration/OOM issue."
            )

    print(
        f"[worker {worker_index}] {stage.name} target reached; exiting GPU {gpu_id}."
    )


def run_parallel_study(
    *,
    name: str,
    journal_path: Path,
    base_config: dict[str, Any],
    family: str,
    space: dict[str, Any],
    stage: Stage,
    device: str,
    study_tag: str,
    keep_artifacts: bool,
    model_table_root: str | None,
    target_complete_trials: int,
    sampler_seed: int,
    enqueued: Iterable[dict[str, Any]],
    gpus: list[str],
    worker_count: int,
    threads_per_worker: int,
    module_name: str,
    monitor_seconds: float,
    max_consecutive_failures: int,
    sample_cache_entries: dict[str, str] | None,
    cache_prepared_xy: bool,
) -> optuna.Study:
    study = load_journal_study(
        name=name,
        journal_path=journal_path,
        sampler_seed=sampler_seed,
    )

    _fail_stale_running_trials(study)

    # Only seed an entirely new stage. A resumed study keeps its original queue.
    if len(study.trials) == 0:
        for params in enqueued:
            enqueue_trial_params(study, params, family=family)

    already_complete = _complete_count(study)
    if already_complete >= target_complete_trials:
        print(
            f"[coordinator] {stage.name}: already has {already_complete} complete "
            f"trials (target={target_complete_trials}); skipping workers."
        )
        return study

    active_gpus = gpus[:worker_count]
    print(
        f"[coordinator] {stage.name}: launching {len(active_gpus)} workers on "
        f"GPUs {active_gpus}; complete={already_complete}/{target_complete_trials}."
    )

    ctx = mp.get_context("spawn")
    processes: list[mp.Process] = []

    for worker_index, gpu_id in enumerate(active_gpus):
        process = ctx.Process(
            target=_optuna_worker,
            kwargs={
                "worker_index": worker_index,
                "worker_count": len(active_gpus),
                "gpu_id": gpu_id,
                "threads_per_worker": threads_per_worker,
                "module_name": module_name,
                "base_config": base_config,
                "family": family,
                "space": space,
                "stage": stage,
                "device": device,
                "study_tag": study_tag,
                "keep_artifacts": keep_artifacts,
                "model_table_root": model_table_root,
                "study_name": name,
                "journal_path": str(journal_path),
                "target_complete_trials": target_complete_trials,
                "sampler_seed": sampler_seed,
                "max_consecutive_failures": max_consecutive_failures,
                "sample_cache_entries": sample_cache_entries,
                "cache_prepared_xy": cache_prepared_xy,
            },
            name=f"{stage.name}-gpu-{gpu_id}",
        )
        process.start()
        processes.append(process)

    next_report = 0.0
    while any(process.is_alive() for process in processes):
        now = time.monotonic()
        if now >= next_report:
            refreshed = load_journal_study(
                name=name,
                journal_path=journal_path,
                sampler_seed=sampler_seed,
            )
            counts = _study_counts(refreshed)
            completed = counts.get("COMPLETE", 0)
            try:
                best = f"{refreshed.best_value:.12f}"
            except ValueError:
                best = "n/a"
            print(
                f"[coordinator] {stage.name}: complete={completed}/"
                f"{target_complete_trials}, states={counts}, best={best}"
            )
            next_report = now + max(5.0, monitor_seconds)

        for process in processes:
            process.join(timeout=0.25)
        time.sleep(0.25)

    for process in processes:
        process.join()

    final_study = load_journal_study(
        name=name,
        journal_path=journal_path,
        sampler_seed=sampler_seed,
    )
    complete = _complete_count(final_study)
    bad_exits = [
        (process.name, process.exitcode)
        for process in processes
        if process.exitcode not in (0, None)
    ]

    if complete < target_complete_trials:
        raise RuntimeError(
            f"{stage.name} ended with only {complete}/{target_complete_trials} "
            f"successful trials. Worker exits={bad_exits}. If workers were killed "
            "for host RAM, lower the stage worker count and rerun; JournalStorage "
            "will resume the completed trials."
        )

    if bad_exits:
        print(
            f"[coordinator] WARNING: some {stage.name} workers exited nonzero "
            f"after the global target was reached: {bad_exits}"
        )

    # Concurrent workers can overshoot the target by at most roughly workers-1.
    print(
        f"[coordinator] {stage.name} complete: {_complete_count(final_study)} "
        f"successful trials."
    )
    return final_study


def _tournament_worker(
    *,
    worker_index: int,
    worker_count: int,
    gpu_id: str,
    threads_per_worker: int,
    module_name: str,
    base_config: dict[str, Any],
    family: str,
    stage: Stage,
    device: str,
    study_tag: str,
    keep_artifacts: bool,
    model_table_root: str | None,
    sample_cache_entries: dict[str, str] | None,
    cache_prepared_xy: bool,
    task_queue,
    result_queue,
) -> None:
    configure_worker_environment(
        gpu_id=gpu_id,
        worker_index=worker_index,
        worker_count=worker_count,
        threads_per_worker=threads_per_worker,
    )
    module = importlib.import_module(module_name)
    if sample_cache_entries:
        install_worker_stage_cache(
            module=module,
            family=family,
            sample_cache_entries=sample_cache_entries,
            cache_prepared_xy=cache_prepared_xy,
            worker_index=worker_index,
        )

    while True:
        task = task_queue.get()
        if task is None:
            return

        task_id = task["task_id"]
        try:
            score, metrics, best_iteration = run_train_once(
                module=module,
                base_config=base_config,
                family=family,
                params=task["params"],
                stage=stage,
                device=device,
                run_label=task["run_label"],
                seed=int(task["seed"]),
                keep_artifacts=keep_artifacts,
                model_table_root=model_table_root,
            )
            result_queue.put(
                {
                    "task_id": task_id,
                    "ok": True,
                    "finalist_index": int(task["finalist_index"]),
                    "seed": int(task["seed"]),
                    "params": task["params"],
                    "score": score,
                    "best_iteration": best_iteration,
                    "metrics": metrics,
                    "gpu_id": gpu_id,
                }
            )
        except Exception:
            result_queue.put(
                {
                    "task_id": task_id,
                    "ok": False,
                    "finalist_index": int(task["finalist_index"]),
                    "seed": int(task["seed"]),
                    "params": task["params"],
                    "gpu_id": gpu_id,
                    "error": traceback.format_exc()[-16000:],
                }
            )


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True)
    temp.replace(path)


def run_parallel_tournament(
    *,
    base_config: dict[str, Any],
    family: str,
    finalists: list[dict[str, Any]],
    seeds: list[int],
    stage: Stage,
    device: str,
    study_tag: str,
    keep_artifacts: bool,
    model_table_root: str | None,
    gpus: list[str],
    worker_count: int,
    threads_per_worker: int,
    module_name: str,
    monitor_seconds: float,
    state_path: Path,
    sample_cache_entries: dict[str, str] | None,
    cache_prepared_xy: bool,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for finalist_index, params in enumerate(finalists, start=1):
        for seed in seeds:
            task_id = f"f{finalist_index:02d}_seed_{seed}"
            tasks.append(
                {
                    "task_id": task_id,
                    "finalist_index": finalist_index,
                    "seed": seed,
                    "params": params,
                    "run_label": f"{study_tag}_tournament_{task_id}",
                }
            )

    saved: dict[str, dict[str, Any]] = {}
    if state_path.exists():
        try:
            raw = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                saved = {
                    str(key): value
                    for key, value in raw.items()
                    if isinstance(value, dict) and value.get("ok") is True
                }
        except Exception:
            print(
                f"[coordinator] WARNING: could not read tournament resume state "
                f"{state_path}; starting tournament state from scratch."
            )

    pending = [task for task in tasks if task["task_id"] not in saved]
    print(
        f"[coordinator] tournament: {len(saved)}/{len(tasks)} tasks already complete; "
        f"{len(pending)} pending."
    )

    if pending:
        active_gpus = gpus[: min(worker_count, len(pending))]
        ctx = mp.get_context("spawn")
        task_queue = ctx.Queue()
        result_queue = ctx.Queue()

        for task in pending:
            task_queue.put(task)
        for _ in active_gpus:
            task_queue.put(None)

        processes: list[mp.Process] = []
        for worker_index, gpu_id in enumerate(active_gpus):
            process = ctx.Process(
                target=_tournament_worker,
                kwargs={
                    "worker_index": worker_index,
                    "worker_count": len(active_gpus),
                    "gpu_id": gpu_id,
                    "threads_per_worker": threads_per_worker,
                    "module_name": module_name,
                    "base_config": base_config,
                    "family": family,
                    "stage": stage,
                    "device": device,
                    "study_tag": study_tag,
                    "keep_artifacts": keep_artifacts,
                    "model_table_root": model_table_root,
                    "sample_cache_entries": sample_cache_entries,
                    "cache_prepared_xy": cache_prepared_xy,
                    "task_queue": task_queue,
                    "result_queue": result_queue,
                },
                name=f"tournament-gpu-{gpu_id}",
            )
            process.start()
            processes.append(process)

        received = 0
        expected = len(pending)
        errors: list[dict[str, Any]] = []

        while received < expected:
            try:
                result = result_queue.get(timeout=max(5.0, monitor_seconds))
            except queue_module.Empty:
                alive = sum(process.is_alive() for process in processes)
                print(
                    f"[coordinator] tournament: received={received}/{expected}, "
                    f"workers_alive={alive}"
                )
                if alive == 0:
                    break
                continue

            received += 1
            if result.get("ok") is True:
                saved[result["task_id"]] = result
                _atomic_write_json(state_path, saved)
                print(
                    f"[coordinator] tournament {result['task_id']} complete on "
                    f"GPU {result['gpu_id']}: {objective_metric(family)}="
                    f"{float(result['score']):.12f}"
                )
            else:
                errors.append(result)
                print(
                    f"[coordinator] tournament {result['task_id']} FAILED on "
                    f"GPU {result.get('gpu_id')}"
                )

        for process in processes:
            process.join()

        missing = [task["task_id"] for task in tasks if task["task_id"] not in saved]
        if errors or missing:
            details = "\n\n".join(
                f"{item.get('task_id')}:\n{item.get('error', 'missing result')}"
                for item in errors
            )
            raise RuntimeError(
                "Tournament did not complete every finalist/seed task. "
                f"Missing={missing}.\n{details}\n"
                "Successful tournament tasks were persisted and will be skipped on rerun."
            )

    records: list[dict[str, Any]] = []
    for finalist_index, params in enumerate(finalists, start=1):
        seed_records = [
            saved[f"f{finalist_index:02d}_seed_{seed}"]
            for seed in seeds
        ]
        seed_scores = [float(item["score"]) for item in seed_records]
        mean_score = sum(seed_scores) / len(seed_scores)
        variance = sum((x - mean_score) ** 2 for x in seed_scores) / len(seed_scores)

        records.append(
            {
                "rank_input": finalist_index,
                "params": params,
                "mean_score": mean_score,
                "std_score": variance ** 0.5,
                "seed_results": [
                    {
                        "seed": int(item["seed"]),
                        "score": float(item["score"]),
                        "best_iteration": int(item["best_iteration"]),
                        "metrics": item["metrics"],
                        "gpu_id": item.get("gpu_id"),
                    }
                    for item in seed_records
                ],
            }
        )

    records.sort(key=lambda item: item["mean_score"])
    for rank, record in enumerate(records, start=1):
        record["rank"] = rank

    return records


def write_yaml(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
            width=100,
        )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()

    base_config = load_yaml(args.config)
    family: str = args.family
    seeds = parse_seeds(args.tournament_seeds)
    gpus = parse_gpus(args.gpus)
    study_tag = sanitize_name(args.study_name)

    explore_workers = resolved_worker_count(args.explore_workers, gpus)
    refine_workers = resolved_worker_count(args.refine_workers, gpus)
    tournament_workers = resolved_worker_count(args.tournament_workers, gpus)

    # Resolve once in the coordinator so import errors fail fast. Spawned workers
    # import this module again only after their CUDA/thread environment is isolated.
    resolved_module = resolve_training_module(
        base_config,
        family=family,
        explicit=args.module,
    )
    module_name = resolved_module.__name__
    del resolved_module

    space = build_space(base_config, family)

    root = args.output_dir / study_tag
    root.mkdir(parents=True, exist_ok=True)

    stage_cache_dir = (
        args.stage_cache_dir
        if args.stage_cache_dir is not None
        else root / "stage_cache"
    )
    stage_cache_dir = stage_cache_dir.resolve()
    hpo_table_root = _hpo_table_root(base_config, args.hpo_model_table_root)
    cache_prepared_xy = not args.no_prepared_xy_cache

    explore_stage = Stage(
        name="explore",
        train_fraction=args.explore_train_fraction,
        validation_fraction=args.explore_validation_fraction,
        num_boost_round=args.explore_rounds,
        early_stopping_rounds=args.explore_early_stop,
    )
    refine_stage = Stage(
        name="refine",
        train_fraction=args.refine_train_fraction,
        validation_fraction=args.refine_validation_fraction,
        num_boost_round=args.refine_rounds,
        early_stopping_rounds=args.refine_early_stop,
    )
    tournament_stage = Stage(
        name="tournament",
        train_fraction=1.0,
        validation_fraction=1.0,
        num_boost_round=args.tournament_rounds,
        early_stopping_rounds=args.tournament_early_stop,
    )

    print(f"Family:              {family}")
    print(f"Base config:         {args.config}")
    print(f"Base model:          {base_config['model']['name']}")
    print(f"Training module:     {module_name}")
    print(f"Objective:           {objective_metric(family)}")
    print(f"Device:              {args.device}")
    print(f"Physical GPUs:       {gpus}")
    print(f"Explore workers:     {explore_workers}")
    print(f"Refine workers:      {refine_workers}")
    print(f"Tournament workers:  {tournament_workers}")
    print(f"Threads/worker:      {'auto' if args.threads_per_worker <= 0 else args.threads_per_worker}")
    print(f"Study output:        {root.resolve()}")
    print(f"HPO table override:  {args.hpo_model_table_root or 'none (base config)'}")
    print(f"HPO source table:    {hpo_table_root}")
    print(f"Stage cache:         {'disabled' if args.disable_stage_cache else stage_cache_dir}")
    print(f"Prepared XY cache:   {cache_prepared_xy and not args.disable_stage_cache}")
    print(f"Search depth:        {space['depth_low']}..{space['depth_high']}")
    print(f"Explore target:      {args.explore_trials} successful trials")
    print(f"Refine target:       {args.refine_trials} successful trials")
    print(f"Finalists:           {args.finalists}")
    print(f"Tournament seeds:    {seeds}")
    print("Parallel mode:       one independent XGBoost trial per GPU")
    print("TEST SPLIT WILL NOT BE ACCESSED.")

    if (root / "explore.db").exists() and not (root / "explore.journal").exists():
        print(
            "WARNING: found an older SQLite exploration study. The distributed "
            "runner uses JournalStorage and intentionally starts a separate journal study."
        )

    base_seed_params = params_for_enqueue(
        base_config,
        family=family,
        space=space,
    )
    initial_enqueue = [base_seed_params] if base_seed_params is not None else []

    # Stage 1: broad distributed exploration. Build each deterministic sample
    # once; resumed studies reuse the immutable Arrow cache.
    explore_cache = (
        None
        if args.disable_stage_cache
        else prepare_stage_sample_cache(
            module_name=module_name,
            base_config=base_config,
            family=family,
            stage=explore_stage,
            model_table_root=hpo_table_root,
            cache_dir=stage_cache_dir,
            rebuild=args.rebuild_stage_cache,
        )
    )

    explore_study = run_parallel_study(
        name=f"{study_tag}__explore",
        journal_path=root / "explore.journal",
        base_config=base_config,
        family=family,
        space=space,
        stage=explore_stage,
        device=args.device,
        study_tag=study_tag,
        keep_artifacts=args.keep_trial_artifacts,
        model_table_root=args.hpo_model_table_root,
        target_complete_trials=args.explore_trials,
        sampler_seed=42,
        enqueued=initial_enqueue,
        gpus=gpus,
        worker_count=explore_workers,
        threads_per_worker=args.threads_per_worker,
        module_name=module_name,
        monitor_seconds=args.monitor_seconds,
        max_consecutive_failures=args.max_consecutive_failures,
        sample_cache_entries=explore_cache,
        cache_prepared_xy=cache_prepared_xy,
    )

    explore_complete = complete_trials(explore_study)
    if not explore_complete:
        raise RuntimeError("Exploration produced no successful trials.")

    top_explore = unique_top_params(
        explore_complete,
        family=family,
        limit=args.seed_top_k,
    )

    print(
        f"\nExploration best: {explore_study.best_value:.12f}\n"
        f"{json.dumps(normalized_params_from_trial(explore_study.best_trial.params, family=family), indent=2, sort_keys=True)}"
    )

    # Stage 2: distributed refinement. Full-train cache is built once; the
    # 25% validation cache is automatically reused from exploration.
    refine_cache = (
        None
        if args.disable_stage_cache
        else prepare_stage_sample_cache(
            module_name=module_name,
            base_config=base_config,
            family=family,
            stage=refine_stage,
            model_table_root=hpo_table_root,
            cache_dir=stage_cache_dir,
            rebuild=args.rebuild_stage_cache,
        )
    )

    refine_study = run_parallel_study(
        name=f"{study_tag}__refine",
        journal_path=root / "refine.journal",
        base_config=base_config,
        family=family,
        space=space,
        stage=refine_stage,
        device=args.device,
        study_tag=study_tag,
        keep_artifacts=args.keep_trial_artifacts,
        model_table_root=args.hpo_model_table_root,
        target_complete_trials=args.refine_trials,
        sampler_seed=1337,
        enqueued=top_explore,
        gpus=gpus,
        worker_count=refine_workers,
        threads_per_worker=args.threads_per_worker,
        module_name=module_name,
        monitor_seconds=args.monitor_seconds,
        max_consecutive_failures=args.max_consecutive_failures,
        sample_cache_entries=refine_cache,
        cache_prepared_xy=cache_prepared_xy,
    )

    refine_complete = complete_trials(refine_study)
    if not refine_complete:
        raise RuntimeError("Refinement produced no successful trials.")

    finalists = unique_top_params(
        refine_complete,
        family=family,
        limit=args.finalists,
    )

    print(
        f"\nRefinement best: {refine_study.best_value:.12f}\n"
        f"{json.dumps(normalized_params_from_trial(refine_study.best_trial.params, family=family), indent=2, sort_keys=True)}"
    )

    # Stage 3: 100% fractions make sampled rows seed-invariant, so tournament
    # reuses refinement's full-train cache and builds full validation once.
    tournament_cache = (
        None
        if args.disable_stage_cache
        else prepare_stage_sample_cache(
            module_name=module_name,
            base_config=base_config,
            family=family,
            stage=tournament_stage,
            model_table_root=hpo_table_root,
            cache_dir=stage_cache_dir,
            rebuild=args.rebuild_stage_cache,
        )
    )

    tournament = run_parallel_tournament(
        base_config=base_config,
        family=family,
        finalists=finalists,
        seeds=seeds,
        stage=tournament_stage,
        device=args.device,
        study_tag=study_tag,
        keep_artifacts=args.keep_trial_artifacts,
        model_table_root=args.hpo_model_table_root,
        gpus=gpus,
        worker_count=tournament_workers,
        threads_per_worker=args.threads_per_worker,
        module_name=module_name,
        monitor_seconds=args.monitor_seconds,
        state_path=root / "tournament_state.json",
        sample_cache_entries=tournament_cache,
        cache_prepared_xy=cache_prepared_xy,
    )

    best = tournament[0]
    best_params = best["params"]

    # Export the final official config. Crucially, do NOT preserve an HPO-only
    # local model-table override: the canonical base config path is restored.
    final_cfg = copy.deepcopy(base_config)
    apply_params(final_cfg, best_params, family=family)
    configure_stage(
        final_cfg,
        stage=tournament_stage,
        device=args.device,
        seed=seeds[0],
        model_table_root=None,
    )

    final_cfg["model"]["name"] = final_model_name(
        str(base_config["model"]["name"]),
        args.final_model_name,
    )

    final_cfg["hpo"] = {
        "study_name": args.study_name,
        "family": family,
        "selection_metric": objective_metric(family),
        "tournament_mean_score": best["mean_score"],
        "tournament_std_score": best["std_score"],
        "tournament_seeds": seeds,
        "source_config": str(args.config),
        "parallel_strategy": "one_trial_per_gpu",
        "physical_gpus": gpus,
        "explore_workers": explore_workers,
        "refine_workers": refine_workers,
        "tournament_workers": tournament_workers,
        "optuna_storage": "JournalStorage/JournalFileBackend",
        "hpo_model_table_root": args.hpo_model_table_root,
        "stage_cache_enabled": not args.disable_stage_cache,
        "stage_cache_dir": str(stage_cache_dir) if not args.disable_stage_cache else None,
        "prepared_xy_cache": cache_prepared_xy and not args.disable_stage_cache,
        "test_split_used": False,
    }

    best_yaml = root / "best_config.yaml"
    tournament_json = root / "tournament_results.json"
    summary_json = root / "summary.json"

    write_yaml(best_yaml, final_cfg)
    write_json(tournament_json, tournament)

    summary = {
        "family": family,
        "study_name": args.study_name,
        "objective": objective_metric(family),
        "base_config": str(args.config),
        "base_model_name": base_config["model"]["name"],
        "explore_complete_trials": len(explore_complete),
        "refine_complete_trials": len(refine_complete),
        "explore_best": float(explore_study.best_value),
        "refine_best": float(refine_study.best_value),
        "tournament_best_mean": float(best["mean_score"]),
        "tournament_best_std": float(best["std_score"]),
        "best_params": best_params,
        "best_config": str(best_yaml),
        "gpus": gpus,
        "stage_cache_enabled": not args.disable_stage_cache,
        "stage_cache_dir": str(stage_cache_dir) if not args.disable_stage_cache else None,
        "prepared_xy_cache": cache_prepared_xy and not args.disable_stage_cache,
        "test_split_used": False,
    }
    write_json(summary_json, summary)

    print("\n" + "=" * 80)
    print("DISTRIBUTED HPO COMPLETE")
    print("=" * 80)
    print(f"Best tournament mean: {best['mean_score']:.12f}")
    print(f"Best tournament std:  {best['std_score']:.12f}")
    print("Best parameters:")
    print(json.dumps(best_params, indent=2, sort_keys=True))
    print(f"\nExported config:      {best_yaml}")
    print(f"Tournament results:   {tournament_json}")
    print(f"Summary:              {summary_json}")
    print("\nTEST SPLIT HAS NOT BEEN ACCESSED.")
    print(
        "\nNext: run the exported YAML once through the normal "
        "machine_learning.experiments.orchestrator to create the official "
        "MLflow baseline run, then run full validation and finally the frozen test."
    )


if __name__ == "__main__":
    main()

    