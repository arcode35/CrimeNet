from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib
import json
import re
import shutil
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
        description="Aggressive Optuna tuning for CrimeNet XGBoost baselines."
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
        help="Stable study name. Reusing it resumes SQLite studies.",
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
        help="Root for Optuna DBs, summaries, and exported YAML.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="XGBoost device written into architecture.device.",
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
) -> tuple[float, dict[str, float], int]:
    cfg = copy.deepcopy(base_config)
    apply_params(cfg, params, family=family)
    configure_stage(cfg, stage=stage, device=device, seed=seed)

    base_model_name = str(base_config["model"]["name"])
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
            )
        except Exception:
            trial.set_user_attr("traceback", traceback.format_exc()[-12000:])
            raise

        trial.set_user_attr("best_iteration", best_iteration)

        # Useful diagnostics without changing the primary proper-scoring objective.
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


def study_storage_url(path: Path) -> str:
    # SQLite URL wants POSIX-style path even on Windows.
    return f"sqlite:///{path.resolve().as_posix()}"


def run_study(
    *,
    name: str,
    db_path: Path,
    objective,
    n_trials: int,
    seed: int,
    enqueued: Iterable[dict[str, Any]],
    family: str,
) -> optuna.Study:
    study = optuna.create_study(
        study_name=name,
        storage=study_storage_url(db_path),
        direction="minimize",
        sampler=make_sampler(seed),
        load_if_exists=True,
    )

    # Only enqueue when the study is empty, so a resume does not duplicate seeds.
    if len(study.trials) == 0:
        for params in enqueued:
            enqueue_trial_params(study, params, family=family)

    study.optimize(
        objective,
        n_trials=n_trials,
        gc_after_trial=True,
        # Failed/OOM parameter combinations should not kill a long search.
        # KeyboardInterrupt/SystemExit remain interruptible.
        catch=(Exception,),
    )

    return study


def run_tournament(
    *,
    module,
    base_config: dict[str, Any],
    family: str,
    finalists: list[dict[str, Any]],
    seeds: list[int],
    stage: Stage,
    device: str,
    study_tag: str,
    keep_artifacts: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for index, params in enumerate(finalists, start=1):
        seed_scores: list[float] = []
        seed_records: list[dict[str, Any]] = []

        print(
            f"\n{'=' * 80}\n"
            f"TOURNAMENT FINALIST {index}/{len(finalists)}\n"
            f"{json.dumps(params, indent=2, sort_keys=True)}\n"
            f"{'=' * 80}"
        )

        for seed in seeds:
            label = f"{study_tag}_tournament_f{index:02d}_seed_{seed}"

            score, metrics, best_iteration = run_train_once(
                module=module,
                base_config=base_config,
                family=family,
                params=params,
                stage=stage,
                device=device,
                run_label=label,
                seed=seed,
                keep_artifacts=keep_artifacts,
            )

            seed_scores.append(score)
            seed_records.append(
                {
                    "seed": seed,
                    "score": score,
                    "best_iteration": best_iteration,
                    "metrics": metrics,
                }
            )

            print(
                f"Finalist {index} seed {seed}: "
                f"{objective_metric(family)}={score:.12f}, "
                f"best_iteration={best_iteration}"
            )

        mean_score = sum(seed_scores) / len(seed_scores)
        variance = (
            sum((x - mean_score) ** 2 for x in seed_scores) / len(seed_scores)
        )
        std_score = variance ** 0.5

        records.append(
            {
                "rank_input": index,
                "params": params,
                "mean_score": mean_score,
                "std_score": std_score,
                "seed_results": seed_records,
            }
        )

    records.sort(key=lambda x: x["mean_score"])

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
    study_tag = sanitize_name(args.study_name)

    module = resolve_training_module(
        base_config,
        family=family,
        explicit=args.module,
    )

    space = build_space(base_config, family)

    root = args.output_dir / study_tag
    root.mkdir(parents=True, exist_ok=True)

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

    print(f"Family:            {family}")
    print(f"Base config:       {args.config}")
    print(f"Base model:        {base_config['model']['name']}")
    print(f"Objective:         {objective_metric(family)}")
    print(f"Device:            {args.device}")
    print(f"Study output:      {root.resolve()}")
    print(f"Search depth:      {space['depth_low']}..{space['depth_high']}")
    print(f"Explore trials:    {args.explore_trials}")
    print(f"Refine trials:     {args.refine_trials}")
    print(f"Finalists:         {args.finalists}")
    print(f"Tournament seeds:  {seeds}")
    print("TEST SPLIT WILL NOT BE ACCESSED.")

    base_seed_params = params_for_enqueue(
        base_config,
        family=family,
        space=space,
    )
    initial_enqueue = [base_seed_params] if base_seed_params is not None else []

    # ------------------------------------------------------------------
    # Stage 1: broad exploration
    # ------------------------------------------------------------------
    explore_objective = trial_objective(
        module=module,
        base_config=base_config,
        family=family,
        space=space,
        stage=explore_stage,
        device=args.device,
        study_tag=study_tag,
        keep_artifacts=args.keep_trial_artifacts,
    )

    explore_study = run_study(
        name=f"{study_tag}__explore",
        db_path=root / "explore.db",
        objective=explore_objective,
        n_trials=args.explore_trials,
        seed=42,
        enqueued=initial_enqueue,
        family=family,
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

    # ------------------------------------------------------------------
    # Stage 2: refine with full training data.
    # Seed it with top Stage-1 configs, then let TPE continue exploring.
    # ------------------------------------------------------------------
    refine_objective = trial_objective(
        module=module,
        base_config=base_config,
        family=family,
        space=space,
        stage=refine_stage,
        device=args.device,
        study_tag=study_tag,
        keep_artifacts=args.keep_trial_artifacts,
    )

    refine_study = run_study(
        name=f"{study_tag}__refine",
        db_path=root / "refine.db",
        objective=refine_objective,
        n_trials=args.refine_trials,
        seed=1337,
        enqueued=top_explore,
        family=family,
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

    # ------------------------------------------------------------------
    # Stage 3: full 2014-2023 train + full 2024 validation tournament.
    # Multiple seeds distinguish real hyperparameter signal from RNG noise.
    # ------------------------------------------------------------------
    tournament = run_tournament(
        module=module,
        base_config=base_config,
        family=family,
        finalists=finalists,
        seeds=seeds,
        stage=tournament_stage,
        device=args.device,
        study_tag=study_tag,
        keep_artifacts=args.keep_trial_artifacts,
    )

    best = tournament[0]
    best_params = best["params"]

    # Export final official config.
    final_cfg = copy.deepcopy(base_config)
    apply_params(final_cfg, best_params, family=family)
    configure_stage(
        final_cfg,
        stage=tournament_stage,
        device=args.device,
        # Official baseline uses the canonical seed. Tournament ranking itself
        # used multiple seeds to make hyperparameter selection robust.
        seed=seeds[0],
    )

    final_cfg["model"]["name"] = final_model_name(
        str(base_config["model"]["name"]),
        args.final_model_name,
    )

    # HPO metadata is additive; model trainers can ignore it.
    final_cfg["hpo"] = {
        "study_name": args.study_name,
        "family": family,
        "selection_metric": objective_metric(family),
        "tournament_mean_score": best["mean_score"],
        "tournament_std_score": best["std_score"],
        "tournament_seeds": seeds,
        "source_config": str(args.config),
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
        "explore_best": float(explore_study.best_value),
        "refine_best": float(refine_study.best_value),
        "tournament_best_mean": float(best["mean_score"]),
        "tournament_best_std": float(best["std_score"]),
        "best_params": best_params,
        "best_config": str(best_yaml),
        "test_split_used": False,
    }
    write_json(summary_json, summary)

    print("\n" + "=" * 80)
    print("HPO COMPLETE")
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