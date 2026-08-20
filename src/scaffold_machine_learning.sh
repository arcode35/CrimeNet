#!/usr/bin/env bash
set -euo pipefail

# Run this script FROM the repository's src/ directory.

if [[ ! -d "crimenet_data" ]]; then
  echo "ERROR: Run this from src/. Expected ./crimenet_data to exist."
  exit 1
fi

echo "Scaffolding machine_learning/ ..."

BACKUP_ROOT="machine_learning/.scaffold_backup/$(date +%Y%m%d_%H%M%S)"

backup_if_exists() {
  local target="$1"

  if [[ -f "$target" ]]; then
    local relative="${target#machine_learning/}"
    local destination="$BACKUP_ROOT/$relative"

    mkdir -p "$(dirname "$destination")"
    cp "$target" "$destination"

    echo "Backed up: $target"
  fi
}

mkdir -p \
  machine_learning/artifacts/experiments \
  machine_learning/artifacts/mlflow/artifacts \
  machine_learning/experiments/log \
  machine_learning/models/xgboost/configs

touch \
  machine_learning/__init__.py \
  machine_learning/experiments/__init__.py \
  machine_learning/models/__init__.py \
  machine_learning/models/xgboost/__init__.py \
  machine_learning/experiments/log/experiments.jsonl

backup_if_exists machine_learning/experiments/mlflow_config.py
backup_if_exists machine_learning/experiments/experiment_logging.py
backup_if_exists machine_learning/experiments/orchestrator.py
backup_if_exists machine_learning/models/xgboost/configs/baseline_v1.yaml
backup_if_exists machine_learning/models/xgboost/model.py
backup_if_exists machine_learning/README.md
backup_if_exists machine_learning/.gitignore

cat > machine_learning/experiments/mlflow_config.py <<'PY_MLFLOW'
from __future__ import annotations

from pathlib import Path

import mlflow


MACHINE_LEARNING_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

MLFLOW_ROOT = (
    MACHINE_LEARNING_ROOT
    / "artifacts"
    / "mlflow"
)

MLFLOW_DB = (
    MLFLOW_ROOT
    / "mlflow.db"
)

MLFLOW_ARTIFACT_ROOT = (
    MLFLOW_ROOT
    / "artifacts"
)

EXPERIMENT_NAME = (
    "crimenet-model-development"
)

EXPERIMENT_TAGS = {
    "project": "CrimeNet",
}


MLFLOW_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

MLFLOW_ARTIFACT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


TRACKING_URI = (
    f"sqlite:///{MLFLOW_DB}"
)

EXPECTED_ARTIFACT_URI = (
    MLFLOW_ARTIFACT_ROOT
    .resolve()
    .as_uri()
)


mlflow.set_tracking_uri(
    TRACKING_URI
)


experiment = (
    mlflow.get_experiment_by_name(
        EXPERIMENT_NAME
    )
)


if experiment is None:
    EXPERIMENT_ID = (
        mlflow.create_experiment(
            name=EXPERIMENT_NAME,
            artifact_location=
                EXPECTED_ARTIFACT_URI,
            tags=EXPERIMENT_TAGS,
        )
    )

else:
    actual_artifact_uri = (
        experiment
        .artifact_location
        .rstrip("/")
    )

    expected_artifact_uri = (
        EXPECTED_ARTIFACT_URI
        .rstrip("/")
    )

    if (
        actual_artifact_uri
        != expected_artifact_uri
    ):
        raise RuntimeError(
            "Existing MLflow experiment uses the wrong "
            "artifact location.\n"
            f"Expected: {expected_artifact_uri}\n"
            f"Actual:   {actual_artifact_uri}"
        )

    EXPERIMENT_ID = (
        experiment.experiment_id
    )


def start_run(
    *,
    run_name: str,
):
    return mlflow.start_run(
        experiment_id=
            EXPERIMENT_ID,
        run_name=
            run_name,
    )
PY_MLFLOW

cat > machine_learning/experiments/experiment_logging.py <<'PY_LOGGING'
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import yaml


MACHINE_LEARNING_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

EXPERIMENT_LOG = (
    MACHINE_LEARNING_ROOT
    / "experiments"
    / "log"
    / "experiments.jsonl"
)


def canonical_json(
    data: dict[str, Any],
) -> str:
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def config_hash(
    config: dict[str, Any],
) -> str:
    return (
        hashlib
        .sha256(
            canonical_json(
                config
            ).encode()
        )
        .hexdigest()[:12]
    )


def flatten(
    data: dict[str, Any],
    prefix: str = "",
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for key, value in data.items():
        name = (
            f"{prefix}.{key}"
            if prefix
            else key
        )

        if isinstance(
            value,
            dict,
        ):
            result.update(
                flatten(
                    value,
                    name,
                )
            )

        elif isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
            ),
        ) or value is None:
            result[name] = value

        else:
            result[name] = json.dumps(
                value,
                sort_keys=True,
                default=str,
            )

    return result


def git_commit() -> str | None:
    try:
        return (
            subprocess
            .check_output(
                [
                    "git",
                    "rev-parse",
                    "HEAD",
                ],
                text=True,
                stderr=
                    subprocess.DEVNULL,
            )
            .strip()
        )
    except Exception:
        return None


def git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
            ],
            text=True,
            capture_output=True,
            check=True,
        )

        return bool(
            result.stdout.strip()
        )
    except Exception:
        return None


def log_mlflow_result(
    *,
    config: dict[str, Any],
    result: dict[str, Any],
    config_hash: str,
) -> None:
    model = config[
        "model"
    ]

    mlflow.set_tags(
        {
            "model":
                model["name"],

            "model_family":
                model.get(
                    "family",
                    "unknown",
                ),

            "config_hash":
                config_hash,

            "git_commit":
                git_commit()
                or "unknown",

            "git_dirty":
                str(
                    git_dirty()
                ),

            "run_status":
                "completed",
        }
    )

    mlflow.log_params(
        flatten(
            config
        )
    )

    mlflow.log_params(
        {
            "runtime.python":
                sys.version.split()[0],

            "runtime.platform":
                platform.platform(),
        }
    )

    mlflow.log_text(
        yaml.safe_dump(
            config,
            sort_keys=False,
        ),
        "config/resolved_config.yaml",
    )

    metrics = result.get(
        "metrics",
        {},
    )

    if metrics:
        mlflow.log_metrics(
            {
                key:
                    float(value)

                for key, value
                in metrics.items()
            }
        )

    history = result.get(
        "history",
        {},
    )

    for dataset, metrics in (
        history.items()
    ):
        for metric, values in (
            metrics.items()
        ):
            for step, value in enumerate(
                values
            ):
                mlflow.log_metric(
                    f"{dataset}.{metric}",
                    float(value),
                    step=step,
                )

    for raw_path in result.get(
        "artifacts",
        [],
    ):
        path = Path(
            raw_path
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing experiment artifact: {path}"
            )

        mlflow.log_artifact(
            str(path),
            artifact_path=
                "native_artifacts",
        )


def log_experiment(
    *,
    config: dict[str, Any],
    result: dict[str, Any] | None,
    config_path: Path,
    run_id: str,
    config_hash: str,
    status: str,
    error: Exception | None = None,
) -> None:
    EXPERIMENT_LOG.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    entry = {
        "timestamp":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "run_id":
            run_id,

        "status":
            status,

        "model":
            config[
                "model"
            ][
                "name"
            ],

        "model_family":
            config[
                "model"
            ].get(
                "family"
            ),

        "config_path":
            str(
                config_path
            ),

        "config_hash":
            config_hash,

        "git_commit":
            git_commit(),

        "git_dirty":
            git_dirty(),
    }

    if result is not None:
        entry[
            "metrics"
        ] = result.get(
            "metrics",
            {},
        )

        entry[
            "summary"
        ] = result.get(
            "summary",
            {},
        )

        entry[
            "artifacts"
        ] = result.get(
            "artifacts",
            [],
        )

    if error is not None:
        entry[
            "error_type"
        ] = type(
            error
        ).__name__

        entry[
            "error"
        ] = str(
            error
        )

        try:
            mlflow.set_tag(
                "run_status",
                "failed",
            )

            mlflow.set_tag(
                "failure_type",
                type(
                    error
                ).__name__,
            )
        except Exception:
            pass

    with EXPERIMENT_LOG.open(
        "a"
    ) as file:
        file.write(
            json.dumps(
                entry,
                sort_keys=True,
                default=str,
            )
            + "\n"
        )
PY_LOGGING

cat > machine_learning/experiments/orchestrator.py <<'PY_ORCHESTRATOR'
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import yaml

from machine_learning.experiments.experiment_logging import (
    config_hash,
    log_experiment,
    log_mlflow_result,
)
from machine_learning.experiments.mlflow_config import (
    start_run,
)


def load_config(
    path: Path,
) -> dict:
    with path.open() as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Expected YAML mapping in {path}"
        )

    return config


def run_experiment(
    config_path: Path,
) -> dict:
    config = load_config(
        config_path
    )

    model_config = config[
        "model"
    ]

    model_module = importlib.import_module(
        model_config[
            "module"
        ]
    )

    train = model_module.train

    digest = config_hash(
        config
    )

    run_name = (
        f"{model_config['name']}__{digest}"
    )

    with start_run(
        run_name=run_name
    ) as run:
        run_id = run.info.run_id

        try:
            result = train(
                config,
                run_id=run_id,
                config_hash=digest,
            )

            log_mlflow_result(
                config=config,
                result=result,
                config_hash=digest,
            )

            log_experiment(
                config=config,
                result=result,
                config_path=config_path,
                run_id=run_id,
                config_hash=digest,
                status="completed",
            )

            return result

        except Exception as exc:
            log_experiment(
                config=config,
                result=None,
                config_path=config_path,
                run_id=run_id,
                config_hash=digest,
                status="failed",
                error=exc,
            )

            raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one CrimeNet model experiment."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        type=Path,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_experiment(
        args.config
    )


if __name__ == "__main__":
    main()
PY_ORCHESTRATOR

cat > machine_learning/models/xgboost/configs/baseline_v1.yaml <<'YAML_BASELINE'
model:
  family: xgboost
  module: machine_learning.models.xgboost.model
  name: xgb_pp_baseline_v1
  description: Full-feature XGBoost Poisson point-process baseline

data:
  model_table_root: gs://crimenet/gold_staging_/model_table_nyc_timestamp_fix
  train_split: train
  validation_split: validation
  train_fraction: 0.05
  validation_fraction: 0.10
  seed: 42

architecture:
  objective: point_process_poisson
  tree_method: hist
  device: cpu
  max_bin: 256
  max_depth: 6
  max_cat_to_onehot: 4

optimization:
  learning_rate: 0.03
  subsample: 0.90
  colsample_bytree: 0.90
  min_child_weight: 50.0
  max_delta_step: 1.0
  reg_lambda: 10.0
  reg_alpha: 0.0

training:
  num_boost_round: 1000
  early_stopping_rounds: 50
  verbose_eval: 10

features:
  feature_set: full_v1
  include_source_city: true
  include_calendar: true
  include_context: true
  include_lighting: true
  include_history: true

numerics:
  min_log_intensity: -30.0
  max_log_intensity: 15.0
  hessian_floor: 1.0e-6
  event_exposure_tolerance: 1.0e-12

artifacts:
  # Resolved relative to machine_learning/, not the shell CWD.
  output_root: artifacts/experiments
YAML_BASELINE

cat > machine_learning/models/xgboost/model.py <<'PY_XGBOOST_MODEL'
from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

from crimenet_data.assets.model_table.transformations import (
    CONTEXT_FEATURE_COLUMNS,
    HISTORY_FEATURE_COLUMNS,
    LIGHTING_FEATURE_COLUMNS,
)


MACHINE_LEARNING_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


CALENDAR_FEATURE_COLUMNS = [
    "local_hour",
    "local_day_of_week",
    "local_hour_sin",
    "local_hour_cos",
    "local_day_of_week_sin",
    "local_day_of_week_cos",
]

AUXILIARY_COLUMNS = [
    "event_count",
    "integration_weight_cell_seconds",
]


def _safe_margin(
    x: np.ndarray,
    *,
    min_log_intensity: float,
    max_log_intensity: float,
) -> np.ndarray:
    return np.clip(
        np.asarray(
            x,
            dtype=np.float64,
        ),
        min_log_intensity,
        max_log_intensity,
    )


def _resolve_feature_columns(
    config: dict[str, Any],
) -> tuple[list[str], list[str]]:
    feature_config = config["features"]

    feature_columns: list[str] = []

    if feature_config.get(
        "include_source_city",
        True,
    ):
        feature_columns.append(
            "source_city"
        )

    if feature_config.get(
        "include_calendar",
        True,
    ):
        feature_columns.extend(
            CALENDAR_FEATURE_COLUMNS
        )

    if feature_config.get(
        "include_context",
        True,
    ):
        feature_columns.extend(
            CONTEXT_FEATURE_COLUMNS
        )

    if feature_config.get(
        "include_lighting",
        True,
    ):
        feature_columns.extend(
            LIGHTING_FEATURE_COLUMNS
        )

    if feature_config.get(
        "include_history",
        True,
    ):
        feature_columns.extend(
            HISTORY_FEATURE_COLUMNS
        )

    # Preserve order while removing accidental duplicates.
    feature_columns = list(
        dict.fromkeys(
            feature_columns
        )
    )

    categorical_columns = [
        column
        for column in (
            "source_city",
            "lighting_condition",
        )
        if column in feature_columns
    ]

    return (
        feature_columns,
        categorical_columns,
    )


def _deterministic_split_sample(
    *,
    table: pl.LazyFrame,
    split: str,
    fraction: float,
    seed: int,
    feature_columns: list[str],
) -> pl.DataFrame:
    if not (
        0.0
        < fraction
        <= 1.0
    ):
        raise ValueError(
            f"Sampling fraction must be in (0, 1], "
            f"got {fraction}."
        )

    buckets = 1_000_000

    threshold = int(
        fraction
        * buckets
    )

    return (
        table

        .filter(
            pl.col("split")
            == split
        )

        .filter(
            (
                pl.col("model_row_id")
                .hash(seed=seed)
                % buckets
            )
            < threshold
        )

        .select(
            [
                *feature_columns,
                *AUXILIARY_COLUMNS,
            ]
        )

        .collect()
    )


def _sample_summary(
    name: str,
    frame: pl.DataFrame,
) -> dict[str, float]:
    rows = int(
        frame.height
    )

    observed_events = float(
        frame
        .get_column(
            "event_count"
        )
        .sum()
    )

    integration_rows = int(
        (
            frame
            .get_column(
                "event_count"
            )
            == 0
        )
        .sum()
    )

    integration_weight = float(
        frame
        .get_column(
            "integration_weight_cell_seconds"
        )
        .sum()
    )

    print(
        f"\n{name.upper()}"
    )

    print(
        f"Rows:               "
        f"{rows:,}"
    )

    print(
        f"Observed events:    "
        f"{observed_events:,.0f}"
    )

    print(
        f"Integration rows:   "
        f"{integration_rows:,}"
    )

    print(
        f"Integration weight: "
        f"{integration_weight:.6e}"
    )

    return {
        "rows":
            float(rows),

        "observed_events":
            observed_events,

        "integration_rows":
            float(
                integration_rows
            ),

        "integration_weight":
            integration_weight,
    }


def _prepare_xy(
    frame: pl.DataFrame,
    *,
    feature_columns: list[str],
    categorical_columns: list[str],
    category_levels:
        dict[str, list[str]]
        | None = None,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    dict[str, list[str]],
]:
    y = (
        frame
        .get_column(
            "event_count"
        )
        .to_numpy()
        .astype(
            np.float32,
            copy=False,
        )
    )

    exposure = (
        frame
        .get_column(
            "integration_weight_cell_seconds"
        )
        .to_numpy()
        .astype(
            np.float64,
            copy=False,
        )
    )

    X = (
        frame
        .select(
            feature_columns
        )
        .to_pandas()
    )

    learned_categories: dict[
        str,
        list[str],
    ] = {}

    for column in categorical_columns:
        if category_levels is None:
            categories = sorted(
                X[column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        else:
            categories = (
                category_levels[
                    column
                ]
            )

        learned_categories[
            column
        ] = categories

        X[column] = pd.Categorical(
            X[column],
            categories=categories,
        )

    for column in feature_columns:
        if column in categorical_columns:
            continue

        X[column] = (
            pd.to_numeric(
                X[column],
                errors="coerce",
            )
            .astype(
                np.float32
            )
        )

    return (
        X,
        y,
        exposure,
        learned_categories,
    )


def _validate_point_process_rows(
    *,
    name: str,
    y: np.ndarray,
    exposure: np.ndarray,
    event_exposure_tolerance: float,
) -> None:
    if not np.isfinite(
        exposure
    ).all():
        raise ValueError(
            f"{name}: exposure contains "
            f"NaN or infinity."
        )

    if not (
        exposure >= 0
    ).all():
        raise ValueError(
            f"{name}: exposure contains "
            f"negative values."
        )

    if not np.isfinite(
        y
    ).all():
        raise ValueError(
            f"{name}: event_count contains "
            f"NaN or infinity."
        )

    if not set(
        np.unique(y)
    ).issubset(
        {0.0, 1.0}
    ):
        raise ValueError(
            f"{name}: event_count is not binary."
        )

    event_mask = (
        y == 1
    )

    integration_mask = (
        y == 0
    )

    if (
        event_mask.any()
        and float(
            exposure[
                event_mask
            ].max()
        )
        > event_exposure_tolerance
    ):
        raise ValueError(
            f"{name}: observed-event rows "
            f"contain non-zero exposure."
        )

    if (
        integration_mask.any()
        and not (
            exposure[
                integration_mask
            ]
            > 0
        ).all()
    ):
        raise ValueError(
            f"{name}: integration rows "
            f"contain zero or negative exposure."
        )


def _constant_nll_per_event(
    *,
    y: np.ndarray,
    exposure: np.ndarray,
    log_lambda: float,
) -> float:
    nll = np.sum(
        exposure
        * np.exp(
            log_lambda
        )
        -
        y
        * log_lambda
    )

    return float(
        nll
        /
        max(
            float(
                y.sum()
            ),
            1.0,
        )
    )


def _evaluate_split(
    *,
    name: str,
    booster: xgb.Booster,
    dmatrix: xgb.DMatrix,
    exposure: np.ndarray,
    min_log_intensity: float,
    max_log_intensity: float,
) -> dict[str, float]:
    y = (
        dmatrix
        .get_label()
        .astype(
            np.float64,
            copy=False,
        )
    )

    margin = booster.predict(
        dmatrix,
        output_margin=True,
    )

    margin = _safe_margin(
        margin,
        min_log_intensity=
            min_log_intensity,
        max_log_intensity=
            max_log_intensity,
    )

    intensity = np.exp(
        margin
    )

    expected_events = float(
        np.sum(
            exposure
            * intensity
        )
    )

    observed_events = float(
        y.sum()
    )

    nll = float(
        np.sum(
            exposure
            * intensity
            -
            y
            * margin
        )
    )

    nll_per_event = (
        nll
        /
        max(
            observed_events,
            1.0,
        )
    )

    expected_observed_ratio = (
        expected_events
        /
        max(
            observed_events,
            1.0,
        )
    )

    calibration_error_pct = (
        (
            expected_observed_ratio
            - 1.0
        )
        * 100.0
    )

    mean_log_lambda = float(
        margin.mean()
    )

    print(
        f"\n{name.upper()}"
    )

    print(
        f"Observed events:  "
        f"{observed_events:,.0f}"
    )

    print(
        f"Expected events:  "
        f"{expected_events:,.2f}"
    )

    print(
        f"Expected/actual:  "
        f"{expected_observed_ratio:.4f}"
    )

    print(
        f"NLL/event:        "
        f"{nll_per_event:.6f}"
    )

    print(
        f"Mean log lambda:  "
        f"{mean_log_lambda:.6f}"
    )

    return {
        "observed_events":
            observed_events,

        "expected_events":
            expected_events,

        "expected_observed_ratio":
            expected_observed_ratio,

        "calibration_error_pct":
            calibration_error_pct,

        "nll":
            nll,

        "nll_per_event":
            nll_per_event,

        "mean_log_lambda":
            mean_log_lambda,
    }


def train(
    config: dict[str, Any],
    *,
    run_id: str,
    config_hash: str,
) -> dict[str, Any]:
    """Train one XGBoost CrimeNet experiment.

    Model-specific code lives here. This function does not create or
    manipulate MLflow runs. It returns plain metrics, history, artifact
    paths, and a summary for the generic experiment orchestrator.
    """

    model_config = config["model"]
    data_config = config["data"]
    architecture_config = config[
        "architecture"
    ]
    optimization_config = config[
        "optimization"
    ]
    training_config = config["training"]
    numerics_config = config["numerics"]
    artifact_config = config["artifacts"]

    seed = int(
        data_config["seed"]
    )

    train_fraction = float(
        data_config[
            "train_fraction"
        ]
    )

    validation_fraction = float(
        data_config[
            "validation_fraction"
        ]
    )

    max_bin = int(
        architecture_config[
            "max_bin"
        ]
    )

    min_log_intensity = float(
        numerics_config[
            "min_log_intensity"
        ]
    )

    max_log_intensity = float(
        numerics_config[
            "max_log_intensity"
        ]
    )

    hessian_floor = float(
        numerics_config[
            "hessian_floor"
        ]
    )

    event_exposure_tolerance = float(
        numerics_config.get(
            "event_exposure_tolerance",
            1e-12,
        )
    )

    (
        feature_columns,
        categorical_columns,
    ) = _resolve_feature_columns(
        config
    )

    if not feature_columns:
        raise ValueError(
            "Resolved feature set is empty."
        )

    artifact_dir = (
        MACHINE_LEARNING_ROOT
        / artifact_config[
            "output_root"
        ]
        / model_config["name"]
        / run_id
    )

    artifact_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    model_path = (
        artifact_dir
        / "model.json"
    )

    metadata_path = (
        artifact_dir
        / "metadata.json"
    )

    importance_path = (
        artifact_dir
        / "feature_importance.json"
    )

    training_history_path = (
        artifact_dir
        / "training_history.json"
    )

    print(
        f"Experiment artifacts: "
        f"{artifact_dir}"
    )

    credentials = (
        pl.CredentialProviderGCP()
    )

    table = pl.scan_delta(
        data_config[
            "model_table_root"
        ],
        credential_provider=
            credentials,
    )

    print(
        "Loading deterministic "
        "training sample..."
    )

    train_frame = (
        _deterministic_split_sample(
            table=table,
            split=data_config[
                "train_split"
            ],
            fraction=train_fraction,
            seed=seed,
            feature_columns=
                feature_columns,
        )
    )

    print(
        "Loading deterministic "
        "validation sample..."
    )

    validation_frame = (
        _deterministic_split_sample(
            table=table,
            split=data_config[
                "validation_split"
            ],
            fraction=
                validation_fraction,
            seed=seed,
            feature_columns=
                feature_columns,
        )
    )

    train_summary = _sample_summary(
        "train",
        train_frame,
    )

    validation_summary = (
        _sample_summary(
            "validation",
            validation_frame,
        )
    )

    (
        X_train,
        y_train,
        exposure_train,
        categories,
    ) = _prepare_xy(
        train_frame,
        feature_columns=
            feature_columns,
        categorical_columns=
            categorical_columns,
    )

    (
        X_validation,
        y_validation,
        exposure_validation,
        _,
    ) = _prepare_xy(
        validation_frame,
        feature_columns=
            feature_columns,
        categorical_columns=
            categorical_columns,
        category_levels=
            categories,
    )

    del train_frame
    del validation_frame

    gc.collect()

    _validate_point_process_rows(
        name="train",
        y=y_train,
        exposure=exposure_train,
        event_exposure_tolerance=
            event_exposure_tolerance,
    )

    _validate_point_process_rows(
        name="validation",
        y=y_validation,
        exposure=
            exposure_validation,
        event_exposure_tolerance=
            event_exposure_tolerance,
    )

    train_events = float(
        y_train.sum()
    )

    train_exposure = float(
        exposure_train.sum()
    )

    if (
        train_events <= 0
        or train_exposure <= 0
    ):
        raise ValueError(
            "Training sample must contain "
            "positive events and exposure."
        )

    initial_lambda = (
        train_events
        /
        train_exposure
    )

    initial_log_lambda = float(
        np.log(
            initial_lambda
        )
    )

    print(
        "\nInitial constant intensity"
    )

    print(
        f"events:       "
        f"{train_events:,.0f}"
    )

    print(
        f"exposure:     "
        f"{train_exposure:,.3e}"
    )

    print(
        f"lambda0:      "
        f"{initial_lambda:.8e}"
    )

    print(
        f"log(lambda0): "
        f"{initial_log_lambda:.6f}"
    )

    train_margin = np.full(
        y_train.shape,
        initial_log_lambda,
        dtype=np.float32,
    )

    validation_margin = np.full(
        y_validation.shape,
        initial_log_lambda,
        dtype=np.float32,
    )

    dtrain = xgb.QuantileDMatrix(
        X_train,
        label=y_train,
        base_margin=train_margin,
        enable_categorical=True,
        max_bin=max_bin,
        nthread=-1,
    )

    dvalidation = (
        xgb.QuantileDMatrix(
            X_validation,
            label=y_validation,
            base_margin=
                validation_margin,
            enable_categorical=True,
            max_bin=max_bin,
            ref=dtrain,
            nthread=-1,
        )
    )

    del X_train
    del X_validation

    gc.collect()

    def point_process_poisson_objective(
        predt: np.ndarray,
        dtrain_: xgb.DMatrix,
    ):
        y = (
            dtrain_
            .get_label()
            .astype(
                np.float64,
                copy=False,
            )
        )

        f_safe = _safe_margin(
            predt,
            min_log_intensity=
                min_log_intensity,
            max_log_intensity=
                max_log_intensity,
        )

        integrated_intensity = (
            exposure_train
            * np.exp(
                f_safe
            )
        )

        grad = (
            integrated_intensity
            - y
        )

        hess = np.maximum(
            integrated_intensity,
            hessian_floor,
        )

        return (
            grad.astype(
                np.float32
            ),
            hess.astype(
                np.float32
            ),
        )

    exposure_lookup = {
        id(dtrain):
            exposure_train,

        id(dvalidation):
            exposure_validation,
    }

    def point_process_nll(
        predt: np.ndarray,
        dmatrix: xgb.DMatrix,
    ):
        y = (
            dmatrix
            .get_label()
            .astype(
                np.float64,
                copy=False,
            )
        )

        exposure = (
            exposure_lookup[
                id(dmatrix)
            ]
        )

        f_safe = _safe_margin(
            predt,
            min_log_intensity=
                min_log_intensity,
            max_log_intensity=
                max_log_intensity,
        )

        nll = np.sum(
            exposure
            * np.exp(
                f_safe
            )
            -
            y
            * f_safe
        )

        return (
            "pp_nll_per_event",
            float(
                nll
                /
                max(
                    float(
                        y.sum()
                    ),
                    1.0,
                )
            ),
        )

    params = {
        "tree_method":
            architecture_config[
                "tree_method"
            ],

        "device":
            architecture_config[
                "device"
            ],

        "max_bin":
            max_bin,

        "max_depth":
            int(
                architecture_config[
                    "max_depth"
                ]
            ),

        "eta":
            float(
                optimization_config[
                    "learning_rate"
                ]
            ),

        "subsample":
            float(
                optimization_config[
                    "subsample"
                ]
            ),

        "colsample_bytree":
            float(
                optimization_config[
                    "colsample_bytree"
                ]
            ),

        "min_child_weight":
            float(
                optimization_config[
                    "min_child_weight"
                ]
            ),

        "max_delta_step":
            float(
                optimization_config[
                    "max_delta_step"
                ]
            ),

        "reg_lambda":
            float(
                optimization_config[
                    "reg_lambda"
                ]
            ),

        "reg_alpha":
            float(
                optimization_config[
                    "reg_alpha"
                ]
            ),

        "max_cat_to_onehot":
            int(
                architecture_config[
                    "max_cat_to_onehot"
                ]
            ),

        "seed":
            seed,

        "nthread":
            -1,

        "disable_default_eval_metric":
            True,
    }

    constant_train_nll = (
        _constant_nll_per_event(
            y=y_train,
            exposure=exposure_train,
            log_lambda=
                initial_log_lambda,
        )
    )

    constant_validation_nll = (
        _constant_nll_per_event(
            y=y_validation,
            exposure=
                exposure_validation,
            log_lambda=
                initial_log_lambda,
        )
    )

    print(
        "Constant train NLL/event:",
        constant_train_nll,
    )

    print(
        "Constant validation NLL/event:",
        constant_validation_nll,
    )

    print(
        "\nTraining XGBoost "
        "Poisson point-process baseline...\n"
    )

    evals_result: dict[
        str,
        dict[
            str,
            list[float],
        ],
    ] = {}

    booster = xgb.train(
        params=params,

        dtrain=dtrain,

        num_boost_round=int(
            training_config[
                "num_boost_round"
            ]
        ),

        evals=[
            (
                dtrain,
                "train",
            ),
            (
                dvalidation,
                "validation",
            ),
        ],

        obj=
            point_process_poisson_objective,

        custom_metric=
            point_process_nll,

        early_stopping_rounds=int(
            training_config[
                "early_stopping_rounds"
            ]
        ),

        evals_result=
            evals_result,

        verbose_eval=
            training_config[
                "verbose_eval"
            ],
    )

    best_iteration = getattr(
        booster,
        "best_iteration",
        None,
    )

    if best_iteration is not None:
        booster = booster[
            : best_iteration + 1
        ]

    train_metrics = _evaluate_split(
        name="train",
        booster=booster,
        dmatrix=dtrain,
        exposure=exposure_train,
        min_log_intensity=
            min_log_intensity,
        max_log_intensity=
            max_log_intensity,
    )

    validation_metrics = (
        _evaluate_split(
            name="validation",
            booster=booster,
            dmatrix=dvalidation,
            exposure=
                exposure_validation,
            min_log_intensity=
                min_log_intensity,
            max_log_intensity=
                max_log_intensity,
        )
    )

    train_nll_gain = (
        constant_train_nll
        -
        train_metrics[
            "nll_per_event"
        ]
    )

    validation_nll_gain = (
        constant_validation_nll
        -
        validation_metrics[
            "nll_per_event"
        ]
    )

    train_bits_per_event = (
        train_nll_gain
        /
        np.log(2.0)
    )

    validation_bits_per_event = (
        validation_nll_gain
        /
        np.log(2.0)
    )

    importance = {
        "gain":
            booster.get_score(
                importance_type="gain"
            ),

        "weight":
            booster.get_score(
                importance_type="weight"
            ),

        "cover":
            booster.get_score(
                importance_type="cover"
            ),
    }

    print(
        "\nTOP 30 FEATURES BY GAIN"
    )

    sorted_gain = sorted(
        importance[
            "gain"
        ].items(),
        key=lambda item:
            item[1],
        reverse=True,
    )

    for (
        feature,
        gain,
    ) in sorted_gain[:30]:
        print(
            f"{feature:<50} "
            f"{gain:,.6f}"
        )

    booster.save_model(
        model_path
    )

    metadata = {
        "model_name":
            model_config[
                "name"
            ],

        "run_id":
            run_id,

        "config_hash":
            config_hash,

        "model_table_root":
            data_config[
                "model_table_root"
            ],

        "train_fraction":
            train_fraction,

        "validation_fraction":
            validation_fraction,

        "seed":
            seed,

        "features":
            feature_columns,

        "feature_set":
            config[
                "features"
            ].get(
                "feature_set"
            ),

        "categorical_columns":
            categorical_columns,

        "category_levels":
            categories,

        "initial_lambda":
            initial_lambda,

        "initial_log_lambda":
            initial_log_lambda,

        "best_iteration":
            best_iteration,

        "test_split_used":
            False,
    }

    with metadata_path.open(
        "w"
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    with importance_path.open(
        "w"
    ) as file:
        json.dump(
            importance,
            file,
            indent=2,
        )

    with training_history_path.open(
        "w"
    ) as file:
        json.dump(
            evals_result,
            file,
            indent=2,
        )

    print(
        f"\nSaved model: "
        f"{model_path}"
    )

    print(
        "TEST SPLIT HAS NOT BEEN ACCESSED."
    )

    metrics = {
        "sample_train_rows":
            train_summary[
                "rows"
            ],

        "sample_validation_rows":
            validation_summary[
                "rows"
            ],

        "sample_train_events":
            train_summary[
                "observed_events"
            ],

        "sample_validation_events":
            validation_summary[
                "observed_events"
            ],

        "sample_train_exposure":
            train_summary[
                "integration_weight"
            ],

        "sample_validation_exposure":
            validation_summary[
                "integration_weight"
            ],

        "initial_lambda":
            initial_lambda,

        "initial_log_lambda":
            initial_log_lambda,

        "constant_train_nll_per_event":
            constant_train_nll,

        "constant_validation_nll_per_event":
            constant_validation_nll,

        "sample_train_nll_per_event":
            train_metrics[
                "nll_per_event"
            ],

        "sample_validation_nll_per_event":
            validation_metrics[
                "nll_per_event"
            ],

        "sample_train_expected_observed":
            train_metrics[
                "expected_observed_ratio"
            ],

        "sample_validation_expected_observed":
            validation_metrics[
                "expected_observed_ratio"
            ],

        "sample_train_calibration_error_pct":
            train_metrics[
                "calibration_error_pct"
            ],

        "sample_validation_calibration_error_pct":
            validation_metrics[
                "calibration_error_pct"
            ],

        "sample_train_nll_gain_per_event":
            train_nll_gain,

        "sample_validation_nll_gain_per_event":
            validation_nll_gain,

        "sample_train_bits_per_event":
            train_bits_per_event,

        "sample_validation_bits_per_event":
            validation_bits_per_event,

        "best_iteration":
            float(
                best_iteration
                if best_iteration
                is not None
                else -1
            ),
    }

    return {
        "metrics":
            metrics,

        "history":
            evals_result,

        "artifacts": [
            str(
                model_path
            ),
            str(
                metadata_path
            ),
            str(
                importance_path
            ),
            str(
                training_history_path
            ),
        ],

        "summary": {
            "best_iteration":
                best_iteration,

            "feature_count":
                len(
                    feature_columns
                ),

            "artifact_dir":
                str(
                    artifact_dir
                ),

            "test_split_used":
                False,
        },
    }
PY_XGBOOST_MODEL

cat > machine_learning/README.md <<'README_EOF'
# CrimeNet machine learning

## Layout

```text
machine_learning/
├── artifacts/
│   ├── experiments/
│   └── mlflow/
├── experiments/
│   ├── experiment_logging.py
│   ├── mlflow_config.py
│   ├── orchestrator.py
│   └── log/
│       └── experiments.jsonl
└── models/
    └── xgboost/
        ├── model.py
        └── configs/
            └── baseline_v1.yaml
```

## Run

From `src/`:

```bash
python -m machine_learning.experiments.orchestrator \
  --config machine_learning/models/xgboost/configs/baseline_v1.yaml
```

## MLflow UI

From `src/`:

```bash
mlflow server \
  --backend-store-uri sqlite:////ABSOLUTE/PATH/TO/src/machine_learning/artifacts/mlflow/mlflow.db \
  --default-artifact-root file:///ABSOLUTE/PATH/TO/src/machine_learning/artifacts/mlflow/artifacts \
  --port 5000
```

The Python experiment runner does not need the server running; it writes directly to
the SQLite tracking store.
README_EOF

cat > machine_learning/.gitignore <<'GITIGNORE_EOF'
__pycache__/
*.py[cod]
artifacts/
GITIGNORE_EOF

echo
echo "Syntax-checking Python files..."

python -m py_compile \
  machine_learning/experiments/mlflow_config.py \
  machine_learning/experiments/experiment_logging.py \
  machine_learning/experiments/orchestrator.py \
  machine_learning/models/xgboost/model.py

echo
echo "Scaffold complete."
echo
echo "src/ should now contain:"
printf "  - crimenet_data/\n"
printf "  - machine_learning/\n"
echo
echo "Run the baseline with:"
echo "python -m machine_learning.experiments.orchestrator \\"
echo "  --config machine_learning/models/xgboost/configs/baseline_v1.yaml"
echo
echo "MLflow DB will be created automatically at:"
echo "machine_learning/artifacts/mlflow/mlflow.db"
