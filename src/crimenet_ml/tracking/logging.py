"""Explicit, reusable CrimeNet MLflow logging."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import yaml


def flatten_parameters(value: Any, prefix: str = "") -> dict[str, str | int | float | bool]:
    if is_dataclass(value):
        value = asdict(value)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    flattened: dict[str, str | int | float | bool] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_parameters(item, name))
    elif isinstance(value, (list, tuple, set)):
        flattened[prefix] = json.dumps(list(value), sort_keys=True, default=str)
    elif value is not None:
        flattened[prefix] = value if isinstance(value, (str, int, float, bool)) else str(value)
    return flattened


def flatten_metrics(value: Any, prefix: str = "") -> dict[str, float]:
    flattened: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_metrics(item, name))
    elif isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric):
            flattened[prefix] = numeric
    return flattened


def git_state(root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", False


def log_json_artifact(value: Any, filename: str, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    mlflow.log_artifact(str(path))
    return path


def log_yaml_artifact(value: Any, filename: str, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    mlflow.log_artifact(str(path))
    return path


def log_evaluation_history(history: dict[str, dict[str, list[float]]]) -> None:
    for dataset, metrics in history.items():
        for metric, values in metrics.items():
            for step, value in enumerate(values):
                if math.isfinite(float(value)):
                    mlflow.log_metric(f"iteration.{dataset}.{metric}", float(value), step=step)


def feature_importance_frame(model: Any, feature_names: tuple[str, ...]) -> pd.DataFrame:
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    cover = booster.get_score(importance_type="cover")
    return pd.DataFrame(
        {
            "feature": feature_names,
            "gain": [float(gain.get(name, 0.0)) for name in feature_names],
            "split_count": [float(weight.get(name, 0.0)) for name in feature_names],
            "cover": [float(cover.get(name, 0.0)) for name in feature_names],
        }
    ).sort_values("gain", ascending=False)


def log_dataset_inputs(frames: dict[str, pd.DataFrame], source: str, digest: str | None) -> None:
    for split, frame in frames.items():
        dataset = mlflow.data.from_pandas(
            frame,
            source=source,
            name=f"crimenet_{split}",
            targets="event_multiplicity" if "event_multiplicity" in frame else None,
            digest=digest[:36] if digest else None,
        )
        mlflow.log_input(dataset, context=split)
