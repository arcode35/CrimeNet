"""Typed, layered CrimeNet configuration."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(StrictModel):
    backend: Literal["local", "databricks"]
    path: Path | None = None
    table: str | None = None
    split_column: str = "dataset_split"
    event_count_column: str = "event_multiplicity"
    integration_weight_column: str = "importance_weight"
    categorical_columns: list[str] = Field(default_factory=lambda: ["offense_mark"])
    identifier_columns: list[str] = Field(default_factory=lambda: ["example_id"])
    metadata_columns: list[str] = Field(default_factory=lambda: ["source_city"])
    allow_missing_features: list[str] = Field(default_factory=list)
    collection_row_limit: int = 5_000_000
    allow_large_collection: bool = False

    @model_validator(mode="after")
    def source_is_configured(self) -> DataConfig:
        if self.backend == "local" and self.path is None:
            raise ValueError("Local backend requires data.path")
        if self.backend == "databricks" and not self.table:
            raise ValueError("Databricks backend requires data.table")
        return self


class ModelConfig(StrictModel):
    family: Literal["xgb_poisson"] = "xgb_poisson"
    objective: Literal["count:poisson"] = "count:poisson"
    eval_metric: Literal["poisson-nloglik"] = "poisson-nloglik"
    n_estimators: int = 3000
    learning_rate: float = 0.03
    max_depth: int = 8
    min_child_weight: float = 1.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    gamma: float = 0.0
    max_delta_step: float = 0.7
    max_bin: int = 256
    tree_method: Literal["hist"] = "hist"
    device: Literal["cpu", "cuda"] = "cpu"
    n_jobs: int = -1
    early_stopping_rounds: int = 100


class TrackingConfig(StrictModel):
    enabled: bool = True
    tracking_uri: str | None = None
    registry_uri: str | None = None
    experiment_name: str = "crimenet/local/xgb-history-poisson"
    register_model: bool = False
    registered_model_name: str = "crimenet_xgb_history_poisson"
    calibrated_model_artifact_name: str = "model"
    raw_model_artifact_name: str = "raw_xgboost_model"


class OutputConfig(StrictModel):
    artifact_dir: Path = Path("artifacts")
    write_test_predictions: bool = False


class AppConfig(StrictModel):
    environment: str
    feature_set: Literal[
    "history_v1",
    "history_no_k1_v1",
    "core_v1",
    ] = "history_v1"
    feature_definition_version: str = "1.0.0"
    split_definition_version: str = "1.0.0"
    random_seed: int = 42
    data: DataConfig
    model: ModelConfig
    tracking: TrackingConfig
    output: OutputConfig = Field(default_factory=OutputConfig)


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?}")


def _expand(value: object) -> object:
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.getenv(m.group(1), m.group(2) or ""), value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def _deep_merge(base: dict, overlay: dict) -> dict:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path).expanduser().resolve()
    overlay = yaml.safe_load(path.read_text()) or {}
    extends = overlay.pop("extends", None)
    base = {}
    if extends:
        base_path = (path.parent / extends).resolve()
        base = yaml.safe_load(base_path.read_text()) or {}
        base.pop("extends", None)
    raw = _expand(_deep_merge(base, overlay))
    repo_root = path.parent.parent if path.parent.name == "configs" else path.parent
    data_path = raw.get("data", {}).get("path")
    if data_path and not Path(data_path).is_absolute():
        raw["data"]["path"] = (repo_root / data_path).resolve()
    artifact_dir = raw.get("output", {}).get("artifact_dir")
    if artifact_dir and not Path(artifact_dir).is_absolute():
        raw["output"]["artifact_dir"] = (repo_root / artifact_dir).resolve()
    raw["tracking"]["tracking_uri"] = os.getenv(
        "MLFLOW_TRACKING_URI", raw["tracking"].get("tracking_uri")
    )
    raw["tracking"]["registry_uri"] = os.getenv(
        "MLFLOW_REGISTRY_URI", raw["tracking"].get("registry_uri")
    )
    raw["tracking"]["experiment_name"] = os.getenv(
        "MLFLOW_EXPERIMENT_NAME", raw["tracking"]["experiment_name"]
    )
    return AppConfig.model_validate(raw)
