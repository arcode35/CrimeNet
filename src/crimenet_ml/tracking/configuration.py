"""MLflow configuration helpers."""

from __future__ import annotations

import mlflow

from crimenet_ml.config import TrackingConfig


def configure_mlflow(config: TrackingConfig) -> None:
    if not config.enabled:
        return
    if config.tracking_uri:
        mlflow.set_tracking_uri(config.tracking_uri)
    if config.registry_uri:
        mlflow.set_registry_uri(config.registry_uri)
    mlflow.set_experiment(config.experiment_name)
