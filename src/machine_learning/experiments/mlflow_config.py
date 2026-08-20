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

MLFLOW_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

MLFLOW_DB = (
    MLFLOW_ROOT
    / "mlflow.db"
)

MLFLOW_ARTIFACT_ROOT = (
    MLFLOW_ROOT
    / "artifacts"
)

MLFLOW_ARTIFACT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

TRACKING_URI = (
    f"sqlite:///{MLFLOW_DB}"
)

EXPERIMENT_NAME = (
    "crimenet-model-development"
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
            artifact_location=(
                MLFLOW_ARTIFACT_ROOT
                .resolve()
                .as_uri()
            ),
        )
    )
else:
    EXPERIMENT_ID = (
        experiment.experiment_id
    )


def start_run(
    *,
    run_name: str,
):
    return mlflow.start_run(
        experiment_id=EXPERIMENT_ID,
        run_name=run_name,
    )


def resume_run(
    *,
    run_id: str,
):
    return mlflow.start_run(
        run_id=run_id,
    )