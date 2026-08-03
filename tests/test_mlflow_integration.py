from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml

from crimenet_ml.config import load_config
from crimenet_ml.features.feature_set import get_feature_set
from crimenet_ml.training.train import train


def _synthetic_frame(split: str, city: str, categories: list[str]) -> pd.DataFrame:
    features = get_feature_set("history_v1").features
    rows = len(categories)
    data: dict[str, object] = {}
    for index, feature in enumerate(features):
        if feature == "offense_mark":
            data[feature] = categories
        elif feature.startswith("is_"):
            data[feature] = [(row + index) % 2 == 0 for row in range(rows)]
        else:
            data[feature] = [float((row + index) % 7) for row in range(rows)]
    data.update(
        example_id=[f"{split}-{row}" for row in range(rows)],
        example_timestamp_utc=pd.date_range("2025-01-01", periods=rows, freq="h"),
        osm_h3_cell_id=[f"cell-{row % 3}" for row in range(rows)],
        is_observed_event=[row % 4 == 0 for row in range(rows)],
        event_multiplicity=[1 if row % 4 == 0 else 0 for row in range(rows)],
        importance_weight=[1.0 + row % 3 for row in range(rows)],
    )
    return pd.DataFrame(data)


def _write_split(root: Path, split: str, categories: list[str]) -> None:
    for city in ("a", "b"):
        directory = root / f"dataset_split={split}" / f"source_city={city}"
        directory.mkdir(parents=True)
        _synthetic_frame(split + city, city, categories).to_parquet(
            directory / "part.parquet", index=False
        )


def test_tiny_training_logs_loadable_calibrated_model(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "dataset"
    _write_split(dataset, "train", ["a", "b"] * 8)
    _write_split(dataset, "validation", ["a", "unseen"] * 8)
    _write_split(dataset, "test", ["b", "also-unseen"] * 8)
    base = yaml.safe_load((Path(__file__).parents[1] / "configs/base.yml").read_text())
    base["model"].update(n_estimators=8, early_stopping_rounds=3, max_depth=2, n_jobs=1)
    base["tracking"].update(
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        registry_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        experiment_name="test/crimenet",
    )
    base["data"].update(path=str(dataset), backend="local")
    base["output"].update(artifact_dir=str(tmp_path / "artifacts"))
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(base))
    outcome = train(load_config(config_path), run_name="integration-test")
    assert outcome.run_id and outcome.model_uri
    runs = mlflow.search_runs(experiment_names=["test/crimenet"])
    assert len(runs) == 1
    run = mlflow.get_run(outcome.run_id)
    assert run.data.tags["objective"] == "count:poisson"
    assert run.data.params["target_column"] == "event_multiplicity"
    assert "validation_raw.point_process_nll" in run.data.metrics
    assert "intensity_calibration_factor" in run.data.metrics
    artifacts = {item.path for item in mlflow.MlflowClient().list_artifacts(outcome.run_id)}
    for expected in {
        "resolved_config.yaml",
        "metrics.json",
        "evals_result.json",
        "metadata.json",
        "feature_importance.csv",
        "dataset_identity.json",
        "model",
        "raw_xgboost_model",
    }:
        assert expected in artifacts
    logged = mlflow.pyfunc.load_model(outcome.model_uri)
    input_example = logged.metadata.load_input_example()
    prediction = logged.predict(input_example)["predicted_intensity"].to_numpy()
    np.testing.assert_allclose(prediction, outcome.in_process_example_prediction)
