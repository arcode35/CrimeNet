from __future__ import annotations

import argparse
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import yaml

from machine_learning.experiments.mlflow_config import resume_run


EXPERIMENT_LOG = Path(__file__).resolve().parent / "log" / "experiments.jsonl"


def load_config(path: Path) -> dict:
    with path.open() as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML mapping in {path}")

    return config


def run_validation(*, config_path: Path, run_id: str) -> dict:
    config = load_config(config_path)
    model_config = config["model"]

    validation_module_name = model_config.get("validation_module")
    if not validation_module_name:
        raise ValueError("model.validation_module is required")

    validation_module = importlib.import_module(validation_module_name)
    validate = validation_module.validate

    with resume_run(run_id=run_id):
        try:
            result = validate(config, run_id=run_id)

            mlflow.log_metrics(
                {
                    f"full_validation.{key}": float(value)
                    for key, value in result.get("metrics", {}).items()
                }
            )

            for city, metrics in result.get("city_metrics", {}).items():
                city_key = city.lower().replace(" ", "_")

                mlflow.log_metrics(
                    {
                        f"full_validation.city.{city_key}.{key}": float(value)
                        for key, value in metrics.items()
                    }
                )

            for artifact in result.get("artifacts", []):
                mlflow.log_artifact(
                    artifact,
                    artifact_path="full_validation",
                )

            validation_split = result["summary"]["validation_split"]

            mlflow.set_tags(
                {
                    "full_validation_status": "completed",
                    "full_validation_split": str(validation_split),
                    "test_split_used": "false",
                }
            )

            EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)

            ledger_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": "full_validation",
                "run_id": run_id,
                "model": model_config["name"],
                "validation_split": validation_split,
                "metrics": result.get("metrics", {}),
                "city_metrics": result.get("city_metrics", {}),
                "artifacts": result.get("artifacts", []),
                "summary": result.get("summary", {}),
            }

            with EXPERIMENT_LOG.open("a") as file:
                file.write(json.dumps(ledger_entry, sort_keys=True) + "\n")

            return result

        except Exception as exc:
            mlflow.set_tags(
                {
                    "full_validation_status": "failed",
                    "full_validation_failure_type": type(exc).__name__,
                }
            )
            raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full validation for an existing CrimeNet experiment."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_validation(
        config_path=args.config,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
