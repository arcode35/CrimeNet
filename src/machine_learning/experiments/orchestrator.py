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
from machine_learning.data.features import resolve_feature_contract
from machine_learning.data.model_table import (
    enrich_config_with_lineage,
    resolve_model_table_from_config,
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

    data_config = config.setdefault("data", {})
    table = resolve_model_table_from_config(data_config)
    config = enrich_config_with_lineage(config, table)
    contract = resolve_feature_contract(
        config["features"],
        available_columns=table.scan_split(
            str(config["data"].get("train_split", "train"))
        ).collect_schema().names(),
    )
    config["features"]["resolved_numeric"] = list(contract.numeric)
    config["features"]["resolved_categorical"] = list(contract.categorical)
    config["features"]["feature_contract_hash"] = contract.contract_hash

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
