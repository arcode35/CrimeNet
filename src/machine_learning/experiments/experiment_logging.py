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
