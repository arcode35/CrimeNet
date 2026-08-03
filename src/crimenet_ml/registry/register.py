"""Register a calibrated model logged by a completed CrimeNet run."""

from __future__ import annotations

import argparse

import mlflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-artifact", default="model")
    parser.add_argument("--registered-model-name", required=True)
    parser.add_argument("--registry-uri")
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    if arguments.registry_uri:
        mlflow.set_registry_uri(arguments.registry_uri)
    result = mlflow.register_model(
        f"runs:/{arguments.run_id}/{arguments.model_artifact}",
        arguments.registered_model_name,
    )
    print(f"registered_model={result.name}")
    print(f"version={result.version}")


if __name__ == "__main__":
    main()
