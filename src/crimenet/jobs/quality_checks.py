"""Databricks entry point for Silver data-quality validation."""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--silver-schema", required=True)
    parser.add_argument("--data-quality-schema", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raise NotImplementedError(
        f"Implement Silver quality checks for catalog {args.catalog!r}."
    )


if __name__ == "__main__":
    main()
