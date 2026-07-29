"""Bootstrap and validate workspace resources required by the bundle."""

from __future__ import annotations

import argparse
import importlib

from pyspark.sql import SparkSession

from crimenet.config.validation import validate_identifier
from crimenet.observability.logging import get_logger
from crimenet.observability.run_context import resolve_pipeline_run_id

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create managed schemas/volumes and validate secrets."
    )
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--bronze-schema", default="bronze")
    parser.add_argument("--silver-schema", default="silver")
    parser.add_argument("--gold-schema", default="gold")
    parser.add_argument("--ops-schema", default="ops")
    parser.add_argument("--data-quality-schema", default="data_quality")
    parser.add_argument("--raw-files-schema", default="raw_files")
    parser.add_argument("--landing-volume", default="landing")
    parser.add_argument(
        "--autoloader-schemas-volume",
        default="autoloader_schemas",
    )
    parser.add_argument("--checkpoints-volume", default="checkpoints")
    parser.add_argument(
        "--preflight-mode",
        choices=("create", "validate"),
        default="create",
        help=(
            "Create missing managed resources before validation, or only "
            "validate resources provisioned outside the job."
        ),
    )
    parser.add_argument("--secret-scope")
    parser.add_argument("--census-api-key-secret")
    parser.add_argument("--pipeline-run-id")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate resources without attempting managed-resource creation.",
    )
    return parser.parse_args()


def _validate_secret(
    spark: SparkSession,
    *,
    secret_scope: str | None,
    census_api_key_secret: str | None,
) -> None:
    if bool(secret_scope) != bool(census_api_key_secret):
        raise ValueError(
            "secret_scope and census_api_key_secret must be supplied together."
        )
    if not secret_scope:
        LOGGER.info(
            "Census API key secret is not configured; the public unauthenticated "
            "Census API quota will be used."
        )
        return

    module = importlib.import_module("pyspark.dbutils")
    dbutils_type = module.DBUtils
    dbutils = dbutils_type(spark)
    value = dbutils.secrets.get(
        scope=secret_scope,
        key=census_api_key_secret,
    )
    if not value:
        raise RuntimeError(
            "The configured Census API key secret exists but is empty."
        )
    LOGGER.info(
        "Validated configured Census API key secret",
        secret_scope=secret_scope,
        secret_key=census_api_key_secret,
    )


def run(
    spark: SparkSession,
    *,
    catalog: str,
    schemas: tuple[str, ...],
    volumes: tuple[tuple[str, str], ...],
    secret_scope: str | None,
    census_api_key_secret: str | None,
    validate_only: bool,
) -> None:
    catalog = validate_identifier(catalog, label="catalog")
    schemas = tuple(
        validate_identifier(schema, label="schema") for schema in schemas
    )
    volumes = tuple(
        (
            validate_identifier(schema, label="volume schema"),
            validate_identifier(volume, label="volume"),
        )
        for schema, volume in volumes
    )

    if not validate_only:
        spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
        for schema in schemas:
            spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
        for schema, volume in volumes:
            spark.sql(
                f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{volume}"
            )

    missing_schemas = [
        schema
        for schema in schemas
        if not spark.catalog.databaseExists(f"{catalog}.{schema}")
    ]
    if missing_schemas:
        raise RuntimeError(
            "Missing required schemas: "
            + ", ".join(f"{catalog}.{name}" for name in missing_schemas)
            + ". Grant CREATE SCHEMA on the catalog or provision them first."
        )

    missing_volumes: list[str] = []
    for schema, volume in volumes:
        rows = spark.sql(f"SHOW VOLUMES IN {catalog}.{schema}").collect()
        names = {
            str(row["volume_name"])
            for row in rows
            if "volume_name" in row.asDict()
        }
        if volume not in names:
            missing_volumes.append(f"{catalog}.{schema}.{volume}")
    if missing_volumes:
        raise RuntimeError(
            "Missing required Unity Catalog volumes: "
            + ", ".join(missing_volumes)
            + ". Grant CREATE VOLUME or provision them before deployment."
        )

    _validate_secret(
        spark,
        secret_scope=secret_scope,
        census_api_key_secret=census_api_key_secret,
    )
    LOGGER.info(
        "CrimeNet workspace preflight passed",
        catalog=catalog,
        schemas=list(schemas),
        volumes=[f"{catalog}.{schema}.{volume}" for schema, volume in volumes],
    )


def main() -> None:
    args = parse_args()
    run_id = resolve_pipeline_run_id(args.pipeline_run_id)
    spark = (
        SparkSession.getActiveSession()
        or SparkSession.builder.getOrCreate()
    )
    run(
        spark,
        catalog=args.catalog,
        schemas=(
            args.bronze_schema,
            args.silver_schema,
            args.gold_schema,
            args.ops_schema,
            args.data_quality_schema,
            args.raw_files_schema,
        ),
        volumes=(
            (args.raw_files_schema, args.landing_volume),
            (args.ops_schema, args.autoloader_schemas_volume),
            (args.ops_schema, args.checkpoints_volume),
        ),
        secret_scope=args.secret_scope,
        census_api_key_secret=args.census_api_key_secret,
        validate_only=(
            args.validate_only
            or args.preflight_mode == "validate"
        ),
    )
    LOGGER.info(
        "CrimeNet preflight task completed",
        pipeline_run_id=run_id,
        catalog=args.catalog,
    )


if __name__ == "__main__":
    main()
