"""Python-wheel entry point for ACS landing ingestion."""

from __future__ import annotations

import argparse

from pyspark.sql import SparkSession

from crimenet.observability.logging import get_logger
from crimenet.observability.run_context import resolve_pipeline_run_id
from crimenet.socioeconomic.acs_client import TEXAS_STATE_FIPS
from crimenet.socioeconomic.acs_ingestion import ingest_acs5_tract_vintages

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--landing-path",
        required=True,
    )
    parser.add_argument(
        "--start-vintage",
        type=int,
        default=2012,
    )
    parser.add_argument(
        "--end-vintage",
        type=int,
        default=2024,
    )
    parser.add_argument(
        "--state-fips",
        default=TEXAS_STATE_FIPS,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--minimum-tract-records",
        type=int,
        default=1,
    )
    parser.add_argument("--secret-scope")
    parser.add_argument("--api-key-secret")
    parser.add_argument("--pipeline-run-id")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = resolve_pipeline_run_id(args.pipeline_run_id)

    spark = (
        SparkSession.getActiveSession()
        or SparkSession.builder.getOrCreate()
    )

    logger.info(
        "Starting ACS landing ingestion",
        landing_path=args.landing_path,
        start_vintage=args.start_vintage,
        end_vintage=args.end_vintage,
        state_fips=args.state_fips,
        overwrite=args.overwrite,
        minimum_tract_records=args.minimum_tract_records,
        pipeline_run_id=run_id,
    )

    try:
        api_key = None
        if bool(args.secret_scope) != bool(args.api_key_secret):
            raise ValueError(
                "--secret-scope and --api-key-secret must be supplied together."
            )
        if args.secret_scope:
            import importlib

            module = importlib.import_module("pyspark.dbutils")
            dbutils_type = module.DBUtils
            dbutils = dbutils_type(spark)
            api_key = dbutils.secrets.get(
                scope=args.secret_scope,
                key=args.api_key_secret,
            )

        ingest_acs5_tract_vintages(
            landing_directory=args.landing_path,
            start_vintage=args.start_vintage,
            end_vintage=args.end_vintage,
            state_fips=args.state_fips,
            api_key=api_key,
            overwrite=args.overwrite,
            minimum_record_count=args.minimum_tract_records,
        )

    except Exception:
        logger.exception(
            "ACS landing ingestion failed",
            start_vintage=args.start_vintage,
            end_vintage=args.end_vintage,
            state_fips=args.state_fips,
            pipeline_run_id=run_id,
        )
        raise

    logger.info(
        "ACS landing ingestion completed",
        start_vintage=args.start_vintage,
        end_vintage=args.end_vintage,
        state_fips=args.state_fips,
        pipeline_run_id=run_id,
    )


if __name__ == "__main__":
    main()
