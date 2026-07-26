"""Python-wheel entry point for ACS landing ingestion."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from crimenet.socioeconomic.acs_client import (
    TEXAS_STATE_FIPS,
)
from crimenet.socioeconomic.acs_ingestion import (
    ingest_acs5_tract_vintages,
)
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

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

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    args = parse_args()

    spark = (
        SparkSession.getActiveSession()
        or SparkSession.builder.getOrCreate()
    )

    dbutils = DBUtils(spark)

    api_key = dbutils.secrets.get(
        scope="crimenet-dev",
        key="census-api-key",
    )

    ingest_acs5_tract_vintages(
        landing_directory=args.landing_path,
        start_vintage=args.start_vintage,
        end_vintage=args.end_vintage,
        state_fips=args.state_fips,
        api_key=api_key,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()