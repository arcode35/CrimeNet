"""Python-wheel entry point for ACS 5-year tract acquisition."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path

from pyspark.sql import SparkSession

from crimenet.observability.logging import (
    get_logger,
)
from crimenet.socioeconomic.acs_client import (
    DEFAULT_END_VINTAGE,
    DEFAULT_START_VINTAGE,
    METRO_GEOGRAPHIES,
    CensusApiError,
    build_census_session,
)
from crimenet.socioeconomic.acs_ingestion import (
    ingest_acs5_tracts,
)


LOGGER = get_logger(__name__)


DEFAULT_METROS = (
    "new_york",
    "chicago",
    "san_francisco",
    "seattle",
    "baltimore",
    "washington_dc",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Acquire ACS 5-year tract data into "
            "a Unity Catalog volume."
        )
    )

    parser.add_argument(
        "--metros",
        nargs="+",
        choices=sorted(
            METRO_GEOGRAPHIES
        ),
        default=list(
            DEFAULT_METROS
        ),
    )

    parser.add_argument(
        "--start-vintage",
        type=int,
        default=DEFAULT_START_VINTAGE,
    )

    parser.add_argument(
        "--end-vintage",
        type=int,
        default=DEFAULT_END_VINTAGE,
    )

    parser.add_argument(
        "--output-root",
        required=True,
    )

    parser.add_argument(
        "--secret-scope",
        required=True,
    )

    parser.add_argument(
        "--secret-key",
        required=True,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--max-partitions",
        type=int,
        default=None,
        help=(
            "Limit actual Census API attempts. "
            "Verified cached partitions do not "
            "count toward this limit."
        ),
    )

    return parser.parse_args()


def get_census_api_key(
    spark: SparkSession,
    *,
    secret_scope: str,
    secret_key: str,
) -> str:
    """Read the Census API key from Databricks Secrets."""
    DBUtils = import_module(
        "pyspark.dbutils"
    ).DBUtils

    dbutils = DBUtils(
        spark
    )

    api_key = dbutils.secrets.get(
        scope=secret_scope,
        key=secret_key,
    )

    if not api_key.strip():
        raise CensusApiError(
            "The Census API secret is empty"
        )

    return api_key


def run(
    spark: SparkSession,
    *,
    metros: Sequence[str],
    start_vintage: int,
    end_vintage: int,
    output_root: str,
    secret_scope: str,
    secret_key: str,
    overwrite: bool,
    pause_seconds: float,
    maximum_partitions: int | None,
) -> None:
    if start_vintage > end_vintage:
        raise ValueError(
            "start_vintage cannot exceed end_vintage"
        )

    if pause_seconds < 0:
        raise ValueError(
            "pause_seconds cannot be negative"
        )

    if (
        maximum_partitions is not None
        and maximum_partitions <= 0
    ):
        raise ValueError(
            "maximum_partitions must be positive"
        )

    spark.conf.set(
        "spark.sql.session.timeZone",
        "UTC",
    )

    api_key = get_census_api_key(
        spark,
        secret_scope=secret_scope,
        secret_key=secret_key,
    )

    resolved_output_root = Path(
        output_root
    )

    resolved_output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOGGER.info(
        "Starting ACS 5-year tract acquisition",
        metros=list(
            metros
        ),
        start_vintage=start_vintage,
        end_vintage=end_vintage,
        output_root=str(
            resolved_output_root
        ),
        overwrite=overwrite,
        maximum_partitions=(
            maximum_partitions
        ),
    )

    session = build_census_session()

    try:
        summary = ingest_acs5_tracts(
            spark,
            output_root=(
                resolved_output_root
            ),
            metros=metros,
            start_vintage=start_vintage,
            end_vintage=end_vintage,
            api_key=api_key,
            overwrite=overwrite,
            pause_seconds=pause_seconds,
            maximum_partitions=(
                maximum_partitions
            ),
            session=session,
        )

    finally:
        session.close()

    LOGGER.info(
        "ACS 5-year tract acquisition completed",
        examined=summary.examined,
        attempted=summary.attempted,
        downloaded=summary.downloaded,
        cached=summary.cached,
        failed=summary.failed,
        output_root=str(
            resolved_output_root
        ),
    )

    if summary.failed:
        failure_preview = "\n".join(
            summary.failures[:10]
        )

        raise RuntimeError(
            f"{summary.failed} ACS partitions "
            "failed. First failures:\n"
            f"{failure_preview}"
        )


def main() -> None:
    args = parse_args()

    spark = (
        SparkSession.getActiveSession()
        or SparkSession.builder.getOrCreate()
    )

    try:
        run(
            spark,
            metros=args.metros,
            start_vintage=(
                args.start_vintage
            ),
            end_vintage=(
                args.end_vintage
            ),
            output_root=(
                args.output_root
            ),
            secret_scope=(
                args.secret_scope
            ),
            secret_key=(
                args.secret_key
            ),
            overwrite=args.overwrite,
            pause_seconds=(
                args.pause_seconds
            ),
            maximum_partitions=(
                args.max_partitions
            ),
        )

    except Exception:
        LOGGER.exception(
            "ACS 5-year tract acquisition failed",
            output_root=args.output_root,
            start_vintage=(
                args.start_vintage
            ),
            end_vintage=(
                args.end_vintage
            ),
        )

        raise


if __name__ == "__main__":
    main()