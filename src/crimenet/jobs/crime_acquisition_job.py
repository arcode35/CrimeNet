"""Python-wheel entry point for crime-source acquisition."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path

from pyspark.sql import SparkSession

from crimenet.ingestion.crime_acquisition import (
    CITY_CHOICES,
    acquire_crime_data,
    build_crime_session,
)
from crimenet.observability.logging import (
    get_logger,
)


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download supported city crime data "
            "into raw-normalized Parquet landing files."
        )
    )

    parser.add_argument(
        "--cities",
        nargs="+",
        choices=CITY_CHOICES,
        default=list(
            CITY_CHOICES
        ),
    )

    parser.add_argument(
        "--output-root",
        required=True,
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help=(
            "Optional common starting year. "
            "When omitted, source-specific defaults apply."
        ),
    )

    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
    )

    parser.add_argument(
        "--socrata-page-size",
        type=int,
        default=25_000,
    )

    parser.add_argument(
        "--arcgis-page-size",
        type=int,
        default=2_000,
    )

    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--socrata-secret-scope",
        default=None,
    )

    parser.add_argument(
        "--socrata-secret-key",
        default=None,
    )

    parser.add_argument(
        "--acquisition-run-id",
        default=None,
    )

    return parser.parse_args()


def get_optional_socrata_token(
    spark: SparkSession,
    *,
    secret_scope: str | None,
    secret_key: str | None,
) -> str:
    if (
        secret_scope is None
        and secret_key is None
    ):
        return ""

    if (
        secret_scope is None
        or secret_key is None
    ):
        raise ValueError(
            "--socrata-secret-scope and "
            "--socrata-secret-key must be "
            "provided together"
        )

    DBUtils = import_module(
        "pyspark.dbutils"
    ).DBUtils

    dbutils = DBUtils(
        spark
    )

    return dbutils.secrets.get(
        scope=secret_scope,
        key=secret_key,
    )


def run(
    spark: SparkSession,
    *,
    cities: Sequence[str],
    output_root: str,
    start_year: int | None,
    end_year: int,
    socrata_page_size: int,
    arcgis_page_size: int,
    pause_seconds: float,
    socrata_secret_scope: str | None,
    socrata_secret_key: str | None,
    acquisition_run_id: str | None,
) -> None:
    if (
        start_year is not None
        and start_year > end_year
    ):
        raise ValueError(
            "start_year cannot exceed end_year"
        )

    if socrata_page_size <= 0:
        raise ValueError(
            "socrata_page_size must be positive"
        )

    if arcgis_page_size <= 0:
        raise ValueError(
            "arcgis_page_size must be positive"
        )

    if pause_seconds < 0:
        raise ValueError(
            "pause_seconds cannot be negative"
        )

    spark.conf.set(
        "spark.sql.session.timeZone",
        "UTC",
    )

    token = get_optional_socrata_token(
        spark,
        secret_scope=(
            socrata_secret_scope
        ),
        secret_key=(
            socrata_secret_key
        ),
    )

    resolved_output_root = Path(
        output_root
    )

    resolved_output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOGGER.info(
        "Starting crime acquisition",
        cities=list(
            cities
        ),
        output_root=str(
            resolved_output_root
        ),
        start_year=start_year,
        end_year=end_year,
        acquisition_run_id=(
            acquisition_run_id
        ),
    )

    session = build_crime_session(
        socrata_app_token=token,
    )

    try:
        summary = acquire_crime_data(
            spark,
            session=session,
            cities=cities,
            output_root=(
                resolved_output_root
            ),
            start_year=start_year,
            end_year=end_year,
            socrata_page_size=(
                socrata_page_size
            ),
            arcgis_page_size=(
                arcgis_page_size
            ),
            pause_seconds=(
                pause_seconds
            ),
            acquisition_run_id=(
                acquisition_run_id
            ),
        )

    finally:
        session.close()

    LOGGER.info(
        "Crime acquisition completed",
        attempted_partitions=(
            summary.attempted_partitions
        ),
        completed_partitions=(
            summary.completed_partitions
        ),
        failed_partitions=(
            summary.failed_partitions
        ),
        downloaded_rows=(
            summary.downloaded_rows
        ),
    )

    if summary.failed_partitions:
        raise RuntimeError(
            f"{summary.failed_partitions} crime "
            "partitions failed. First failures:\n"
            + "\n".join(
                summary.failures[:10]
            )
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
            cities=args.cities,
            output_root=(
                args.output_root
            ),
            start_year=(
                args.start_year
            ),
            end_year=args.end_year,
            socrata_page_size=(
                args.socrata_page_size
            ),
            arcgis_page_size=(
                args.arcgis_page_size
            ),
            pause_seconds=(
                args.pause_seconds
            ),
            socrata_secret_scope=(
                args.socrata_secret_scope
            ),
            socrata_secret_key=(
                args.socrata_secret_key
            ),
            acquisition_run_id=(
                args.acquisition_run_id
            ),
        )

    except Exception:
        LOGGER.exception(
            "Crime acquisition job failed",
            cities=args.cities,
            output_root=args.output_root,
            end_year=args.end_year,
        )

        raise


if __name__ == "__main__":
    main()