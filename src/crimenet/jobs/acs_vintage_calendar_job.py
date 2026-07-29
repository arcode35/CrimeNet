"""Build the versioned ACS release calendar with validation-before-promotion."""

from __future__ import annotations

import argparse

from pyspark.sql import SparkSession

from crimenet.config.validation import validate_identifier
from crimenet.observability.logging import get_logger
from crimenet.observability.run_context import resolve_pipeline_run_id
from crimenet.socioeconomic.acs_calendar import (
    ACS_CALENDAR_DEFINITION_VERSION,
    ACS_VINTAGE_RELEASES,
    create_calendar_dataframe,
    select_vintage_releases,
    validate_calendar_dataframe,
)
from crimenet.utils.promotion import promote_staged_table

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the release-aware ACS 5-year vintage calendar."
    )
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--silver-schema", default="silver")
    parser.add_argument("--start-vintage", type=int, default=2012)
    parser.add_argument("--end-vintage", type=int, default=2024)
    parser.add_argument(
        "--calendar-definition-version",
        default=ACS_CALENDAR_DEFINITION_VERSION,
    )
    parser.add_argument("--pipeline-run-id")
    return parser.parse_args()


def run(
    spark: SparkSession,
    *,
    catalog: str,
    silver_schema: str,
    start_vintage: int,
    end_vintage: int,
    calendar_definition_version: str,
    pipeline_run_id: str,
) -> str:
    """Create, stage, validate, and promote the release calendar."""

    validate_identifier(catalog, label="catalog")
    validate_identifier(silver_schema, label="silver_schema")
    target_table = f"{catalog}.{silver_schema}.acs_vintage_calendar"

    releases = select_vintage_releases(
        start_vintage=start_vintage,
        end_vintage=end_vintage,
        releases=ACS_VINTAGE_RELEASES,
    )
    candidate = create_calendar_dataframe(
        spark,
        releases,
        definition_version=calendar_definition_version,
    )

    LOGGER.info(
        "Staging ACS release calendar",
        pipeline_run_id=pipeline_run_id,
        target_table=target_table,
        start_vintage=start_vintage,
        end_vintage=end_vintage,
        calendar_definition_version=calendar_definition_version,
    )

    promote_staged_table(
        spark,
        candidate=candidate,
        target_table=target_table,
        pipeline_run_id=pipeline_run_id,
        validate=validate_calendar_dataframe,
    )

    LOGGER.info(
        "Promoted ACS release calendar",
        pipeline_run_id=pipeline_run_id,
        target_table=target_table,
        row_count=len(releases),
    )
    return target_table


def main() -> None:
    args = parse_args()
    pipeline_run_id = resolve_pipeline_run_id(args.pipeline_run_id)
    spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")

    run(
        spark,
        catalog=args.catalog,
        silver_schema=args.silver_schema,
        start_vintage=args.start_vintage,
        end_vintage=args.end_vintage,
        calendar_definition_version=(args.calendar_definition_version),
        pipeline_run_id=pipeline_run_id,
    )


if __name__ == "__main__":
    main()
