"""Python-wheel entry point for solar lighting features."""

from __future__ import annotations

import argparse

from pyspark.sql import SparkSession

from crimenet.contracts.lighting import (
    LIGHTING_DEFINITION_VERSION,
)
from crimenet.observability.logging import get_logger
from crimenet.observability.run_context import (
    resolve_pipeline_run_id,
)
from crimenet.silver.lighting import (
    materialize_lighting_conditions,
)

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute solar position and lighting conditions for Silver crime records."
        )
    )

    parser.add_argument(
        "--catalog",
        required=True,
    )
    parser.add_argument(
        "--silver-schema",
        default="silver",
    )
    parser.add_argument(
        "--full-rebuild",
        action="store_true",
        help=("Recompute and overwrite the complete lighting conditions table."),
    )
    parser.add_argument(
        "--lighting-definition-version",
        default=LIGHTING_DEFINITION_VERSION,
        help=(
            "Version included in the physical lighting key. "
            "Changing it makes every source cell/hour eligible "
            "for recomputation."
        ),
    )
    parser.add_argument(
        "--pipeline-run-id",
        help=(
            "Identifier used to isolate the validated staging "
            "table. A random identifier is generated when omitted."
        ),
    )

    return parser.parse_args()


def run(
    spark: SparkSession,
    *,
    catalog: str,
    silver_schema: str,
    full_rebuild: bool,
    lighting_definition_version: str = (LIGHTING_DEFINITION_VERSION),
    pipeline_run_id: str | None = None,
) -> None:
    # Spark-to-pandas conversion must use UTC because pvlib receives
    # timezone-aware UTC timestamps.
    spark.conf.set(
        "spark.sql.session.timeZone",
        "UTC",
    )

    crime_table = f"{catalog}.{silver_schema}.crime_offenses"

    target_table = f"{catalog}.{silver_schema}.solar_lighting_conditions"

    materialize_lighting_conditions(
        spark,
        crime_table=crime_table,
        target_table=target_table,
        full_rebuild=full_rebuild,
        definition_version=(lighting_definition_version),
        pipeline_run_id=pipeline_run_id,
    )


def main() -> None:
    args = parse_args()
    pipeline_run_id = resolve_pipeline_run_id(args.pipeline_run_id)

    spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()

    LOGGER.info(
        "Starting Silver lighting job",
        catalog=args.catalog,
        silver_schema=args.silver_schema,
        full_rebuild=args.full_rebuild,
        pipeline_run_id=pipeline_run_id,
    )

    try:
        run(
            spark,
            catalog=args.catalog,
            silver_schema=args.silver_schema,
            full_rebuild=args.full_rebuild,
            lighting_definition_version=(args.lighting_definition_version),
            pipeline_run_id=pipeline_run_id,
        )
    except Exception:
        LOGGER.exception(
            "Silver lighting job failed",
            catalog=args.catalog,
            silver_schema=args.silver_schema,
            full_rebuild=args.full_rebuild,
        )
        raise

    LOGGER.info(
        "Silver lighting job completed",
        catalog=args.catalog,
        silver_schema=args.silver_schema,
        full_rebuild=args.full_rebuild,
    )


if __name__ == "__main__":
    main()
