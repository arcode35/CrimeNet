"""Materialize version-aware crime-location to Census-tract mappings."""

from __future__ import annotations

import argparse

from pyspark.sql import DataFrame, SparkSession

from crimenet.boundaries.tiger_line import BOUNDARY_DEFINITION_VERSION
from crimenet.config.validation import (
    validate_identifier,
    validate_qualified_table_name,
)
from crimenet.observability.logging import get_logger
from crimenet.observability.run_context import resolve_pipeline_run_id
from crimenet.spatial.tract_mapping import (
    LOCATION_KEY_COLUMNS,
    MAPPING_DEFINITION_VERSION,
    attach_release_aware_boundary_year,
    mapping_issues_to_quarantine,
    merge_spatial_quarantine,
    select_stale_or_missing_locations,
    spatially_map_locations,
    split_location_candidates,
    validate_mapping_dataframe,
    validate_spatial_boundary_inputs,
)
from crimenet.utils.promotion import (
    promote_staged_table,
    staging_table_name,
)

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the version-aware crime coordinate/year-to-tract "
            "mapping independently of the Gold feature rebuild."
        )
    )
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--silver-schema", default="silver")
    parser.add_argument("--data-quality-schema", default="data_quality")
    parser.add_argument(
        "--boundary-definition-version",
        default=BOUNDARY_DEFINITION_VERSION,
    )
    parser.add_argument(
        "--mapping-definition-version",
        default=MAPPING_DEFINITION_VERSION,
    )
    parser.add_argument(
        "--maximum-invalid-location-rate",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--maximum-unmatched-rate",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--maximum-ambiguous-matches",
        type=int,
        default=0,
    )
    parser.add_argument("--full-rebuild", action="store_true")
    parser.add_argument("--pipeline-run-id")
    return parser.parse_args()


def _merge_mapping_updates(
    spark: SparkSession,
    *,
    candidate: DataFrame,
    expected_locations: DataFrame,
    target_table: str,
    pipeline_run_id: str,
    maximum_ambiguous_matches: int,
    maximum_unmatched_rate: float,
) -> None:
    validate_qualified_table_name(target_table)
    stage = staging_table_name(
        target_table,
        f"{pipeline_run_id}_mapping_updates",
    )
    (
        candidate.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(stage)
    )
    try:
        staged = spark.table(stage)
        validate_mapping_dataframe(
            staged,
            expected_locations=expected_locations,
            maximum_ambiguous_matches=maximum_ambiguous_matches,
            maximum_unmatched_rate=maximum_unmatched_rate,
        )
        spark.sql(
            f"""
            MERGE INTO {target_table} AS target
            USING {stage} AS source
            ON target.tiger_line_year = source.tiger_line_year
            AND target.latitude = source.latitude
            AND target.longitude = source.longitude
            WHEN MATCHED THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *
            """
        )
    finally:
        spark.sql(f"DROP TABLE IF EXISTS {stage}")


def run(
    spark: SparkSession,
    *,
    catalog: str,
    silver_schema: str,
    data_quality_schema: str,
    boundary_definition_version: str,
    mapping_definition_version: str,
    maximum_invalid_location_rate: float,
    maximum_unmatched_rate: float,
    maximum_ambiguous_matches: int,
    full_rebuild: bool,
    pipeline_run_id: str,
) -> str:
    """Build missing/stale mappings or safely replace a full candidate."""

    validate_identifier(catalog, label="catalog")
    validate_identifier(silver_schema, label="silver_schema")
    validate_identifier(
        data_quality_schema,
        label="data_quality_schema",
    )
    if not 0.0 <= maximum_invalid_location_rate <= 1.0:
        raise ValueError("maximum_invalid_location_rate must be between 0 and 1.")

    silver = f"{catalog}.{silver_schema}"
    crime_table = f"{silver}.crime_offenses"
    calendar_table = f"{silver}.acs_vintage_calendar"
    boundary_table = f"{silver}.census_tract_boundaries"
    target_table = f"{silver}.crime_location_tract_mapping"
    quarantine_table = f"{catalog}.{data_quality_schema}.spatial_mapping_quarantine"

    crimes = spark.table(crime_table)
    calendar = spark.table(calendar_table)
    boundaries = spark.table(boundary_table)
    validate_spatial_boundary_inputs(
        boundaries,
        boundary_definition_version=(boundary_definition_version),
    )
    crime_with_calendar = attach_release_aware_boundary_year(
        crimes,
        calendar,
    )
    location_frames = split_location_candidates(
        crime_with_calendar,
        pipeline_run_id=pipeline_run_id,
    )
    if not location_frames.quarantine.isEmpty():
        merge_spatial_quarantine(
            spark,
            location_frames.quarantine,
            target_table=quarantine_table,
            pipeline_run_id=pipeline_run_id,
        )

    invalid_rate = (
        location_frames.invalid_row_count / location_frames.source_row_count
        if location_frames.source_row_count
        else 0.0
    )
    if invalid_rate > maximum_invalid_location_rate:
        raise RuntimeError(
            "Invalid spatial-input rate exceeded its threshold: "
            f"observed={invalid_rate:.8f}, "
            f"maximum={maximum_invalid_location_rate:.8f}. "
            "Rejected rows were quarantined and the final mapping "
            "was not modified."
        )
    if location_frames.candidates.isEmpty():
        raise RuntimeError(
            "No valid crime locations are available for tract mapping. "
            "Rejected rows were quarantined and the final mapping was "
            "not modified."
        )

    target_exists = spark.catalog.tableExists(target_table)
    if full_rebuild or not target_exists:
        locations_to_map = location_frames.candidates
    else:
        existing_mapping = spark.table(target_table)
        duplicate_existing_keys = (
            existing_mapping.groupBy(*LOCATION_KEY_COLUMNS)
            .count()
            .filter("count != 1")
            .limit(1)
            .count()
        )
        if duplicate_existing_keys:
            raise RuntimeError(
                "Existing location mapping contains duplicate physical "
                "keys. Run a full rebuild after investigating the table."
            )
        locations_to_map = select_stale_or_missing_locations(
            location_frames.candidates,
            boundaries,
            existing_mapping,
            boundary_definition_version=(boundary_definition_version),
            mapping_definition_version=(mapping_definition_version),
        )

    if locations_to_map.isEmpty():
        LOGGER.info(
            "No new or version-stale location keys require mapping",
            pipeline_run_id=pipeline_run_id,
            target_table=target_table,
            mapping_definition_version=mapping_definition_version,
            boundary_definition_version=boundary_definition_version,
        )
        return target_table

    candidate_mapping = spatially_map_locations(
        locations_to_map,
        boundaries,
        boundary_definition_version=boundary_definition_version,
        mapping_definition_version=mapping_definition_version,
        pipeline_run_id=pipeline_run_id,
    )
    mapping_quarantine = mapping_issues_to_quarantine(candidate_mapping)
    if not mapping_quarantine.isEmpty():
        merge_spatial_quarantine(
            spark,
            mapping_quarantine,
            target_table=quarantine_table,
            pipeline_run_id=pipeline_run_id,
        )

    # Validate before either the one-table replacement or atomic Delta MERGE.
    validate_mapping_dataframe(
        candidate_mapping,
        expected_locations=locations_to_map,
        maximum_ambiguous_matches=maximum_ambiguous_matches,
        maximum_unmatched_rate=maximum_unmatched_rate,
    )

    if full_rebuild or not target_exists:

        def validate(staged: DataFrame) -> None:
            validate_mapping_dataframe(
                staged,
                expected_locations=location_frames.candidates,
                maximum_ambiguous_matches=(maximum_ambiguous_matches),
                maximum_unmatched_rate=maximum_unmatched_rate,
            )

        promote_staged_table(
            spark,
            candidate=candidate_mapping,
            target_table=target_table,
            pipeline_run_id=pipeline_run_id,
            validate=validate,
        )
        operation = "full_promotion"
    else:
        _merge_mapping_updates(
            spark,
            candidate=candidate_mapping,
            expected_locations=locations_to_map,
            target_table=target_table,
            pipeline_run_id=pipeline_run_id,
            maximum_ambiguous_matches=(maximum_ambiguous_matches),
            maximum_unmatched_rate=maximum_unmatched_rate,
        )
        operation = "incremental_merge"

    final_table = spark.table(target_table)
    duplicate_keys = (
        final_table.groupBy(*LOCATION_KEY_COLUMNS)
        .count()
        .filter("count != 1")
        .limit(1)
        .count()
    )
    if duplicate_keys:
        raise RuntimeError(
            "Final location mapping contains duplicate physical keys after promotion."
        )

    LOGGER.info(
        "Completed location-to-tract materialization",
        pipeline_run_id=pipeline_run_id,
        target_table=target_table,
        operation=operation,
        mapped_location_count=candidate_mapping.count(),
        invalid_source_row_count=(location_frames.invalid_row_count),
        invalid_source_row_rate=invalid_rate,
        mapping_definition_version=mapping_definition_version,
        boundary_definition_version=boundary_definition_version,
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
        data_quality_schema=args.data_quality_schema,
        boundary_definition_version=(args.boundary_definition_version),
        mapping_definition_version=(args.mapping_definition_version),
        maximum_invalid_location_rate=(args.maximum_invalid_location_rate),
        maximum_unmatched_rate=args.maximum_unmatched_rate,
        maximum_ambiguous_matches=(args.maximum_ambiguous_matches),
        full_rebuild=args.full_rebuild,
        pipeline_run_id=pipeline_run_id,
    )


if __name__ == "__main__":
    main()
