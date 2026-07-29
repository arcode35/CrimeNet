"""Land and normalize every TIGER/Line tract vintage required by ACS."""

from __future__ import annotations

import argparse
from functools import reduce

from pyspark.sql import DataFrame, SparkSession

from crimenet.boundaries.tiger_line import (
    BOUNDARY_DEFINITION_VERSION,
    TIGER_BASE_URL,
    BoundaryIssue,
    NormalizationResult,
    boundary_issues_to_dataframe,
    create_boundary_dataframe,
    land_tiger_archives,
    merge_boundary_quarantine,
    normalization_failure_issue,
    normalize_tiger_archive,
    validate_boundary_dataframe,
)
from crimenet.config.validation import validate_identifier
from crimenet.observability.logging import get_logger
from crimenet.observability.run_context import resolve_pipeline_run_id
from crimenet.utils.promotion import promote_staged_table

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Atomically land, normalize, validate, and promote Census "
            "TIGER/Line tract boundaries required by the ACS calendar."
        )
    )
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--silver-schema", default="silver")
    parser.add_argument("--data-quality-schema", default="data_quality")
    parser.add_argument("--landing-path", required=True)
    parser.add_argument("--tiger-base-url", default=TIGER_BASE_URL)
    parser.add_argument("--state-fips", default="48")
    parser.add_argument(
        "--boundary-definition-version",
        default=BOUNDARY_DEFINITION_VERSION,
    )
    parser.add_argument(
        "--minimum-tracts-per-vintage",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--maximum-quarantine-records",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--download-timeout-seconds",
        type=float,
        default=180.0,
    )
    parser.add_argument(
        "--maximum-archive-bytes",
        type=int,
        default=1_000_000_000,
    )
    parser.add_argument("--overwrite-landing", action="store_true")
    parser.add_argument("--pipeline-run-id")
    return parser.parse_args()


def _calendar_boundary_years(
    spark: SparkSession,
    *,
    calendar_table: str,
) -> tuple[int, ...]:
    rows = (
        spark.table(calendar_table)
        .select("tiger_line_year")
        .distinct()
        .orderBy("tiger_line_year")
        .collect()
    )
    years = tuple(int(row["tiger_line_year"]) for row in rows)
    if not years:
        raise ValueError(
            f"ACS calendar contains no TIGER/Line years: {calendar_table}."
        )
    return years


def run(
    spark: SparkSession,
    *,
    catalog: str,
    silver_schema: str,
    data_quality_schema: str,
    landing_path: str,
    tiger_base_url: str,
    state_fips: str,
    boundary_definition_version: str,
    minimum_tracts_per_vintage: int,
    maximum_quarantine_records: int,
    download_timeout_seconds: float,
    maximum_archive_bytes: int,
    overwrite_landing: bool,
    pipeline_run_id: str,
) -> str:
    """Build a complete candidate without touching the last good table."""

    validate_identifier(catalog, label="catalog")
    validate_identifier(silver_schema, label="silver_schema")
    validate_identifier(
        data_quality_schema,
        label="data_quality_schema",
    )
    if maximum_quarantine_records < 0:
        raise ValueError("maximum_quarantine_records cannot be negative.")

    calendar_table = f"{catalog}.{silver_schema}.acs_vintage_calendar"
    target_table = f"{catalog}.{silver_schema}.census_tract_boundaries"
    quarantine_table = f"{catalog}.{data_quality_schema}.boundary_quarantine"
    years = _calendar_boundary_years(
        spark,
        calendar_table=calendar_table,
    )

    LOGGER.info(
        "Landing required TIGER/Line tract archives",
        pipeline_run_id=pipeline_run_id,
        years=years,
        state_fips=state_fips,
        landing_path=landing_path,
        boundary_definition_version=boundary_definition_version,
    )
    landing = land_tiger_archives(
        years=years,
        state_fips=state_fips,
        landing_directory=landing_path,
        overwrite=overwrite_landing,
        timeout_seconds=download_timeout_seconds,
        maximum_archive_bytes=maximum_archive_bytes,
        base_url=tiger_base_url,
    )

    candidate_parts: list[DataFrame] = []
    record_count = 0
    issues: list[BoundaryIssue] = list(landing.issues)
    for archive in landing.archives:
        try:
            normalized: NormalizationResult = normalize_tiger_archive(
                archive,
                definition_version=boundary_definition_version,
            )
        except Exception as exc:
            issues.append(normalization_failure_issue(archive, exc))
        else:
            record_count += len(normalized.records)
            candidate_parts.append(
                create_boundary_dataframe(
                    spark,
                    normalized.records,
                )
            )
            issues.extend(normalized.issues)

    if issues:
        quarantine = boundary_issues_to_dataframe(
            spark,
            issues,
            pipeline_run_id=pipeline_run_id,
            definition_version=boundary_definition_version,
        )
        merge_boundary_quarantine(
            spark,
            quarantine,
            target_table=quarantine_table,
            pipeline_run_id=pipeline_run_id,
        )

    if len(issues) > maximum_quarantine_records:
        reason_counts: dict[str, int] = {}
        for issue in issues:
            reason_counts[issue.reason_code] = (
                reason_counts.get(issue.reason_code, 0) + 1
            )
        raise RuntimeError(
            "TIGER/Line boundary quarantine threshold exceeded: "
            f"observed={len(issues)}, "
            f"maximum={maximum_quarantine_records}, "
            f"reasons={reason_counts}. The previous final table was "
            "not modified."
        )

    if candidate_parts:
        candidate = reduce(
            lambda left, right: left.unionByName(right),
            candidate_parts,
        )
    else:
        candidate = create_boundary_dataframe(spark, ())

    def validate(candidate_table: DataFrame) -> None:
        validate_boundary_dataframe(
            candidate_table,
            expected_years=years,
            state_fips=state_fips,
            minimum_tracts_per_vintage=(minimum_tracts_per_vintage),
        )

    promote_staged_table(
        spark,
        candidate=candidate,
        target_table=target_table,
        pipeline_run_id=pipeline_run_id,
        validate=validate,
    )

    LOGGER.info(
        "Promoted Census tract boundaries",
        pipeline_run_id=pipeline_run_id,
        target_table=target_table,
        boundary_vintages=len(years),
        row_count=record_count,
        quarantine_count=len(issues),
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
        landing_path=args.landing_path,
        tiger_base_url=args.tiger_base_url,
        state_fips=args.state_fips,
        boundary_definition_version=(args.boundary_definition_version),
        minimum_tracts_per_vintage=(args.minimum_tracts_per_vintage),
        maximum_quarantine_records=(args.maximum_quarantine_records),
        download_timeout_seconds=(args.download_timeout_seconds),
        maximum_archive_bytes=args.maximum_archive_bytes,
        overwrite_landing=args.overwrite_landing,
        pipeline_run_id=pipeline_run_id,
    )


if __name__ == "__main__":
    main()
