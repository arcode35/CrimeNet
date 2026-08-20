import dagster as dg
import polars as pl

from crimenet_data.assets.tract_resources.transformations import (
    build_tract_lookup_workload,
    resolve_tract_mappings,
)
from crimenet_data.observability.context import log_context
from crimenet_data.observability.logger import get_logger
from crimenet_data.resources.crime_lake import CrimeLakeResources
from crimenet_data.resources.duckdb import DuckDBResource


log = get_logger(__name__)

GCP_CREDENTIALS = pl.CredentialProviderGCP()

TRACT_RESOURCE_ROOT = (
    "gs://crimenet/raw_files/landing/"
    "tract_resources"
)

ACS_CALENDAR_URI = (
    f"{TRACT_RESOURCE_ROOT}/"
    "acs_vintage_calendar"
)

TRACT_BOUNDARIES_URI = (
    f"{TRACT_RESOURCE_ROOT}/"
    "census_tract_boundaries_polars_v2"
)

MIN_MAPPING_RATE = 0.99


@dg.asset(
    name="crime_location_tract_mapping",
    group_name="tract_resources",
    deps=["silver_crime_offenses"],
)
def crime_location_tract_mapping(
    context: dg.AssetExecutionContext,
    crime_lake: CrimeLakeResources,
    duckdb_resource: DuckDBResource,
) -> dg.MaterializeResult:

    with log_context(
        run_id=context.run_id,
        asset_key=context.asset_key.to_user_string(),
    ):
        crime_uri = (
            f"{crime_lake.silver_root.rstrip('/')}/"
            "crime_offenses"
        )

        target_uri = (
            f"{crime_lake.silver_root.rstrip('/')}/"
            "tract_resources/"
            "crime_location_tract_mapping"
        )

        log.info(
            "processing_started",
            crime_uri=crime_uri,
            target_uri=target_uri,
        )

        crime_lf = (
            pl.scan_delta(
                crime_uri,
                credential_provider=GCP_CREDENTIALS,
            )
            .select(
                "occurrence_timestamp",
                "latitude",
                "longitude",
            )
        )

        calendar_lf = (
            pl.scan_delta(
                ACS_CALENDAR_URI,
                credential_provider=GCP_CREDENTIALS,
            )
            .select(
                "acs_vintage",
                "acs_release_date",
                "tiger_line_year",
                "tract_definition_vintage",
            )
        )

        boundaries_lf = (
            pl.scan_delta(
                TRACT_BOUNDARIES_URI,
                credential_provider=GCP_CREDENTIALS,
            )
            .select(
                "geoid",
                "boundary_vintage",
                "tract_geometry_wkb",
            )
        )

        lookup_lf = build_tract_lookup_workload(
            crime_lf=crime_lf,
            calendar_lf=calendar_lf,
        )

        lookup_count = (
            lookup_lf
            .select(pl.len())
            .collect()
            .item()
        )

        log.info(
            "lookup_workload_created",
            unique_lookup_keys=lookup_count,
        )

        with duckdb_resource.get_connection() as con:
            mapping_lf, ambiguous_keys = (
                resolve_tract_mappings(
                    con=con,
                    lookup_lf=lookup_lf,
                    boundaries_lf=boundaries_lf,
                )
            )

            mapped_count = (
                mapping_lf
                .select(pl.len())
                .collect()
                .item()
            )

            mapping_rate = (
                mapped_count / lookup_count
                if lookup_count
                else 0.0
            )

            log.info(
                "mapping_validation_completed",
                lookup_keys=lookup_count,
                mapped_keys=mapped_count,
                mapping_rate=mapping_rate,
                ambiguous_keys=ambiguous_keys,
            )

            if mapping_rate < MIN_MAPPING_RATE:
                raise ValueError(
                    "Crime-to-tract mapping coverage is below "
                    f"threshold: {mapping_rate:.6%} < "
                    f"{MIN_MAPPING_RATE:.2%}"
                )

            log.info(
                "write_started",
                target_uri=target_uri,
            )

            # Keep this inside the DuckDB context because mapping_lf
            # references a DuckDB temporary relation.
            crime_lake.write_crimenet_table(
                lf=mapping_lf,
                target_uri=target_uri,
                partitioning_columns=[
                    "tiger_line_year",
                ],
            )

        log.info(
            "processing_completed",
            target_uri=target_uri,
            lookup_keys=lookup_count,
            mapped_keys=mapped_count,
            mapping_rate=mapping_rate,
        )

        return dg.MaterializeResult(
            metadata={
                "target_uri": target_uri,
                "lookup_keys": lookup_count,
                "mapped_keys": mapped_count,
                "mapping_rate": mapping_rate,
                "ambiguous_keys": ambiguous_keys,
            }
        )


tract_resource_silver_assets = [
    crime_location_tract_mapping,
]