import dagster as dg
import polars as pl

from crimenet_data.assets.crime.transformations.canonical import (
    add_canonical_crime,
    cleanse_data,
    convert_dallas_coordinates,
    deduplicate_city,
    normalize_dc_timestamps,
    project_canonical_crime_schema,
)
from crimenet_data.observability.context import log_context
from crimenet_data.observability.logger import get_logger
from crimenet_data.resources.crime_lake import CrimeLakeResources
from crimenet_data.resources.duckdb import DuckDBResource


log = get_logger(__name__)

GCP_CREDENTIALS = pl.CredentialProviderGCP()

ALL_CITIES = [
    "dallas",
    "new_york",
    "chicago",
    "baltimore",
    "seattle",
    "san_francisco",
    "washington_dc",
    "fort_worth",
]


@dg.asset(
    name="silver_crime_offenses",
    group_name="silver_crime",
    deps=[f"silver_{city}" for city in ALL_CITIES],
)
def silver_crime_offenses(
    context: dg.AssetExecutionContext,
    crime_lake: CrimeLakeResources,
) -> dg.MaterializeResult:

    with log_context(
        run_id=context.run_id,
        asset_key=context.asset_key.to_user_string(),
    ):
        log.info(
            "processing_started",
            cities=ALL_CITIES,
        )

        lfs = [
            pl.scan_delta(
                crime_lake.resolve_city_path(city, "silver"),
                credential_provider=GCP_CREDENTIALS,
            )
            for city in ALL_CITIES
        ]

        silver_lf = pl.union(
            lfs,
            how="vertical",
        )

        target_uri = (
            f"{crime_lake.silver_root.rstrip('/')}/"
            "crime_offenses"
        )

        log.info(
            "write_started",
            target_uri=target_uri,
        )

        crime_lake.write_crimenet_table(
            lf=silver_lf,
            target_uri=target_uri,
            partitioning_columns=[
                "source_city",
                "occurrence_year",
            ],
        )

        log.info(
            "processing_completed",
            target_uri=target_uri,
            cities_processed=len(ALL_CITIES),
        )

        return dg.MaterializeResult(
            metadata={
                "target_uri": target_uri,
                "cities_processed": len(ALL_CITIES),
            }
        )


@dg.asset(
    name="silver_dallas",
    group_name="silver_crime",
    deps=[
        "canonical_crime_crosswalk",
        "bronze_dallas",
    ],
)
def silver_dallas(
    context: dg.AssetExecutionContext,
    duckdb_resource: DuckDBResource,
    crime_lake: CrimeLakeResources,
) -> dg.MaterializeResult:

    city = "dallas"

    with log_context(
        run_id=context.run_id,
        asset_key=context.asset_key.to_user_string(),
        source_city=city,
    ):
        source_uri = crime_lake.resolve_city_path(
            city,
            "bronze",
        )

        target_uri = (
            f"{crime_lake.silver_root.rstrip('/')}/"
            f"crime/{city}"
        )

        log.info(
            "processing_started",
            source_uri=source_uri,
            target_uri=target_uri,
        )

        lf = pl.scan_delta(
            source_uri,
            credential_provider=GCP_CREDENTIALS,
        )

        lf = deduplicate_city(
            lf=lf,
            city=city,
        )

        with duckdb_resource.get_connection() as connection:
            lf = convert_dallas_coordinates(
                lf=lf,
                con=connection,
            )

            lf = add_canonical_crime(
                lf=lf,
                city=city,
            )

            lf = cleanse_data(
                lf=lf,
                city=city,
            )

            lf = project_canonical_crime_schema(
                lf=lf,
                city=city,
            )

            log.info(
                "write_started",
                target_uri=target_uri,
            )

            crime_lake.write_crimenet_table(
                lf=lf,
                target_uri=target_uri,
                partitioning_columns=[
                    "occurrence_year",
                ],
            )

        log.info(
            "processing_completed",
            target_uri=target_uri,
        )

        return dg.MaterializeResult(
            metadata={
                "city": city,
                "target_uri": target_uri,
            }
        )


def build_silver_city_asset(
    city: str,
) -> dg.AssetsDefinition:

    @dg.asset(
        name=f"silver_{city}",
        group_name="silver_crime",
        deps=[
            "canonical_crime_crosswalk",
            f"bronze_{city}",
        ],
    )
    def _silver_asset(
        context: dg.AssetExecutionContext,
        crime_lake: CrimeLakeResources,
    ) -> dg.MaterializeResult:

        with log_context(
            run_id=context.run_id,
            asset_key=context.asset_key.to_user_string(),
            source_city=city,
        ):
            source_uri = crime_lake.resolve_city_path(
                city,
                "bronze",
            )

            target_uri = (
                f"{crime_lake.silver_root.rstrip('/')}/"
                f"crime/{city}"
            )

            log.info(
                "processing_started",
                source_uri=source_uri,
                target_uri=target_uri,
            )

            lf = pl.scan_delta(
                source_uri,
                credential_provider=GCP_CREDENTIALS,
            )

            if city == "washington_dc":
                log.info("dc_timestamp_normalization_started")
                lf = normalize_dc_timestamps(lf)

            lf = deduplicate_city(
                lf=lf,
                city=city,
            )

            lf = add_canonical_crime(
                lf=lf,
                city=city,
            )

            lf = cleanse_data(
                lf=lf,
                city=city,
            )

            lf = project_canonical_crime_schema(
                lf=lf,
                city=city,
            )

            log.info(
                "write_started",
                target_uri=target_uri,
            )

            crime_lake.write_crimenet_table(
                lf=lf,
                target_uri=target_uri,
                partitioning_columns=[
                    "occurrence_year",
                ],
            )

            log.info(
                "processing_completed",
                target_uri=target_uri,
            )

            return dg.MaterializeResult(
                metadata={
                    "city": city,
                    "target_uri": target_uri,
                }
            )

    return _silver_asset


CITIES = [
    "new_york",
    "chicago",
    "baltimore",
    "seattle",
    "san_francisco",
    "washington_dc",
    "fort_worth",
]


crime_silver_assets = [
    silver_dallas,
    *[
        build_silver_city_asset(city)
        for city in CITIES
    ],
    silver_crime_offenses,
]