from datetime import UTC, datetime

import dagster as dg
import polars as pl

from crimenet_data.assets.socioeconomic.transformations.silver import (
    build_tract_socioeconomic_silver,
    count_duplicate_socioeconomic_keys,
    count_invalid_acs_periods,
    count_invalid_geoids,
    count_invalid_geography_types,
    count_invalid_rates,
)
from crimenet_data.observability.context import log_context
from crimenet_data.observability.logger import get_logger
from crimenet_data.resources.crime_lake import CrimeLakeResources


log = get_logger(__name__)

GCP = pl.CredentialProviderGCP()


@dg.asset(
    name="silver_tract_socioeconomic",
    group_name="silver_socioeconomic",
    deps=["bronze_socioeconomic"],
)
def silver_tract_socioeconomic(
    context: dg.AssetExecutionContext,
    crime_lake: CrimeLakeResources,
) -> dg.MaterializeResult:
    with log_context(
        run_id=context.run_id,
        asset_key=context.asset_key.to_user_string(),
        source_system="census_acs5",
    ):
        processed_at = datetime.now(UTC)

        source_uri = (
            f"{crime_lake.bronze_root.rstrip('/')}/"
            "acs5/tract"
        )

        target_uri = (
            f"{crime_lake.silver_root.rstrip('/')}/"
            "socioeconomic/tract"
        )

        log.info(
            "processing_started",
            source_uri=source_uri,
            target_uri=target_uri,
        )

        bronze_lf = pl.scan_delta(
            source_uri,
            credential_provider=GCP,
        )

        silver_lf = build_tract_socioeconomic_silver(
            bronze_lf,
            processed_at=processed_at,
        )

        # -------------------------------------------------------------
        # Structural validation
        # -------------------------------------------------------------

        duplicate_keys = count_duplicate_socioeconomic_keys(
            silver_lf
        )

        if duplicate_keys:
            raise ValueError(
                "Socioeconomic Silver contains duplicate "
                "(acs_vintage, geoid) keys: "
                f"{duplicate_keys:,}"
            )

        invalid_geoids = count_invalid_geoids(
            silver_lf
        )

        if invalid_geoids:
            raise ValueError(
                "Socioeconomic Silver contains invalid "
                f"Census tract GEOIDs: {invalid_geoids:,}"
            )

        invalid_geography_types = (
            count_invalid_geography_types(
                silver_lf
            )
        )

        if invalid_geography_types:
            raise ValueError(
                "Socioeconomic Silver contains non-tract "
                "geographies: "
                f"{invalid_geography_types:,}"
            )

        invalid_acs_periods = count_invalid_acs_periods(
            silver_lf
        )

        if invalid_acs_periods:
            raise ValueError(
                "Socioeconomic Silver contains ACS periods "
                "inconsistent with their vintage: "
                f"{invalid_acs_periods:,}"
            )

        invalid_rates = count_invalid_rates(
            silver_lf
        )

        if invalid_rates:
            raise ValueError(
                "Socioeconomic Silver contains rates outside "
                f"[0, 1]: {invalid_rates:,}"
            )

        # -------------------------------------------------------------
        # Materialization metadata
        # -------------------------------------------------------------

        stats = (
            silver_lf
            .select(
                pl.len().alias("rows"),

                pl.col("geoid")
                .n_unique()
                .alias("unique_geoids"),

                pl.col("acs_vintage")
                .n_unique()
                .alias("acs_vintages"),

                pl.col("acs_vintage")
                .min()
                .alias("minimum_acs_vintage"),

                pl.col("acs_vintage")
                .max()
                .alias("maximum_acs_vintage"),

                pl.col("state_fips")
                .n_unique()
                .alias("states"),
            )
            .collect()
            .row(0, named=True)
        )

        log.info(
            "silver_write_started",
            target_uri=target_uri,
            rows=stats["rows"],
            unique_geoids=stats["unique_geoids"],
            acs_vintages=stats["acs_vintages"],
        )

        crime_lake.write_crimenet_table(
            silver_lf,
            target_uri=target_uri,
            partitioning_columns=[
                "acs_vintage",
                "state_fips",
            ],
        )

        log.info(
            "silver_ingestion_completed",
            target_uri=target_uri,
            rows=stats["rows"],
        )

        return dg.MaterializeResult(
            metadata={
                "rows": stats["rows"],
                "unique_geoids": stats["unique_geoids"],
                "acs_vintages": stats["acs_vintages"],
                "minimum_acs_vintage":
                    stats["minimum_acs_vintage"],
                "maximum_acs_vintage":
                    stats["maximum_acs_vintage"],
                "states": stats["states"],
                "duplicate_keys": duplicate_keys,
                "invalid_geoids": invalid_geoids,
                "invalid_geography_types":
                    invalid_geography_types,
                "invalid_acs_periods":
                    invalid_acs_periods,
                "invalid_rates": invalid_rates,
                "source_uri": source_uri,
                "target_uri": target_uri,
            }
        )


socioeconomic_silver_assets = [
    silver_tract_socioeconomic,
]