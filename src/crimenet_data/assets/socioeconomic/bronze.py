from datetime import UTC, datetime

import dagster as dg
import deltalake
import polars as pl

from crimenet_data.observability.context import log_context
from crimenet_data.observability.logger import get_logger
from crimenet_data.resources.crime_lake import CrimeLakeResources


log = get_logger(__name__)


ACS_TRACT_SCHEMA = pl.Schema(
    {
        "geoid": pl.String,
        "geography_name": pl.String,
        "county_fips": pl.String,
        "tract_code": pl.String,
        "period_start_year": pl.Int64,
        "period_end_year": pl.Int64,
        "geography_type": pl.String,
        "population": pl.Int64,
        "population_moe": pl.Int64,
        "median_age": pl.Float64,
        "median_age_moe": pl.Float64,
        "median_household_income": pl.Float64,
        "median_household_income_moe": pl.Float64,
        "poverty_universe": pl.Int64,
        "poverty_universe_moe": pl.Int64,
        "population_below_poverty": pl.Int64,
        "population_below_poverty_moe": pl.Int64,
        "civilian_labor_force": pl.Int64,
        "civilian_labor_force_moe": pl.Int64,
        "unemployed_population": pl.Int64,
        "unemployed_population_moe": pl.Int64,
        "housing_units": pl.Int64,
        "housing_units_moe": pl.Int64,
        "occupied_housing_units": pl.Int64,
        "occupied_housing_units_moe": pl.Int64,
        "vacant_housing_units": pl.Int64,
        "vacant_housing_units_moe": pl.Int64,
        "occupied_units_tenure_universe": pl.Int64,
        "occupied_units_tenure_universe_moe": pl.Int64,
        "renter_occupied_units": pl.Int64,
        "renter_occupied_units_moe": pl.Int64,
        "households_vehicle_universe": pl.Int64,
        "households_vehicle_universe_moe": pl.Int64,
        "households_no_vehicle": pl.Int64,
        "households_no_vehicle_moe": pl.Int64,
        "poverty_rate": pl.Float64,
        "unemployment_rate": pl.Float64,
        "vacancy_rate": pl.Float64,
        "renter_occupied_rate": pl.Float64,
        "no_vehicle_rate": pl.Float64,
        "source_file": pl.String,
        "bronze_ingested_at": pl.Datetime("ns"),
        "metro": pl.String,
        "acs_vintage": pl.Int64,
        "state_fips": pl.String,
    }
)


def build_socioeconomic_bronze(
    raw_df: pl.LazyFrame,
    *,
    run_id: str,
) -> pl.LazyFrame:
    ingested_at = datetime.now(UTC)

    return raw_df.with_columns(
        pl.lit("census_acs5").alias("_source_system"),
        pl.lit(run_id).alias("_ingestion_run_id"),
        pl.lit(ingested_at).alias("_ingested_at_utc"),
    )


@dg.asset(
    name="bronze_socioeconomic",
    group_name="bronze_socioeconomic",
)
def bronze_socioeconomic(
    context: dg.AssetExecutionContext,
    crime_lake: CrimeLakeResources,
) -> dg.MaterializeResult:
    with log_context(
        run_id=context.run_id,
        asset_key=context.asset_key.to_user_string(),
        source_system="census_acs5",
    ):
        source_uri = crime_lake.resolve_socioeconomic_path()

        target_uri = (
            f"{crime_lake.bronze_root.rstrip('/')}/"
            "acs5/tract"
        )

        log.info(
            "processing_started",
            source_uri=source_uri,
            target_uri=target_uri,
        )

        raw_lf = pl.scan_parquet(
            source_uri,
            schema=ACS_TRACT_SCHEMA,
            hive_partitioning=True,
            include_file_paths="_source_file_uri",
            credential_provider=pl.CredentialProviderGCP(),
        )

        bronze_lf = build_socioeconomic_bronze(
            raw_df=raw_lf,
            run_id=context.run_id,
        )

        schema = bronze_lf.collect_schema()

        required_partition_columns = {
            "acs_vintage",
            "state_fips",
        }

        missing_partition_columns = (
            required_partition_columns - set(schema.names())
        )

        if missing_partition_columns:
            log.error(
                "partition_validation_failed",
                missing_partition_columns=sorted(
                    missing_partition_columns
                ),
                available_columns=schema.names(),
            )

            raise ValueError(
                "Required socioeconomic partition columns are missing: "
                f"{sorted(missing_partition_columns)}. "
                f"Available columns: {schema.names()}"
            )

        log.info(
            "partition_validation_completed",
            partition_columns=sorted(
                required_partition_columns
            ),
        )
        log.info(
            "bronze_write_started",
            target_uri=target_uri,
        )
        crime_lake.write_crimenet_table(bronze_lf, target_uri=target_uri, partitioning_columns=["acs_vintage", "state_fips"])

        log.info(
            "bronze_ingestion_completed",
            target_uri=target_uri,
        )

        return dg.MaterializeResult(
            metadata={
                "source_system": "census_acs5",
                "ingestion_run_id": context.run_id,
                "source_uri": source_uri,
                "target_uri": target_uri,
                "partition_columns": [
                    "acs_vintage",
                    "state_fips",
                ],
            }
        )


socioeconomic_bronze_assets = [
    bronze_socioeconomic,
]