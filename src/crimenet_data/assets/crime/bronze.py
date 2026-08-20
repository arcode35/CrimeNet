from datetime import UTC, datetime

import dagster as dg
import deltalake
import polars as pl

from crimenet_data.observability.context import log_context
from crimenet_data.observability.logger import get_logger
from crimenet_data.resources.crime_lake import (
    CITIES,
    CrimeLakeResources,
)


log = get_logger(__name__)


def build_bronze(
    raw_df: pl.LazyFrame,
    *,
    run_id: str,
    source_city: str,
) -> pl.LazyFrame:
    ingested_at = datetime.now(UTC)

    log.info(
        "bronze_transformation_started",
        source_city=source_city,
    )

    result = raw_df.with_columns(
        pl.lit(source_city).alias("source_city"),
        pl.lit(run_id).alias("_ingestion_run_id"),
        pl.lit(ingested_at).alias("_ingested_at_utc"),
    )

    log.info(
        "bronze_transformation_completed",
        source_city=source_city,
    )

    return result


def build_bronze_city_asset(city: str) -> dg.AssetsDefinition:
    @dg.asset(
        name=f"bronze_{city}",
        group_name="bronze_crime",
    )
    def _bronze_asset(
        context: dg.AssetExecutionContext,
        crime_lake: CrimeLakeResources,
    ) -> dg.MaterializeResult:

        with log_context(
            run_id=context.run_id,
            asset_key=context.asset_key.to_user_string(),
            source_city=city,
        ):
            bronze_root = crime_lake.bronze_root

            source_uri = crime_lake.source_uri(city)

            target_uri = (
                f"{bronze_root}/"
                f"crime/"
                f"{city}"
            )

            log.info(
                "bronze_ingestion_started",
                source_uri=source_uri,
                target_uri=target_uri,
            )

            raw_df = pl.scan_parquet(
                source_uri,
                hive_partitioning=True,
                credential_provider=pl.CredentialProviderGCP(),
            )

            bronze_lf = build_bronze(
                raw_df=raw_df,
                run_id=context.run_id,
                source_city=city,
            )

            log.info(
                "bronze_write_started",
                target_uri=target_uri,
            )

            crime_lake.write_crimenet_table(
                lf=bronze_lf,
                target_uri=target_uri,
                partitioning_columns=["occurrence_year"]
            )

            log.info(
                "bronze_ingestion_completed",
                target_uri=target_uri,
            )

            return dg.MaterializeResult(
                metadata={
                    "source_city": city,
                    "ingestion_run_id": context.run_id,
                    "source_uri": source_uri,
                    "target_uri": target_uri,
                }
            )

    return _bronze_asset


crime_bronze_assets = [
    build_bronze_city_asset(city)
    for city in CITIES
]