import dagster as dg
import polars as pl

from crimenet_data.assets.crime.canonical import (
    apply_canonical_crosswalk,
    cleanse_canonical_source,
    project_canonical_schema,
)
from crimenet_data.assets.crime.sources import SOURCE_KEYS, AdapterContext, get_source
from crimenet_data.observability.context import log_context
from crimenet_data.observability.logger import get_logger
from crimenet_data.resources.crime_lake import CrimeLakeResources
from crimenet_data.resources.duckdb import DuckDBResource

log = get_logger(__name__)
CROSSWALK_ASSET_KEY = dg.AssetKey(["reference", "canonical_crime_crosswalk"])


def build_silver(
    bronze_lf: pl.LazyFrame,
    crosswalk_lf: pl.LazyFrame,
    *,
    source_key: str,
    adapter_context: AdapterContext,
) -> pl.LazyFrame:
    source = get_source(source_key)
    adapted = source.adapt_to_silver(bronze_lf, adapter_context)
    mapped = apply_canonical_crosswalk(adapted, crosswalk_lf, source_key)
    cleansed = cleanse_canonical_source(mapped, source_key)
    return project_canonical_schema(cleansed, source_key)


def build_silver_source_asset(source_key: str) -> dg.AssetsDefinition:
    source = get_source(source_key)

    @dg.asset(
        name=f"silver_{source_key}",
        group_name="silver_crime",
        deps=[CROSSWALK_ASSET_KEY, f"bronze_{source_key}"],
        pool=f"crime_silver_{source_key}_writer",
    )
    def _silver_asset(
        context: dg.AssetExecutionContext,
        crime_lake: CrimeLakeResources,
        duckdb_resource: DuckDBResource,
    ) -> dg.MaterializeResult:
        source_uri = crime_lake.resolve_current_bronze_snapshot(source_key)
        target_uri = crime_lake.resolve_source_path(source_key, "silver")
        with log_context(
            run_id=context.run_id,
            asset_key=context.asset_key.to_user_string(),
            source_city=source_key,
        ):
            log.info(
                "silver_normalization_started",
                source_uri=source_uri,
                target_uri=target_uri,
            )
            with duckdb_resource.get_connection() as connection:
                silver_lf = build_silver(
                    crime_lake.scan_bronze_snapshot(
                        source_key,
                        snapshot_uri=source_uri,
                    ),
                    crime_lake.resolve_crosswalk(),
                    source_key=source_key,
                    adapter_context=AdapterContext(duckdb=connection),
                )
                crime_lake.write_delta_table(
                    lf=silver_lf,
                    target_uri=target_uri,
                    partitioning_columns=["occurrence_year"],
                )
            log.info("silver_normalization_completed", target_uri=target_uri)
            return dg.MaterializeResult(
                metadata={
                    "source_key": source_key,
                    "source_system": source.config.source_system,
                    "source_uri": source_uri,
                    "target_uri": target_uri,
                    "event_grain": "source offense/event row",
                }
            )

    return _silver_asset


@dg.asset(
    name="silver_crime_offenses",
    group_name="silver_crime",
    deps=[f"silver_{source_key}" for source_key in SOURCE_KEYS],
    pool="crime_silver_offenses_writer",
)
def silver_crime_offenses(
    context: dg.AssetExecutionContext,
    crime_lake: CrimeLakeResources,
) -> dg.MaterializeResult:
    with log_context(
        run_id=context.run_id,
        asset_key=context.asset_key.to_user_string(),
    ):
        silver_lf = pl.concat(
            [
                crime_lake.scan_source_delta(source_key, "silver")
                for source_key in SOURCE_KEYS
            ],
            how="vertical",
        )
        target_uri = f"{crime_lake.silver_root}/crime_offenses"
        crime_lake.write_delta_table(
            lf=silver_lf,
            target_uri=target_uri,
            partitioning_columns=["source_city", "occurrence_year"],
        )
        return dg.MaterializeResult(
            metadata={
                "target_uri": target_uri,
                "sources_processed": len(SOURCE_KEYS),
                "event_grain": "source offense/event row",
            }
        )


crime_silver_assets = [
    *[build_silver_source_asset(source_key) for source_key in SOURCE_KEYS],
    silver_crime_offenses,
]
