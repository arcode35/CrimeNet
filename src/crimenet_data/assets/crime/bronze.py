from datetime import UTC, datetime

import dagster as dg

from crimenet_data.assets.crime.ingestion import prepare_bronze_source
from crimenet_data.assets.crime.sources import SOURCE_KEYS, get_source
from crimenet_data.observability.context import log_context
from crimenet_data.observability.logger import get_logger
from crimenet_data.resources.crime_lake import CrimeLakeResources

log = get_logger(__name__)


def build_bronze_source_asset(source_key: str) -> dg.AssetsDefinition:
    source = get_source(source_key)

    @dg.asset(
        name=f"bronze_{source_key}",
        group_name="bronze_crime",
        pool=f"crime_bronze_{source_key}_writer",
    )
    def _bronze_asset(
        context: dg.AssetExecutionContext,
        crime_lake: CrimeLakeResources,
    ) -> dg.MaterializeResult:
        source_uris = crime_lake.source_uris(source_key)
        target_uri = crime_lake.resolve_source_path(source_key, "bronze")
        with log_context(
            run_id=context.run_id,
            asset_key=context.asset_key.to_user_string(),
            source_city=source_key,
        ):
            log.info(
                "bronze_ingestion_started",
                source_uris=source_uris,
                target_uri=target_uri,
            )
            bronze_lf = prepare_bronze_source(
                crime_lake.scan_source(source_key),
                source,
                run_id=context.run_id,
                ingested_at=datetime.now(UTC),
            )
            if "occurrence_year" not in bronze_lf.collect_schema():
                raise KeyError(
                    f"Source {source_key!r} did not derive the Bronze partition year"
                )
            crime_lake.write_crimenet_table(
                lf=bronze_lf,
                target_uri=target_uri,
                partitioning_columns=["occurrence_year"],
            )
            log.info("bronze_ingestion_completed", target_uri=target_uri)
            return dg.MaterializeResult(
                metadata={
                    "source_key": source_key,
                    "source_format": sorted(
                        {pattern.format for pattern in source.config.patterns}
                    ),
                    "source_uris": list(source_uris),
                    "target_uri": target_uri,
                    "ingestion_run_id": context.run_id,
                    "source_layer": "landing",
                }
            )

    return _bronze_asset


crime_bronze_assets = [
    build_bronze_source_asset(source_key) for source_key in SOURCE_KEYS
]
