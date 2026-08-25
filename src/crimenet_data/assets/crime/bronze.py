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
        retry_policy=dg.RetryPolicy(
            max_retries=5,
            delay=5,
            backoff=dg.Backoff.EXPONENTIAL,
            jitter=dg.Jitter.PLUS_MINUS,
        ),
    )
    def _bronze_asset(
        context: dg.AssetExecutionContext,
        crime_lake: CrimeLakeResources,
    ) -> dg.MaterializeResult:
        source_uris = crime_lake.source_uris(source_key)
        snapshot_id = context.run_id
        snapshot_uri = crime_lake.bronze_snapshot_uri(source_key, snapshot_id)
        ingested_at = datetime.now(UTC)
        with log_context(
            run_id=context.run_id,
            asset_key=context.asset_key.to_user_string(),
            source_city=source_key,
            snapshot_id=snapshot_id,
        ):
            log.info(
                "bronze_ingestion_started",
                source_uris=source_uris,
                snapshot_uri=snapshot_uri,
            )
            bronze_lf = prepare_bronze_source(
                crime_lake.scan_source(source_key),
                source,
                run_id=context.run_id,
                ingested_at=ingested_at,
            )
            if "occurrence_year" not in bronze_lf.collect_schema():
                raise KeyError(
                    f"Source {source_key!r} did not derive the Bronze partition year"
                )
            log.info(
                "bronze_snapshot_write_started",
                snapshot_uri=snapshot_uri,
                partition_column="occurrence_year",
            )
            crime_lake.write_bronze_snapshot(
                lf=bronze_lf,
                source_key=source_key,
                snapshot_id=snapshot_id,
                partitioning_columns=["occurrence_year"],
            )
            log.info(
                "bronze_snapshot_write_completed",
                snapshot_uri=snapshot_uri,
            )
            completed_at = datetime.now(UTC)
            crime_lake.complete_bronze_snapshot(
                source_key=source_key,
                snapshot_id=snapshot_id,
                created_at=completed_at,
            )
            log.info(
                "bronze_snapshot_pointer_updated",
                snapshot_uri=snapshot_uri,
            )
            log.info(
                "bronze_ingestion_completed",
                snapshot_uri=snapshot_uri,
            )
            return dg.MaterializeResult(
                metadata={
                    "source_key": source_key,
                    "source_format": sorted(
                        {pattern.format for pattern in source.config.patterns}
                    ),
                    "source_uris": list(source_uris),
                    "snapshot_id": snapshot_id,
                    "snapshot_uri": snapshot_uri,
                    "partition_column": "occurrence_year",
                    "ingestion_run_id": context.run_id,
                    "source_layer": "landing",
                    "storage_format": "parquet",
                    "completed_at": completed_at.isoformat(),
                }
            )

    return _bronze_asset


crime_bronze_assets = [
    build_bronze_source_asset(source_key) for source_key in SOURCE_KEYS
]
