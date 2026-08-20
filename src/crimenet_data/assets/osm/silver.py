from datetime import UTC, datetime

import dagster as dg
import polars as pl

from crimenet_data.assets.osm.transformations import (
    OSM_H3_KEY,
    assert_valid_source,
    build_osm_h3_silver,
    collect_silver_validation,
    collect_source_validation,
    normalize_osm_h3_source,
)
from crimenet_data.observability.context import (
    log_context,
)
from crimenet_data.observability.logger import (
    get_logger,
)
from crimenet_data.resources.crime_lake import (
    CrimeLakeResources,
)


log = get_logger(__name__)

GCP = pl.CredentialProviderGCP()

RAW_OSM_H3_ROOT = (
    "gs://crimenet/raw_files/landing/h3_features"
)


@dg.asset(
    name="silver_osm_h3_features",
    group_name="silver_osm",
)
def silver_osm_h3_features(
    context: dg.AssetExecutionContext,
    crime_lake: CrimeLakeResources,
) -> dg.MaterializeResult:
    with log_context(
        run_id=context.run_id,
        asset_key=(
            context.asset_key.to_user_string()
        ),
        source_system="openstreetmap",
    ):
        processed_at = datetime.now(UTC)

        source_uri = RAW_OSM_H3_ROOT

        target_uri = (
            f"{crime_lake.silver_root.rstrip('/')}/"
            "osm_h3_features"
        )

        log.info(
            "processing_started",
            source_uri=source_uri,
            target_uri=target_uri,
        )

        # -------------------------------------------------------------
        # Read authoritative raw H3-9 feature table
        # -------------------------------------------------------------

        raw_lf = pl.scan_parquet(
            f"{source_uri}/**/*.parquet",
            hive_partitioning=True,
            credential_provider=GCP,
        )

        # -------------------------------------------------------------
        # Normalize — still NO filtering
        # -------------------------------------------------------------

        normalized_lf = (
            normalize_osm_h3_source(
                raw_lf
            )
        )

        # -------------------------------------------------------------
        # Source validation
        # -------------------------------------------------------------

        source_stats = (
            collect_source_validation(
                normalized_lf
            )
        )

        log.info(
            "source_validation_completed",
            **source_stats,
        )

        assert_valid_source(
            source_stats
        )

        source_rows = source_stats["rows"]

        # -------------------------------------------------------------
        # Feature derivation
        # -------------------------------------------------------------

        silver_lf = build_osm_h3_silver(
            normalized_lf,
            run_id=context.run_id,
            processed_at=processed_at,
        )

        # -------------------------------------------------------------
        # Silver validation
        # -------------------------------------------------------------

        silver_stats = (
            collect_silver_validation(
                silver_lf
            )
        )

        log.info(
            "silver_validation_completed",
            **silver_stats,
        )

        silver_rows = silver_stats["rows"]
        silver_unique_keys = (
            silver_stats["unique_keys"]
        )

        # -------------------------------------------------------------
        # HARD INVARIANT:
        #
        # Silver transformations are not permitted to discard H3 rows.
        # -------------------------------------------------------------

        if silver_rows != source_rows:
            raise ValueError(
                "OSM Silver changed source row count: "
                f"source={source_rows:,}, "
                f"silver={silver_rows:,}"
            )

        if silver_unique_keys != source_rows:
            raise ValueError(
                "OSM Silver key cardinality is invalid: "
                f"source_rows={source_rows:,}, "
                f"silver_unique_keys="
                f"{silver_unique_keys:,}"
            )

        invalid_derived = (
            silver_stats[
                "invalid_derived_float_rows"
            ]
        )

        if invalid_derived:
            raise ValueError(
                "OSM Silver generated invalid "
                "derived feature values: "
                f"{invalid_derived:,} rows"
            )

        # -------------------------------------------------------------
        # Write
        # -------------------------------------------------------------

        log.info(
            "silver_write_started",
            target_uri=target_uri,
            rows=silver_rows,
        )

        crime_lake.write_crimenet_table(
            silver_lf,
            target_uri=target_uri,
            partitioning_columns=[
                "snapshot_year",
                "source_city",
            ],
        )

        log.info(
            "silver_write_completed",
            target_uri=target_uri,
            rows=silver_rows,
        )

        return dg.MaterializeResult(
            metadata={
                "source_uri": source_uri,
                "target_uri": target_uri,

                "rows": silver_rows,
                "unique_keys":
                    silver_unique_keys,

                "key_columns":
                    OSM_H3_KEY,

                "invalid_key_rows":
                    source_stats[
                        "invalid_key_rows"
                    ],

                "invalid_count_rows":
                    source_stats[
                        "invalid_count_rows"
                    ],

                "invalid_length_rows":
                    source_stats[
                        "invalid_length_rows"
                    ],

                "invalid_network_rows":
                    source_stats[
                        "invalid_network_rows"
                    ],

                "duplicate_keys":
                    source_stats[
                        "duplicate_keys"
                    ],

                "minimum_cell_area_km2":
                    silver_stats[
                        "minimum_cell_area_km2"
                    ],

                "maximum_cell_area_km2":
                    silver_stats[
                        "maximum_cell_area_km2"
                    ],

                "h3_resolution": 9,
            }
        )


osm_h3_silver_assets = [
    silver_osm_h3_features,
]