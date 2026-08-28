import dagster as dg

from crimenet_data.assets.crime.bronze import crime_bronze_assets
from crimenet_data.assets.crime.silver import crime_silver_assets
from crimenet_data.assets.environmental import (
    environmental_asset_checks,
    environmental_assets,
    published_integration_sampling,
    raw_model_weather_v2,
)
from crimenet_data.assets.event_spine import event_spine_gold_assets
from crimenet_data.assets.final_model_table import (
    final_model_table_asset_checks,
    final_model_table_assets,
)
from crimenet_data.assets.integration import integration_sampling_job
from crimenet_data.assets.references.external import canonical_crime_crosswalk
from crimenet_data.observability.config import configure_logging
from crimenet_data.resources.crime_lake import CrimeLakeResources
from crimenet_data.resources.duckdb import DuckDBResource


crime_bronze_job = dg.define_asset_job(
    name="crime_bronze_job",
    selection=dg.AssetSelection.groups("bronze_crime"),
    config={
        "execution": {
            "config": {
                "multiprocess": {
                    "max_concurrent": 3,
                }
            }
        }
    },
)

crime_silver_job = dg.define_asset_job(
    name="crime_silver_job",
    selection=dg.AssetSelection.groups("silver_crime"),
)

environmental_features_job = dg.define_asset_job(
    name="environmental_features_job",
    selection=dg.AssetSelection.groups(
        "silver_environmental",
        "gold_environmental",
    ),
)

final_model_table_job = dg.define_asset_job(
    name="final_model_table_job",
    selection=dg.AssetSelection.groups("gold_model"),
)


configure_logging()

defs = dg.Definitions(
    assets=[
        canonical_crime_crosswalk,
        raw_model_weather_v2,
        published_integration_sampling,
        *crime_bronze_assets,
        *crime_silver_assets,
        *event_spine_gold_assets,
        *environmental_assets,
        *final_model_table_assets,
    ],
    asset_checks=[
        *environmental_asset_checks,
        *final_model_table_asset_checks,
    ],
    jobs=[
        crime_bronze_job,
        crime_silver_job,
        integration_sampling_job,
        environmental_features_job,
        final_model_table_job,
    ],
    resources={
        "crime_lake": CrimeLakeResources(),
        "duckdb_resource": DuckDBResource(
            database=":memory:",
            enable_spatial=True,
        ),
    },
)