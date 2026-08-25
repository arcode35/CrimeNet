import dagster as dg

from crimenet_data.assets.crime.bronze import crime_bronze_assets
from crimenet_data.assets.crime.silver import crime_silver_assets
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

configure_logging()

defs = dg.Definitions(
    assets=[
        canonical_crime_crosswalk,
        *crime_bronze_assets,
        *crime_silver_assets,
    ],
    jobs=[crime_bronze_job, crime_silver_job],
    resources={
        "crime_lake": CrimeLakeResources(),
        "duckdb_resource": DuckDBResource(
            database=":memory:",
            enable_spatial=True,
        ),
    },
)
