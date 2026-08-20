import dagster as dg


from crimenet_data.assets.crime.bronze import crime_bronze_assets
from crimenet_data.assets.crime.silver import crime_silver_assets
from crimenet_data.assets.weather.bronze import weather_bronze_assets
from crimenet_data.assets.weather.silver import weather_silver_assets
from crimenet_data.assets.socioeconomic.bronze import (
    socioeconomic_bronze_assets,
)
from crimenet_data.assets.socioeconomic.silver import (
    socioeconomic_silver_assets,
)
from crimenet_data.assets.references.external import (
    canonical_crime_crosswalk,
)
from crimenet_data.assets.osm.silver import (
    osm_h3_silver_assets,
)
from crimenet_data.assets.tract_resources.silver import (
    tract_resource_silver_assets,
)
from crimenet_data.assets.event_spine import (
    event_spine,
)
from crimenet_data.assets.lighting import (
    lighting_required_keys,
    solar_lighting_conditions,
)

from crimenet_data.resources.crime_lake import (
    CrimeLakeResources,
)
from crimenet_data.resources.duckdb import (
    DuckDBResource,
)
from crimenet_data.observability.config import (
    configure_logging,
)
from crimenet_data.assets.integration import (
    integration_context,
    integration_samples,
)
from crimenet_data.assets.model_table import final_model_table
# =============================================================================
# Jobs
# =============================================================================


crime_bronze_job = dg.define_asset_job(
    name="crime_bronze_job",
    selection=dg.AssetSelection.groups(
        "bronze_crime"
    ),
)


crime_silver_job = dg.define_asset_job(
    name="crime_silver_job",
    selection=dg.AssetSelection.groups(
        "silver_crime"
    ),
)


weather_bronze_job = dg.define_asset_job(
    name="weather_bronze_job",
    selection=dg.AssetSelection.groups(
        "bronze_weather"
    ),
)


weather_silver_job = dg.define_asset_job(
    name="weather_silver_job",
    selection=dg.AssetSelection.groups(
        "silver_weather"
    ),
)


socioeconomic_bronze_job = dg.define_asset_job(
    name="socioeconomic_bronze_job",
    selection=dg.AssetSelection.groups(
        "bronze_socioeconomic"
    ),
)


socioeconomic_silver_job = dg.define_asset_job(
    name="socioeconomic_silver_job",
    selection=dg.AssetSelection.groups(
        "silver_socioeconomic"
    ),
)

lighting_job = dg.define_asset_job(
    name="lighting_job",
    selection=dg.AssetSelection.assets(
        lighting_required_keys,
        solar_lighting_conditions,
    ),
)

osm_h3_silver_job = dg.define_asset_job(
    name="osm_h3_silver_job",
    selection=dg.AssetSelection.groups(
        "silver_osm"
    ),
)


tract_resources_job = dg.define_asset_job(
    name="tract_resources_job",
    selection=dg.AssetSelection.groups(
        "tract_resources"
    ),
)


event_spine_job = dg.define_asset_job(
    name="event_spine_job",
    selection=dg.AssetSelection.assets(
        event_spine
    ),
)

integration_job = dg.define_asset_job(
    name="integration_job",
    selection=dg.AssetSelection.assets(
        integration_samples
    ),
)
integration_context_job = dg.define_asset_job(
    name="integration_context_job",
    selection=dg.AssetSelection.assets(
        integration_context
    ),
)

final_model_job = (
    dg.define_asset_job(
        name=
            "final_model_job",

        selection=
            dg.AssetSelection.assets(
                final_model_table
            ),
    )
)


# =============================================================================
# Logging
# =============================================================================


configure_logging()


# =============================================================================
# Definitions
# =============================================================================


defs = dg.Definitions(
    assets=[
        canonical_crime_crosswalk,

        *crime_bronze_assets,
        *crime_silver_assets,

        *weather_bronze_assets,
        *weather_silver_assets,

        *socioeconomic_bronze_assets,
        *socioeconomic_silver_assets,

        *osm_h3_silver_assets,
        *tract_resource_silver_assets,

        event_spine,
        integration_samples,
        integration_context,
        lighting_required_keys,
        solar_lighting_conditions,
        final_model_table,
    ],
    jobs=[
        crime_bronze_job,
        crime_silver_job,

        weather_bronze_job,
        weather_silver_job,

        socioeconomic_bronze_job,
        socioeconomic_silver_job,

        tract_resources_job,
        osm_h3_silver_job,
        lighting_job,

        event_spine_job,
        integration_job,
        integration_context_job,
        final_model_job
    ],
    resources={
        "crime_lake": CrimeLakeResources(),
        "duckdb_resource": DuckDBResource(
            database=":memory:",
            enable_spatial=True,
        ),
    },
)