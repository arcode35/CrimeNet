import dagster as dg 

from .resources import CrimeLakeResources

from .bronze import bronze_assets

defs = dg.Definitions(
    assets=[*bronze_assets],
    resources={
        "crimenet_data": CrimeLakeResources()
    }
)