from __future__ import annotations

import dagster as dg

from crimenet_data.assets.crime.bronze import bronze_assets
from crimenet_data.resources import CrimeLakeResources


defs = dg.Definitions(
    assets=[
        *bronze_assets,
    ],
    resources={
        "crime_lake": CrimeLakeResources(),
    },
)