from __future__ import annotations

from types import MappingProxyType

from crimenet_data.assets.crime.sources.atlanta import SOURCE as ATLANTA
from crimenet_data.assets.crime.sources.baltimore import SOURCE as BALTIMORE
from crimenet_data.assets.crime.sources.base import SourceDefinition
from crimenet_data.assets.crime.sources.baton_rouge import SOURCE as BATON_ROUGE
from crimenet_data.assets.crime.sources.boston import SOURCE as BOSTON
from crimenet_data.assets.crime.sources.chandler_az import SOURCE as CHANDLER_AZ
from crimenet_data.assets.crime.sources.chicago import SOURCE as CHICAGO
from crimenet_data.assets.crime.sources.dallas import SOURCE as DALLAS
from crimenet_data.assets.crime.sources.denver import SOURCE as DENVER
from crimenet_data.assets.crime.sources.east_baton_rouge_parish_sheriff_la import (
    SOURCE as EAST_BATON_ROUGE_PARISH_SHERIFF_LA,
)
from crimenet_data.assets.crime.sources.fort_worth import SOURCE as FORT_WORTH
from crimenet_data.assets.crime.sources.gainesville_fl import SOURCE as GAINESVILLE_FL
from crimenet_data.assets.crime.sources.los_angeles_county_sheriff import (
    SOURCE as LOS_ANGELES_COUNTY_SHERIFF,
)
from crimenet_data.assets.crime.sources.marin_county_sheriff_ca import (
    SOURCE as MARIN_COUNTY_SHERIFF_CA,
)
from crimenet_data.assets.crime.sources.montgomery_county_md import (
    SOURCE as MONTGOMERY_COUNTY_MD,
)
from crimenet_data.assets.crime.sources.new_york import SOURCE as NEW_YORK
from crimenet_data.assets.crime.sources.san_francisco import SOURCE as SAN_FRANCISCO
from crimenet_data.assets.crime.sources.seattle import SOURCE as SEATTLE
from crimenet_data.assets.crime.sources.sonoma_county_sheriff_ca import (
    SOURCE as SONOMA_COUNTY_SHERIFF_CA,
)
from crimenet_data.assets.crime.sources.washington_dc import SOURCE as WASHINGTON_DC

_SOURCE_DEFINITIONS = (
    ATLANTA,
    BALTIMORE,
    BATON_ROUGE,
    BOSTON,
    CHANDLER_AZ,
    CHICAGO,
    DALLAS,
    DENVER,
    EAST_BATON_ROUGE_PARISH_SHERIFF_LA,
    FORT_WORTH,
    GAINESVILLE_FL,
    LOS_ANGELES_COUNTY_SHERIFF,
    MARIN_COUNTY_SHERIFF_CA,
    MONTGOMERY_COUNTY_MD,
    NEW_YORK,
    SAN_FRANCISCO,
    SEATTLE,
    SONOMA_COUNTY_SHERIFF_CA,
    WASHINGTON_DC,
)

_keys = tuple(source.config.key for source in _SOURCE_DEFINITIONS)
if len(_keys) != len(set(_keys)):
    raise ValueError("Duplicate crime source keys are registered")

SOURCES = MappingProxyType(
    {source.config.key: source for source in _SOURCE_DEFINITIONS}
)
SOURCE_KEYS = tuple(SOURCES)

SILVER_SOURCE_KEYS = (
    "atlanta",
    "baltimore",
    "chandler_az",
    "chicago",
    "dallas",
    "denver",
    "fort_worth",
    "los_angeles_county_sheriff",
    "marin_county_sheriff_ca",
    "montgomery_county_md",
    "new_york",
    "san_francisco",
    "seattle",
    "sonoma_county_sheriff_ca",
    "washington_dc",
)

_unknown_silver_sources = set(SILVER_SOURCE_KEYS) - set(SOURCES)
if _unknown_silver_sources:
    raise ValueError(
        "Silver-enabled crime sources are not registered: "
        f"{sorted(_unknown_silver_sources)}"
    )


def get_source(source_key: str) -> SourceDefinition:
    try:
        return SOURCES[source_key]
    except KeyError as error:
        raise KeyError(
            f"Unknown crime source {source_key!r}. Valid sources: {sorted(SOURCES)}"
        ) from error
