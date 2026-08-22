import dagster as dg

from .silver import (
    imagery_temporal_index_integrity_check,
    silver_imagery_h3_candidates,
    silver_imagery_h3_temporal_index,
)


IMAGERY_ASSETS = [
    silver_imagery_h3_candidates,
    silver_imagery_h3_temporal_index,
]

IMAGERY_ASSET_CHECKS = [
    imagery_temporal_index_integrity_check,
]

# If this package is its own Dagster code location, `dagster dev -m <package>`
# can discover these definitions directly. If your repository already has a
# top-level Definitions object, import IMAGERY_ASSETS / IMAGERY_ASSET_CHECKS
# there instead of using this `defs` object.
defs = dg.Definitions(
    assets=IMAGERY_ASSETS,
    asset_checks=IMAGERY_ASSET_CHECKS,
)

__all__ = [
    "IMAGERY_ASSETS",
    "IMAGERY_ASSET_CHECKS",
    "defs",
    "silver_imagery_h3_candidates",
    "silver_imagery_h3_temporal_index",
    "imagery_temporal_index_integrity_check",
]