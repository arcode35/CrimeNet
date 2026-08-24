from __future__ import annotations

import duckdb
import polars as pl

from crimenet_data.assets.crime.legacy.high_roi import HIGH_ROI_ADAPTERS
from crimenet_data.assets.crime.legacy.legacy import LEGACY_ADAPTERS
from crimenet_data.assets.crime.sources import get_source


def adapt_city_source(
    lf: pl.LazyFrame,
    city: str,
    *,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> pl.LazyFrame:
    """Convert one municipal bronze representation to canonical source fields."""

    config = get_source(city).config
    available = set(lf.collect_schema().names())
    if config.deduplication_keys:
        missing = set(config.deduplication_keys) - available
        if missing:
            raise KeyError(
                f"Cannot normalize {city!r}; deduplication keys are missing: "
                f"{sorted(missing)}"
            )
        lf = lf.unique(subset=list(config.deduplication_keys), keep="last")

    if city == "dallas":
        return LEGACY_ADAPTERS[city](lf, connection)
    if city in LEGACY_ADAPTERS:
        return LEGACY_ADAPTERS[city](lf)
    if city in HIGH_ROI_ADAPTERS:
        return HIGH_ROI_ADAPTERS[city](lf)
    raise KeyError(f"No source adapter registered for {city!r}")
