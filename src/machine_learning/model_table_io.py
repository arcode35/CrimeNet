"""Compatibility wrapper for split-pruned canonical Parquet model tables."""

from __future__ import annotations

import polars as pl

from crimenet_data.resources.crime_lake import CrimeLakeResources


def scan_model_table(root: str, *, split: str = "train") -> pl.LazyFrame:
    """Scan one non-test partition; obsolete Delta roots are unsupported."""

    if split not in {"train", "validation"}:
        raise ValueError("ML model-table access is limited to train/validation")
    lake = CrimeLakeResources()
    glob = f"{root.rstrip('/')}/split={split}/source_city=*/*.parquet"
    return pl.scan_parquet(
        glob,
        hive_partitioning=True,
        storage_options=lake.storage_options_for(root),
    )


__all__ = ["scan_model_table"]
