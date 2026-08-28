"""Shared lazy reader for legacy Delta and canonical Parquet model tables."""

from __future__ import annotations

import polars as pl


def scan_model_table(root: str) -> pl.LazyFrame:
    """Scan a canonical snapshot or retain compatibility with legacy Delta roots."""

    if "final_model_table" in root or "snapshot_id=" in root:
        glob = f"{root.rstrip('/')}/split=*/source_city=*/*.parquet"
        options: dict[str, object] = {
            "hive_partitioning": True,
            "hive_schema": {
                "snapshot_id": pl.String,
                "split": pl.String,
                "source_city": pl.String,
            },
        }
        if root.startswith("gs://"):
            options["credential_provider"] = pl.CredentialProviderGCP()
        return pl.scan_parquet(glob, **options)
    if root.startswith("gs://"):
        return pl.scan_delta(
            root,
            credential_provider=pl.CredentialProviderGCP(),
        )
    return pl.scan_delta(root)


__all__ = ["scan_model_table"]
