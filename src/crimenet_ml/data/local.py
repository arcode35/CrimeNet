"""Local Hive-partitioned Parquet adapter using Polars lazy scans."""

from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from crimenet_ml.data.interfaces import DatasetIdentity, LoadedSplit


class LocalParquetLoader:
    def __init__(self, root: Path, split_column: str, feature_set: str, seed: int = 42) -> None:
        self.root = root.resolve()
        self.split_column = split_column
        self.feature_set = feature_set
        self.seed = seed
        self._counts: dict[str, int] = {}
        self._cities: set[str] = set()

    def _scan(self) -> pl.LazyFrame:
        if not self.root.exists():
            raise FileNotFoundError(f"Parquet root does not exist: {self.root}")
        return pl.scan_parquet(
            str(self.root / "**" / "*.parquet"), hive_partitioning=True, glob=True
        )

    def load_split(self, split: str, columns: list[str], limit: int | None = None) -> LoadedSplit:
        selected = list(dict.fromkeys(columns + [self.split_column]))
        lazy = self._scan().filter(pl.col(self.split_column) == split).select(selected)
        if limit is not None and "source_city" in selected:
            if limit <= 0:
                raise ValueError("limit_per_split must be greater than zero")
            cities = (
                lazy.select("source_city")
                .unique()
                .collect()
                .get_column("source_city")
                .drop_nulls()
                .sort()
                .to_list()
            )
            if not cities:
                raise ValueError(f"No cities found in split {split!r}")
            rows_per_city, remainder = divmod(limit, len(cities))
            pieces = [
                lazy.filter(pl.col("source_city") == city)
                .head(rows_per_city + (1 if index < remainder else 0))
                .collect()
                for index, city in enumerate(cities)
                if rows_per_city + (1 if index < remainder else 0) > 0
            ]
            collected = pl.concat(pieces) if pieces else lazy.limit(0).collect()
        elif limit is not None:
            collected = lazy.limit(limit).collect()
        else:
            collected = lazy.collect()
        frame = collected.to_pandas()
        self._counts[split] = len(frame)
        if "source_city" in frame:
            self._cities.update(map(str, frame["source_city"].dropna().unique()))
        return LoadedSplit(split, frame)

    def identity(self) -> DatasetIdentity:
        digest = hashlib.sha256()
        files = sorted(self.root.rglob("*.parquet"))
        for file in files:
            stat = file.stat()
            metadata = pq.ParquetFile(file).metadata
            manifest_entry = (
                f"{file.relative_to(self.root)}\0{stat.st_size}\0{metadata.num_rows}\0"
                f"{metadata.num_row_groups}\0{metadata.serialized_size}\n"
            )
            digest.update(manifest_entry.encode())
        return DatasetIdentity(
            backend="local",
            source_location=str(self.root),
            table_or_path=str(self.root),
            fingerprint_or_delta_version=digest.hexdigest(),
            row_count=sum(self._counts.values()),
            split_counts=dict(self._counts),
            selected_feature_set=self.feature_set,
            file_count=len(files),
            selected_cities=sorted(self._cities),
        )
