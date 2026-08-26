from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

import duckdb
import polars as pl

SourceFormat = Literal["csv", "parquet", "geojson"]
BronzeHook = Callable[[pl.LazyFrame], pl.LazyFrame]
OccurrenceHook = Callable[[pl.LazyFrame], pl.Expr]
SilverAdapter = Callable[[pl.LazyFrame, "AdapterContext"], pl.LazyFrame]


def identity_bronze(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf


@dataclass(frozen=True)
class SourcePattern:
    glob: str
    format: SourceFormat
    read_options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.glob:
            raise ValueError("A crime source glob cannot be empty")
        if self.format not in {"csv", "parquet", "geojson"}:
            raise ValueError(f"Unsupported crime source format: {self.format!r}")
        object.__setattr__(
            self,
            "read_options",
            MappingProxyType(dict(self.read_options)),
        )


@dataclass(frozen=True)
class CrimeSourceConfig:
    key: str
    source_system: str
    patterns: tuple[SourcePattern, ...]
    timezone: str
    crosswalk_keys: tuple[str, ...]
    coordinates_required: bool = True
    deduplication_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("A crime source key cannot be empty")
        if not self.patterns:
            raise ValueError(f"Crime source {self.key!r} has no input patterns")
        if not self.crosswalk_keys:
            raise ValueError(f"Crime source {self.key!r} has no crosswalk keys")


@dataclass(frozen=True)
class AdapterContext:
    duckdb: duckdb.DuckDBPyConnection | None = None


@dataclass(frozen=True)
class SourceDefinition:
    config: CrimeSourceConfig
    occurrence_timestamp: OccurrenceHook
    adapt_to_silver: SilverAdapter
    prepare_bronze: BronzeHook = identity_bronze
