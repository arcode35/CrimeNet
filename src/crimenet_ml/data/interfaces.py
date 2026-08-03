"""Backend-neutral dataset contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd


@dataclass
class DatasetIdentity:
    backend: str
    source_location: str
    table_or_path: str
    fingerprint_or_delta_version: str | int | None
    row_count: int = 0
    split_counts: dict[str, int] = field(default_factory=dict)
    selected_feature_set: str = ""
    file_count: int | None = None
    selected_columns: list[str] = field(default_factory=list)
    selected_cities: list[str] = field(default_factory=list)
    catalog: str | None = None
    schema_name: str | None = None
    table_name: str | None = None


@dataclass
class LoadedSplit:
    name: str
    frame: pd.DataFrame


@dataclass
class DatasetBundle:
    train: LoadedSplit
    validation: LoadedSplit
    test: LoadedSplit
    identity: DatasetIdentity


class DatasetLoader(Protocol):
    def load_split(
        self, split: str, columns: list[str], limit: int | None = None
    ) -> LoadedSplit: ...

    def identity(self) -> DatasetIdentity: ...
