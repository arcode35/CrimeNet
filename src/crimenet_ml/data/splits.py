"""Canonical train/validation/test bundle construction."""

from __future__ import annotations

from crimenet_ml.data.interfaces import DatasetBundle, DatasetLoader
from crimenet_ml.data.validation import validate_no_split_overlap, validate_split

TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
SPLIT_NAMES = (TRAIN, VALIDATION, TEST)


def build_dataset_bundle(
    loader: DatasetLoader,
    columns: list[str],
    features: list[str],
    event_column: str,
    weight_column: str,
    identifiers: list[str],
    allow_missing: list[str],
    limit_per_split: int | None = None,
) -> DatasetBundle:
    loaded = {name: loader.load_split(name, columns, limit_per_split) for name in SPLIT_NAMES}
    for split in loaded.values():
        validate_split(split.frame, features, event_column, weight_column, allow_missing)
    validate_no_split_overlap({name: split.frame for name, split in loaded.items()}, identifiers)
    return DatasetBundle(loaded[TRAIN], loaded[VALIDATION], loaded[TEST], loader.identity())
