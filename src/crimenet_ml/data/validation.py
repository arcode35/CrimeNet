"""Reusable data validation for CrimeNet splits."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_required_columns(frame: pd.DataFrame, required: list[str]) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_event_counts(values: pd.Series) -> None:
    numeric = values.to_numpy(dtype=float)
    if not np.all(np.isfinite(numeric)) or np.any(numeric < 0):
        raise ValueError("Event counts must be finite and nonnegative")
    if numeric.sum() <= 0:
        raise ValueError("Split must have positive total event mass")


def validate_weights(values: pd.Series) -> None:
    numeric = values.to_numpy(dtype=float)
    if not np.all(np.isfinite(numeric)) or np.any(numeric <= 0):
        raise ValueError("Integration weights must be finite and strictly positive")
    if numeric.sum() <= 0:
        raise ValueError("Split must have positive total integration weight")


def validate_features(frame: pd.DataFrame, features: list[str], allow_missing: list[str]) -> None:
    forbidden = [
        name for name in features if name not in allow_missing and frame[name].isna().any()
    ]
    if forbidden:
        raise ValueError(f"Null values in features that disallow missing values: {forbidden}")


def validate_split(
    frame: pd.DataFrame,
    features: list[str],
    event_column: str,
    weight_column: str,
    allow_missing: list[str] | None = None,
) -> None:
    if frame.empty:
        raise ValueError("Dataset split is empty")
    validate_required_columns(frame, features + [event_column, weight_column])
    validate_event_counts(frame[event_column])
    validate_weights(frame[weight_column])
    validate_features(frame, features, allow_missing or [])


def validate_no_split_overlap(splits: dict[str, pd.DataFrame], identifiers: list[str]) -> None:
    usable = [column for column in identifiers if all(column in frame for frame in splits.values())]
    if not usable:
        return
    seen: set[tuple] = set()
    for name, frame in splits.items():
        keys = set(frame[usable].itertuples(index=False, name=None))
        overlap = seen.intersection(keys)
        if overlap:
            raise ValueError(f"Identifier overlap detected in {name} split")
        seen.update(keys)
