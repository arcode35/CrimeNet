"""Split-safe whole-city holdout selection."""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl


def geographic_frames(
    *,
    train: pl.LazyFrame,
    validation: pl.LazyFrame,
    holdout_cities: Iterable[str],
    report_in_domain: bool = True,
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame | None]:
    holdouts = tuple(sorted(set(holdout_cities)))
    if not holdouts:
        return train, validation, None
    training = train.filter(~pl.col("source_city").is_in(holdouts))
    geographic = validation.filter(pl.col("source_city").is_in(holdouts))
    in_domain = (
        validation.filter(~pl.col("source_city").is_in(holdouts))
        if report_in_domain
        else None
    )
    return training, geographic, in_domain


def deterministic_sample(
    frame: pl.LazyFrame, *, fraction: float, seed: int
) -> pl.LazyFrame:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    if fraction == 1.0:
        return frame
    return frame.filter(
        (pl.col("model_row_id").hash(seed=seed) % 1_000_000)
        < int(fraction * 1_000_000)
    )


def validate_holdout_membership(
    *,
    training: pl.DataFrame,
    validation: pl.DataFrame,
    holdout_cities: Iterable[str],
    expected_modeling_cities: Iterable[str] | None = None,
) -> None:
    holdouts = set(map(str, holdout_cities))
    if not holdouts:
        return
    train_cities = set(training["source_city"].cast(pl.String).to_list())
    validation_cities = set(validation["source_city"].cast(pl.String).to_list())
    leaked = sorted(train_cities & holdouts)
    if leaked:
        raise ValueError(f"Held-out cities leaked into training: {leaked}")
    if validation_cities != holdouts:
        raise ValueError(
            "Geographic validation city mismatch: "
            f"actual={sorted(validation_cities)}, expected={sorted(holdouts)}"
        )
    if expected_modeling_cities is not None:
        expected_training = set(map(str, expected_modeling_cities)) - holdouts
        if train_cities != expected_training:
            raise ValueError(
                "Geographic training city mismatch: "
                f"actual={sorted(train_cities)}, expected={sorted(expected_training)}"
            )


__all__ = [
    "deterministic_sample",
    "geographic_frames",
    "validate_holdout_membership",
]
