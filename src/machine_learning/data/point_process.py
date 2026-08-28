"""Point-process target/exposure normalization and row-contract checks."""

from __future__ import annotations

import numpy as np
import polars as pl


def prepare_target_exposure(
    frame: pl.DataFrame, *, event_exposure_tolerance: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    required = {
        "event_count", "integration_weight_cell_seconds", "event_indicator",
        "is_observed_event", "row_type",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Point-process controls missing: {missing}")
    target = frame["event_count"].to_numpy().astype(np.float64, copy=False)
    exposure = (
        frame["integration_weight_cell_seconds"]
        .fill_null(0.0)
        .to_numpy()
        .astype(np.float64, copy=False)
    )
    if not np.isfinite(target).all() or not np.isin(target, [0.0, 1.0]).all():
        raise ValueError("event_count must be finite and binary")
    if not np.isfinite(exposure).all() or (exposure < 0).any():
        raise ValueError("Exposure must be finite and non-negative")
    observed = target == 1.0
    integration = ~observed
    if not observed.any():
        raise ValueError("Point-process frame contains no observed events")
    if not integration.any():
        raise ValueError("Point-process frame contains no integration rows")
    if (np.abs(exposure[observed]) > event_exposure_tolerance).any():
        raise ValueError("Observed events must have zero mathematical exposure")
    if (exposure[integration] <= 0).any():
        raise ValueError("Integration rows must have positive exposure")
    indicator = frame["event_indicator"].to_numpy()
    observed_flag = frame["is_observed_event"].to_numpy()
    row_type = frame["row_type"].cast(pl.String).to_numpy()
    if not np.array_equal(indicator, observed.astype(indicator.dtype)):
        raise ValueError("event_indicator disagrees with event_count")
    if not np.array_equal(observed_flag, observed):
        raise ValueError("is_observed_event disagrees with event_count")
    if not np.array_equal(row_type == "event", observed):
        raise ValueError("row_type disagrees with event_count")
    return target, exposure


__all__ = ["prepare_target_exposure"]
