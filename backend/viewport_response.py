from __future__ import annotations

import numpy as np


def build_viewport_rows(
    cells: list[str],
    positions: np.ndarray,
    intensity_per_second: np.ndarray,
    child_counts: np.ndarray,
) -> list[dict]:
    """Format matched cells without changing the public numeric semantics."""
    events_per_hour = intensity_per_second.astype(np.float64) * 3600.0
    mean_r9_events_per_hour = np.divide(
        events_per_hour,
        child_counts,
        out=np.zeros_like(events_per_hour),
        where=child_counts != 0,
    )
    selected_cells = [cells[int(position)] for position in positions]
    return [
        {
            "h3": cell,
            "events_per_hour": float(hourly),
            "mean_r9_events_per_hour": float(mean_hourly),
            "modeled_r9_cells": int(child_count),
        }
        for cell, hourly, mean_hourly, child_count in zip(
            selected_cells,
            events_per_hour,
            mean_r9_events_per_hour,
            child_counts,
            strict=True,
        )
    ]
