"""Point-process and secondary ranking metrics."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def point_process_nll(events: np.ndarray, weights: np.ndarray, intensity: np.ndarray) -> float:
    events = np.asarray(events, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    if np.any(~np.isfinite(intensity)) or np.any(intensity < 0):
        raise ValueError("Intensity must be finite and nonnegative")
    safe = np.maximum(intensity, 1e-15)
    return float(np.sum(weights * intensity - events * np.log(safe)))


def point_process_metrics(
    events: np.ndarray,
    weights: np.ndarray,
    intensity: np.ndarray,
) -> dict[str, float | None]:
    events = np.asarray(events, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    observed = float(events.sum())
    predicted = float(np.sum(weights * intensity))
    nll = point_process_nll(events, weights, intensity)
    baseline_intensity = observed / float(weights.sum())
    baseline_prediction = np.full_like(intensity, baseline_intensity)
    baseline_nll = point_process_nll(events, weights, baseline_prediction)
    labels = events > 0
    average_precision: float | None = None
    roc_auc: float | None = None
    if np.unique(labels).size == 2:
        average_precision = float(average_precision_score(labels, intensity))
        roc_auc = float(roc_auc_score(labels, intensity))
    return {
        "point_process_nll": nll,
        "baseline_point_process_nll": baseline_nll,
        "nll_improvement_over_baseline": baseline_nll - nll,
        "nll_per_observed_event": nll / observed if observed > 0 else None,
        "observed_event_mass": observed,
        "predicted_event_mass": predicted,
        "predicted_to_observed_ratio": predicted / observed if observed > 0 else None,
        "mean_predicted_intensity": float(np.mean(intensity)),
        "baseline_intensity": float(baseline_intensity),
        "average_precision": average_precision,
        "roc_auc": roc_auc,
    }


def is_loggable_metric(value: object) -> bool:
    return isinstance(value, (int, float, np.number)) and math.isfinite(float(value))
