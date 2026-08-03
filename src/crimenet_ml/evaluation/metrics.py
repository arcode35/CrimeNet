from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def _safe_metric(function: Any, *args: Any, **kwargs: Any) -> float | None:
    try:
        return float(function(*args, **kwargs))
    except ValueError:
        return None


def evaluate_binary_predictions(
    y_true: np.ndarray,
    probability: np.ndarray,
    sample_weight: np.ndarray | None,
) -> dict[str, float | int | None]:
    probability = np.asarray(probability, dtype=np.float64)
    probability = np.clip(probability, 1e-7, 1.0 - 1e-7)

    metrics: dict[str, float | int | None] = {
        "rows": int(y_true.size),
        "positives": int(y_true.sum()),
        "positive_rate": float(y_true.mean()),
        "average_precision": _safe_metric(
            average_precision_score,
            y_true,
            probability,
        ),
        "roc_auc": _safe_metric(
            roc_auc_score,
            y_true,
            probability,
        ),
        "log_loss": _safe_metric(
            log_loss,
            y_true,
            probability,
            labels=[0, 1],
        ),
        "brier_score": _safe_metric(
            brier_score_loss,
            y_true,
            probability,
        ),
    }

    if sample_weight is not None:
        metrics.update(
            {
                "weighted_average_precision": _safe_metric(
                    average_precision_score,
                    y_true,
                    probability,
                    sample_weight=sample_weight,
                ),
                "weighted_roc_auc": _safe_metric(
                    roc_auc_score,
                    y_true,
                    probability,
                    sample_weight=sample_weight,
                ),
                "weighted_log_loss": _safe_metric(
                    log_loss,
                    y_true,
                    probability,
                    labels=[0, 1],
                    sample_weight=sample_weight,
                ),
                "weighted_brier_score": _safe_metric(
                    brier_score_loss,
                    y_true,
                    probability,
                    sample_weight=sample_weight,
                ),
            }
        )

    return metrics
