"""Geographic point-process validation aggregations."""

from __future__ import annotations

import math

import numpy as np
import polars as pl


def geographic_point_process_metrics(
    frame: pl.DataFrame,
    *,
    log_intensity: np.ndarray,
    constant_log_intensity: float,
    min_log_intensity: float = -30.0,
    max_log_intensity: float = 15.0,
) -> dict[str, object]:
    """Return pooled, per-city, and unweighted macro-city metrics."""

    if len(log_intensity) != frame.height:
        raise ValueError("prediction length mismatch")
    margin = np.clip(
        np.asarray(log_intensity, dtype=np.float64),
        min_log_intensity,
        max_log_intensity,
    )
    exposure = (
        frame["integration_weight_cell_seconds"]
        .fill_null(0.0)
        .to_numpy()
        .astype(np.float64, copy=False)
    )
    expected_contribution = exposure * np.exp(margin)
    working = frame.select("source_city", "event_count").with_columns(
        pl.Series("_margin", margin),
        pl.Series("_exposure", exposure),
        pl.Series("_expected", expected_contribution),
    )
    rows: list[dict[str, object]] = []
    for city, city_frame in working.partition_by("source_city", as_dict=True).items():
        city_name = city[0] if isinstance(city, tuple) else city
        observed = float(city_frame["event_count"].sum())
        expected = float(city_frame["_expected"].sum())
        exposure = float(city_frame["_exposure"].sum())
        nll = float(
            (city_frame["_expected"] - city_frame["event_count"] * city_frame["_margin"]).sum()
        )
        nll_per_event = nll / observed if observed else math.nan
        constant_nll = float(
            (
                city_frame["_exposure"] * math.exp(constant_log_intensity)
                - city_frame["event_count"] * constant_log_intensity
            ).sum()
        )
        constant_nll_per_event = constant_nll / observed if observed else math.nan
        gain = constant_nll_per_event - nll_per_event
        rows.append({
            "source_city": city_name, "rows": city_frame.height,
            "observed_events": observed, "integration_rows": int((city_frame["event_count"] == 0).sum()),
            "total_exposure": exposure, "expected_events": expected,
            "exposure": exposure,
            "expected_observed_ratio": expected / observed if observed else math.nan,
            "calibration_error_pct": 100.0 * (expected - observed) / observed if observed else math.nan,
            "nll": nll, "nll_per_event": nll_per_event,
            "constant_nll_per_event": constant_nll_per_event,
            "nll_gain_per_event": gain,
            "bits_per_event": gain / math.log(2.0),
        })
    finite = [row for row in rows if math.isfinite(float(row["nll_per_event"]))]
    if not finite:
        raise ValueError("Geographic validation has no cities with observed events")
    observed = float(working["event_count"].sum())
    expected = float(working["_expected"].sum())
    pooled_nll = float(
        (working["_expected"] - working["event_count"] * working["_margin"]).sum()
    )
    pooled_constant_nll = float(
        (
            working["_exposure"] * math.exp(constant_log_intensity)
            - working["event_count"] * constant_log_intensity
        ).sum()
    )
    pooled_nll_per_event = pooled_nll / observed
    pooled_constant_per_event = pooled_constant_nll / observed
    bits = [float(row["bits_per_event"]) for row in finite]
    return {
        "per_city": rows,
        "global": {
            "rows": working.height,
            "observed_events": observed,
            "integration_rows": int((working["event_count"] == 0).sum()),
            "total_exposure": float(working["_exposure"].sum()),
            "expected_events": expected,
            "expected_observed_ratio": expected / observed,
            "calibration_error_pct": 100.0 * (expected - observed) / observed,
            "nll": pooled_nll,
            "nll_per_event": pooled_nll_per_event,
            "constant_nll_per_event": pooled_constant_per_event,
            "nll_gain_per_event": pooled_constant_per_event - pooled_nll_per_event,
            "bits_per_event": (pooled_constant_per_event - pooled_nll_per_event) / math.log(2.0),
        },
        "macro_city": {
            "mean_nll_per_event": float(np.mean([row["nll_per_event"] for row in finite])),
            "median_nll_per_event": float(np.median([row["nll_per_event"] for row in finite])),
            "mean_bits_per_event": float(np.mean(bits)),
            "median_bits_per_event": float(np.median(bits)),
            "p10_bits_per_event": float(np.percentile(bits, 10)),
            "worst_city_bits_per_event": float(np.min(bits)),
            "mean_absolute_calibration_error_pct": float(np.mean([abs(row["calibration_error_pct"]) for row in finite])),
        },
    }


def geographic_mark_metrics(
    *,
    source_cities: list[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
    training_priors: np.ndarray,
) -> dict[str, object]:
    """Compute pooled and unweighted city mark-classification metrics."""

    y = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if len(source_cities) != len(y) or probabilities.shape[0] != len(y):
        raise ValueError("mark prediction length mismatch")
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    eps = 1e-15
    losses = -np.log(np.clip(probabilities[np.arange(len(y)), y], eps, 1.0))
    baseline_losses = -np.log(np.clip(training_priors[y], eps, 1.0))
    predicted = probabilities.argmax(axis=1)
    rows: list[dict[str, object]] = []
    for city in sorted(set(source_cities)):
        mask = np.asarray([value == city for value in source_cities])
        loss = float(losses[mask].mean())
        baseline = float(baseline_losses[mask].mean())
        rows.append(
            {
                "source_city": city,
                "rows": int(mask.sum()),
                "log_loss": loss,
                "baseline_log_loss": baseline,
                "log_loss_gain": baseline - loss,
                "bits_gain": (baseline - loss) / math.log(2.0),
                "accuracy": float((predicted[mask] == y[mask]).mean()),
            }
        )
    return {
        "global": {
            "rows": len(y),
            "log_loss": float(losses.mean()),
            "baseline_log_loss": float(baseline_losses.mean()),
            "bits_gain": float((baseline_losses.mean() - losses.mean()) / math.log(2.0)),
            "accuracy": float((predicted == y).mean()),
        },
        "macro_city": {
            "mean_log_loss": float(np.mean([row["log_loss"] for row in rows])),
            "median_log_loss": float(np.median([row["log_loss"] for row in rows])),
            "mean_bits_gain": float(np.mean([row["bits_gain"] for row in rows])),
            "mean_accuracy": float(np.mean([row["accuracy"] for row in rows])),
        },
        "per_city": rows,
    }


__all__ = ["geographic_mark_metrics", "geographic_point_process_metrics"]
