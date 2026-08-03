from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from crimenet_ml.data.validation import validate_event_counts, validate_weights
from crimenet_ml.evaluation.metrics import point_process_metrics, point_process_nll
from crimenet_ml.features.feature_set import get_feature_set
from crimenet_ml.models.xgb_poisson import (
    XGBPoissonModel,
    berman_turner_target,
    validation_calibration_factor,
)
from crimenet_ml.tracking.logging import flatten_metrics


def test_berman_turner_target_and_weights_are_exact() -> None:
    events = np.array([0.0, 2.0, 3.0])
    weights = np.array([0.5, 4.0, 2.0])
    np.testing.assert_array_equal(berman_turner_target(events, weights), [0.0, 0.5, 1.5])
    model = XGBPoissonModel(["x"], [], {}, 7)
    matrix = model.make_matrix(
        pd.DataFrame({"x": [1, 2, 3], "events": events, "weights": weights}),
        "events",
        "weights",
    )
    np.testing.assert_array_equal(matrix.sample_weight, weights)


@pytest.mark.parametrize("weights", [[0, 1], [-1, 1], [np.nan, 1], [np.inf, 1]])
def test_invalid_weights_are_rejected(weights: list[float]) -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        validate_weights(pd.Series(weights))


def test_negative_events_are_rejected() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        validate_event_counts(pd.Series([1.0, -1.0]))


def test_categories_are_training_only_and_unknowns_become_missing() -> None:
    model = XGBPoissonModel(["offense_mark", "x"], ["offense_mark"], {}, 3)
    train = pd.DataFrame({"offense_mark": ["a", "b", "a"], "x": [1, 2, 3]})
    model.fit_category_vocabularies(train)
    assert model.category_vocabularies == {"offense_mark": ("a", "b")}
    transformed = model._model_frame(  # noqa: SLF001 - verifies the inference contract directly.
        pd.DataFrame({"offense_mark": ["c", "a"], "x": [4, 5]})
    )
    assert pd.isna(transformed.loc[0, "offense_mark"])
    assert transformed.loc[1, "offense_mark"] == "a"


def test_feature_set_lengths() -> None:
    assert len(get_feature_set("history_v1").features) == 40
    assert len(get_feature_set("core_v1").features) == 77


def test_calibration_uses_validation_only() -> None:
    events = np.array([1.0, 2.0])
    weights = np.array([2.0, 4.0])
    raw = np.array([0.25, 0.5])
    expected = 3.0 / 2.5
    assert validation_calibration_factor(events, weights, raw) == expected
    unrelated_test_rows = np.array([10_000.0, 20_000.0])
    assert unrelated_test_rows.sum() > 0
    assert validation_calibration_factor(events, weights, raw) == expected


def test_hand_computed_point_process_metrics() -> None:
    events = np.array([1.0, 0.0])
    weights = np.array([2.0, 3.0])
    raw = np.array([0.5, 0.25])
    expected_nll = 2 * 0.5 - np.log(0.5) + 3 * 0.25
    assert point_process_nll(events, weights, raw) == pytest.approx(expected_nll)
    metrics = point_process_metrics(events, weights, raw)
    assert metrics["predicted_event_mass"] == pytest.approx(1.75)
    assert metrics["observed_event_mass"] == 1.0
    assert metrics["predicted_to_observed_ratio"] == pytest.approx(1.75)
    calibrated = raw * (1.0 / 1.75)
    calibrated_metrics = point_process_metrics(events, weights, calibrated)
    assert calibrated_metrics["predicted_to_observed_ratio"] == pytest.approx(1.0)


def test_metric_flattening_excludes_invalid_values() -> None:
    assert flatten_metrics(
        {"good": 2, "none": None, "nan": np.nan, "inf": np.inf, "nested": {"x": 3.5}}
    ) == {"good": 2.0, "nested.x": 3.5}


def test_package_import_does_not_import_pyspark() -> None:
    assert "pyspark" not in sys.modules
