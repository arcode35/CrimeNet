"""Weighted Poisson XGBoost using the Berman--Turner construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


@dataclass(frozen=True)
class ModelMatrix:
    features: pd.DataFrame
    poisson_target: np.ndarray
    sample_weight: np.ndarray
    event_count: np.ndarray


@dataclass
class XGBPoissonTrainingResult:
    model: XGBRegressor
    feature_names: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    category_vocabularies: dict[str, tuple[Any, ...]]
    baseline_intensity: float
    calibration_factor: float
    evals_result: dict[str, dict[str, list[float]]]


def berman_turner_target(event_count: np.ndarray, integration_weight: np.ndarray) -> np.ndarray:
    events = np.asarray(event_count, dtype=np.float64)
    weights = np.asarray(integration_weight, dtype=np.float64)
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Integration weights must be finite and strictly positive")
    if np.any(~np.isfinite(events)) or np.any(events < 0):
        raise ValueError("Event counts must be finite and nonnegative")
    return events / weights


def validation_calibration_factor(
    validation_events: np.ndarray,
    validation_weights: np.ndarray,
    validation_raw_intensity: np.ndarray,
) -> float:
    observed = float(np.sum(validation_events))
    predicted = float(np.sum(validation_weights * validation_raw_intensity))
    if not np.isfinite(predicted) or predicted <= 0:
        raise ValueError("Validation raw predicted event mass must be finite and positive")
    factor = observed / predicted
    if not np.isfinite(factor) or factor < 0:
        raise ValueError("Calibration factor must be finite and nonnegative")
    return factor


class XGBPoissonModel:
    def __init__(
        self,
        feature_names: list[str],
        categorical_columns: list[str],
        model_parameters: dict[str, Any],
        random_seed: int,
    ) -> None:
        self.feature_names = tuple(feature_names)
        self.categorical_columns = tuple(categorical_columns)
        self.model_parameters = dict(model_parameters)
        self.random_seed = random_seed
        self.category_vocabularies: dict[str, tuple[Any, ...]] = {}
        self.model: XGBRegressor | None = None
        self.baseline_intensity: float | None = None
        self.calibration_factor: float | None = None

    def fit_category_vocabularies(self, training_frame: pd.DataFrame) -> None:
        self.category_vocabularies = {
            column: tuple(sorted(training_frame[column].dropna().unique().tolist(), key=str))
            for column in self.categorical_columns
        }

    def _model_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(set(self.feature_names) - set(frame.columns))
        if missing:
            raise ValueError(f"Missing inference features: {missing}")
        result = frame.loc[:, self.feature_names].copy()
        for column in self.categorical_columns:
            result[column] = pd.Categorical(
                result[column], categories=self.category_vocabularies[column]
            )
        return result

    def make_matrix(
        self, frame: pd.DataFrame, event_column: str, weight_column: str
    ) -> ModelMatrix:
        events = frame[event_column].to_numpy(dtype=np.float64)
        weights = frame[weight_column].to_numpy(dtype=np.float64)
        return ModelMatrix(
            features=self._model_frame(frame),
            poisson_target=berman_turner_target(events, weights),
            sample_weight=weights.copy(),
            event_count=events,
        )

    def fit(self, train: ModelMatrix, validation: ModelMatrix) -> XGBPoissonTrainingResult:
        total_weight = float(train.sample_weight.sum())
        total_events = float(train.event_count.sum())
        if total_weight <= 0 or total_events <= 0:
            raise ValueError("Training totals must be positive")
        self.baseline_intensity = total_events / total_weight
        parameters = dict(self.model_parameters)
        parameters.update(
            random_state=self.random_seed,
            base_score=self.baseline_intensity,
            enable_categorical=True,
        )
        self.model = XGBRegressor(**parameters)
        self.model.fit(
            train.features,
            train.poisson_target,
            sample_weight=train.sample_weight,
            eval_set=[
                (train.features, train.poisson_target),
                (validation.features, validation.poisson_target),
            ],
            sample_weight_eval_set=[train.sample_weight, validation.sample_weight],
            verbose=False,
        )
        validation_raw = np.asarray(self.model.predict(validation.features), dtype=np.float64)
        self.calibration_factor = validation_calibration_factor(
            validation.event_count, validation.sample_weight, validation_raw
        )
        return XGBPoissonTrainingResult(
            model=self.model,
            feature_names=self.feature_names,
            categorical_columns=self.categorical_columns,
            category_vocabularies=self.category_vocabularies,
            baseline_intensity=self.baseline_intensity,
            calibration_factor=self.calibration_factor,
            evals_result=self.model.evals_result(),
        )

    def predict_raw(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fit")
        prediction = np.asarray(self.model.predict(self._model_frame(frame)), dtype=np.float64)
        if np.any(~np.isfinite(prediction)) or np.any(prediction < 0):
            raise ValueError("Model produced invalid intensity")
        return prediction

    def predict_calibrated(self, frame: pd.DataFrame) -> np.ndarray:
        if self.calibration_factor is None:
            raise RuntimeError("Calibration has not been fit")
        prediction = self.predict_raw(frame) * self.calibration_factor
        if np.any(~np.isfinite(prediction)) or np.any(prediction < 0):
            raise ValueError("Calibrated model produced invalid intensity")
        return prediction
