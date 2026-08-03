"""MLflow pyfunc contract for complete calibrated CrimeNet inference."""

from __future__ import annotations

from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd


class CalibratedXGBPoissonPyfunc(mlflow.pyfunc.PythonModel):
    def __init__(
        self,
        model: Any,
        feature_names: tuple[str, ...],
        categorical_columns: tuple[str, ...],
        category_vocabularies: dict[str, tuple[Any, ...]],
        calibration_factor: float,
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.categorical_columns = categorical_columns
        self.category_vocabularies = category_vocabularies
        self.calibration_factor = calibration_factor

    def predict(
        self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame, params=None
    ) -> pd.DataFrame:
        missing = sorted(set(self.feature_names) - set(model_input.columns))
        if missing:
            raise ValueError(f"Missing inference features: {missing}")
        frame = model_input.loc[:, self.feature_names].copy()
        for column in self.categorical_columns:
            frame[column] = pd.Categorical(
                frame[column], categories=self.category_vocabularies[column]
            )
        raw_prediction = np.asarray(self.model.predict(frame), dtype=np.float64)
        prediction = raw_prediction * self.calibration_factor
        if np.any(~np.isfinite(prediction)) or np.any(prediction < 0):
            raise ValueError("Model produced invalid calibrated intensity")
        return pd.DataFrame({"predicted_intensity": prediction}, index=model_input.index)
