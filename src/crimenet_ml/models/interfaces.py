"""Model-neutral training contracts."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class IntensityModel(Protocol):
    def predict_raw(self, frame: pd.DataFrame) -> np.ndarray: ...

    def predict_calibrated(self, frame: pd.DataFrame) -> np.ndarray: ...
