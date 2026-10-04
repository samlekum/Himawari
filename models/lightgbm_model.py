"""LightGBM model for AI Himawari Forecasting."""
from __future__ import annotations
import logging
import numpy as np
from lightgbm import LGBMRegressor

from config import RANDOM_SEED
from models.base import BaseModel

logger = logging.getLogger(__name__)


class LightGBMModel(BaseModel):
    requires_flat_input: bool = True

    def __init__(self, n_estimators: int = 300, max_depth: int = 6,
                 learning_rate: float = 0.05, subsample: float = 0.8,
                 random_state: int = RANDOM_SEED, **kwargs: object) -> None:
        self._model = LGBMRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=subsample, subsample_freq=1,
            n_jobs=-1, random_state=random_state, verbose=-1, **kwargs,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._validate_2d(X_train)
        logger.info("Fitting LightGBMModel pada %s.", X_train.shape)
        self._model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._validate_2d(X)
        return self._model.predict(X)