"""CatBoost model for AI Himawari Forecasting."""
from __future__ import annotations
import logging
import numpy as np
from catboost import CatBoostRegressor

from config import RANDOM_SEED
from models.base import BaseModel

logger = logging.getLogger(__name__)


class CatBoostModel(BaseModel):
    requires_flat_input: bool = True

    def __init__(self, iterations: int = 300, depth: int = 6,
                 learning_rate: float = 0.05,
                 random_state: int = RANDOM_SEED, **kwargs: object) -> None:
        self._model = CatBoostRegressor(
            iterations=iterations, depth=depth, learning_rate=learning_rate,
            random_seed=random_state, thread_count=-1, verbose=0,
            allow_writing_files=False, **kwargs,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._validate_2d(X_train)
        logger.info("Fitting CatBoostModel pada %s.", X_train.shape)
        self._model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._validate_2d(X)
        return np.asarray(self._model.predict(X)).ravel()