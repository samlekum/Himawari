"""Gradient Boosting model implementation (scikit-learn) for AI Himawari Forecasting.

Implements the ``BaseModel`` interface (see ``models.base``) using
scikit-learn's ``GradientBoostingRegressor`` as the underlying
estimator.

``GradientBoostingRegressor`` operates on flat feature vectors, just
like the other classical models in this project. ``requires_flat_input``
is therefore ``True``: callers are expected to build the dataset with
``dataset_builder.build_dataset(..., flatten=True)`` before calling
``fit``/``predict``.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from config import RANDOM_SEED
from models.base import BaseModel

logger = logging.getLogger(__name__)


class GradientBoostingModel(BaseModel):
    """Gradient Boosting forecasting model backed by scikit-learn's GradientBoostingRegressor.

    Args:
        n_estimators: Number of boosting stages, forwarded to
            ``GradientBoostingRegressor``.
        learning_rate: Shrinkage applied to each tree's contribution,
            forwarded to ``GradientBoostingRegressor``.
        max_depth: Maximum depth of each regression tree, forwarded to
            ``GradientBoostingRegressor``.
        random_state: Random seed for reproducibility. Defaults to
            ``config.RANDOM_SEED``.
        **kwargs: Additional keyword arguments forwarded to
            ``GradientBoostingRegressor`` (e.g. ``subsample``,
            ``min_samples_leaf``).
    """

    requires_flat_input: bool = True

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        random_state: int = RANDOM_SEED,
        **kwargs: object,
    ) -> None:
        self._model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the underlying GradientBoostingRegressor.

        Args:
            X_train: Flattened training features, shape
                ``(n_samples, window_size * n_features)``.
            y_train: Training targets, shape ``(n_samples,)``.

        Raises:
            ValueError: If ``X_train`` does not have 2 dimensions.
        """
        self._validate_2d(X_train)
        logger.info(
            "Fitting GradientBoostingModel on %d sample(s), X%s.",
            X_train.shape[0], X_train.shape,
        )
        self._model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets using the fitted GradientBoostingRegressor.

        Args:
            X: Flattened features, shape ``(n_samples, window_size *
                n_features)``.

        Returns:
            Predicted targets, shape ``(n_samples,)``.

        Raises:
            ValueError: If ``X`` does not have 2 dimensions.
        """
        self._validate_2d(X)
        return self._model.predict(X)
