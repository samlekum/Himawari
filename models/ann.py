"""ANN model implementation (scikit-learn MLPRegressor) for AI Himawari Forecasting.

Implements the ``BaseModel`` interface (see ``models.base``) using
scikit-learn's ``MLPRegressor`` as the underlying estimator.

``MLPRegressor`` is a plain feedforward network -- not a recurrent /
sequential model -- so it operates on flat feature vectors just like
the classical models in this project. ``requires_flat_input`` is
therefore ``True``: callers are expected to build the dataset with
``dataset_builder.build_dataset(..., flatten=True)`` before calling
``fit``/``predict``. If a truly sequential ANN (e.g. an RNN/LSTM) ever
replaces ``MLPRegressor``, this flag should flip to ``False`` and the
3D handling would move inside this class.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.neural_network import MLPRegressor

from config import RANDOM_SEED
from models.base import BaseModel

logger = logging.getLogger(__name__)


class ANNModel(BaseModel):
    """Feedforward ANN forecasting model backed by scikit-learn's MLPRegressor.

    Args:
        hidden_layer_sizes: Sizes of the hidden layers, forwarded to
            ``MLPRegressor``.
        random_state: Random seed for reproducibility. Defaults to
            ``config.RANDOM_SEED``.
        **kwargs: Additional keyword arguments forwarded to
            ``MLPRegressor`` (e.g. ``max_iter``, ``learning_rate_init``).
    """

    requires_flat_input: bool = True

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (64, 32),
        random_state: int = RANDOM_SEED,
        **kwargs: object,
    ) -> None:
        self._model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            random_state=random_state,
            **kwargs,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the underlying MLPRegressor.

        Args:
            X_train: Flattened training features, shape
                ``(n_samples, window_size * n_features)``.
            y_train: Training targets, shape ``(n_samples,)``.

        Raises:
            ValueError: If ``X_train`` does not have 2 dimensions.
        """
        self._validate_2d(X_train)
        logger.info(
            "Fitting ANNModel on %d sample(s), X%s.",
            X_train.shape[0], X_train.shape,
        )
        self._model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets using the fitted MLPRegressor.

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
