"""Common model interface for AI Himawari Forecasting.

Every forecasting model used by this project (ANN, Gradient Boosting,
Gaussian Process, ...) must implement ``BaseModel`` so that
orchestration modules (``experiment.py``, ``training.py``,
``forecasting.py``) can remain model-agnostic: they depend only on
``fit``/``predict``/``requires_flat_input``, never on a concrete model
class.

This module defines the contract only. It contains no concrete model
logic and no training/forecasting/evaluation logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    """Abstract base class defining the common forecasting model interface.

    Attributes:
        requires_flat_input: Whether this model expects 2D input of
            shape ``(n_samples, window_size * n_features)`` (``True``,
            e.g. classical models such as Gradient Boosting, Gaussian
            Process, or a plain feedforward ANN) or 3D input of shape
            ``(n_samples, window_size, n_features)`` (``False``, e.g. a
            truly sequential model such as an RNN/LSTM). Callers (e.g.
            ``experiment.py``) read this attribute to decide whether to
            build the dataset with
            ``dataset_builder.build_dataset(..., flatten=...)``.
    """

    requires_flat_input: bool = True

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the model on training data.

        Args:
            X_train: Training features. Shape follows
                ``requires_flat_input``: ``(n_samples, window_size *
                n_features)`` if ``True``, ``(n_samples, window_size,
                n_features)`` if ``False``.
            y_train: Training targets, shape ``(n_samples,)``.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets for the given features.

        Args:
            X: Features following the same shape convention as
                ``X_train`` in :meth:`fit`.

        Returns:
            Predicted targets, shape ``(n_samples,)``.
        """
        raise NotImplementedError

    @staticmethod
    def _validate_2d(X: np.ndarray) -> None:
        """Raise ValueError if ``X`` is not a 2D array.

        Shared by every ``requires_flat_input=True`` concrete model
        (ANN, Gradient Boosting, Gaussian Process, ...) so the same
        check/message isn't duplicated in each one.

        Args:
            X: Array to validate.

        Raises:
            ValueError: If ``X.ndim != 2``.
        """
        if X.ndim != 2:
            raise ValueError(
                f"{X.__class__.__name__} input must be 2D (n_samples, "
                f"window_size * n_features); got shape {X.shape}. Build "
                "the dataset with dataset_builder.build_dataset(..., "
                "flatten=True)."
            )
