"""Gaussian Process model implementation (scikit-learn) for AI Himawari Forecasting.

Implements the ``BaseModel`` interface (see ``models.base``) using
scikit-learn's ``GaussianProcessRegressor`` as the underlying
estimator.

``GaussianProcessRegressor`` operates on flat feature vectors, just
like the other classical models in this project. ``requires_flat_input``
is therefore ``True``: callers are expected to build the dataset with
``dataset_builder.build_dataset(..., flatten=True)`` before calling
``fit``/``predict``.

CAUTION -- scalability: exact Gaussian Process regression is O(n^3) in
time and O(n^2) in memory for ``n`` training samples (it factorizes the
full kernel matrix), unlike the other models in this project. Fitting
it directly on a full TRAIN_MONTHS sample count (which can run into the
tens or hundreds of thousands here) is impractical -- it will be very
slow or exhaust memory, not just "slower than the other models".
:meth:`GaussianProcessModel.fit` logs a warning above
``LARGE_SAMPLE_WARNING_THRESHOLD`` but does not subsample on your
behalf (that would silently change what the model was actually fit
on); if you hit that warning, pass a deliberately reduced/subsampled
``X_train``/``y_train`` yourself.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from config import RANDOM_SEED
from models.base import BaseModel

logger = logging.getLogger(__name__)

#: Above this many training samples, GaussianProcessModel.fit() logs a
#: warning about the O(n^3)/O(n^2) cost of exact GP regression (see
#: module docstring) instead of fitting silently.
LARGE_SAMPLE_WARNING_THRESHOLD = 5_000

#: predict() computes a dense (n_query, n_train) kernel matrix -- with a
#: training set already capped (see fit()'s warning) this is still
#: O(n_query * n_train) in memory, so predicting on a large test/query
#: set (e.g. a full held-out split, common in this project) in one call
#: can exhaust memory even though fitting itself was fine. Predicting in
#: chunks of this many rows at a time keeps peak memory bounded by
#: (PREDICT_BATCH_SIZE * n_train) regardless of how large X is, with the
#: exact same output as predicting all at once.
PREDICT_BATCH_SIZE = 5_000


class GaussianProcessModel(BaseModel):
    """Gaussian Process forecasting model backed by scikit-learn's GaussianProcessRegressor.

    Default kernel is ``ConstantKernel * RBF + WhiteKernel`` -- a
    standard smooth-signal-plus-noise choice -- with scikit-learn's
    default hyperparameter bounds, so the kernel's own hyperparameters
    (length scale, output scale, noise level) are optimized during
    :meth:`fit` rather than fixed upfront.

    Args:
        alpha: Value added to the kernel matrix diagonal during
            fitting, forwarded to ``GaussianProcessRegressor`` (extra
            numerical regularization on top of the ``WhiteKernel``
            noise term).
        n_restarts_optimizer: Number of restarts of the kernel
            hyperparameter optimizer, forwarded to
            ``GaussianProcessRegressor``.
        random_state: Random seed for reproducibility. Defaults to
            ``config.RANDOM_SEED``.
        **kwargs: Additional keyword arguments forwarded to
            ``GaussianProcessRegressor`` (e.g. ``kernel`` to override
            the default entirely, ``normalize_y``).
    """

    requires_flat_input: bool = True

    def __init__(
        self,
        alpha: float = 1e-10,
        n_restarts_optimizer: int = 0,
        random_state: int = RANDOM_SEED,
        **kwargs: object,
    ) -> None:
        kernel = kwargs.pop("kernel", None)
        if kernel is None:
            kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1.0)
        self._model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=random_state,
            **kwargs,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the underlying GaussianProcessRegressor.

        Args:
            X_train: Flattened training features, shape
                ``(n_samples, window_size * n_features)``.
            y_train: Training targets, shape ``(n_samples,)``.

        Raises:
            ValueError: If ``X_train`` does not have 2 dimensions.
        """
        self._validate_2d(X_train)
        if X_train.shape[0] > LARGE_SAMPLE_WARNING_THRESHOLD:
            logger.warning(
                "Fitting GaussianProcessModel on %d sample(s) -- exact GP "
                "regression is O(n^3) in time / O(n^2) in memory (see "
                "models/gaussian_process.py's module docstring); this may "
                "be very slow or exhaust memory. Consider fitting on a "
                "deliberately reduced/subsampled X_train/y_train instead.",
                X_train.shape[0],
            )
        logger.info(
            "Fitting GaussianProcessModel on %d sample(s), X%s.",
            X_train.shape[0], X_train.shape,
        )
        self._model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets using the fitted GaussianProcessRegressor.

        Predicts in chunks of ``PREDICT_BATCH_SIZE`` rows rather than
        passing all of ``X`` to scikit-learn in one call, since GP
        prediction's dense ``(n_query, n_train)`` kernel matrix can
        otherwise exhaust memory for a large ``X`` (e.g. a full
        held-out test split) even when the training set itself was
        kept small -- see ``PREDICT_BATCH_SIZE``'s docstring. Output is
        numerically identical to predicting all at once.

        Args:
            X: Flattened features, shape ``(n_samples, window_size *
                n_features)``.

        Returns:
            Predicted targets (posterior mean), shape ``(n_samples,)``.

        Raises:
            ValueError: If ``X`` does not have 2 dimensions.
        """
        self._validate_2d(X)
        if X.shape[0] <= PREDICT_BATCH_SIZE:
            return self._model.predict(X)

        predictions = np.empty(X.shape[0], dtype=float)
        for start in range(0, X.shape[0], PREDICT_BATCH_SIZE):
            end = start + PREDICT_BATCH_SIZE
            predictions[start:end] = self._model.predict(X[start:end])
        return predictions