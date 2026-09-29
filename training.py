"""Model lifecycle management for AI Himawari Forecasting.

This module is responsible ONLY for the lifecycle of a model instance:
obtaining it from the model registry, training it, saving it to disk,
and loading it back. It never loads datasets, never preprocesses,
never builds sliding windows, never splits datasets, never searches
for hyperparameters, never forecasts, never evaluates, and never
visualizes -- all of that is the responsibility of other modules
(``loader.py``, ``preprocessing.py``, ``windowing.py``,
``dataset_builder.py``, ``experiment.py``, ``forecasting.py``,
``evaluation.py``, ``visualization.py``).

Models are always obtained through ``models.get_model`` (the registry
factory). This module never imports or instantiates a concrete model
class (e.g. ``models.ann.ANNModel``) directly, so it stays
model-agnostic in the same way ``experiment.py`` does.

Model artifacts are persisted with ``joblib``. Save/load paths are
always derived from ``config.MODEL_OUTPUT_PATH`` and never hardcoded.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np

from config import DEFAULT_MODEL, MODEL_OUTPUT_PATH, SAVE_MODEL
from models import BaseModel, get_model

logger = logging.getLogger(__name__)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str = DEFAULT_MODEL,
) -> BaseModel:
    """Train a model obtained from the model registry.

    Args:
        X_train: Training features. Shape depends on the selected
            model's ``requires_flat_input``: ``(n_samples, window_size
            * n_features)`` if ``True``, ``(n_samples, window_size,
            n_features)`` if ``False``. As produced by
            ``dataset_builder.build_dataset``.
        y_train: Training targets, shape ``(n_samples,)``.
        model_name: Name of the model to train, must be registered in
            ``models.registry``. Defaults to ``config.DEFAULT_MODEL``.

    Returns:
        The fitted model instance (implements ``models.BaseModel``).

    Raises:
        ValueError: If ``X_train`` and ``y_train`` do not have the
            same number of samples, or if ``model_name`` is not
            registered (propagated from ``models.get_model``).
    """
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            "X_train and y_train must have the same number of samples; "
            f"got X_train.shape[0]={X_train.shape[0]} vs "
            f"y_train.shape[0]={y_train.shape[0]}."
        )

    model = get_model(model_name)
    logger.info(
        "Training model '%s' on %d sample(s), X_train%s.",
        model_name, X_train.shape[0], X_train.shape,
    )
    model.fit(X_train, y_train)
    logger.info("Finished training model '%s'.", model_name)
    return model


def save_model(
    model: BaseModel,
    model_name: str = DEFAULT_MODEL,
    output_path: Path = MODEL_OUTPUT_PATH,
) -> Path:
    """Persist a trained model to disk using joblib.

    If ``config.SAVE_MODEL`` is ``False``, the actual write is skipped
    (only logged), and the path that would have been used is still
    returned, so callers can rely on a consistent return type
    regardless of the toggle.

    Args:
        model: The model instance to persist (typically already
            fitted via :func:`train_model`).
        model_name: Name used to derive the output filename
            (``f"{model_name}.joblib"``). Defaults to
            ``config.DEFAULT_MODEL``.
        output_path: Directory where the model file is written.
            Defaults to ``config.MODEL_OUTPUT_PATH``.

    Returns:
        The full path of the model file (whether or not it was
        actually written -- see ``config.SAVE_MODEL`` above).

    Raises:
        ValueError: If ``model`` is not an instance of
            ``models.BaseModel``.
    """
    if not isinstance(model, BaseModel):
        raise ValueError(
            f"Expected an instance of models.BaseModel, got {type(model)!r}."
        )

    file_path = output_path / f"{model_name}.joblib"

    if not SAVE_MODEL:
        logger.info(
            "SAVE_MODEL is disabled in config; skipping write to '%s'.",
            file_path,
        )
        return file_path

    output_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, file_path)
    logger.info("Saved model '%s' to '%s'.", model_name, file_path)
    return file_path


def load_model(
    model_name: str = DEFAULT_MODEL,
    output_path: Path = MODEL_OUTPUT_PATH,
) -> BaseModel:
    """Load a previously saved model from disk.

    Args:
        model_name: Name used to derive the input filename
            (``f"{model_name}.joblib"``). Defaults to
            ``config.DEFAULT_MODEL``.
        output_path: Directory the model file is read from. Defaults
            to ``config.MODEL_OUTPUT_PATH``.

    Returns:
        The loaded model instance.

    Raises:
        FileNotFoundError: If no model file exists at the resolved
            path.
        ValueError: If the loaded object is not an instance of
            ``models.BaseModel``.
    """
    file_path = output_path / f"{model_name}.joblib"

    if not file_path.exists():
        raise FileNotFoundError(f"No saved model found at '{file_path}'.")

    model = joblib.load(file_path)

    if not isinstance(model, BaseModel):
        raise ValueError(
            f"Loaded object from '{file_path}' is not a models.BaseModel "
            f"instance (got {type(model)!r})."
        )

    logger.info("Loaded model '%s' from '%s'.", model_name, file_path)
    return model