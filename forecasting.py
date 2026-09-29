"""Inference / forecasting for AI Himawari Forecasting.

This module is responsible ONLY for running inference with an already
trained model: single-step prediction, and multi-step forecasting
using a pluggable window-update method. It never loads NetCDF files,
never preprocesses, never builds sliding windows, never splits
datasets, never trains a model or searches for hyperparameters, never
computes evaluation metrics, never turns results into classes, never
exports, and never visualizes -- all of that is the responsibility of
other modules (``loader.py``, ``preprocessing.py``, ``windowing.py``,
``dataset_builder.py``, ``experiment.py``, ``training.py``,
``classification.py``, ``evaluation.py``, an export module, and
``visualization.py``).

Models are used strictly through the ``models.BaseModel`` interface
(``fit``/``predict``/``requires_flat_input``) -- this module never
imports or checks for a concrete model class (e.g. ``models.ann``),
so it stays model-agnostic. A model is either passed in directly by
the caller, or obtained beforehand via ``training.load_model()``.

Forecast update methods are implemented as a simple Strategy pattern:
each method is a plain function registered in ``_STRATEGY_REGISTRY``
by name, and the high-level :func:`forecast` dispatches to it via dict
lookup. The registered names are kept in lockstep with
``config.SUPPORTED_UPDATE_METHODS`` -- checked once at import time --
so the two can never silently drift apart.

Per the project's current priority, ``"bias_correction"`` is the main,
fully-worked-out method (see :func:`bias_correction`); ``"forecast_only"``
is a simple one-step baseline; ``"teacher_forcing"`` is a secondary
method, implemented but not the current focus.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

from config import FORECAST_HORIZON, FORECAST_UPDATE_METHOD, SUPPORTED_UPDATE_METHODS
from models import BaseModel
from windowing import flatten_windows

logger = logging.getLogger(__name__)

__all__ = [
    "predict",
    "bias_correction",
    "forecast_only",
    "teacher_forcing",
    "forecast",
]


def predict(model: BaseModel, X: np.ndarray) -> np.ndarray:
    """Run a single (one-step) inference pass through a trained model.

    Automatically flattens ``X`` via ``windowing.flatten_windows`` when
    the model requires flat input (``model.requires_flat_input``), so
    callers can always pass the canonical 3D windowed array regardless
    of which concrete model is used underneath.

    Args:
        model: Trained model instance implementing ``models.BaseModel``.
        X: Feature array, shape ``(n_samples, window_size, n_features)``.

    Returns:
        Predicted targets, shape ``(n_samples,)``.

    Raises:
        ValueError: If ``model`` is not a ``models.BaseModel`` instance,
            or if ``X`` does not have 3 dimensions.
    """
    _validate_model(model)
    if X.ndim != 3:
        raise ValueError(
            "Expected X with 3 dimensions (n_samples, window_size, "
            f"n_features), got shape {X.shape}."
        )

    X_input = flatten_windows(X) if model.requires_flat_input else X
    logger.info(
        "Running predict() on %d sample(s) (requires_flat_input=%s).",
        X.shape[0], model.requires_flat_input,
    )
    return model.predict(X_input)


def bias_correction(
    model: BaseModel,
    X: np.ndarray,
    horizon: int = FORECAST_HORIZON,
    bias: float | np.ndarray = 0.0,
    step_bias: np.ndarray | None = None,
) -> np.ndarray:
    """Perform bias-corrected multi-step forecasting (primary method).

    At each step, the model's raw prediction is corrected by adding
    ``bias`` (and, if given, that step's entry in ``step_bias``) before
    the corrected value is fed back into the window as the newest
    timestep (dropping the oldest one) for the next step. This is the
    project's primary forecasting method.

    ``bias`` corrects a *constant* offset, applied identically at every
    step. ``step_bias`` is a separate, additive correction indexed by
    rollout step -- for drift that specifically grows or shrinks with
    how many autoregressive steps have been taken, which a single
    constant ``bias`` cannot capture. The two compose: total correction
    at step ``i`` is ``bias + step_bias[i]``.

    TODO: this function *applies* bias corrections that are given to it
    via the ``bias``/``step_bias`` arguments; it does not estimate them
    itself. Estimating them (e.g. from residuals measured on a held-out
    validation set) depends on a calibration step / ``evaluation.py``
    that does not exist yet in this project, and is out of scope for
    ``forecasting.py``. Until that is wired in, callers must supply
    ``bias``/``step_bias`` explicitly -- both default to no correction,
    which makes this equivalent to an uncorrected iterative rollout.

    Args:
        model: Trained model instance implementing ``models.BaseModel``.
        X: Most recent observed window per pixel, shape ``(n_pixels,
            window_size, n_features)``.
        horizon: Number of future timesteps to forecast. Defaults to
            ``config.FORECAST_HORIZON``.
        bias: Constant bias correction added to every raw prediction
            before it is used to update the window. Either a scalar
            (applied to every pixel) or an array of shape
            ``(n_pixels,)`` for a per-pixel correction. Defaults to
            ``0.0`` (see TODO above).
        step_bias: Optional per-rollout-step correction, shape
            ``(horizon,)`` -- ``step_bias[i]`` is added (on top of
            ``bias``) to every pixel's prediction at step ``i``.
            Defaults to ``None`` (no per-step correction; see TODO
            above).

    Returns:
        Forecast array of shape ``(horizon, n_pixels)``, containing
        the bias-corrected predictions.

    Raises:
        ValueError: If ``X`` does not have 3 dimensions, has more than
            one feature, if ``horizon < 1``, if ``bias`` is an array
            whose shape is not ``(n_pixels,)``, or if ``step_bias`` is
            given and its shape is not ``(horizon,)``.
    """
    _validate_model(model)
    n_pixels, _window_size, _n_features = _validate_window(X)
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}.")

    bias_arr = np.asarray(bias, dtype=float)
    if bias_arr.ndim == 0:
        bias_arr = np.full(n_pixels, float(bias_arr))
    elif bias_arr.shape != (n_pixels,):
        raise ValueError(
            f"bias must be a scalar or have shape (n_pixels,) = "
            f"({n_pixels},), got shape {bias_arr.shape}."
        )

    if step_bias is not None:
        step_bias_arr = np.asarray(step_bias, dtype=float)
        if step_bias_arr.shape != (horizon,):
            raise ValueError(
                f"step_bias must have shape (horizon,) = ({horizon},), "
                f"got shape {step_bias_arr.shape}."
            )
    else:
        step_bias_arr = np.zeros(horizon, dtype=float)

    window = X.copy()
    forecasts = np.empty((horizon, n_pixels), dtype=float)

    for step in range(horizon):
        raw_pred = predict(model, window)
        corrected_pred = raw_pred + bias_arr + step_bias_arr[step]
        forecasts[step] = corrected_pred
        window = _slide_window(window, corrected_pred)

    logger.info(
        "bias_correction: produced %d step(s) for %d pixel(s) "
        "(bias=%s, step_bias=%s).",
        horizon, n_pixels,
        "0.0" if np.all(bias_arr == 0.0) else "custom",
        "none" if step_bias is None else "custom",
    )
    return forecasts


def forecast_only(model: BaseModel, X: np.ndarray) -> np.ndarray:
    """Baseline forecast: a single one-step prediction, no window update.

    ``window -> model.predict() -> prediction``. No multi-step rollout,
    no bias correction, no teacher forcing.

    Args:
        model: Trained model instance implementing ``models.BaseModel``.
        X: Feature array, shape ``(n_samples, window_size, n_features)``.

    Returns:
        Predicted targets, shape ``(n_samples,)``.

    Raises:
        ValueError: Propagated from :func:`predict`.
    """
    return predict(model, X)


def teacher_forcing(
    model: BaseModel,
    X: np.ndarray,
    y_true: np.ndarray,
    horizon: int = FORECAST_HORIZON,
) -> np.ndarray:
    """Perform multi-step forecasting using ground-truth observations.

    At each step, the model still produces its own one-step-ahead
    prediction (which is what gets returned), but the window is
    advanced using the actual observed value (``y_true``) rather than
    the model's own prediction. Not the current focus of the project --
    kept as a working secondary method rather than a stub, since it was
    already implemented and validated in an earlier phase.

    Args:
        model: Trained model instance implementing ``models.BaseModel``.
        X: Most recent observed window per pixel, shape ``(n_pixels,
            window_size, n_features)``.
        y_true: Actual observed target values for each of the
            ``horizon`` future steps, shape ``(horizon, n_pixels)``,
            used to update the window instead of the model's own
            prediction.
        horizon: Number of future timesteps to forecast. Defaults to
            ``config.FORECAST_HORIZON``.

    Returns:
        Forecast array of shape ``(horizon, n_pixels)`` -- the model's
        own one-step-ahead predictions at each step (not the ground
        truth used to update the window).

    Raises:
        ValueError: If ``X`` does not have 3 dimensions, has more than
            one feature, if ``horizon < 1``, or if ``y_true`` does not
            have shape ``(horizon, n_pixels)``.
    """
    _validate_model(model)
    n_pixels, _window_size, _n_features = _validate_window(X)
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}.")
    if y_true.shape != (horizon, n_pixels):
        raise ValueError(
            f"y_true must have shape (horizon, n_pixels) = "
            f"({horizon}, {n_pixels}), got {y_true.shape}."
        )

    window = X.copy()
    forecasts = np.empty((horizon, n_pixels), dtype=float)

    for step in range(horizon):
        y_pred = predict(model, window)
        forecasts[step] = y_pred
        window = _slide_window(window, y_true[step])

    logger.info(
        "teacher_forcing: produced %d step(s) for %d pixel(s).",
        horizon, n_pixels,
    )
    return forecasts


def forecast(
    model: BaseModel,
    X: np.ndarray,
    update_method: str = FORECAST_UPDATE_METHOD,
    **kwargs: Any,
) -> np.ndarray:
    """High-level forecasting entry point, dispatching by update method.

    This function performs validation and dispatch only; the actual
    forecasting logic lives in the registered method function (see
    ``_STRATEGY_REGISTRY``). Adding a new method never requires
    changing this function -- only registering a new function and
    adding it to ``config.SUPPORTED_UPDATE_METHODS``.

    Args:
        model: Trained model instance implementing ``models.BaseModel``.
        X: Input array for the selected method. For ``"bias_correction"``
            and ``"teacher_forcing"``, the most recent observed window
            per pixel, shape ``(n_pixels, window_size, n_features)``.
            For ``"forecast_only"``, a batch of windows, shape
            ``(n_samples, window_size, n_features)``.
        update_method: Name of the registered update method to use.
            Must be one of ``config.SUPPORTED_UPDATE_METHODS``. Defaults
            to ``config.FORECAST_UPDATE_METHOD`` (currently
            ``"bias_correction"``).
        **kwargs: Additional keyword arguments forwarded to the
            selected method function (e.g. ``horizon``/``bias`` for
            ``"bias_correction"``, ``horizon``/``y_true`` for
            ``"teacher_forcing"``).

    Returns:
        The selected method function's forecast output.

    Raises:
        ValueError: If ``update_method`` is not one of
            ``config.SUPPORTED_UPDATE_METHODS``, or propagated from the
            selected method function's own validation.
    """
    if update_method not in SUPPORTED_UPDATE_METHODS:
        raise ValueError(
            f"Unknown update_method '{update_method}'. Supported methods "
            f"(config.SUPPORTED_UPDATE_METHODS): {sorted(SUPPORTED_UPDATE_METHODS)}."
        )

    strategy_fn = _STRATEGY_REGISTRY[update_method]
    logger.info(
        "forecast(): dispatching to update_method '%s'.", update_method,
    )
    return strategy_fn(model=model, X=X, **kwargs)


def _validate_model(model: BaseModel) -> None:
    """Validate that ``model`` implements the ``models.BaseModel`` contract.

    Args:
        model: Object to validate.

    Raises:
        ValueError: If ``model`` is not a ``models.BaseModel`` instance.
    """
    if not isinstance(model, BaseModel):
        raise ValueError(
            f"Expected an instance of models.BaseModel, got {type(model)!r}."
        )


def _validate_window(X: np.ndarray) -> tuple[int, int, int]:
    """Validate and unpack a window array's shape.

    Args:
        X: Array expected to have shape ``(n_pixels, window_size,
            n_features)``.

    Returns:
        Tuple ``(n_pixels, window_size, n_features)``.

    Raises:
        ValueError: If ``X`` does not have 3 dimensions, or if it has
            more than one feature. Multivariate windows are not yet
            supported by the window-update step, since only the
            forecasted target value -- not future values of the other
            input features -- is available to build the next timestep.
            This matches the project's current config
            (``INPUT_FEATURES == (TARGET_VARIABLE,)``).
    """
    if X.ndim != 3:
        raise ValueError(
            "Expected X with 3 dimensions (n_pixels, window_size, "
            f"n_features), got shape {X.shape}."
        )
    n_pixels, window_size, n_features = X.shape
    if n_features != 1:
        raise ValueError(
            "bias_correction/teacher_forcing currently only support a "
            f"single input feature (n_features=1), got n_features="
            f"{n_features}. Multivariate windows would require a way "
            "to obtain future values of the other input features, "
            "which forecasting.py does not have."
        )
    return n_pixels, window_size, n_features


def _slide_window(window: np.ndarray, new_values: np.ndarray) -> np.ndarray:
    """Advance a window by one timestep, appending ``new_values``.

    Drops the oldest timestep and appends ``new_values`` as the newest
    one, keeping ``window_size`` constant.

    Args:
        window: Current window, shape ``(n_pixels, window_size,
            n_features)``.
        new_values: New timestep values per pixel, shape ``(n_pixels,)``.

    Returns:
        The updated window, same shape as ``window``.
    """
    n_pixels = window.shape[0]
    new_step = new_values.reshape(n_pixels, 1, 1)
    return np.concatenate([window[:, 1:, :], new_step], axis=1)


#: Strategy registry mapping an ``update_method`` name to its
#: forecasting function. Keys are kept in lockstep with
#: ``config.SUPPORTED_UPDATE_METHODS`` (checked below at import time).
_STRATEGY_REGISTRY: dict[str, Callable[..., np.ndarray]] = {
    "bias_correction": bias_correction,
    "forecast_only": forecast_only,
    "teacher_forcing": teacher_forcing,
}

if set(_STRATEGY_REGISTRY) != set(SUPPORTED_UPDATE_METHODS):
    raise RuntimeError(
        "forecasting._STRATEGY_REGISTRY keys must exactly match "
        f"config.SUPPORTED_UPDATE_METHODS; got registry="
        f"{sorted(_STRATEGY_REGISTRY)} vs config="
        f"{sorted(SUPPORTED_UPDATE_METHODS)}."
    )

