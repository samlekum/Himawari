"""Sliding-window dataset construction for AI Himawari Forecasting.

This module is responsible ONLY for turning the per-timestamp feature/
target arrays produced by ``preprocessing.py`` into a supervised
sliding-window dataset ready for model training/forecasting.

Pipeline position:

    preprocessing.py (X, y per day)
        -> create_windows
        -> supervised (X_windowed, y_windowed, pixel_index, time_index,
                        window_start, window_end)

Each spatial pixel is treated as an independent time series: windows are
built per pixel, never mixing values across different pixels. This
module uses a standard sliding window with a fixed stride of 1 (i.e.
every valid window start index is used; no configurable stride).

Window size and forecast horizon are read from ``config.py``
(``WINDOW_SIZE``, ``FORECAST_HORIZON``) and are never hardcoded here.

This module never trains models, never forecasts, never evaluates, and
never visualizes. It only builds supervised arrays from whatever
``(X, y)`` arrays it is given.
"""

from __future__ import annotations

import logging

import numpy as np

from config import FORECAST_HORIZON, WINDOW_SIZE

logger = logging.getLogger(__name__)


def create_windows(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int = WINDOW_SIZE,
    horizon: int = FORECAST_HORIZON,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a supervised sliding-window dataset from per-timestamp arrays.

    Each spatial pixel is treated as an independent time series. For a
    pixel ``p`` and a window start index ``i``, one sample consists of:

    - features: ``X[i : i + window_size, p, :]``
    - target:   ``y[i + window_size + horizon - 1, p]``

    Windows advance one timestep at a time (standard sliding window,
    stride = 1), independently for every pixel (i.e. pixels are never
    mixed within a single window).

    Args:
        X: Feature array of shape ``(n_timesteps, n_pixels, n_features)``,
            e.g. as returned by ``preprocessing.preprocess_day``.
        y: Target array of shape ``(n_timesteps, n_pixels)``, e.g. as
            returned by ``preprocessing.preprocess_day``.
        window_size: Number of past timesteps used as model input (lag
            window). Defaults to ``config.WINDOW_SIZE``.
        horizon: Number of timesteps ahead of the window to forecast.
            Defaults to ``config.FORECAST_HORIZON``.

    Returns:
        Tuple ``(X_windowed, y_windowed, pixel_index, time_index,
        window_start, window_end)``:

        - ``X_windowed``: shape ``(n_samples, window_size, n_features)``.
        - ``y_windowed``: shape ``(n_samples,)``.
        - ``pixel_index``: shape ``(n_samples,)``, ``int64``. The source
          pixel of each sample (index into the ``n_pixels`` axis of
          ``X``/``y``) -- needed to reconstruct forecasts back onto the
          spatial grid (e.g. for GIF/summary output).
        - ``time_index``: shape ``(n_samples,)``, ``int64``. The
          timestep index (into the ``n_timesteps`` axis) that each
          sample's target corresponds to -- needed by forecast update
          strategies (teacher forcing / bias correction) to look up the
          matching observation.
        - ``window_start``: shape ``(n_samples,)``, ``int64``. The first
          timestep index (inclusive) included in each sample's input
          window -- kept for debugging/traceability.
        - ``window_end``: shape ``(n_samples,)``, ``int64``. The last
          timestep index (inclusive) included in each sample's input
          window, i.e. ``window_start + window_size - 1`` -- kept for
          debugging/traceability.

    Raises:
        ValueError: If ``X``/``y`` do not have the expected number of
            dimensions or consistent ``(n_timesteps, n_pixels)`` shapes,
            if ``window_size`` or ``horizon`` are not positive integers,
            or if there are not enough timesteps to build a single
            window.
    """
    if X.ndim != 3:
        raise ValueError(
            "Expected X with 3 dimensions (n_timesteps, n_pixels, "
            f"n_features), got shape {X.shape}."
        )
    if y.ndim != 2:
        raise ValueError(
            "Expected y with 2 dimensions (n_timesteps, n_pixels), "
            f"got shape {y.shape}."
        )
    if X.shape[0] != y.shape[0] or X.shape[1] != y.shape[1]:
        raise ValueError(
            "X and y must share the same (n_timesteps, n_pixels); "
            f"got X{X.shape} vs y{y.shape}."
        )
    if window_size < 1 or horizon < 1:
        raise ValueError(
            "window_size and horizon must both be >= 1; got "
            f"window_size={window_size}, horizon={horizon}."
        )

    n_timesteps, n_pixels, _ = X.shape
    span = window_size + horizon
    if n_timesteps < span:
        raise ValueError(
            f"Not enough timesteps ({n_timesteps}) to build a single "
            f"window with window_size={window_size} and horizon={horizon} "
            f"(need at least {span})."
        )

    starts = range(0, n_timesteps - span + 1)  # stride = 1

    X_windows: list[np.ndarray] = []
    y_windows: list[float] = []
    pixel_index: list[int] = []
    time_index: list[int] = []
    window_start: list[int] = []
    window_end: list[int] = []

    for p in range(n_pixels):
        for i in starts:
            end = i + window_size - 1
            target_t = end + horizon
            X_windows.append(X[i : i + window_size, p, :])
            y_windows.append(y[target_t, p])
            pixel_index.append(p)
            time_index.append(target_t)
            window_start.append(i)
            window_end.append(end)

    X_windowed = np.stack(X_windows, axis=0)
    y_windowed = np.asarray(y_windows)
    pixel_index_arr = np.asarray(pixel_index, dtype=np.int64)
    time_index_arr = np.asarray(time_index, dtype=np.int64)
    window_start_arr = np.asarray(window_start, dtype=np.int64)
    window_end_arr = np.asarray(window_end, dtype=np.int64)

    logger.info(
        "Built %d window(s) from %d pixel(s) x %d timestep(s) "
        "(window_size=%d, horizon=%d, stride=1) -> X%s, y%s.",
        X_windowed.shape[0], n_pixels, n_timesteps,
        window_size, horizon, X_windowed.shape, y_windowed.shape,
    )
    return (
        X_windowed,
        y_windowed,
        pixel_index_arr,
        time_index_arr,
        window_start_arr,
        window_end_arr,
    )


def flatten_windows(X_windowed: np.ndarray) -> np.ndarray:
    """Flatten a windowed feature array for classical (non-sequential) models.

    Reshapes the 3D output of :func:`create_windows` from
    ``(n_samples, window_size, n_features)`` into a 2D array of shape
    ``(n_samples, window_size * n_features)`` by concatenating each
    sample's window along a single feature axis, in timestep-major
    order (i.e. all features of timestep 0, then all features of
    timestep 1, and so on).

    This is an optional helper for models that require 2D input, such
    as Gradient Boosting or Gaussian Process. Sequential models (e.g.
    ANN) should keep using the original 3D output of
    :func:`create_windows` directly.

    Args:
        X_windowed: Feature array of shape
            ``(n_samples, window_size, n_features)``, as returned by
            :func:`create_windows`.

    Returns:
        Flattened feature array of shape
        ``(n_samples, window_size * n_features)``.

    Raises:
        ValueError: If ``X_windowed`` does not have exactly 3
            dimensions.
    """
    if X_windowed.ndim != 3:
        raise ValueError(
            "Expected X_windowed with 3 dimensions (n_samples, "
            f"window_size, n_features), got shape {X_windowed.shape}."
        )

    n_samples, window_size, n_features = X_windowed.shape
    X_flat = X_windowed.reshape(n_samples, window_size * n_features)

    logger.info(
        "Flattened windowed features: %s -> %s.",
        X_windowed.shape, X_flat.shape,
    )
    return X_flat