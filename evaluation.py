"""Performance evaluation for AI Himawari Forecasting.

This module is responsible ONLY for computing performance metrics from
already-produced predictions: regression metrics for continuous
predictions (``forecasting.py`` output) and classification metrics for
discrete class predictions (``classification.py`` output). It never
loads files, never preprocesses, never builds sliding windows, never
trains or forecasts, never classifies continuous values into classes
itself, never exports results to disk, and never visualizes -- all of
that is the responsibility of other modules (``loader.py``,
``preprocessing.py``, ``windowing.py``, ``dataset_builder.py``,
``training.py``, ``forecasting.py``, ``classification.py``, an export
module, and ``visualization.py``).

Pipeline position:

    forecasting.py (y_true, y_pred)          -> evaluate_regression
    classification.py (y_true_class,
                        y_pred_class)        -> evaluate_classification
        -> evaluate() (wrapper over both)
        -> an export module / visualization.py

This module is intentionally scheme-agnostic: it has no notion of the
"cloud" or "risk" classification schemes defined in ``config.py``. It
only ever operates on plain ``class_index`` integer arrays and an
optional ``labels`` list for display/ordering. Callers that need to
evaluate multiple schemes (e.g. both cloud and risk) call
:func:`evaluate` once per scheme, passing in that scheme's own
``class_index`` arrays and labels -- this module never imports or
inspects ``classification.py``'s output structure directly, so the two
modules stay decoupled.

All metrics are computed with ``sklearn.metrics``, the same library
already used by ``experiment.py`` for its regression metrics.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

logger = logging.getLogger(__name__)

__all__ = [
    "evaluate_regression",
    "evaluate_classification",
    "evaluate",
    "estimate_bias",
    "estimate_step_bias",
    "safe_mape",
]

#: Below this absolute value (in real target units, e.g. Kelvin for
#: tbb_13), a y_true sample is treated as a missing-data fallback
#: artifact -- see preprocessing.handle_missing_values's last-resort
#: 0.0 fill for an entire-timestamp-missing variable -- rather than a
#: genuine measurement: Himawari brightness temperature is never
#: actually this close to zero. Such samples are still included in
#: MAE/RMSE/R^2 as usual (an artifact still needs to be predicted
#: decently), but excluded from MAPE, whose denominator is y_true
#: itself -- a handful of near-zero denominators can inflate MAPE by
#: many orders of magnitude (e.g. 1e14) and make it meaningless as a
#: report metric.
MAPE_MIN_ABS_TARGET = 1.0


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error, excluding near-zero y_true samples.

    Behaves like ``sklearn.metrics.mean_absolute_percentage_error``
    except that samples with ``abs(y_true) < MAPE_MIN_ABS_TARGET`` are
    dropped before computing the mean, since MAPE divides by
    ``y_true`` and a near-zero true value (a missing-data fallback
    artifact in this project, not a real tbb_13 reading) inflates the
    metric by orders of magnitude and makes it useless for reporting.
    Any excluded samples are logged.

    Args:
        y_true: Ground-truth continuous values, any shape.
        y_pred: Predicted continuous values, same shape as ``y_true``.

    Returns:
        MAPE over the retained samples, or ``NaN`` if every sample was
        excluded.
    """
    y_true_flat = np.asarray(y_true).reshape(-1)
    y_pred_flat = np.asarray(y_pred).reshape(-1)

    mask = np.abs(y_true_flat) >= MAPE_MIN_ABS_TARGET
    n_excluded = int((~mask).sum())
    if n_excluded:
        logger.warning(
            "safe_mape: excluding %d/%d sample(s) with |y_true| < %.3g "
            "from MAPE (near-zero denominator -- likely a missing-data "
            "fallback artifact, not a real measurement; see "
            "preprocessing.handle_missing_values). MAE/RMSE/R^2 are "
            "unaffected by this exclusion.",
            n_excluded, y_true_flat.size, MAPE_MIN_ABS_TARGET,
        )
    if not np.any(mask):
        logger.warning("safe_mape: every sample was excluded; returning NaN.")
        return float("nan")

    return float(mean_absolute_percentage_error(y_true_flat[mask], y_pred_flat[mask]))


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute regression metrics for continuous predictions.

    Args:
        y_true: Ground-truth continuous values (e.g. tbb_13), any
            shape, typically ``(n_samples,)`` or ``(horizon,
            n_pixels)`` as produced by ``forecasting.py``.
        y_pred: Predicted continuous values, same shape as ``y_true``.

    Returns:
        Dictionary with keys ``"mae"``, ``"rmse"``, ``"mse"``,
        ``"r2"``, ``"mape"``.

    Raises:
        ValueError: If ``y_true`` and ``y_pred`` do not have the same
            shape.
    """
    y_true, y_pred = _validate_pair(y_true, y_pred, "y_true", "y_pred")

    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    mse = float(mean_squared_error(y_true_flat, y_pred_flat))
    metrics = {
        "mae": float(mean_absolute_error(y_true_flat, y_pred_flat)),
        "rmse": float(np.sqrt(mse)),
        "mse": mse,
        "r2": float(r2_score(y_true_flat, y_pred_flat)),
        "mape": safe_mape(y_true_flat, y_pred_flat),
    }

    logger.info(
        "evaluate_regression: mae=%.4f, rmse=%.4f, mse=%.4f, r2=%.4f, "
        "mape=%.4f (n=%d).",
        metrics["mae"], metrics["rmse"], metrics["mse"], metrics["r2"],
        metrics["mape"], y_true_flat.size,
    )
    return metrics


def evaluate_classification(
    y_true_class: np.ndarray,
    y_pred_class: np.ndarray,
    labels: list[str] | None = None,
    average: str = "macro",
) -> dict[str, Any]:
    """Compute classification metrics for a single classification scheme.

    Scheme-agnostic: this function has no notion of "cloud" or "risk"
    -- it only operates on integer class-index arrays. Callers pass in
    whichever scheme's arrays/labels they want evaluated, and call
    this again for any other scheme.

    Args:
        y_true_class: Ground-truth class indices, any shape, typically
            the ``"class_index"`` array from ``classification.py``'s
            output for one scheme.
        y_pred_class: Predicted class indices, same shape as
            ``y_true_class``.
        labels: Optional display labels, ordered by class index (e.g.
            ``config.CLOUD_LABELS`` or ``config.RISK_LABELS``), used
            only to order/label the confusion matrix. If omitted, the
            confusion matrix is ordered by the sorted unique class
            indices found in the data.
        average: Averaging strategy forwarded to
            ``sklearn.metrics``'s ``precision_score``/``recall_score``/
            ``f1_score`` for multi-class data. One of ``"macro"``,
            ``"micro"``, ``"weighted"``.

    Returns:
        Dictionary with keys ``"accuracy"``, ``"precision"``,
        ``"recall"``, ``"f1"``, ``"confusion_matrix"`` (``np.ndarray``),
        and ``"labels"`` (the labels used to order the confusion
        matrix, or ``None`` if not provided).

    Raises:
        ValueError: If ``y_true_class`` and ``y_pred_class`` do not
            have the same shape, or if ``average`` is not one of the
            supported strategies.
    """
    y_true_class, y_pred_class = _validate_pair(
        y_true_class, y_pred_class, "y_true_class", "y_pred_class",
    )
    supported_averages = {"macro", "micro", "weighted"}
    if average not in supported_averages:
        raise ValueError(
            f"Unsupported average '{average}'. Supported: {sorted(supported_averages)}."
        )

    y_true_flat = y_true_class.reshape(-1)
    y_pred_flat = y_pred_class.reshape(-1)

    class_order = list(range(len(labels))) if labels is not None else None

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_flat, y_pred_flat)),
        "precision": float(
            precision_score(y_true_flat, y_pred_flat, average=average, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true_flat, y_pred_flat, average=average, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true_flat, y_pred_flat, average=average, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_true_flat, y_pred_flat, labels=class_order),
        "labels": labels,
    }

    logger.info(
        "evaluate_classification: accuracy=%.4f, precision=%.4f, "
        "recall=%.4f, f1=%.4f (average='%s', n=%d).",
        metrics["accuracy"], metrics["precision"], metrics["recall"],
        metrics["f1"], average, y_true_flat.size,
    )
    return metrics


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_true_class: np.ndarray,
    y_pred_class: np.ndarray,
    labels: list[str] | None = None,
    average: str = "macro",
) -> dict[str, Any]:
    """Compute both regression and classification metrics in one call.

    Thin wrapper combining :func:`evaluate_regression` and
    :func:`evaluate_classification` for a single classification
    scheme. To evaluate more than one scheme (e.g. both cloud and
    risk), call this function once per scheme with that scheme's own
    ``y_true_class``/``y_pred_class``/``labels``.

    Args:
        y_true: Ground-truth continuous values, forwarded to
            :func:`evaluate_regression`.
        y_pred: Predicted continuous values, forwarded to
            :func:`evaluate_regression`.
        y_true_class: Ground-truth class indices for one scheme,
            forwarded to :func:`evaluate_classification`.
        y_pred_class: Predicted class indices for one scheme,
            forwarded to :func:`evaluate_classification`.
        labels: Optional display labels for that scheme, forwarded to
            :func:`evaluate_classification`.
        average: Averaging strategy, forwarded to
            :func:`evaluate_classification`.

    Returns:
        Dictionary with keys ``"regression"`` (see
        :func:`evaluate_regression`) and ``"classification"`` (see
        :func:`evaluate_classification`).

    Raises:
        ValueError: Propagated from :func:`evaluate_regression` /
            :func:`evaluate_classification`.
    """
    return {
        "regression": evaluate_regression(y_true, y_pred),
        "classification": evaluate_classification(
            y_true_class, y_pred_class, labels=labels, average=average,
        ),
    }


def estimate_bias(
    y_true: np.ndarray, y_pred: np.ndarray, pixel_index: np.ndarray | None = None,
) -> float | tuple[float, dict[int, float]]:
    """Estimate a systematic bias from held-out residuals.

    This is the calibration step ``forecasting.bias_correction``'s
    module docstring names but leaves out of scope for
    ``forecasting.py`` itself: it measures, on data the model did not
    train on, how far off its one-step predictions tend to run on
    average. The result is meant to be added back to every raw
    prediction in ``forecasting.bias_correction`` (its ``bias``
    argument) to correct that average offset.

    Always returns the global (scalar) bias. Optionally also returns a
    per-pixel breakdown when ``pixel_index`` is given -- grouped by
    each sample's actual source pixel (``windowing.create_windows``'s
    ``pixel_index``), NOT a plain per-column/``axis=0`` mean: this
    project's held-out validation split is a flat, pixel-tagged sample
    list (not a tidy ``(time, pixel)`` matrix), and with
    ``config.SHUFFLE_DATASET=False`` splitting a pixel-major-ordered
    sample list, that split can easily cover only a contiguous *range*
    of pixels rather than every pixel -- so a per-pixel result here may
    be missing most of the grid entirely. Callers must check coverage
    themselves (the returned dict only has keys for pixels actually
    present in the data) and fall back to the global bias for any
    pixel that's missing -- never assume full coverage.

    Args:
        y_true: Ground-truth continuous values from a held-out
            validation split (data the model was not trained on), any
            shape.
        y_pred: The model's own one-step predictions for the same
            samples (e.g. via ``forecasting.predict``), same shape as
            ``y_true``.
        pixel_index: Optional, same sample count as ``y_true``/
            ``y_pred`` (flattened) -- each sample's source pixel id.
            When given, a per-pixel breakdown is also returned.

    Returns:
        If ``pixel_index`` is ``None``: the global bias (float; mean of
        ``y_true - y_pred``, so positive means the model tends to
        under-predict).
        If ``pixel_index`` is given: tuple ``(global_bias,
        per_pixel_bias)`` where ``per_pixel_bias`` is ``{pixel_id:
        bias}``, covering only pixels actually present in the data.

    Raises:
        ValueError: If ``y_true``/``y_pred`` don't share a shape, or if
            ``pixel_index`` is given with a different sample count.
    """
    y_true, y_pred = _validate_pair(y_true, y_pred, "y_true", "y_pred")
    y_true_flat, y_pred_flat = y_true.reshape(-1), y_pred.reshape(-1)
    residual = y_true_flat - y_pred_flat
    global_bias = float(np.mean(residual))
    logger.info(
        "estimate_bias: mean residual (global bias) = %.4f over %d sample(s).",
        global_bias, y_true_flat.size,
    )
    if pixel_index is None:
        return global_bias

    pixel_index = np.asarray(pixel_index).reshape(-1)
    if pixel_index.shape[0] != y_true_flat.shape[0]:
        raise ValueError(
            f"pixel_index length ({pixel_index.shape[0]}) must match the "
            f"flattened y_true/y_pred sample count ({y_true_flat.shape[0]})."
        )
    per_pixel_bias = {
        int(p): float(np.mean(residual[pixel_index == p]))
        for p in np.unique(pixel_index)
    }
    logger.info(
        "estimate_bias: per-pixel bias computed for %d distinct pixel(s) "
        "(caller must verify this covers every pixel it needs).",
        len(per_pixel_bias),
    )
    return global_bias, per_pixel_bias


def estimate_step_bias(y_true_steps: np.ndarray, y_pred_steps: np.ndarray) -> np.ndarray:
    """Estimate a per-rollout-step bias from an already-rolled-out calibration set.

    ``estimate_bias`` measures a model's *constant* one-step offset;
    this measures drift that specifically grows or shrinks with rollout
    step -- error a constant correction can't capture, since it doesn't
    depend on how many autoregressive steps have already been taken.
    Pure statistics: it does not itself run the model or roll it
    forward -- callers produce ``y_pred_steps`` themselves (e.g. via
    several calls to ``forecasting.bias_correction`` from different
    calibration starting points, ideally already corrected by the
    constant bias from ``estimate_bias`` so this measures only the
    residual drift on top of it) and pass the matching ground truth.

    Args:
        y_true_steps: Ground-truth values at each rollout step, shape
            ``(horizon, n_samples)`` -- ``n_samples`` can combine
            multiple pixels and/or multiple calibration starting
            points; every column is just one more sample of "the
            residual at step i".
        y_pred_steps: The model's own rolled-out predictions for the
            same steps/samples, same shape as ``y_true_steps``.

    Returns:
        Array of shape ``(horizon,)``: mean residual (``y_true -
        y_pred``) at each rollout step. Pass this directly as
        ``forecasting.bias_correction``'s ``step_bias`` argument.

    Raises:
        ValueError: If ``y_true_steps`` and ``y_pred_steps`` do not
            share a shape, or aren't 2D.
    """
    y_true_steps, y_pred_steps = _validate_pair(y_true_steps, y_pred_steps, "y_true_steps", "y_pred_steps")
    if y_true_steps.ndim != 2:
        raise ValueError(
            "y_true_steps/y_pred_steps must be 2D, shape (horizon, "
            f"n_samples); got ndim={y_true_steps.ndim}."
        )
    step_bias = np.nanmean(y_true_steps - y_pred_steps, axis=1)
    logger.info(
        "estimate_step_bias: per-step bias over %d step(s), %d sample(s) "
        "each: %s.",
        step_bias.shape[0], y_true_steps.shape[1], np.array2string(step_bias, precision=4),
    )
    return step_bias


def _validate_pair(
    a: np.ndarray,
    b: np.ndarray,
    a_name: str,
    b_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate that two arrays are numpy arrays with the same shape.

    Args:
        a: First array.
        b: Second array.
        a_name: Name of ``a``, used in error messages.
        b_name: Name of ``b``, used in error messages.

    Returns:
        Tuple ``(a, b)``, coerced to ``numpy.ndarray`` if not already.

    Raises:
        ValueError: If ``a`` and ``b`` do not have the same shape.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(
            f"{a_name} and {b_name} must have the same shape; got "
            f"{a_name}.shape={a.shape} vs {b_name}.shape={b.shape}."
        )
    return a, b