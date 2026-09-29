"""Continuous-to-class classification for AI Himawari Forecasting.

This module is responsible ONLY for turning continuous predictions
(e.g. tbb_13 values produced by ``forecasting.py``) into discrete
classes, using thresholds, labels, and colors defined in
``config.py``. It never loads NetCDF files, never preprocesses, never
builds sliding windows, never trains a model, never forecasts, never
computes evaluation metrics, never exports, and never visualizes --
all of that is the responsibility of other modules (``loader.py``,
``preprocessing.py``, ``windowing.py``, ``dataset_builder.py``,
``training.py``, ``forecasting.py``, ``evaluation.py``, an export
module, and ``visualization.py``).

Pipeline position:

    forecasting.py (continuous prediction)
        -> classification.py
        -> {"class_index", "class_label", "class_color"}
        -> evaluation.py / export.py / visualization.py

Two classification schemes are supported, both read from ``config.py``
and never hardcoded here: the cloud-condition scheme
(``CLOUD_BINS``/``CLOUD_LABELS``/``CLOUD_COLORS``) and the risk-level
scheme (``RISK_BINS``/``RISK_LABELS``/``RISK_COLORS``). Both schemes
share the same binning logic (see :func:`_classify`), since only the
threshold/label/color values differ between them -- duplicating that
logic per scheme would violate separation of concern.

Binning is done with ``numpy.digitize``: for ``n`` bin edges there are
always ``n + 1`` classes -- values below the first edge fall into
class 0, values at/above the last edge fall into class ``n``, and
values between two consecutive edges fall into the class in between.
This means ``len(labels)`` and ``len(colors)`` must always equal
``len(bins) + 1``, which is validated on every call.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from config import (
    CLOUD_BINS,
    CLOUD_COLORS,
    CLOUD_LABELS,
    RISK_BINS,
    RISK_COLORS,
    RISK_LABELS,
)

logger = logging.getLogger(__name__)

__all__ = [
    "classify_cloud",
    "classify_risk",
    "classify",
]


def classify_cloud(
    values: np.ndarray,
    bins: list[int] = CLOUD_BINS,
    labels: list[str] = CLOUD_LABELS,
    colors: list[str] = CLOUD_COLORS,
) -> dict[str, np.ndarray]:
    """Classify continuous predictions into cloud-condition classes.

    Args:
        values: Continuous prediction values (e.g. tbb_13), any shape.
            Typically the output of ``forecasting.predict``/
            ``forecasting.forecast``.
        bins: Ascending bin edges separating the classes. Defaults to
            ``config.CLOUD_BINS``.
        labels: Class labels, ordered to match ``bins`` (must have
            ``len(bins) + 1`` entries). Defaults to
            ``config.CLOUD_LABELS``.
        colors: Display colors (hex) for ``labels``, same order and
            length. Defaults to ``config.CLOUD_COLORS``.

    Returns:
        Dictionary with keys ``"class_index"`` (``int`` array),
        ``"class_label"`` (``str`` array), and ``"class_color"``
        (``str`` array), each with the same shape as ``values``.

    Raises:
        ValueError: Propagated from :func:`_classify` -- see there for
            the specific checks performed.
    """
    return _classify(values, bins=bins, labels=labels, colors=colors, scheme_name="cloud")


def classify_risk(
    values: np.ndarray,
    bins: list[int] = RISK_BINS,
    labels: list[str] = RISK_LABELS,
    colors: list[str] = RISK_COLORS,
) -> dict[str, np.ndarray]:
    """Classify continuous predictions into risk-level classes.

    Args:
        values: Continuous prediction values (e.g. tbb_13), any shape.
            Typically the output of ``forecasting.predict``/
            ``forecasting.forecast``.
        bins: Ascending bin edges separating the classes. Defaults to
            ``config.RISK_BINS``.
        labels: Class labels, ordered to match ``bins`` (must have
            ``len(bins) + 1`` entries). Defaults to
            ``config.RISK_LABELS``.
        colors: Display colors (hex) for ``labels``, same order and
            length. Defaults to ``config.RISK_COLORS``.

    Returns:
        Dictionary with keys ``"class_index"`` (``int`` array),
        ``"class_label"`` (``str`` array), and ``"class_color"``
        (``str`` array), each with the same shape as ``values``.

    Raises:
        ValueError: Propagated from :func:`_classify` -- see there for
            the specific checks performed.
    """
    return _classify(values, bins=bins, labels=labels, colors=colors, scheme_name="risk")


def classify(values: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Classify continuous predictions using every supported scheme.

    Convenience wrapper around :func:`classify_cloud` and
    :func:`classify_risk` for callers (typically ``evaluation.py``,
    an export module, or ``visualization.py``) that need both
    schemes for the same predictions, without having to know each
    scheme's individual bins/labels/colors.

    Args:
        values: Continuous prediction values (e.g. tbb_13), any shape.

    Returns:
        Dictionary with keys ``"cloud"`` and ``"risk"``, each mapping
        to the corresponding scheme's output (see
        :func:`classify_cloud` / :func:`classify_risk`).

    Raises:
        ValueError: Propagated from :func:`classify_cloud` /
            :func:`classify_risk`.
    """
    return {
        "cloud": classify_cloud(values),
        "risk": classify_risk(values),
    }


def _classify(
    values: np.ndarray,
    bins: list[int],
    labels: list[str],
    colors: list[str],
    scheme_name: str,
) -> dict[str, np.ndarray]:
    """Shared binning logic used by every classification scheme.

    Args:
        values: Continuous prediction values, any shape.
        bins: Ascending bin edges separating the classes.
        labels: Class labels, ordered to match ``bins`` (must have
            ``len(bins) + 1`` entries).
        colors: Display colors (hex) for ``labels``, same order and
            length.
        scheme_name: Name of the calling scheme (``"cloud"`` or
            ``"risk"``), used only for logging/error messages.

    Returns:
        Dictionary with keys ``"class_index"``, ``"class_label"``,
        and ``"class_color"``, each with the same shape as ``values``.

    Raises:
        ValueError: If ``values`` is not a ``numpy.ndarray``, if
            ``bins`` is not strictly ascending, or if ``labels``/
            ``colors`` do not have exactly ``len(bins) + 1`` entries.
    """
    if not isinstance(values, np.ndarray):
        raise ValueError(
            f"Expected values to be a numpy.ndarray, got {type(values)!r}."
        )
    if list(bins) != sorted(set(bins)):
        raise ValueError(
            f"bins for scheme '{scheme_name}' must be strictly ascending "
            f"with no duplicates, got {bins}."
        )
    n_classes = len(bins) + 1
    if len(labels) != n_classes:
        raise ValueError(
            f"labels for scheme '{scheme_name}' must have len(bins) + 1 = "
            f"{n_classes} entries, got {len(labels)}."
        )
    if len(colors) != n_classes:
        raise ValueError(
            f"colors for scheme '{scheme_name}' must have len(bins) + 1 = "
            f"{n_classes} entries, got {len(colors)}."
        )

    class_index = np.digitize(values, bins)
    class_label = np.asarray(labels, dtype=object)[class_index]
    class_color = np.asarray(colors, dtype=object)[class_index]

    unique, counts = np.unique(class_index, return_counts=True)
    distribution = ", ".join(
        f"{labels[idx]}={count}" for idx, count in zip(unique, counts)
    )
    logger.info(
        "classify (%s): classified %d value(s) into %d class(es) [%s].",
        scheme_name, class_index.size, n_classes, distribution,
    )

    return {
        "class_index": class_index,
        "class_label": class_label,
        "class_color": class_color,
    }
