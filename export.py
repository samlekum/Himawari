"""Result export for AI Himawari Forecasting.

This module is responsible ONLY for persisting already-produced
results to disk, in JSON, CSV, and NumPy (.npy) format. It never
loads files, never preprocesses, never builds sliding windows, never
trains or forecasts, never classifies, never evaluates, and never
visualizes -- all of that is the responsibility of other modules
(``loader.py``, ``preprocessing.py``, ``windowing.py``,
``dataset_builder.py``, ``training.py``, ``forecasting.py``,
``classification.py``, ``evaluation.py``, ``visualization.py``).

Pipeline position:

    forecasting.py / classification.py / evaluation.py (results)
        -> export.py
        -> files on disk under config.*_OUTPUT_PATH

This module is decoupled from ``evaluation.py`` and ``visualization.py``:
it never imports or calls them. It only receives whatever data the
caller passes in, and writes it out. Every export function checks the
relevant toggle (``config.SAVE_JSON``/``config.SAVE_CSV``/
``config.SAVE_NUMPY``) before writing; when a toggle is ``False`` the
write is skipped (only logged), but the resolved file path is still
returned, so callers can rely on a consistent return type regardless
of the toggle -- the same convention already used by
``training.save_model``.

``export_all`` accepts grouped payload objects (``forecast``,
``classification``, ``evaluation``, ``metadata``) rather than many
individual parameters, so new fields can be added to a payload without
changing this module's signature.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    CLASSIFICATION_OUTPUT_PATH,
    EVALUATION_OUTPUT_PATH,
    FORECAST_OUTPUT_PATH,
    SAVE_CSV,
    SAVE_JSON,
    SAVE_NUMPY,
)

logger = logging.getLogger(__name__)

__all__ = [
    "export_json",
    "export_csv",
    "export_numpy",
    "export_all",
]


def export_json(data: dict[str, Any], filepath: Path) -> Path:
    """Persist a dictionary as a JSON file.

    Values that are not natively JSON-serializable (``numpy.ndarray``,
    ``numpy.integer``, ``numpy.floating``, ``pathlib.Path``) are
    converted first via :func:`_make_json_safe`, so callers can pass
    dictionaries containing raw numpy arrays (e.g.
    ``dataset_builder.create_metadata``'s output) directly.

    If ``config.SAVE_JSON`` is ``False``, the actual write is skipped
    (only logged), and the path that would have been used is still
    returned.

    Args:
        data: Dictionary to serialize.
        filepath: Destination file path (including filename).

    Returns:
        The full path of the JSON file (whether or not it was
        actually written -- see ``config.SAVE_JSON`` above).

    Raises:
        ValueError: If ``data`` is not a ``dict``.
    """
    if not isinstance(data, dict):
        raise ValueError(f"Expected data to be a dict, got {type(data)!r}.")

    if not SAVE_JSON:
        logger.info("SAVE_JSON is disabled in config; skipping write to '%s'.", filepath)
        return filepath

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(_make_json_safe(data), f, indent=2)

    logger.info("Exported JSON to '%s'.", filepath)
    return filepath


def export_csv(data: dict[str, np.ndarray] | pd.DataFrame, filepath: Path) -> Path:
    """Persist tabular data as a CSV file.

    Args:
        data: Either a ``pandas.DataFrame``, or a dictionary mapping
            column name to a 1D array of equal length (converted to a
            ``DataFrame`` internally).
        filepath: Destination file path (including filename).

    Returns:
        The full path of the CSV file (whether or not it was actually
        written -- see ``config.SAVE_CSV`` below).

    Raises:
        ValueError: If ``data`` is neither a ``dict`` nor a
            ``pandas.DataFrame``, or if the dict's arrays do not all
            have the same length.
    """
    if isinstance(data, pd.DataFrame):
        frame = data
    elif isinstance(data, dict):
        lengths = {len(np.asarray(values)) for values in data.values()}
        if len(lengths) > 1:
            raise ValueError(
                f"All columns in data must have the same length; got lengths {lengths}."
            )
        frame = pd.DataFrame(data)
    else:
        raise ValueError(
            f"Expected data to be a dict or pandas.DataFrame, got {type(data)!r}."
        )

    if not SAVE_CSV:
        logger.info("SAVE_CSV is disabled in config; skipping write to '%s'.", filepath)
        return filepath

    filepath.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(filepath, index=False)

    logger.info("Exported CSV to '%s' (%d row(s)).", filepath, len(frame))
    return filepath


def export_numpy(array: np.ndarray, filepath: Path) -> Path:
    """Persist a numpy array as a .npy file.

    Args:
        array: Array to persist.
        filepath: Destination file path (including filename).

    Returns:
        The full path of the .npy file (whether or not it was
        actually written -- see ``config.SAVE_NUMPY`` below).

    Raises:
        ValueError: If ``array`` is not a ``numpy.ndarray``.
    """
    if not isinstance(array, np.ndarray):
        raise ValueError(f"Expected array to be a numpy.ndarray, got {type(array)!r}.")

    if not SAVE_NUMPY:
        logger.info("SAVE_NUMPY is disabled in config; skipping write to '%s'.", filepath)
        return filepath

    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.save(filepath, array)

    logger.info("Exported NumPy array to '%s' (shape=%s).", filepath, array.shape)
    return filepath


def export_all(
    forecast: dict[str, Any] | None = None,
    classification: dict[str, Any] | None = None,
    evaluation: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    filename_suffix: str = "",
) -> dict[str, Path]:
    """Export a full set of run artifacts in one call.

    Each argument is a grouped payload dictionary rather than many
    individual parameters, so new fields can be added to a payload
    without changing this function's signature. Any argument left as
    ``None`` is skipped entirely (no error).

    Args:
        forecast: Dictionary with any of the following keys:
            ``"y_true"`` (``np.ndarray``, ground-truth TBB),
            ``"y_pred"`` (``np.ndarray``, forecast TBB). Written under
            ``config.FORECAST_OUTPUT_PATH`` as
            ``ground_truth.npy``/``.csv`` and
            ``forecast.npy``/``.csv``.
        classification: Dictionary with any of the following keys:
            ``"cloud"``, ``"risk"`` -- each, if present, a dict with
            ``"y_true_class"``/``"y_pred_class"`` (``np.ndarray``).
            Written under ``config.CLASSIFICATION_OUTPUT_PATH`` as
            ``cloud_class.npy``/``.csv`` and
            ``risk_class.npy``/``.csv``.
        evaluation: Dictionary with any of the following keys:
            ``"regression"`` (a :func:`evaluation.evaluate_regression`
            result), ``"classification"`` -- a dict keyed by scheme
            name (e.g. ``"cloud"``, ``"risk"``), each a
            :func:`evaluation.evaluate_classification` result. Written
            under ``config.EVALUATION_OUTPUT_PATH`` as
            ``regression_metrics.json``,
            ``classification_metrics_<scheme>.json``, and
            ``confusion_matrix_<scheme>.npy``.
        metadata: Arbitrary metadata dictionary (e.g.
            ``dataset_builder.create_metadata``'s output, which already
            includes ``pixel_index``/``time_index``). Written under
            ``config.FORECAST_OUTPUT_PATH`` as ``metadata.json``.
        filename_suffix: Inserted before every written file's extension
            (e.g. ``"_10min"`` -> ``ground_truth_10min.npy``). Defaults
            to ``""`` (no change, the original filenames) -- set this
            when calling ``export_all`` more than once per run (e.g.
            once per forecast-horizon scenario) so later calls don't
            overwrite earlier ones.

    Returns:
        Dictionary mapping a descriptive artifact name (e.g.
        ``"forecast_npy"``, ``"cloud_class_csv"``,
        ``"regression_metrics_json"``) to the path it was written to
        (or would have been written to, per the relevant ``SAVE_*``
        toggle). Only keys for artifacts actually requested (non-``None``
        groups with the relevant sub-keys present) are included.

    Raises:
        ValueError: Propagated from :func:`export_json` /
            :func:`export_csv` / :func:`export_numpy`.
    """
    def _name(stem: str, ext: str) -> str:
        return f"{stem}{filename_suffix}.{ext}"

    written: dict[str, Path] = {}

    if forecast is not None:
        y_true = forecast.get("y_true")
        y_pred = forecast.get("y_pred")
        if y_true is not None:
            written["ground_truth_npy"] = export_numpy(
                y_true, FORECAST_OUTPUT_PATH / _name("ground_truth", "npy"),
            )
            written["ground_truth_csv"] = export_csv(
                {"ground_truth": np.asarray(y_true).reshape(-1)},
                FORECAST_OUTPUT_PATH / _name("ground_truth", "csv"),
            )
        if y_pred is not None:
            written["forecast_npy"] = export_numpy(
                y_pred, FORECAST_OUTPUT_PATH / _name("forecast", "npy"),
            )
            written["forecast_csv"] = export_csv(
                {"forecast": np.asarray(y_pred).reshape(-1)},
                FORECAST_OUTPUT_PATH / _name("forecast", "csv"),
            )

    if classification is not None:
        for scheme_name, scheme_data in classification.items():
            y_true_class = scheme_data.get("y_true_class")
            y_pred_class = scheme_data.get("y_pred_class")
            if y_true_class is None and y_pred_class is None:
                continue

            npy_columns: dict[str, np.ndarray] = {}
            if y_true_class is not None:
                npy_columns["true_class"] = np.asarray(y_true_class).reshape(-1)
            if y_pred_class is not None:
                npy_columns["pred_class"] = np.asarray(y_pred_class).reshape(-1)

            stacked = np.stack(list(npy_columns.values()), axis=-1)
            written[f"{scheme_name}_class_npy"] = export_numpy(
                stacked, CLASSIFICATION_OUTPUT_PATH / _name(f"{scheme_name}_class", "npy"),
            )
            written[f"{scheme_name}_class_csv"] = export_csv(
                npy_columns, CLASSIFICATION_OUTPUT_PATH / _name(f"{scheme_name}_class", "csv"),
            )

    if evaluation is not None:
        regression_result = evaluation.get("regression")
        if regression_result is not None:
            written["regression_metrics_json"] = export_json(
                regression_result,
                EVALUATION_OUTPUT_PATH / _name("regression_metrics", "json"),
            )

        classification_results = evaluation.get("classification")
        if classification_results is not None:
            for scheme_name, scheme_result in classification_results.items():
                confusion = scheme_result.get("confusion_matrix")
                metrics_only = {
                    key: value for key, value in scheme_result.items()
                    if key != "confusion_matrix"
                }
                written[f"classification_metrics_{scheme_name}_json"] = export_json(
                    metrics_only,
                    EVALUATION_OUTPUT_PATH / _name(f"classification_metrics_{scheme_name}", "json"),
                )
                if confusion is not None:
                    written[f"confusion_matrix_{scheme_name}_npy"] = export_numpy(
                        np.asarray(confusion),
                        EVALUATION_OUTPUT_PATH / _name(f"confusion_matrix_{scheme_name}", "npy"),
                    )

    if metadata is not None:
        written["metadata_json"] = export_json(
            metadata, FORECAST_OUTPUT_PATH / _name("metadata", "json"),
        )

    logger.info("export_all: wrote %d artifact(s): %s.", len(written), sorted(written))
    return written


def _make_json_safe(value: Any) -> Any:
    """Recursively convert a value into a JSON-serializable form.

    Converts ``numpy.ndarray`` to nested lists, ``numpy.integer``/
    ``numpy.floating`` to plain ``int``/``float``, and
    ``pathlib.Path`` to ``str``. Dicts, lists, and tuples are
    recursed into; every other value is passed through unchanged (and
    left to ``json.dump`` to accept or reject).

    Args:
        value: Value to convert.

    Returns:
        A JSON-serializable equivalent of ``value``.
    """
    if isinstance(value, dict):
        return {key: _make_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _make_json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value