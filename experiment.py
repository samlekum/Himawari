"""Hyperparameter-search orchestration for AI Himawari Forecasting.

This module is responsible ONLY for orchestrating experiments that
compare configurations (window size; and, separately, cross-validating
a chosen (window size, model) combination) before final training. It
never loads data or preprocesses it itself -- it only calls the public
API of ``windowing.py``, ``dataset_builder.py``, ``models`` (via
``models.get_model``), and, for the single purpose of reporting metrics
in real units when the caller passes a fitted scaler,
``preprocessing.inverse_transform`` -- then aggregates the results.

``experiment.py`` is model-agnostic: it never assumes a specific model
(ANN, Gradient Boosting, Gaussian Process, ...). Model selection is
driven entirely by ``config.DEFAULT_MODEL`` (or an explicit
``model_name`` argument) and resolved through ``models.get_model``,
which returns an object implementing the common ``models.BaseModel``
contract (``fit``, ``predict``, ``requires_flat_input``). Adding a new
model to the project never requires changing this module -- only
registering the new model in ``models/registry.py``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from classification import classify_cloud
from config import (
    CLOUD_LABELS,
    CV_N_SPLITS,
    DEFAULT_MODEL,
    FORECAST_HORIZON,
    FORECAST_SCENARIOS,
    GP_MAX_TRAIN_SAMPLES,
    RANDOM_SEED,
    SAVE_JSON,
    SHUFFLE_DATASET,
    SUMMARY_OUTPUT_PATH,
    TRAIN_RATIO,
    WINDOW_CANDIDATES,
)
from dataset_builder import (
    build_dataset,
    coverage_then_random_subsample,
    split_dataset,
    time_series_kfold_indices,
)
from evaluation import evaluate_classification, safe_mape
from export import export_csv
from models import get_model
from preprocessing import inverse_transform
from windowing import create_windows, flatten_windows

logger = logging.getLogger(__name__)

#: Metrics for which a lower value is better.
_LOWER_IS_BETTER = {"rmse", "mae"}
#: Metrics for which a higher value is better.
_HIGHER_IS_BETTER = {"r2"}


def run_window_size_search(
    X: np.ndarray,
    y: np.ndarray,
    window_candidates: tuple[int, ...] = WINDOW_CANDIDATES,
    model_name: str = DEFAULT_MODEL,
    horizon: int = FORECAST_HORIZON,
    train_ratio: float = TRAIN_RATIO,
    shuffle: bool = SHUFFLE_DATASET,
    random_seed: int = RANDOM_SEED,
    selection_metric: str = "rmse",
    target_scaler: Any = None,
    horizon_minutes: int | None = None,
    export_results: bool = True,
) -> dict[str, Any]:
    """Search for the best-performing window size for a given model.

    For every candidate in ``window_candidates``, this builds a
    sliding-window dataset (via ``windowing.create_windows`` and
    ``dataset_builder.build_dataset``), fits a fresh model instance
    (via ``models.get_model``), predicts on the held-out test split,
    and records RMSE/MAE/MAPE/R^2 (regression) plus accuracy/
    precision/recall/F1 (cloud-condition classification, via
    ``classification.classify_cloud`` + ``evaluation.
    evaluate_classification``). The candidate with the best
    ``selection_metric`` is reported as the winner.

    A fresh model instance is created for every candidate (rather than
    reused across the loop), since input dimensionality depends on
    ``window_size``.

    Args:
        X: Preprocessed feature array of shape ``(n_timesteps,
            n_pixels, n_features)``, e.g. as returned by
            ``preprocessing.preprocess_day`` (concatenated across
            days/months as needed by the caller). Not loaded or
            preprocessed by this function. In ``target_scaler``'s
            scaled space if it's given.
        y: Preprocessed target array of shape ``(n_timesteps,
            n_pixels)``. Same scaled-space note as ``X``.
        window_candidates: Window sizes to evaluate. Defaults to
            ``config.WINDOW_CANDIDATES``.
        model_name: Name of the model to use for every candidate, must
            be registered in ``models.registry``. Defaults to
            ``config.DEFAULT_MODEL``.
        horizon: Forecast horizon, held constant across all candidates
            (only window size is varied by this experiment). Defaults
            to ``config.FORECAST_HORIZON``.
        train_ratio: Train/test split ratio forwarded to
            ``dataset_builder.build_dataset``. Defaults to
            ``config.TRAIN_RATIO``.
        shuffle: Whether to shuffle before splitting, forwarded to
            ``dataset_builder.build_dataset``. Defaults to
            ``config.SHUFFLE_DATASET``.
        random_seed: Random seed forwarded to
            ``dataset_builder.build_dataset``. Defaults to
            ``config.RANDOM_SEED``.
        selection_metric: Metric used to pick the best candidate. One
            of ``"rmse"``, ``"mae"`` (lower is better), or ``"r2"``
            (higher is better).
        target_scaler: The fitted scaler ``y``/the model's predictions
            are in the scaled space of (e.g. from
            ``preprocessing.fit_scaler``), or ``None`` if the target
            was never scaled. Fitting/predicting stays in scaled space
            throughout (unaffected by this argument -- consistent with
            what the model actually trains on); only ``y_true``/
            ``y_pred`` are converted back to real units, via
            ``preprocessing.inverse_transform``, right before
            RMSE/MAE/MAPE/R^2 are computed, so the reported metrics --
            and ``window_size_search.json``/``.csv`` -- are in real
            units, comparable to the rest of the pipeline's reporting
            (e.g. the dashboard's MAE). Passing the wrong (or no)
            scaler when the target actually was scaled silently
            reports metrics in the wrong units -- it will not raise an
            error.
        horizon_minutes: Optional label recorded on every result record
            as ``"horizon_minutes"`` (e.g. ``10``/``30``/``60``) when
            this search is one scenario of a multi-horizon run (see
            ``run_multi_horizon_window_size_search``). Left as ``None``
            (the default) for a plain single-horizon search.
        export_results: Whether this call writes its own
            ``window_size_search.csv``/``.json``. Set to ``False`` when
            called from ``run_multi_horizon_window_size_search``, which
            combines every scenario's results into one export instead
            of letting each scenario overwrite the other's file.

    Returns:
        Dictionary with keys:

        - ``"results"``: list of per-candidate result dicts, each with
          ``window_size``, ``model``, ``n_train``, ``n_test``,
          ``rmse``, ``mae``, ``mape``, ``r2``, ``accuracy``,
          ``precision``, ``recall``, ``f1``. Also written, one row per
          candidate (every window size 1..24 with the full metric
          set), to ``window_size_search.csv`` under
          ``config.SUMMARY_OUTPUT_PATH`` (subject to
          ``config.SAVE_CSV``) -- handy for a report/spreadsheet.
        - ``"best"``: the result dict with the best
          ``selection_metric`` value.

    Raises:
        ValueError: If ``window_candidates`` is empty, or if
            ``selection_metric`` is not one of the supported metrics.
    """
    if not window_candidates:
        raise ValueError("window_candidates must contain at least one window size.")
    supported_metrics = _LOWER_IS_BETTER | _HIGHER_IS_BETTER
    if selection_metric not in supported_metrics:
        raise ValueError(
            f"Unsupported selection_metric '{selection_metric}'. "
            f"Supported: {sorted(supported_metrics)}."
        )

    logger.info(
        "Starting window size search: candidates=%s, model='%s'.",
        window_candidates, model_name,
    )

    results: list[dict[str, Any]] = []

    for window_size in window_candidates:
        logger.info("Evaluating window_size=%d.", window_size)

        X_window, y_window, pixel_index, time_index, _, _ = create_windows(
            X, y, window_size=window_size, horizon=horizon,
        )

        model = get_model(model_name)

        dataset = build_dataset(
            X_window,
            y_window,
            pixel_index,
            time_index,
            flatten=model.requires_flat_input,
            train_ratio=train_ratio,
            shuffle=shuffle,
            random_seed=random_seed,
            horizon=horizon,
        )

        X_train_fit, y_train_fit = dataset["X_train"], dataset["y_train"]
        if model_name == "gaussian_process":
            # Exact GP regression is O(n^3)/O(n^2) -- fitting it on the
            # full training split is impractical (see
            # models/gaussian_process.py's module docstring). Recover
            # which physical pixel each training sample belongs to by
            # splitting pixel_index with the exact same (deterministic)
            # parameters build_dataset used internally, then subsample.
            pixel_index_train_split = split_dataset(
                X_window, pixel_index,
                train_ratio=train_ratio, shuffle=shuffle, random_seed=random_seed,
            )[2]
            X_train_fit, y_train_fit = coverage_then_random_subsample(
                X_train_fit, y_train_fit, pixel_index_train_split,
                max_samples=GP_MAX_TRAIN_SAMPLES, random_seed=random_seed,
            )

        model.fit(X_train_fit, y_train_fit)
        y_pred = model.predict(dataset["X_test"])
        y_true = dataset["y_test"]

        # Model fitting/prediction above stays entirely in target_scaler's
        # scaled space (consistent with what it was trained on); convert
        # back to real units only now, right before computing metrics --
        # never touches X_train_fit/y_train_fit/dataset["X_test"] itself.
        if target_scaler is not None:
            y_true = inverse_transform(y_true.reshape(-1, 1), target_scaler).reshape(y_true.shape)
            y_pred = inverse_transform(y_pred.reshape(-1, 1), target_scaler).reshape(y_pred.shape)

        metrics = _compute_metrics(y_true, y_pred)

        # Classification metrics (cloud-condition scheme) alongside the
        # regression ones above, so window_size_search.csv/.json report
        # a full picture per candidate -- not just RMSE/MAE/MAPE/R^2.
        # Reuses the same real-unit y_true/y_pred already converted
        # above.
        cloud_true_class = classify_cloud(y_true)["class_index"]
        cloud_pred_class = classify_cloud(y_pred)["class_index"]
        classification_metrics = evaluate_classification(
            cloud_true_class, cloud_pred_class, labels=CLOUD_LABELS,
        )

        record = {
            "window_size": window_size,
            "model": model_name,
            "horizon_minutes": horizon_minutes,
            "horizon_steps": horizon,
            "n_train": int(X_train_fit.shape[0]),
            "n_test": int(dataset["X_test"].shape[0]),
            **metrics,
            "accuracy": classification_metrics["accuracy"],
            "precision": classification_metrics["precision"],
            "recall": classification_metrics["recall"],
            "f1": classification_metrics["f1"],
        }
        results.append(record)

        logger.info(
            "window_size=%d -> rmse=%.4f, mae=%.4f, mape=%.4f, r2=%.4f, "
            "accuracy=%.4f, precision=%.4f, recall=%.4f, f1=%.4f.",
            window_size, metrics["rmse"], metrics["mae"], metrics["mape"],
            metrics["r2"], classification_metrics["accuracy"],
            classification_metrics["precision"], classification_metrics["recall"],
            classification_metrics["f1"],
        )

    best = _select_best(results, selection_metric)
    logger.info(
        "Best window_size=%d (selection_metric='%s').",
        best["window_size"], selection_metric,
    )

    if export_results:
        # Full metric table -- every window size candidate (1..24 by
        # default) with every metric in one row each -- for a report or
        # spreadsheet, independent of window_size_search.json below.
        export_csv(pd.DataFrame(results), SUMMARY_OUTPUT_PATH / "window_size_search.csv")

        if SAVE_JSON:
            _save_experiment_results({"results": results, "best": best})

    return {"results": results, "best": best}


def run_multi_horizon_window_size_search(
    X: np.ndarray,
    y: np.ndarray,
    scenarios: tuple[tuple[int, int], ...] = FORECAST_SCENARIOS,
    window_candidates: tuple[int, ...] = WINDOW_CANDIDATES,
    model_name: str = DEFAULT_MODEL,
    train_ratio: float = TRAIN_RATIO,
    shuffle: bool = SHUFFLE_DATASET,
    random_seed: int = RANDOM_SEED,
    selection_metric: str = "rmse",
    target_scaler: Any = None,
) -> dict[str, Any]:
    """Run :func:`run_window_size_search` once per lead-time scenario.

    For every ``(horizon_minutes, horizon_steps)`` pair in
    ``scenarios`` (default ``config.FORECAST_SCENARIOS`` -- 10/30/60
    minute lead times), runs a full window size search (every candidate
    in ``window_candidates``) targeting that scenario's own
    ``horizon_steps``, so the winning window size can differ per
    scenario (a 10-minute-ahead forecast and a 60-minute-ahead forecast
    don't necessarily want the same amount of input history). Every
    scenario's per-candidate records are combined into a single CSV
    (``window_size_search.csv``, one row per ``(horizon_minutes,
    window_size)`` combination -- e.g. 3 scenarios x 24 window sizes =
    72 rows by default) instead of each scenario overwriting the same
    file.

    Args:
        X, y: Preprocessed feature/target arrays -- see
            :func:`run_window_size_search`.
        scenarios: ``(horizon_minutes, horizon_steps)`` pairs to search.
            Defaults to ``config.FORECAST_SCENARIOS``.
        window_candidates, model_name, train_ratio, shuffle,
            random_seed, selection_metric, target_scaler: Forwarded to
            :func:`run_window_size_search` for every scenario -- see
            that function's docstring.

    Returns:
        Dictionary with keys:

        - ``"results"``: combined list of every scenario's per-candidate
          result dicts (each tagged with ``"horizon_minutes"``/
          ``"horizon_steps"``), also written to ``window_size_search.csv``.
        - ``"scenarios"``: dict keyed by ``horizon_minutes``, each value
          the corresponding scenario's ``{"results": ..., "best": ...}``
          (same shape :func:`run_window_size_search` returns on its
          own) -- ``scenarios[minutes]["best"]["window_size"]`` is what
          the rest of the pipeline should use for that scenario.
    """
    if not scenarios:
        raise ValueError("scenarios must contain at least one (minutes, steps) pair.")

    all_results: list[dict[str, Any]] = []
    per_scenario: dict[int, dict[str, Any]] = {}

    for horizon_minutes, horizon_steps in scenarios:
        logger.info(
            "Window size search: skenario horizon=%d menit (%d langkah).",
            horizon_minutes, horizon_steps,
        )
        scenario_result = run_window_size_search(
            X, y,
            window_candidates=window_candidates,
            model_name=model_name,
            horizon=horizon_steps,
            train_ratio=train_ratio,
            shuffle=shuffle,
            random_seed=random_seed,
            selection_metric=selection_metric,
            target_scaler=target_scaler,
            horizon_minutes=horizon_minutes,
            export_results=False,
        )
        all_results.extend(scenario_result["results"])
        per_scenario[horizon_minutes] = scenario_result
        logger.info(
            "Skenario %d menit: best window_size=%d (%s=%.4f).",
            horizon_minutes, scenario_result["best"]["window_size"],
            selection_metric, scenario_result["best"][selection_metric],
        )

    export_csv(pd.DataFrame(all_results), SUMMARY_OUTPUT_PATH / "window_size_search.csv")

    if SAVE_JSON:
        _save_experiment_results({
            "results": all_results,
            "scenarios": {
                str(minutes): result["best"] for minutes, result in per_scenario.items()
            },
        })

    return {"results": all_results, "scenarios": per_scenario}


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE, MAE, MAPE, and R^2 for a set of predictions.

    MAPE is computed via ``evaluation.safe_mape``, which excludes
    near-zero ``y_true`` samples (missing-data fallback artifacts --
    see that function's docstring) before dividing, instead of
    ``sklearn``'s raw ``mean_absolute_percentage_error``: a handful of
    near-zero true values otherwise inflate MAPE by many orders of
    magnitude (e.g. 1e14) and make it meaningless for a report. RMSE/
    MAE/R^2 are unaffected and still use every sample.

    Args:
        y_true: Ground-truth targets, shape ``(n_samples,)``.
        y_pred: Predicted targets, shape ``(n_samples,)``.

    Returns:
        Dictionary with keys ``"rmse"``, ``"mae"``, ``"mape"``, ``"r2"``.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = safe_mape(y_true, y_pred)
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


def _select_best(results: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    """Select the best result record according to ``metric``.

    Args:
        results: List of per-candidate result dicts, each containing
            ``metric`` as a key.
        metric: Metric name to optimize. Lower is better for
            ``"rmse"``/``"mae"``, higher is better for ``"r2"``.

    Returns:
        The single best result dict.
    """
    reverse = metric in _HIGHER_IS_BETTER
    return sorted(results, key=lambda record: record[metric], reverse=reverse)[0]


def _save_experiment_results(payload: dict[str, Any], filename: str = "window_size_search.json") -> None:
    """Persist experiment results as a JSON summary file.

    Args:
        payload: Dictionary to serialize, as produced by
            :func:`run_window_size_search` or
            :func:`run_cross_validation`.
        filename: Output filename within ``config.SUMMARY_OUTPUT_PATH``.
    """
    SUMMARY_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    output_file = SUMMARY_OUTPUT_PATH / filename

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("Saved experiment summary to '%s'.", output_file)


def run_cross_validation(
    X_window: np.ndarray,
    y_window: np.ndarray,
    pixel_index: np.ndarray,
    model_name: str = DEFAULT_MODEL,
    n_splits: int = CV_N_SPLITS,
    random_seed: int = RANDOM_SEED,
    target_scaler: Any = None,
) -> dict[str, Any]:
    """Cross-validate one (window size, model) combination.

    A single train/test split (``dataset_builder.split_dataset``, used
    by ``build_dataset``/``run_window_size_search``) reports performance
    on just one held-out slice, which can look better or worse than
    typical purely by chance. This instead uses
    ``dataset_builder.time_series_kfold_indices`` (blocked per-pixel
    folds, safe for this project's overlapping-window, pixel-major
    samples -- see that function's docstring) to fit and evaluate
    ``n_splits`` independent times, reporting per-fold RMSE/MAE/R^2 plus
    their mean/std -- a more robust estimate of how the (window size,
    model) combination actually performs, and a sense of how much that
    performance varies.

    A fresh model instance is fit for every fold (models don't support
    incremental/reset fitting in this project).

    Args:
        X_window: Already-windowed feature array, shape ``(n_samples,
            window_size, n_features)``, as returned by
            ``windowing.create_windows`` -- e.g. reuse the same
            ``X_window_train`` the main pipeline already built for
            ``config.WINDOW_SIZE``, rather than re-windowing.
        y_window: Already-windowed target array, shape ``(n_samples,)``,
            matching ``X_window``.
        pixel_index: Source-pixel index per sample, shape
            ``(n_samples,)``, as returned by ``windowing.create_windows``
            alongside ``X_window``/``y_window`` -- must be in the same
            (chronological-within-each-pixel) order, not shuffled.
        model_name: Name of the model to cross-validate, must be
            registered in ``models.registry``. Defaults to
            ``config.DEFAULT_MODEL``.
        n_splits: Number of folds. Defaults to ``config.CV_N_SPLITS``.
        random_seed: Base seed; fold ``i`` of a "gaussian_process" model
            seeds its subsample with ``random_seed + i`` so folds don't
            all draw the identical subsample pattern. Defaults to
            ``config.RANDOM_SEED``.
        target_scaler: The fitted scaler ``y_window``/the model's
            predictions are in the scaled space of, or ``None`` if the
            target was never scaled. Fitting/predicting stays in scaled
            space throughout (unaffected by this argument); only
            ``y_true``/``y_pred`` are converted back to real units,
            via ``preprocessing.inverse_transform``, right before
            RMSE/MAE/R^2 are computed for each fold -- same convention
            as :func:`run_window_size_search`.

    Returns:
        Dictionary with keys:

        - ``"window_size"``, ``"model"``, ``"n_splits"``.
        - ``"folds"``: list of per-fold dicts, each with ``fold``,
          ``n_train``, ``n_test``, ``rmse``, ``mae``, ``r2``.
        - ``"summary"``: dict keyed by ``"rmse"``/``"mae"``/``"r2"``,
          each ``{"mean": ..., "std": ...}`` across folds.

    Raises:
        ValueError: If ``n_splits < 2`` or any pixel has fewer than
            ``n_splits`` samples (see
            ``dataset_builder.time_series_kfold_indices``).
    """
    window_size = X_window.shape[1]
    model_probe = get_model(model_name)
    X_for_fit = flatten_windows(X_window) if model_probe.requires_flat_input else X_window

    splits = time_series_kfold_indices(pixel_index, n_splits)

    logger.info(
        "Starting cross-validation: window_size=%d, model='%s', n_splits=%d.",
        window_size, model_name, n_splits,
    )

    fold_results: list[dict[str, Any]] = []
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        model = get_model(model_name)  # fresh instance per fold
        X_train_fit, y_train_fit = X_for_fit[train_idx], y_window[train_idx]

        if model_name == "gaussian_process":
            # Exact GP regression is O(n^3)/O(n^2) -- see
            # models/gaussian_process.py's module docstring.
            X_train_fit, y_train_fit = coverage_then_random_subsample(
                X_train_fit, y_train_fit, pixel_index[train_idx],
                max_samples=GP_MAX_TRAIN_SAMPLES, random_seed=random_seed + fold_i,
            )

        model.fit(X_train_fit, y_train_fit)
        y_pred = model.predict(X_for_fit[test_idx])
        y_true = y_window[test_idx]

        if target_scaler is not None:
            y_true = inverse_transform(y_true.reshape(-1, 1), target_scaler).reshape(y_true.shape)
            y_pred = inverse_transform(y_pred.reshape(-1, 1), target_scaler).reshape(y_pred.shape)

        metrics = _compute_metrics(y_true, y_pred)
        fold_results.append({
            "fold": fold_i,
            "n_train": int(X_train_fit.shape[0]),
            "n_test": int(test_idx.shape[0]),
            **metrics,
        })
        logger.info(
            "fold=%d -> rmse=%.4f, mae=%.4f, r2=%.4f.",
            fold_i, metrics["rmse"], metrics["mae"], metrics["r2"],
        )

    summary = {
        key: {
            "mean": float(np.mean([r[key] for r in fold_results])),
            "std": float(np.std([r[key] for r in fold_results])),
        }
        for key in ("rmse", "mae", "r2")
    }
    logger.info(
        "Cross-validation summary (window_size=%d, model='%s'): "
        "rmse=%.4f\u00b1%.4f, mae=%.4f\u00b1%.4f, r2=%.4f\u00b1%.4f.",
        window_size, model_name,
        summary["rmse"]["mean"], summary["rmse"]["std"],
        summary["mae"]["mean"], summary["mae"]["std"],
        summary["r2"]["mean"], summary["r2"]["std"],
    )

    result = {
        "window_size": window_size,
        "model": model_name,
        "n_splits": n_splits,
        "folds": fold_results,
        "summary": summary,
    }

    if SAVE_JSON:
        _save_experiment_results(result, filename="cross_validation.json")

    return result