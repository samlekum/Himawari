"""Orchestration entry point for AI Himawari Forecasting.

This module is responsible ONLY for wiring together the public API of
every other module in this project, in the order described below. It
contains no business logic of its own: no data transformation, no
modeling logic, no metric computation. Every step is a thin call into
``loader``, ``preprocessing``, ``windowing``, ``dataset_builder``,
``experiment``, ``training``, ``forecasting``, ``classification``,
``evaluation``, ``export``, or ``visualization``.

Pipeline::

    Load Data
        -> Preprocessing
        -> for each (horizon_minutes, horizon_steps) scenario in
           config.FORECAST_SCENARIOS (10/30/60-minute lead times):
               -> Window Size Search (if config.RUN_EXPERIMENT) --
                  picks THIS SCENARIO's window size
               -> Windowing (this scenario's window size + horizon)
               -> Dataset Builder
               -> Cross Validation (if config.RUN_CROSS_VALIDATION)
               -> Training (or Load Model if config.RUN_TRAINING is
                  False) -- this scenario's own model, saved as
                  f"{DEFAULT_MODEL}_{horizon_minutes}min"
               -> Forecasting
               -> Classification
               -> Evaluation
               -> Export (filenames suffixed f"_{horizon_minutes}min")
               -> Visualization (own dashboard/GIF, rolled out exactly
                  horizon_steps native steps)
        -> Finish

See ``_run_scenario`` for the per-scenario body (everything from
Windowing through Visualization above) and ``experiment.
run_multi_horizon_window_size_search`` for the window size search.
Every step is gated by its corresponding ``config.RUN_*`` toggle
(``RUN_EXPERIMENT``, ``RUN_CROSS_VALIDATION``, ``RUN_TRAINING``,
``RUN_FORECAST``, ``RUN_CLASSIFICATION``, ``RUN_EVALUATION``,
``RUN_EXPORT``, ``RUN_VISUALIZATION``), applied identically to every
scenario; a disabled step is skipped (and logged) rather than executed.

Two design decisions worth calling out explicitly, since they are not
fully spelled out by the individual modules' docstrings:

- ``experiment.run_window_size_search`` (called once per scenario by
  ``run_multi_horizon_window_size_search``) operates on preprocessed
  arrays *before* windowing (it calls ``windowing.create_windows``
  internally for every candidate window size). ``_run_scenario`` passes
  that scenario's winning ``window_size`` on to the real
  ``windowing.create_windows`` calls that build
  ``X_window_train``/``X_window_test`` -- so every later step for that
  scenario (dataset builder, cross-validation, training, forecasting,
  the dashboard/GIF) trains and predicts at the window size the search
  actually found best for THAT lead time, not a single
  ``config.WINDOW_SIZE`` shared by every scenario.
  ``config.WINDOW_SIZE`` is only used as a fallback when
  ``config.RUN_EXPERIMENT`` is ``False``.
- ``visualization.py`` exposes a single combined dashboard renderer,
  ``render_six_panel``, that expects 2D ``(n_lat, n_lon)`` grids.
  ``y_pred_test`` cannot supply that directly: ``dataset_builder.
  split_dataset`` operates on samples ordered pixel-major (see
  ``windowing.create_windows``, which loops pixels in the outer loop)
  with ``SHUFFLE_DATASET`` False, so the held-out test fraction is a
  slice of *pixels*, not a full spatial grid at a single timestep.
  Grid-based visualization therefore uses a separate, genuine
  full-grid forecast instead: the most recent ``window_size``
  timesteps of ``X_test_full`` for *every* pixel
  (``_latest_pixel_window`` -- a display-only reshape, not a
  scientific transformation), fed through ``forecasting.
  bias_correction`` (the project's primary forecasting method; see
  ``forecasting.py``'s module docstring), then reshaped back onto the
  ``(n_lat, n_lon)`` grid recovered from a reference test dataset via
  ``preprocessing.subset_bandung`` (``_grid_shape`` -- reuses
  preprocessing's existing public API only; no new spatial logic is
  introduced here). The dashboard's "input" panel is the most recently
  *observed* target grid (``y_test_full[-1]``, reshaped the same way --
  valid because ``config.INPUT_FEATURES == (config.TARGET_VARIABLE,)``,
  so ``X_test_full``'s sole feature and ``y_test_full`` are the same
  variable). The dashboard has no genuine future ground truth to show
  as "actual" (the forecast horizon extends past the end of the loaded
  test period), so ``actual_grid`` is always passed as ``None`` --
  ``render_six_panel`` already renders a "Data aktual belum tersedia"
  placeholder for that case. The same ``frames`` rollout also feeds
  ``visualization.create_forecast_gif`` (animated, using the same
  Bandung-boundary map style as the dashboard's continuous panels),
  and ``evaluation_result``'s per-scheme confusion matrices feed
  ``visualization.plot_confusion_matrix`` (dark-themed to match, but
  not map-based, so no geojson overlay).
- The forecasting step for the test set calls
  ``forecasting.forecast_only`` (a plain one-step prediction), never
  ``forecasting.forecast`` with ``"bias_correction"``/
  ``"teacher_forcing"``. Those two update methods perform an iterative
  multi-step rollout from a single *most-recent window per pixel*
  (shape ``(n_pixels, window_size, n_features)`` -> ``(horizon,
  n_pixels)``). ``X_window_test`` is a different shape of thing
  entirely: ``dataset_builder``'s train/test split of
  ``windowing.create_windows``'s output, i.e. many independent
  windowed samples mixed across pixels and timesteps. Feeding that
  into ``bias_correction``/``teacher_forcing`` would silently
  misinterpret ``n_samples`` as ``n_pixels`` and return a
  ``(horizon, n_samples)`` array instead of the ``(n_samples,)`` shape
  ``evaluation.py`` needs to compare against ``y_window_test``.
  Genuine multi-step rollout forecasting (from an actual latest
  window per pixel) is a distinct use case this pipeline does not
  currently produce test-set metrics for, and would need its own,
  separately-sourced input.
"""

from __future__ import annotations

import logging
import random
import sys
from typing import Any

import numpy as np
import xarray as xr

import classification
import dataset_builder
import evaluation
import export
import experiment
import forecasting
import loader
import preprocessing
import run_metrics
import training
import visualization
import windowing
from config import (
    CLOUD_LABELS,
    CV_N_SPLITS,
    DEFAULT_MODEL,
    EXPERIMENT_ID,
    FORECAST_SCENARIOS,
    GIF_OUTPUT_PATH,
    GP_MAX_TRAIN_SAMPLES,
    LOG_OUTPUT_PATH,
    PROJECT_NAME,
    RANDOM_SEED,
    RISK_LABELS,
    RUN_CLASSIFICATION,
    RUN_CROSS_VALIDATION,
    RUN_EVALUATION,
    RUN_EXPERIMENT,
    RUN_EXPORT,
    RUN_FORECAST,
    RUN_STEP_BIAS_CALIBRATION,
    RUN_TRAINING,
    RUN_VISUALIZATION,
    SAVE_LOG,
    STEP_BIAS_CALIBRATION_SAMPLES,
    SUMMARY_OUTPUT_PATH,
    TEST_MONTHS,
    TRAIN_MONTHS,
    VISUALIZATION_ANCHOR_LAST,
    VISUALIZATION_CONFUSION_MATRIX_PATH,
    VISUALIZATION_DASHBOARD_PATH,
    VISUALIZATION_N_SAMPLES,
    VISUALIZATION_T0_INDICES,
    VISUALIZATION_T0_MODE,
    WINDOW_SIZE,
)
from models import BaseModel, get_model

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure logging for the pipeline run.

    Always logs to stdout. Additionally logs to a file under
    ``config.LOG_OUTPUT_PATH`` when ``config.SAVE_LOG`` is ``True``,
    matching the write/skip-but-log convention used elsewhere in the
    project (e.g. ``training.save_model``).
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if SAVE_LOG:
        LOG_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.FileHandler(LOG_OUTPUT_PATH / f"{EXPERIMENT_ID}.log", encoding="utf-8")
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def _load_months(months: tuple[str, ...]) -> list[xr.Dataset]:
    """Load and flatten every NetCDF dataset for a set of months.

    Thin wrapper around ``loader.load_month``: flattens its
    ``{day: [Dataset, ...]}`` return value into a single chronological
    list, and concatenates that across every requested month in the
    order given.

    Args:
        months: Month folder names (YYYYMM), in the chronological
            order they should appear in the returned list.

    Returns:
        Chronologically ordered list of ``xarray.Dataset`` objects
        across every day of every requested month.
    """
    all_datasets: list[xr.Dataset] = []
    for month in months:
        month_data = loader.load_month(month)
        for day in month_data:
            all_datasets.extend(month_data[day])
    return all_datasets


def _latest_pixel_window(
    X_full: np.ndarray, window_size: int, end_index: int | None = None,
) -> np.ndarray:
    """Extract a per-pixel window ending at ``end_index`` from a full timestep array.

    Display-only reshape (not a scientific transformation): it exists
    purely to assemble the ``(n_pixels, window_size, n_features)`` input shape that
    ``forecasting.bias_correction``/``forecasting.teacher_forcing``
    expect for a "most recent window per pixel" forecast -- used here to
    produce the full-grid forecast that grid-based visualizations
    (dashboard, GIF, interval comparison) need.

    Args:
        X_full: Feature array of shape ``(n_timesteps, n_pixels,
            n_features)``, e.g. ``preprocessing.preprocess_day``'s
            output for the test period.
        window_size: Number of most recent timesteps to keep as the
            window (matches the window size used to train the model).
        end_index: Exclusive end index into ``X_full``'s timestep axis
            -- the window covers ``X_full[end_index - window_size:
            end_index]``. Defaults to ``X_full.shape[0]`` (the very
            last window), which matches this function's original,
            end-of-array-only behavior. Pass an earlier index to build
            a window as of an earlier point in the test period (e.g.
            for backtesting against real, later ground truth).

    Returns:
        Array of shape ``(n_pixels, window_size, n_features)``: for
        every pixel, its ``window_size`` observed timesteps ending
        just before ``end_index``.

    Raises:
        ValueError: If ``end_index`` (or ``X_full``'s length, when
            ``end_index`` is unset) is smaller than ``window_size``.
    """
    if end_index is None:
        end_index = X_full.shape[0]
    if end_index < window_size:
        raise ValueError(
            f"Not enough timesteps ({end_index}) before end_index to build "
            f"a window of size {window_size}."
        )
    return X_full[end_index - window_size:end_index, :, :].transpose(1, 0, 2)


def _grid_shape(reference_dataset: xr.Dataset, n_pixels: int) -> tuple[int, int]:
    """Recover the ``(n_lat, n_lon)`` grid shape backing a flattened pixel axis.

    Display-only lookup (not a scientific transformation): it re-derives
    the spatial grid dimensions from the same Bandung subset that
    ``preprocessing.py`` already flattens onto the ``pixel`` axis, using
    only preprocessing's existing public ``subset_bandung`` function --
    no new spatial logic is introduced here. This is what lets
    grid-based visualization functions (which expect a 2D ``(n_lat,
    n_lon)`` array) receive a correctly-shaped array.

    Args:
        reference_dataset: Any raw single-timestamp dataset covering the
            same region as the flattened pixel axis (e.g. the most
            recent dataset of the test period).
        n_pixels: Expected pixel count (the flattened axis size), used
            only to validate the recovered grid shape.

    Returns:
        Tuple ``(n_lat, n_lon)``.

    Raises:
        ValueError: If ``n_lat * n_lon`` does not match ``n_pixels``,
            meaning ``reference_dataset``'s grid does not match the one
            the flattened pixel axis was actually built from.
    """
    subset = preprocessing.subset_bandung(reference_dataset)
    n_lat = subset.sizes["latitude"]
    n_lon = subset.sizes["longitude"]

    if n_lat * n_lon != n_pixels:
        raise ValueError(
            f"Recovered grid shape ({n_lat}, {n_lon}) = {n_lat * n_lon} "
            f"pixel(s) does not match the flattened pixel axis size "
            f"({n_pixels})."
        )
    return n_lat, n_lon


def _extract_timestamp(ds: xr.Dataset) -> np.datetime64 | None:
    """Read a real datetime off a dataset's time-like coordinate, if any.

    Display-only helper used by ``_forecast_labels``/``_detect_interval_minutes``.
    Tries a few common coordinate names;
    returns ``None`` if none are present or aren't ``datetime64``,
    since ``windowing.py`` itself only tracks an integer
    ``time_index``, not real timestamps, so not every dataset is
    guaranteed to carry one.

    Args:
        ds: A single-timestep dataset (e.g. one entry of
            ``test_datasets``).

    Returns:
        The dataset's timestamp, or ``None`` if it can't be found.
    """
    for key in ("time", "Time", "t"):
        if key in ds.coords:
            value = np.atleast_1d(ds.coords[key].values)[0]
            if np.issubdtype(np.asarray(value).dtype, np.datetime64):
                return value
    return None


def _detect_interval_minutes(test_datasets: list[xr.Dataset]) -> float:
    """Detect the real time interval (minutes) between the two most
    recent test datasets.

    Display-only helper shared by ``_forecast_labels`` and the
    visualization block below (both need to know the real cadence,
    once, before deciding how many rollout steps to request). Falls
    back to 10.0 (Himawari Band-13's standard full-disk cadence) when
    a real ``time`` coordinate isn't present -- adjust if your data's
    actual cadence differs.

    Args:
        test_datasets: Chronologically ordered test-period datasets.

    Returns:
        Interval in minutes.
    """
    t0_time = _extract_timestamp(test_datasets[-1])
    if t0_time is not None and len(test_datasets) > 1:
        prev_time = _extract_timestamp(test_datasets[-2])
        if prev_time is not None:
            delta = (t0_time - prev_time) / np.timedelta64(1, "m")
            if delta > 0:
                return float(delta)
    return 10.0


def _select_t0_indices(
    test_datasets: list[xr.Dataset], horizon_steps: int, min_t0_index: int,
    mode: str, n_samples: int, manual_indices: tuple[int, ...],
    anchor_last: bool = False,
) -> list[int]:
    """Select one or more t0 indices for the forecast dashboard/GIF.

    Implements ``config.VISUALIZATION_T0_MODE``/``VISUALIZATION_N_SAMPLES``/
    ``VISUALIZATION_T0_INDICES``/``VISUALIZATION_ANCHOR_LAST``: lets a
    single run produce dashboards/GIFs for more than one t0 (e.g. for
    research/demo purposes) without code changes. ``test_datasets`` is
    chronologically ordered (see ``loader.load_day``), so t0 indices
    double as a real backtest point whenever enough later data exists;
    ``_render_forecast_sample`` uses that per-frame, falling back to
    "Forecast Mode" placeholders where it doesn't. Never touches array
    values.

    Args:
        test_datasets: Chronologically ordered test-period datasets.
        horizon_steps: Rollout length the main dashboard/GIF will use --
            determines how much "future" room a fully-backtestable t0
            needs.
        min_t0_index: Smallest allowed t0 (needs ``window_size - 1``
            timesteps of history before it).
        mode: One of ``"last"``, ``"first"``, ``"random"``, ``"manual"``
            (see ``config.VISUALIZATION_T0_MODE``).
        n_samples: How many t0 indices to return in total, including the
            anchor when ``anchor_last`` is set (ignored for ``"manual"``,
            where the count is ``len(manual_indices)``).
        manual_indices: Explicit t0 indices, used only when
            ``mode == "manual"``.
        anchor_last: If True, always include the single most recent
            valid t0 as one of the samples (first in the returned
            list), filling the rest per ``mode``. No-op for
            ``mode == "last"`` (redundant) and ``mode == "manual"``.

    Returns:
        List of t0 indices, each within ``[min_t0_index, len(test_datasets)
        - 1]`` -- but not necessarily with ``horizon_steps`` of real future
        data available (degrades gracefully per-frame, see
        ``_render_forecast_sample``).

    Raises:
        ValueError: If ``mode`` isn't recognized, the test period is too
            short to build even one window, or (for ``"manual"``) no
            index in ``manual_indices`` falls in the valid range.
    """
    last_index = len(test_datasets) - 1
    if last_index < min_t0_index:
        raise ValueError(
            f"Test period only has {len(test_datasets)} dataset(s), not "
            f"enough to build even one window (need >= {min_t0_index + 1})."
        )

    if mode == "manual":
        selected = []
        for idx in manual_indices:
            if min_t0_index <= idx <= last_index:
                selected.append(idx)
            else:
                logger.warning(
                    "VISUALIZATION_T0_INDICES: index %d di luar rentang "
                    "valid [%d, %d]; dilewati.", idx, min_t0_index, last_index,
                )
        if not selected:
            raise ValueError(
                "VISUALIZATION_T0_MODE is 'manual' but no index in "
                "VISUALIZATION_T0_INDICES falls in the valid range "
                f"[{min_t0_index}, {last_index}]."
            )
        return selected

    max_valid_t0 = last_index - horizon_steps  # has full backtest room
    if max_valid_t0 < min_t0_index:
        # No t0 anywhere has enough real future data -- fall back to the
        # single most recent index (Forecast Mode, no Aktual/MAE),
        # regardless of mode.
        logger.info(
            "Test period cuma %d dataset, kurang dari %d langkah yang "
            "dibutuhkan (t0 + horizon) untuk backtest; dashboard akan "
            "tampil 'Forecast Mode' tanpa Aktual/MAE.",
            len(test_datasets), horizon_steps + min_t0_index + 1,
        )
        return [last_index]

    n_samples = max(1, n_samples)

    # "last" mode already starts at max_valid_t0, so anchoring would
    # just duplicate its own first pick -- skip it there.
    anchor = [max_valid_t0] if (anchor_last and mode != "last") else []
    remaining = max(0, n_samples - len(anchor))

    if mode == "last":
        indices, idx = [], max_valid_t0
        while idx >= min_t0_index and len(indices) < n_samples:
            indices.append(idx)
            idx -= horizon_steps
        return indices

    if mode == "first":
        indices, idx = [], min_t0_index
        while idx <= max_valid_t0 and len(indices) < remaining:
            if idx not in anchor:
                indices.append(idx)
            idx += horizon_steps
        return anchor + indices

    if mode == "random":
        pool = [i for i in range(min_t0_index, max_valid_t0 + 1) if i not in anchor]
        random.Random(RANDOM_SEED).shuffle(pool)
        picked = sorted(pool[:min(remaining, len(pool))])
        return anchor + picked

    raise ValueError(
        f"Unknown VISUALIZATION_T0_MODE: {mode!r} (expected 'last', "
        "'first', 'random', or 'manual')."
    )


def _calibrate_step_bias(
    model: BaseModel,
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    train_datasets: list[xr.Dataset],
    horizon_steps: int,
    window_size: int,
    baseline_bias: float | np.ndarray,
    n_samples: int,
) -> np.ndarray | None:
    """Estimate, then validate, a per-rollout-step bias from TRAIN_MONTHS data.

    A single scalar/per-pixel bias (``evaluation.estimate_bias``) only
    corrects a model's *constant* one-step offset; it can't fix error
    that grows or shrinks specifically with how many autoregressive
    rollout steps have been taken (drift). This estimates that
    step-specific residual: picks several t0 points within
    TRAIN_MONTHS's own timeline (reusing the same ``_select_t0_indices``/
    ``_latest_pixel_window`` machinery ``_render_forecast_sample`` uses
    for TEST_MONTHS), rolls the model forward ``horizon_steps`` steps
    from each -- already corrected by ``baseline_bias``, so this
    measures only the *additional* step-specific drift on top of it.

    Estimating from only a handful of t0's is noisy, and a pattern
    calibrated on TRAIN_MONTHS isn't guaranteed to transfer to
    TEST_MONTHS (different time period/conditions) -- applying an
    unvalidated correction can easily make things worse, not better.
    So the t0 samples are split in half: one half estimates a
    candidate ``step_bias``, the other half (never used to fit it)
    checks whether actually applying it lowers rollout MAE versus
    ``baseline_bias`` alone. Only returned if it demonstrably helps.

    Args:
        model: Trained forecasting model.
        X_train_full, y_train_full: Full TRAIN_MONTHS feature/target
            arrays, index-aligned with ``train_datasets``.
        train_datasets: Chronologically ordered TRAIN_MONTHS datasets.
        horizon_steps: Rollout length to calibrate (must match what the
            dashboard/GIF actually uses -- interval-comparison scenarios
            with a shorter horizon truncate this instead of
            recalibrating).
        window_size: Model input window size.
        baseline_bias: The already-estimated global/per-pixel bias,
            applied as the calibration rollout's baseline correction so
            this measures only the residual beyond it.
        n_samples: Total t0 starting points to use, split roughly in
            half between fitting the candidate and validating it.

    Returns:
        Array of shape ``(horizon_steps,)`` if a candidate was found
        AND it measurably reduced MAE on its held-out check half.
        ``None`` if TRAIN_MONTHS isn't long enough to calibrate/validate
        at all, or if the candidate did not improve on the check half
        (constant bias alone is kept instead).
    """
    min_t0_index = window_size - 1
    try:
        t0_indices = _select_t0_indices(
            train_datasets, horizon_steps, min_t0_index,
            mode="random", n_samples=max(2, n_samples), manual_indices=(),
        )
    except ValueError:
        logger.warning(
            "Could not select calibration t0(s) from TRAIN_MONTHS for "
            "step-bias estimation; skipping (constant bias only).",
            exc_info=True,
        )
        return None

    split = max(1, len(t0_indices) // 2)
    fit_t0s, check_t0s = t0_indices[:split], t0_indices[split:] or t0_indices[:split]

    def _rollout(t0_list: list[int], step_bias_arr: np.ndarray | None):
        true_list, pred_list = [], []
        for t0_index in t0_list:
            if t0_index + horizon_steps >= len(train_datasets):
                continue  # not enough future room for this t0
            window = _latest_pixel_window(
                X_train_full, window_size=window_size, end_index=t0_index + 1,
            )
            rollout = forecasting.bias_correction(
                model=model, X=window, horizon=horizon_steps,
                bias=baseline_bias, step_bias=step_bias_arr,
            )  # shape (horizon_steps, n_pixels)
            true_future = np.stack(
                [y_train_full[t0_index + 1 + i] for i in range(horizon_steps)],
            )  # shape (horizon_steps, n_pixels)
            true_list.append(true_future)
            pred_list.append(rollout)
        if not true_list:
            return None, None
        return np.concatenate(true_list, axis=1), np.concatenate(pred_list, axis=1)

    y_true_fit, y_pred_fit = _rollout(fit_t0s, None)
    if y_true_fit is None:
        logger.warning(
            "No usable calibration t0 (fit half) had enough future room "
            "in TRAIN_MONTHS; skipping step-bias estimation."
        )
        return None
    step_bias_candidate = evaluation.estimate_step_bias(y_true_fit, y_pred_fit)

    y_true_check, y_pred_check_baseline = _rollout(check_t0s, None)
    if y_true_check is None:
        logger.warning(
            "No usable calibration t0 (check half) had enough future "
            "room in TRAIN_MONTHS; can't validate step-bias, skipping "
            "it to be safe."
        )
        return None
    _, y_pred_check_corrected = _rollout(check_t0s, step_bias_candidate)

    mae_baseline = float(np.mean(np.abs(y_true_check - y_pred_check_baseline)))
    mae_corrected = float(np.mean(np.abs(y_true_check - y_pred_check_corrected)))
    logger.info(
        "Step-bias validasi (held-out t0 dari TRAIN_MONTHS, tidak "
        "dipakai buat fit): MAE tanpa step-bias=%.4f, dengan "
        "step-bias=%.4f.", mae_baseline, mae_corrected,
    )
    if mae_corrected >= mae_baseline:
        logger.warning(
            "Step-bias TIDAK memperbaiki MAE di data held-out (%.4f -> "
            "%.4f); dilewati, tetap pakai bias konstan saja.",
            mae_baseline, mae_corrected,
        )
        return None

    logger.info(
        "Step-bias tervalidasi (MAE held-out %.4f -> %.4f), dipakai. "
        "Nilai: %s", mae_baseline, mae_corrected,
        np.array2string(step_bias_candidate, precision=4, max_line_width=10_000),
    )
    return step_bias_candidate


def _render_forecast_sample(
    model: BaseModel,
    X_test_full: np.ndarray,
    y_test_full: np.ndarray,
    test_datasets: list[xr.Dataset],
    t0_index: int,
    horizon_steps: int,
    window_size: int,
    dashboard_dir: "Path",
    gif_dir: "Path",
    filename_suffix: str = "",
    bias: float | np.ndarray = 0.0,
    step_bias: np.ndarray | None = None,
    target_scaler: "preprocessing.Scaler" = None,
) -> None:
    """Render the dashboard PNG and GIF for one t0.

    Factored out of the main visualization block so
    ``config.VISUALIZATION_N_SAMPLES`` can render more than one t0
    sample per run without duplicating this logic. Ground truth is used
    wherever ``t0_index + step`` still falls inside ``test_datasets``
    (genuine backtest); beyond that, panels fall back to "Forecast Mode"
    placeholders, same as a t0 with no future data at all.

    Args:
        model: Trained forecasting model.
        X_test_full, y_test_full: Full test-period feature/target
            arrays, index-aligned with ``test_datasets``. In
            ``target_scaler``'s scaled space if it's given (the model's
            native training space) -- same as ``model``/``bias``/
            ``step_bias`` all operate in.
        test_datasets: Chronologically ordered test-period datasets.
        t0_index: Index to use as t0 for this sample.
        horizon_steps: Number of rollout steps for the main dashboard/GIF.
        window_size: Model input window size.
        dashboard_dir, gif_dir: Output directories (already created).
        filename_suffix: Appended to output filenames (e.g.
            ``f"_t0-{t0_index}"``) so multiple samples don't overwrite
            each other.
        bias: Scalar or per-pixel (shape ``(n_pixels,)``) bias
            correction forwarded to every ``forecasting.bias_correction``
            call here (see ``evaluation.estimate_bias``). Defaults to
            ``0.0`` (uncorrected rollout).
        step_bias: Optional per-rollout-step correction, shape
            ``(horizon_steps,)`` (see ``evaluation.estimate_step_bias``/
            ``_calibrate_step_bias``). Defaults to ``None`` (no
            per-step correction).
        target_scaler: The fitted scaler ``y_test_full``/the model's
            predictions are in the scaled space of (e.g. from
            ``preprocessing.fit_scaler``), or ``None`` if the target
            was never scaled. Every grid built here is converted back
            to real units via ``preprocessing.inverse_transform`` right
            before it's used for visualization/MAE -- rollout/bias
            correction above stay in scaled space throughout (required
            for the autoregressive window-feeding to match what the
            model was trained on); only the finished output grids get
            converted, once, at this boundary.
    """
    latest_window = _latest_pixel_window(
        X_test_full, window_size=window_size, end_index=t0_index + 1,
    )
    n_lat, n_lon = _grid_shape(test_datasets[t0_index], n_pixels=latest_window.shape[0])
    grid_forecast = forecasting.bias_correction(
        model=model, X=latest_window, horizon=horizon_steps, bias=bias, step_bias=step_bias,
    )
    frames = grid_forecast.reshape(grid_forecast.shape[0], n_lat, n_lon)

    bandung_subset = preprocessing.subset_bandung(test_datasets[t0_index])
    lat_arr = bandung_subset["latitude"].values
    lon_arr = bandung_subset["longitude"].values

    input_grid = y_test_full[t0_index].reshape(n_lat, n_lon)

    # Ground truth asli per langkah kalau datanya ada; None (jadi
    # placeholder) untuk langkah yang tidak ada datanya.
    actual_frames = []
    for i in range(frames.shape[0]):
        future_i = t0_index + 1 + i
        if future_i < len(test_datasets):
            actual_frames.append(y_test_full[future_i].reshape(n_lat, n_lon))
        else:
            actual_frames.append(None)

    if target_scaler is not None:
        input_grid = preprocessing.inverse_transform(
            input_grid.reshape(-1, 1), target_scaler,
        ).reshape(input_grid.shape)
        frames = preprocessing.inverse_transform(
            frames.reshape(-1, 1), target_scaler,
        ).reshape(frames.shape)
        actual_frames = [
            preprocessing.inverse_transform(g.reshape(-1, 1), target_scaler).reshape(g.shape)
            if g is not None else None
            for g in actual_frames
        ]

    pred_grid = frames[-1]  # konsisten dengan forecast_label (+N langkah terakhir)
    actual_grid = actual_frames[-1]

    bias_label = (
        f"per-pixel (rata-rata {float(np.mean(bias)):.4f})"
        if isinstance(bias, np.ndarray) else f"{bias:.4f}"
    )
    if actual_grid is not None:
        sample_mae = float(np.nanmean(np.abs(pred_grid - actual_grid)))
        logger.info(
            "t0=%d: dashboard MAE (+%d langkah, bias=%s) = %.4f K.",
            t0_index, horizon_steps, bias_label, sample_mae,
        )
    else:
        logger.info(
            "t0=%d: tidak ada ground truth; dashboard tampil 'Forecast "
            "Mode' (MAE tidak bisa dihitung).", t0_index,
        )

    t0_label, forecast_label, interval_minutes = _forecast_labels(
        test_datasets, horizon_steps=frames.shape[0], t0_index=t0_index,
    )

    visualization.render_six_panel(
        lat_arr, lon_arr,
        input_grid, pred_grid, actual_grid,
        t0_label, forecast_label, interval_minutes,
        out_path=dashboard_dir / f"{EXPERIMENT_ID}_dashboard{filename_suffix}.png",
    )

    visualization.create_dashboard_gif(
        input_grid, frames, lat_arr, lon_arr,
        t0_label, interval_minutes,
        actual_frames=actual_frames,
        output_dir=gif_dir,
        filename=f"{EXPERIMENT_ID}_dashboard{filename_suffix}.gif",
    )
    # NOTE: this used to also render a "10/30/60 menit" interval-
    # comparison chart here, by rolling THIS SAME model out for 3
    # different step counts (a simulation of 30/60 minutes from a
    # 1-step model, with compounding rollout error). That's superseded
    # by main.py's per-scenario loop (config.FORECAST_SCENARIOS): each
    # of 10/30/60 minutes now has its OWN model, trained and windowed
    # natively for that lead time, and its own genuine dashboard/GIF
    # via this same function -- so there is nothing left to "compare"
    # from a single model here.


def _forecast_labels(
    test_datasets: list[xr.Dataset], horizon_steps: int, t0_index: int | None = None,
) -> tuple[str, str, float]:
    """Derive display labels/timestep interval for the forecast dashboard.

    Display-only helper (not a scientific transformation), in the same
    spirit as ``_latest_pixel_window``/``_grid_shape``: it never
    touches array values, only formats the labels
    ``visualization.render_six_panel`` puts in its title.

    Args:
        test_datasets: Chronologically ordered test-period datasets.
        horizon_steps: Number of forecasted steps (``frames.shape[0]``).
        t0_index: Index into ``test_datasets`` of the actual t0 used to
            build the forecast (may be earlier than the last index,
            e.g. when backtesting). Defaults to the last index.

    Returns:
        Tuple ``(t0_label, forecast_label, interval_minutes)``.
    """
    if t0_index is None:
        t0_index = len(test_datasets) - 1
    t0_time = _extract_timestamp(test_datasets[t0_index])
    interval_minutes = _detect_interval_minutes(test_datasets)

    if t0_time is not None:
        t0_label = str(np.datetime_as_string(t0_time, unit="m"))
        forecast_time = t0_time + np.timedelta64(
            int(round(interval_minutes * horizon_steps)), "m"
        )
        forecast_label = str(np.datetime_as_string(forecast_time, unit="m"))
    else:
        t0_label = f"t0 (data test ke-{t0_index + 1})"
        forecast_label = f"t0 + {horizon_steps} langkah"

    return t0_label, forecast_label, interval_minutes


def _run_scenario(
    horizon_minutes: int,
    horizon_steps: int,
    window_size: int,
    window_size_search_payload: dict[str, Any] | None,
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    X_test_full: np.ndarray,
    y_test_full: np.ndarray,
    train_datasets: list[xr.Dataset],
    test_datasets: list[xr.Dataset],
    target_scaler: "preprocessing.Scaler",
) -> dict[str, Any]:
    """Run windowing through visualization for one forecast-horizon scenario.

    Factored out of ``run_pipeline`` so it can be called once per
    ``config.FORECAST_SCENARIOS`` entry (10/30/60-minute lead times).
    Each scenario gets: its own windowed dataset (``window_size`` +
    ``horizon_steps`` specific to it), its own trained model (saved via
    ``training.save_model`` as ``f"{DEFAULT_MODEL}_{horizon_minutes}min"``
    -- a different file per scenario, so none overwrite each other; the
    model TYPE, via ``models.get_model``/``training.train_model``, is
    still plain ``config.DEFAULT_MODEL``), its own cross-validation, its
    own evaluation/export (filenames suffixed
    ``f"_{horizon_minutes}min"`` -- see ``export.export_all``'s
    ``filename_suffix``), and its own dashboard/GIF, rolled out exactly
    ``horizon_steps`` native steps -- e.g. the 30-minute scenario's
    dashboard genuinely shows +30 minutes from a model trained for that,
    not a relabeled +10-minute step repeated. Every ``config.RUN_*``
    toggle is honored exactly as in the single-scenario pipeline this
    replaces.

    Args:
        horizon_minutes, horizon_steps: This scenario's lead time --
            one entry of ``config.FORECAST_SCENARIOS``.
        window_size: Window size to use for this scenario (typically
            ``window_size_search_payload["best"]["window_size"]``, or
            ``config.WINDOW_SIZE`` when ``RUN_EXPERIMENT`` is False).
        window_size_search_payload: This scenario's own ``{"results":
            ..., "best": ...}`` (one entry of ``experiment.
            run_multi_horizon_window_size_search()["scenarios"]``), or
            ``None`` when ``RUN_EXPERIMENT`` is False. Forwarded only to
            ``visualization.render_window_size_search`` and into the
            returned summary.
        X_train_full, y_train_full, X_test_full, y_test_full: Full
            preprocessed (pre-windowing) feature/target arrays, shared
            across every scenario (preprocessing doesn't depend on
            window size/horizon).
        train_datasets, test_datasets: Chronologically ordered raw
            datasets, shared across every scenario.
        target_scaler: Fitted target scaler (or ``None``), shared
            across every scenario.

    Returns:
        Dictionary summarizing this scenario's run (window size,
        training time, dataset sizes, evaluation/cross-validation/
        window-size-search results) -- one entry of ``summary.json``'s
        ``"scenarios"`` map.

    Raises:
        RuntimeError: If ``config.RUN_FORECAST`` is enabled but no
            model is available (mirrors the single-scenario pipeline's
            original behavior).
    """
    model_label = f"{DEFAULT_MODEL}_{horizon_minutes}min"
    filename_suffix = f"_{horizon_minutes}min"
    training_seconds: float | None = None

    logger.info(
        "=== Skenario %d menit (window_size=%d, horizon_steps=%d) ===",
        horizon_minutes, window_size, horizon_steps,
    )

    # ---- Windowing --------------------------------------------------
    logger.info(
        "Starting windowing (window_size=%d, horizon=%d)...",
        window_size, horizon_steps,
    )
    (
        X_window_train, y_window_train,
        pixel_index_train, time_index_train, _, _,
    ) = windowing.create_windows(
        X_train_full, y_train_full, window_size=window_size, horizon=horizon_steps,
    )
    (
        X_window_test, y_window_test,
        pixel_index_test, time_index_test, _, _,
    ) = windowing.create_windows(
        X_test_full, y_test_full, window_size=window_size, horizon=horizon_steps,
    )
    logger.info("Finished windowing.")

    # ---- Dataset Builder --------------------------------------------------
    logger.info("Starting dataset builder...")
    flatten_required = get_model(DEFAULT_MODEL).requires_flat_input
    train_dataset = dataset_builder.build_dataset(
        X_window_train, y_window_train, pixel_index_train, time_index_train,
        flatten=flatten_required,
    )
    logger.info("Finished dataset builder.")

    # ---- Cross Validation (optional) --------------------------------------------------
    cv_result: dict[str, Any] | None = None
    if RUN_CROSS_VALIDATION:
        logger.info("Starting cross-validation...")
        cv_result = experiment.run_cross_validation(
            X_window_train, y_window_train, pixel_index_train,
            model_name=DEFAULT_MODEL, n_splits=CV_N_SPLITS, target_scaler=target_scaler,
        )
        logger.info("Finished cross-validation.")
    else:
        logger.info("RUN_CROSS_VALIDATION disabled; skipping cross-validation.")

    # ---- Training (or Load Model) --------------------------------------------------
    model: BaseModel | None = None
    if RUN_TRAINING:
        logger.info("Starting training...")
        train_start = run_metrics.now()
        X_train_fit, y_train_fit = train_dataset["X_train"], train_dataset["y_train"]
        if DEFAULT_MODEL == "gaussian_process":
            # Exact GP regression is O(n^3)/O(n^2) -- fitting it on
            # the full training set is impractical (see
            # models/gaussian_process.py's module docstring). Recover
            # which physical pixel each training sample belongs to
            # (same trick the bias-estimation step below uses:
            # split_dataset is deterministic/shuffle=False, so calling
            # it again with pixel_index_train in place of y gives the
            # exact same partition, just pixel-tagged).
            pixel_index_train_split = dataset_builder.split_dataset(
                X_window_train, pixel_index_train,
            )[2]
            X_train_fit, y_train_fit = dataset_builder.coverage_then_random_subsample(
                X_train_fit, y_train_fit, pixel_index_train_split,
                max_samples=GP_MAX_TRAIN_SAMPLES,
            )
        model = training.train_model(
            X_train_fit, y_train_fit, model_name=DEFAULT_MODEL,
        )
        training.save_model(model, model_name=model_label)
        training_seconds = run_metrics.elapsed_seconds(train_start)
        logger.info("Finished training in %.2fs.", training_seconds)
    elif RUN_FORECAST:
        logger.info("RUN_TRAINING disabled; loading existing model '%s'...", model_label)
        model = training.load_model(model_name=model_label)
        logger.info("Finished loading model.")
    else:
        logger.info("RUN_TRAINING and RUN_FORECAST both disabled; skipping model lifecycle.")

    # ---- Bias Estimation (calibration step) ----------------------------------------------
    estimated_bias: float | np.ndarray = 0.0
    if model is not None:
        logger.info("Estimating forecast bias from the held-out validation split...")
        _, X_val, _, y_val = dataset_builder.split_dataset(X_window_train, y_window_train)
        _, _, _, pixel_index_val = dataset_builder.split_dataset(
            X_window_train, pixel_index_train,
        )
        val_pred = forecasting.predict(model, X_val)
        estimated_bias, per_pixel_bias = evaluation.estimate_bias(
            y_val, val_pred, pixel_index=pixel_index_val,
        )
        logger.info(
            "Estimated global bias: %.4f (fallback for any pixel not "
            "covered by the validation split).", estimated_bias,
        )

        bandung_sizes = preprocessing.subset_bandung(test_datasets[-1]).sizes
        n_pixels_test = int(bandung_sizes["latitude"] * bandung_sizes["longitude"])
        missing_pixels = [p for p in range(n_pixels_test) if p not in per_pixel_bias]
        if missing_pixels:
            logger.info(
                "Per-pixel bias hanya mencakup %d/%d pixel grid test "
                "(validation split -- pixel-major, tidak di-shuffle -- "
                "cuma nyentuh sebagian rentang pixel); dipakai bias "
                "global (%.4f) untuk semua pixel, bukan per-pixel.",
                n_pixels_test - len(missing_pixels), n_pixels_test, estimated_bias,
            )
        else:
            estimated_bias = np.array(
                [per_pixel_bias[p] for p in range(n_pixels_test)], dtype=np.float64,
            )
            logger.info(
                "Per-pixel bias mencakup semua %d pixel grid test; "
                "dipakai per-pixel (bukan global).", n_pixels_test,
            )

    # ---- Forecasting --------------------------------------------------
    y_true_test: np.ndarray | None = None
    y_pred_test: np.ndarray | None = None
    if RUN_FORECAST:
        if model is None:
            raise RuntimeError(
                "RUN_FORECAST is enabled but no model is available "
                "(RUN_TRAINING is False and no model could be loaded)."
            )
        logger.info("Starting forecasting...")
        y_pred_test = forecasting.forecast_only(model=model, X=X_window_test)
        y_true_test = y_window_test
        logger.info("Finished forecasting.")

        # forecast_only()/y_window_test are in target_scaler's scaled
        # space (the model's native training space), not real Kelvin --
        # classification.classify()'s CLOUD_BINS/RISK_BINS and
        # evaluation's regression metrics are only meaningful in real
        # units, so convert back here, once, right at this boundary
        # (never mid-rollout -- forecasting/bias_correction must keep
        # working in scaled space; see _render_forecast_sample).
        if target_scaler is not None:
            y_true_test = preprocessing.inverse_transform(
                y_true_test.reshape(-1, 1), target_scaler,
            ).reshape(y_true_test.shape)
            y_pred_test = preprocessing.inverse_transform(
                y_pred_test.reshape(-1, 1), target_scaler,
            ).reshape(y_pred_test.shape)
    else:
        logger.info("RUN_FORECAST disabled; skipping forecasting.")

    # ---- Classification --------------------------------------------------
    classified_true: dict[str, Any] | None = None
    classified_pred: dict[str, Any] | None = None
    if RUN_CLASSIFICATION:
        if y_true_test is None or y_pred_test is None:
            logger.warning(
                "RUN_CLASSIFICATION is enabled but no forecast results "
                "are available; skipping."
            )
        else:
            logger.info("Starting classification...")
            classified_true = classification.classify(y_true_test)
            classified_pred = classification.classify(y_pred_test)
            logger.info("Finished classification.")
    else:
        logger.info("RUN_CLASSIFICATION disabled; skipping classification.")

    # ---- Evaluation --------------------------------------------------
    evaluation_result: dict[str, Any] | None = None
    if RUN_EVALUATION:
        if (
            y_true_test is None or y_pred_test is None
            or classified_true is None or classified_pred is None
        ):
            logger.warning(
                "RUN_EVALUATION is enabled but forecast/classification "
                "results are missing; skipping."
            )
        else:
            logger.info("Starting evaluation...")
            regression_metrics = evaluation.evaluate_regression(y_true_test, y_pred_test)
            classification_metrics = {
                "cloud": evaluation.evaluate_classification(
                    classified_true["cloud"]["class_index"],
                    classified_pred["cloud"]["class_index"],
                    labels=CLOUD_LABELS,
                ),
                "risk": evaluation.evaluate_classification(
                    classified_true["risk"]["class_index"],
                    classified_pred["risk"]["class_index"],
                    labels=RISK_LABELS,
                ),
            }
            evaluation_result = {
                "regression": regression_metrics,
                "classification": classification_metrics,
            }
            logger.info("Finished evaluation.")
    else:
        logger.info("RUN_EVALUATION disabled; skipping evaluation.")

    # ---- Export --------------------------------------------------
    if RUN_EXPORT:
        logger.info("Starting export...")
        test_metadata = dataset_builder.create_metadata(
            X_window_test, pixel_index_test, time_index_test,
        )

        export_forecast = None
        if y_true_test is not None and y_pred_test is not None:
            export_forecast = {"y_true": y_true_test, "y_pred": y_pred_test}

        export_classification = None
        if classified_true is not None and classified_pred is not None:
            export_classification = {
                "cloud": {
                    "y_true_class": classified_true["cloud"]["class_index"],
                    "y_pred_class": classified_pred["cloud"]["class_index"],
                },
                "risk": {
                    "y_true_class": classified_true["risk"]["class_index"],
                    "y_pred_class": classified_pred["risk"]["class_index"],
                },
            }

        export.export_all(
            forecast=export_forecast,
            classification=export_classification,
            evaluation=evaluation_result,
            metadata=test_metadata,
            filename_suffix=filename_suffix,
        )
        logger.info("Finished export.")
    else:
        logger.info("RUN_EXPORT disabled; skipping export.")

    # ---- Visualization --------------------------------------------------
    if RUN_VISUALIZATION:
        if y_true_test is None or y_pred_test is None:
            logger.warning(
                "RUN_VISUALIZATION is enabled but no forecast results "
                "are available; skipping."
            )
        else:
            logger.info("Starting visualization...")
            try:
                min_t0_index = window_size - 1
                t0_indices = _select_t0_indices(
                    test_datasets, horizon_steps, min_t0_index,
                    mode=VISUALIZATION_T0_MODE, n_samples=VISUALIZATION_N_SAMPLES,
                    manual_indices=VISUALIZATION_T0_INDICES,
                    anchor_last=VISUALIZATION_ANCHOR_LAST,
                )
            except ValueError:
                logger.warning(
                    "Could not select t0 sample(s) for visualization; "
                    "skipping the forecast dashboard.",
                    exc_info=True,
                )
            else:
                VISUALIZATION_DASHBOARD_PATH.mkdir(parents=True, exist_ok=True)
                GIF_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

                step_bias: np.ndarray | None = None
                if RUN_STEP_BIAS_CALIBRATION and model is not None:
                    logger.info(
                        "Mengkalibrasi step-bias (%d langkah) dari "
                        "TRAIN_MONTHS, %d titik t0...",
                        horizon_steps, STEP_BIAS_CALIBRATION_SAMPLES,
                    )
                    step_bias = _calibrate_step_bias(
                        model, X_train_full, y_train_full, train_datasets,
                        horizon_steps=horizon_steps, window_size=window_size,
                        baseline_bias=estimated_bias,
                        n_samples=STEP_BIAS_CALIBRATION_SAMPLES,
                    )

                logger.info(
                    "Rendering %d forecast sample(s) (mode=%r) at t0 "
                    "index(es): %s", len(t0_indices), VISUALIZATION_T0_MODE, t0_indices,
                )
                for t0_index in t0_indices:
                    sample_suffix = (
                        f"{filename_suffix}_t0-{t0_index}" if len(t0_indices) > 1
                        else filename_suffix
                    )
                    try:
                        _render_forecast_sample(
                            model, X_test_full, y_test_full, test_datasets,
                            t0_index=t0_index, horizon_steps=horizon_steps,
                            window_size=window_size,
                            dashboard_dir=VISUALIZATION_DASHBOARD_PATH,
                            gif_dir=GIF_OUTPUT_PATH,
                            filename_suffix=sample_suffix,
                            bias=estimated_bias,
                            step_bias=step_bias,
                            target_scaler=target_scaler,
                        )
                    except ValueError:
                        logger.warning(
                            "Could not render the forecast sample for "
                            "t0=%d; skipping it.", t0_index, exc_info=True,
                        )

            if evaluation_result is not None:
                for scheme_name, labels in (("cloud", CLOUD_LABELS), ("risk", RISK_LABELS)):
                    confusion = evaluation_result["classification"][scheme_name]["confusion_matrix"]
                    visualization.plot_confusion_matrix(
                        confusion, labels,
                        output_dir=VISUALIZATION_CONFUSION_MATRIX_PATH,
                        filename=(
                            f"{EXPERIMENT_ID}_confusion_matrix_{scheme_name}"
                            f"{filename_suffix}.png"
                        ),
                    )

            if window_size_search_payload is not None:
                visualization.render_window_size_search(
                    window_size_search_payload,
                    output_dir=VISUALIZATION_DASHBOARD_PATH,
                    filename=f"{EXPERIMENT_ID}_window_size_search{filename_suffix}.png",
                )

            if cv_result is not None:
                visualization.render_cross_validation(
                    cv_result,
                    output_dir=VISUALIZATION_DASHBOARD_PATH,
                    filename=f"{EXPERIMENT_ID}_cross_validation{filename_suffix}.png",
                )

            logger.info("Finished visualization.")
    else:
        logger.info("RUN_VISUALIZATION disabled; skipping visualization.")

    return {
        "horizon_minutes": horizon_minutes,
        "horizon_steps": horizon_steps,
        "window_size": window_size,
        "model": model_label,
        "timing_seconds": {"training": training_seconds},
        "dataset": {
            "train_samples": int(X_window_train.shape[0]),
            "test_samples": int(X_window_test.shape[0]),
        },
        "results": {
            "evaluation": evaluation_result,
            "cross_validation": cv_result,
            "window_size_search": window_size_search_payload,
        },
    }


def run_pipeline() -> None:
    """Run the full AI Himawari Forecasting pipeline end-to-end.

    Orchestrates, in order: data loading, preprocessing (both shared
    across every scenario), then a multi-horizon window-size search
    (one search per ``config.FORECAST_SCENARIOS`` lead time -- 10/30/60
    minutes by default) whose winning window size feeds that
    scenario's own windowing, dataset assembly, cross-validation, model
    training/loading, forecasting, classification, evaluation, export,
    and visualization (see ``_run_scenario``). Every step is still
    gated by its corresponding ``config.RUN_*`` toggle, applied
    identically to every scenario.

    Raises:
        Exception: Any exception raised by an underlying module is
            logged via ``logger.exception`` and re-raised unchanged.
    """
    pipeline_start = run_metrics.now()
    loading_seconds: float | None = None
    training_seconds: float | None = None

    try:
        logger.info("Starting '%s' (experiment_id='%s').", PROJECT_NAME, EXPERIMENT_ID)

        # ---- Load Data ----------------------------------------------------
        logger.info("Starting data loading...")
        load_start = run_metrics.now()
        train_datasets = _load_months(TRAIN_MONTHS)
        test_datasets = _load_months(TEST_MONTHS)
        loading_seconds = run_metrics.elapsed_seconds(load_start)
        logger.info(
            "Finished data loading (%d train timestamp(s), %d test timestamp(s)) "
            "in %.2fs.",
            len(train_datasets), len(test_datasets), loading_seconds,
        )

        # ---- Preprocessing --------------------------------------------------
        logger.info("Starting preprocessing...")
        X_train_raw, y_train_raw = preprocessing.preprocess_day(train_datasets)
        feature_scaler = preprocessing.fit_scaler(X_train_raw.reshape(-1, X_train_raw.shape[-1]))
        target_scaler = preprocessing.fit_scaler(y_train_raw.reshape(-1, 1))

        X_train_full, y_train_full = preprocessing.preprocess_day(
            train_datasets, feature_scaler=feature_scaler, target_scaler=target_scaler,
        )
        X_test_full, y_test_full = preprocessing.preprocess_day(
            test_datasets, feature_scaler=feature_scaler, target_scaler=target_scaler,
        )
        logger.info("Finished preprocessing.")

        # ---- Window Size Search (multi-horizon: 10/30/60 menit) -----------
        # Runs BEFORE windowing/dataset builder (it operates on the
        # pre-windowing, preprocessed arrays above, calling
        # windowing.create_windows internally for every candidate). One
        # search per config.FORECAST_SCENARIOS entry -- each scenario's
        # winning window_size feeds ITS OWN windowing/dataset
        # builder/cross-validation/training/forecasting/dashboard below
        # (see _run_scenario), not a single window size shared by all
        # three lead times.
        experiment_result: dict[str, Any] | None = None
        if RUN_EXPERIMENT:
            logger.info("Starting multi-horizon window size search...")
            experiment_result = experiment.run_multi_horizon_window_size_search(
                X_train_full, y_train_full, target_scaler=target_scaler,
            )
            logger.info("Finished multi-horizon window size search.")
        else:
            logger.info(
                "RUN_EXPERIMENT disabled; skipping window size search. "
                "Every scenario will use config.WINDOW_SIZE=%d.", WINDOW_SIZE,
            )

        # ---- Per-scenario pipeline (windowing -> ... -> visualization) ----
        # One full pass per (horizon_minutes, horizon_steps) in
        # config.FORECAST_SCENARIOS -- see _run_scenario's docstring.
        scenario_summaries: dict[str, Any] = {}
        training_seconds_total = 0.0
        any_training_ran = False
        for horizon_minutes, horizon_steps in FORECAST_SCENARIOS:
            if experiment_result is not None:
                scenario_search = experiment_result["scenarios"][horizon_minutes]
                scenario_window_size = scenario_search["best"]["window_size"]
            else:
                scenario_search = None
                scenario_window_size = WINDOW_SIZE

            scenario_summary = _run_scenario(
                horizon_minutes=horizon_minutes,
                horizon_steps=horizon_steps,
                window_size=scenario_window_size,
                window_size_search_payload=scenario_search,
                X_train_full=X_train_full, y_train_full=y_train_full,
                X_test_full=X_test_full, y_test_full=y_test_full,
                train_datasets=train_datasets, test_datasets=test_datasets,
                target_scaler=target_scaler,
            )
            scenario_summaries[str(horizon_minutes)] = scenario_summary
            scenario_training_seconds = scenario_summary["timing_seconds"]["training"]
            if scenario_training_seconds is not None:
                training_seconds_total += scenario_training_seconds
                any_training_ran = True

        training_seconds = training_seconds_total if any_training_ran else None

        # ---- Run Summary (timing / resource usage / results) --------------
        total_seconds = run_metrics.elapsed_seconds(pipeline_start)
        resource_snapshot = run_metrics.get_resource_snapshot()
        peak_memory_gb = resource_snapshot["peak_memory_gb"]

        logger.info(
            "Run timing -- data loading: %s, training (semua skenario): %s, "
            "total: %.2fs. Peak memory: %s.",
            f"{loading_seconds:.2f}s" if loading_seconds is not None else "n/a",
            f"{training_seconds:.2f}s" if training_seconds is not None else "n/a",
            total_seconds,
            f"{peak_memory_gb:.2f} GB" if peak_memory_gb is not None else "n/a",
        )

        run_summary = {
            "experiment_id": EXPERIMENT_ID,
            "model": DEFAULT_MODEL,
            "timing_seconds": {
                "data_loading": loading_seconds,
                "training_total": training_seconds,
                "total": total_seconds,
            },
            "resource_usage": resource_snapshot,
            "dataset": {
                "train_timesteps": len(train_datasets),
                "test_timesteps": len(test_datasets),
            },
            "scenarios": scenario_summaries,
        }
        export.export_json(run_summary, SUMMARY_OUTPUT_PATH / "summary.json")

        logger.info("Finish. Pipeline completed successfully.")

    except Exception:
        logger.exception("Pipeline failed.")
        raise


def main() -> None:
    """Entry point: seed RNGs, configure logging, then run the pipeline.

    ``config.RANDOM_SEED`` is documented as a global reproducibility
    seed, but no other module actually seeds NumPy's global RNG state
    (individual modules only forward it as an explicit
    ``random_state``/``random_seed`` parameter to scikit-learn calls).
    Seeding it once here, at the process entry point, is an
    orchestration/bootstrap concern, not business logic.
    """
    np.random.seed(RANDOM_SEED)
    _configure_logging()
    run_pipeline()


if __name__ == "__main__":
    main()