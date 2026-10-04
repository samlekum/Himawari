"""Project configuration for AI Himawari Forecasting.

This module centralizes all static configuration values used across the
project: paths, dataset parameters, forecasting parameters, output toggles,
and reproducibility settings.

This version supports ONLY three models: XGBoost, LightGBM and CatBoost.
The active model is chosen with the HIMAWARI_MODEL environment variable, so
the three models can be run one after another without editing this file:

    PowerShell:  $env:HIMAWARI_MODEL = "xgboost"; python main.py
    Linux/macOS: HIMAWARI_MODEL=xgboost python main.py

This file must remain free of processing logic. It only defines constants
that other modules (loader, preprocessing, forecasting, etc.) will import.
"""

import os
from pathlib import Path

# =====================================================================
# PROJECT METADATA
# =====================================================================

#: Human-readable name of the project, used in logs and output artifacts.
PROJECT_NAME: str = "AI Himawari Forecasting"

# =====================================================================
# MODEL REGISTRY / SELECTION
# =====================================================================

#: Identifiers of models supported by this project. Must match the keys
#: registered in models/registry.py.
SUPPORTED_MODELS: tuple[str, ...] = (
    "xgboost",
    "lightgbm",
    "catboost",
)

#: Model used for this run. Read from the HIMAWARI_MODEL environment
#: variable; falls back to "lightgbm" (fastest, good for smoke tests).
DEFAULT_MODEL: str = os.environ.get("HIMAWARI_MODEL", "lightgbm").strip().lower()

if DEFAULT_MODEL not in SUPPORTED_MODELS:
    raise ValueError(
        f"HIMAWARI_MODEL={DEFAULT_MODEL!r} is not supported. "
        f"Choose one of: {SUPPORTED_MODELS}."
    )

#: Maximum training samples for "gaussian_process". This project version no
#: longer includes that model, but main.py still imports this name, so it is
#: kept (unused) to avoid an ImportError.
GP_MAX_TRAIN_SAMPLES: int = 3000

# =====================================================================
# EXPERIMENT MANAGEMENT
# =====================================================================

#: Name of the current experiment. Follows the active model so that the
#: outputs of different models never overwrite each other.
EXPERIMENT_NAME: str = DEFAULT_MODEL

#: Version tag of the current experiment. Change it after a code or
#: hyperparameter change to keep old results.
EXPERIMENT_VERSION: str = "v1"

#: Unique identifier derived from EXPERIMENT_NAME and EXPERIMENT_VERSION.
#: Used to namespace all output artifacts (outputs/<EXPERIMENT_ID>/...).
EXPERIMENT_ID: str = f"{EXPERIMENT_NAME}_{EXPERIMENT_VERSION}"

# =====================================================================
# PATHS
# =====================================================================

#: Root directory of the project (directory containing this file).
BASE_PATH: Path = Path(__file__).resolve().parent

#: Root directory of the raw Himawari-9 NetCDF dataset.
#: Expected structure:
#:   data/data_bandung/jma/netcdf/<YYYYMM>/<DD>/<timestamp>.nc
DATASET_PATH: Path = BASE_PATH / "data" / "data_bandung" / "jma" / "netcdf"

#: Root directory where all generated artifacts are written.
OUTPUT_PATH: Path = BASE_PATH / "outputs"

#: Output directory scoped to the current experiment (EXPERIMENT_ID).
EXPERIMENT_OUTPUT_PATH: Path = OUTPUT_PATH / EXPERIMENT_ID

#: Experiment-scoped output subdirectories.
SUMMARY_OUTPUT_PATH: Path = EXPERIMENT_OUTPUT_PATH / "summary"
MODEL_OUTPUT_PATH: Path = EXPERIMENT_OUTPUT_PATH / "models"
LOG_OUTPUT_PATH: Path = EXPERIMENT_OUTPUT_PATH / "logs"

#: Forecast artifacts (ground truth / forecast TBB, dataset metadata).
FORECAST_OUTPUT_PATH: Path = EXPERIMENT_OUTPUT_PATH / "forecast"

#: Classification artifacts (cloud/risk class arrays).
CLASSIFICATION_OUTPUT_PATH: Path = EXPERIMENT_OUTPUT_PATH / "classification"

#: Evaluation artifacts (regression/classification metrics, confusion matrices).
EVALUATION_OUTPUT_PATH: Path = EXPERIMENT_OUTPUT_PATH / "evaluation"

#: Root directory for all visualization artifacts (plots, maps, GIFs).
VISUALIZATION_OUTPUT_PATH: Path = EXPERIMENT_OUTPUT_PATH / "visualization"

#: Visualization-scoped output subdirectories.
VISUALIZATION_TBB_PATH: Path = VISUALIZATION_OUTPUT_PATH / "tbb"
VISUALIZATION_CLOUD_PATH: Path = VISUALIZATION_OUTPUT_PATH / "cloud"
VISUALIZATION_RISK_PATH: Path = VISUALIZATION_OUTPUT_PATH / "risk"
VISUALIZATION_CONFUSION_MATRIX_PATH: Path = VISUALIZATION_OUTPUT_PATH / "confusion_matrix"
VISUALIZATION_TIMESERIES_PATH: Path = VISUALIZATION_OUTPUT_PATH / "timeseries"

#: Combined 6-panel forecast dashboard (visualization.render_six_panel()).
VISUALIZATION_DASHBOARD_PATH: Path = VISUALIZATION_OUTPUT_PATH / "dashboard"

#: Output directory for animated GIFs.
GIF_OUTPUT_PATH: Path = VISUALIZATION_OUTPUT_PATH / "gif"

#: Which test-period t0(s) to render the forecast dashboard/GIF for
#: (main.py's `_select_t0_indices`):
#:   "last"   -- the most recent t0 that still has enough real future data
#:               to backtest against (default).
#:   "first"  -- the earliest such t0.
#:   "random" -- random t0(s), seeded by RANDOM_SEED (reproducible).
#:   "manual" -- exactly the indices listed in VISUALIZATION_T0_INDICES.
VISUALIZATION_T0_MODE: str = "last"

#: How many t0 samples to render for "last"/"first"/"random" modes.
VISUALIZATION_N_SAMPLES: int = 1

#: Explicit t0 indices, used only when VISUALIZATION_T0_MODE == "manual".
VISUALIZATION_T0_INDICES: tuple[int, ...] = ()

#: When True, always include the single most recent valid t0 as one of the
#: rendered samples (no-op for "last" and "manual").
VISUALIZATION_ANCHOR_LAST: bool = False

# =====================================================================
# DATASET / TARGET PARAMETERS
# =====================================================================

#: NetCDF variable(s) used as model input features.
INPUT_FEATURES: tuple[str, ...] = (
    "tbb_13",
)

#: NetCDF variable name to be forecasted (Band-13 brightness temperature).
TARGET_VARIABLE: str = "tbb_13"

# =====================================================================
# DATASET SPLIT (TIME-SERIES, MONTH-LEVEL)
# =====================================================================

#: Month folders (YYYYMM) used for training. A month-based split preserves
#: temporal ordering and avoids leakage between training and testing.
#: Adjust to the months you actually have under DATASET_PATH.
TRAIN_MONTHS: tuple[str, ...] = (
    "202512",
    "202601",
    "202602",
    "202603",
    "202604",
    "202605",
)

#: Month folders (YYYYMM) held out for testing/evaluation.
TEST_MONTHS: tuple[str, ...] = (
    "202606",
    "202607",
)

# =====================================================================
# DATASET SPLIT (SAMPLE-LEVEL, POST-WINDOWING)
# =====================================================================

#: Fraction of windowed samples allocated to the training set by
#: dataset_builder.split_dataset() (sample-level split after windowing).
TRAIN_RATIO: float = 0.8

#: Whether dataset_builder.split_dataset() shuffles samples before
#: splitting. Keep False: windows of the same pixel overlap, so shuffling
#: would put near-duplicate windows on both sides of the split.
SHUFFLE_DATASET: bool = False

# =====================================================================
# PREPROCESSING
# =====================================================================

#: Feature scaling strategy applied before model training.
#: Supported values: "standard", "minmax", "none".
SCALER: str = "standard"

# =====================================================================
# FORECASTING PARAMETERS
# =====================================================================

#: Number of previous timesteps used as model input (lag window).
#: Used as-is only when RUN_EXPERIMENT is False; otherwise main.py overrides
#: it per scenario with the winner of the window size search.
WINDOW_SIZE: int = 10

#: Number of future timesteps to predict ahead of the input window.
FORECAST_HORIZON: int = 1

#: Step size between consecutive windows (kept for documentation;
#: windowing.py currently uses a fixed stride of 1).
WINDOW_STRIDE: int = 1

#: Real-world minutes represented by one native data timestep
#: (one Himawari .nc file = 10 minutes).
NATIVE_INTERVAL_MINUTES: int = 10

#: Lead-time scenarios as ``(lead_time_minutes, horizon_steps)`` pairs.
#: Each scenario gets its own window size search, its own model
#: (saved as f"{DEFAULT_MODEL}_{lead_time_minutes}min") and its own outputs.
FORECAST_SCENARIOS: tuple[tuple[int, int], ...] = tuple(
    (minutes, max(1, minutes // NATIVE_INTERVAL_MINUTES))
    for minutes in (10, 30, 60)
)

# =====================================================================
# EXPERIMENTATION (WINDOW SIZE SEARCH)
# =====================================================================

#: Window size candidates evaluated by experiment.run_window_size_search().
#: Checks every window size from 1 to 24 timesteps. To speed up a run,
#: use a short list, e.g. (3, 6, 10, 12, 18, 24).
WINDOW_CANDIDATES: tuple[int, ...] = tuple(range(1, 25))

# =====================================================================
# CROSS-VALIDATION
# =====================================================================

#: Number of folds for experiment.run_cross_validation() (blocked,
#: per-pixel time-series folds).
CV_N_SPLITS: int = 5

# =====================================================================
# FORECAST UPDATE STRATEGY
# =====================================================================

#: Strategy used to update/correct forecasts during recursive forecasting.
#: Supported values: "forecast_only", "teacher_forcing", "bias_correction".
FORECAST_UPDATE_METHOD: str = "bias_correction"

#: Forecast update methods supported by this project.
SUPPORTED_UPDATE_METHODS: tuple[str, ...] = (
    "forecast_only",
    "teacher_forcing",
    "bias_correction",
)

# =====================================================================
# OUTPUT TOGGLES
# =====================================================================

#: Whether to save animated GIFs of forecast/observation sequences.
#: Disabled for now: the 30/60-minute dashboards/GIFs roll a direct
#: multi-step model out several times (wrong horizon). Re-enable after
#: patching main.py's _render_forecast_sample.
SAVE_GIF: bool = False

#: Whether to save JSON summaries of runs (metrics, metadata, configs).
SAVE_JSON: bool = True

#: Whether to save execution logs to disk.
SAVE_LOG: bool = True

#: Whether to persist trained model artifacts to disk.
SAVE_MODEL: bool = True

#: Whether to save static PNG plots.
SAVE_PNG: bool = True

#: Whether to save tabular results as CSV.
SAVE_CSV: bool = True

#: Whether to save raw array results as NumPy .npy files.
SAVE_NUMPY: bool = True

# =====================================================================
# REPRODUCIBILITY
# =====================================================================

#: Global random seed used across NumPy and other libraries.
RANDOM_SEED: int = 42

# =====================================================================
# SPATIAL REGION
# =====================================================================

#: Default Bandung region bounding box used by preprocessing.subset_bandung().
#: Note: the boundary is inclusive, and the last longitude column in the
#: sample files is 107.80000305 (> 107.8), so it is dropped (grid 5x6).
#: Use BANDUNG_LON_MAX = 107.81 to keep all 7 longitude columns (grid 5x7).
BANDUNG_LAT_MIN: float = -7.0
BANDUNG_LAT_MAX: float = -6.8
BANDUNG_LON_MIN: float = 107.5
BANDUNG_LON_MAX: float = 107.8

# =====================================================================
# CLASSIFICATION THRESHOLDS
# =====================================================================

# Thresholds, labels and colors used by classification.py to turn
# continuous tbb_13 predictions into discrete classes. Bin edges are in
# ascending order; there is always one more label/color than bin edges.

#: Bin edges (Kelvin, tbb_13) for the cloud-condition classification.
CLOUD_BINS: list[int] = [220, 260]

#: Cloud-condition class labels, ordered to match CLOUD_BINS.
CLOUD_LABELS: list[str] = ["Hujan", "Mendung", "Tidak Hujan"]

#: Display colors (hex) for CLOUD_LABELS, same order.
CLOUD_COLORS: list[str] = ["#08285B", "#6A727D", "#B89300"]

#: Bin edges (Kelvin, tbb_13) for the risk-level classification.
RISK_BINS: list[int] = [240, 265]

#: Risk-level class labels, ordered to match RISK_BINS.
RISK_LABELS: list[str] = ["Bahaya", "Waspada", "Aman"]

#: Display colors (hex) for RISK_LABELS, same order.
RISK_COLORS: list[str] = ["#9B0000", "#B89300", "#217C00"]

# =====================================================================
# PIPELINE EXECUTION TOGGLES
# =====================================================================

# Each toggle gates exactly one pipeline step in main.py; False skips it.
# For a quick smoke test set RUN_EXPERIMENT and RUN_CROSS_VALIDATION to
# False and use only one train month and one test month.

#: Run experiment.run_window_size_search() before training.
RUN_EXPERIMENT: bool = True

#: Run experiment.run_cross_validation() before training
#: (adds CV_N_SPLITS extra model fits per scenario).
RUN_CROSS_VALIDATION: bool = True

#: Train a fresh model. If False, a previously saved model is loaded.
RUN_TRAINING: bool = True

#: Run inference on the held-out test set.
RUN_FORECAST: bool = True

#: Classify forecast results into cloud/risk classes.
RUN_CLASSIFICATION: bool = True

#: Compute regression/classification evaluation metrics.
RUN_EVALUATION: bool = True

#: Export results (forecast, classification, evaluation, metadata) to disk.
RUN_EXPORT: bool = True

#: Render visualization artifacts (plots).
RUN_VISUALIZATION: bool = True

#: Calibrate a per-rollout-step bias from TRAIN_MONTHS data.
#: Disabled for now: it assumes a 1-step model rolled out iteratively,
#: while every scenario here trains a direct multi-step model.
RUN_STEP_BIAS_CALIBRATION: bool = False

#: Total t0 starting points used for step-bias calibration (only relevant
#: when RUN_STEP_BIAS_CALIBRATION is True).
STEP_BIAS_CALIBRATION_SAMPLES: int = 10

# =====================================================================
# VISUALIZATION PARAMETERS
# =====================================================================

#: Resolution (dots per inch) used when saving static plots/maps.
FIGURE_DPI: int = 150

#: Figure size (width, height) in inches used for static plots/maps.
FIGURE_SIZE: tuple[int, int] = (8, 6)