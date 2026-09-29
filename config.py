"""Project configuration for AI Himawari Forecasting.

This module centralizes all static configuration values used across the
project: paths, dataset parameters, forecasting parameters, output toggles,
and reproducibility settings.

This file must remain free of processing logic. It only defines constants
that other modules (loader, preprocessing, forecasting, etc.) will import.
"""

from pathlib import Path

# =====================================================================
# PROJECT METADATA
# =====================================================================

#: Human-readable name of the project, used in logs and output artifacts.
PROJECT_NAME: str = "AI Himawari Forecasting"

# =====================================================================
# EXPERIMENT MANAGEMENT
# =====================================================================

#: Name of the current experiment (e.g. "baseline", "ann_v2", "gpr_tuned").
EXPERIMENT_NAME: str = "gradient_boosting"

#: Version tag of the current experiment, used to distinguish reruns of
#: the same experiment configuration (e.g. after a code or param change).
EXPERIMENT_VERSION: str = "v8_8months"

#: Unique identifier derived from EXPERIMENT_NAME and EXPERIMENT_VERSION.
#: Used to namespace all output artifacts so multiple experiments can
#: coexist without overwriting each other.
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

#: Root directory where all generated artifacts are written, across all
#: experiments.
OUTPUT_PATH: Path = BASE_PATH / "outputs"

#: Output directory scoped to the current experiment (EXPERIMENT_ID).
#: All artifacts produced by a run are namespaced under this path so
#: different experiments never overwrite each other.
EXPERIMENT_OUTPUT_PATH: Path = OUTPUT_PATH / EXPERIMENT_ID

#: Experiment-scoped output subdirectories, derived from
#: EXPERIMENT_OUTPUT_PATH.
SUMMARY_OUTPUT_PATH: Path = EXPERIMENT_OUTPUT_PATH / "summary"
MODEL_OUTPUT_PATH: Path = EXPERIMENT_OUTPUT_PATH / "models"
LOG_OUTPUT_PATH: Path = EXPERIMENT_OUTPUT_PATH / "logs"

#: Forecast artifacts (ground truth / forecast TBB, dataset metadata),
#: written by an export module and consumed by forecasting.py's output.
FORECAST_OUTPUT_PATH: Path = EXPERIMENT_OUTPUT_PATH / "forecast"

#: Classification artifacts (cloud/risk class arrays), written by an
#: export module and consumed by classification.py's output.
CLASSIFICATION_OUTPUT_PATH: Path = EXPERIMENT_OUTPUT_PATH / "classification"

#: Evaluation artifacts (regression/classification metrics, confusion
#: matrices), written by an export module and consumed by
#: evaluation.py's output.
EVALUATION_OUTPUT_PATH: Path = EXPERIMENT_OUTPUT_PATH / "evaluation"

#: Root directory for all visualization artifacts (plots, maps, GIFs).
VISUALIZATION_OUTPUT_PATH: Path = EXPERIMENT_OUTPUT_PATH / "visualization"

#: Visualization-scoped output subdirectories, derived from
#: VISUALIZATION_OUTPUT_PATH.
VISUALIZATION_TBB_PATH: Path = VISUALIZATION_OUTPUT_PATH / "tbb"
VISUALIZATION_CLOUD_PATH: Path = VISUALIZATION_OUTPUT_PATH / "cloud"
VISUALIZATION_RISK_PATH: Path = VISUALIZATION_OUTPUT_PATH / "risk"
VISUALIZATION_CONFUSION_MATRIX_PATH: Path = VISUALIZATION_OUTPUT_PATH / "confusion_matrix"
VISUALIZATION_TIMESERIES_PATH: Path = VISUALIZATION_OUTPUT_PATH / "timeseries"

#: Combined 6-panel forecast dashboard produced by
#: visualization.render_six_panel() (input/prediction/actual/cloud
#: class/risk class/error-map panels in a single figure).
VISUALIZATION_DASHBOARD_PATH: Path = VISUALIZATION_OUTPUT_PATH / "dashboard"

#: Whether to save animated GIFs of forecast/observation sequences.
#: Nested under VISUALIZATION_OUTPUT_PATH since GIF creation is a
#: visualization concern.
GIF_OUTPUT_PATH: Path = VISUALIZATION_OUTPUT_PATH / "gif"

#: Which test-period t0(s) to render the forecast dashboard/GIF/interval
#: comparison for (main.py's `_select_t0_indices`):
#:   "last"   -- the most recent t0 that still has enough real future
#:               data to backtest against (default; closest to a "live"
#:               forecast while still being validatable).
#:   "first"  -- the earliest such t0.
#:   "random" -- random t0(s), seeded by RANDOM_SEED (reproducible).
#:   "manual" -- exactly the indices listed in VISUALIZATION_T0_INDICES.
#: Falls back to the single most recent test-period index (no real
#: future data to compare against -- dashboard shows "Forecast Mode")
#: when the loaded test period is too short to backtest at all.
VISUALIZATION_T0_MODE: str = "last"

#: How many t0 samples to render for "last"/"first"/"random" modes.
#: Each sample gets its own dashboard PNG, GIF, and interval-comparison
#: chart (filenames suffixed with "_t0-<index>" when more than one).
#: Ignored for "manual" mode, where the sample count is
#: len(VISUALIZATION_T0_INDICES) instead.
VISUALIZATION_N_SAMPLES: int = 1

#: Explicit t0 indices into the loaded test_datasets list, used only
#: when VISUALIZATION_T0_MODE == "manual". E.g. (0, 500, 1200).
VISUALIZATION_T0_INDICES: tuple[int, ...] = ()

#: When True, always include the single most recent valid t0 as one of
#: the rendered samples (on top of VISUALIZATION_N_SAMPLES), with the
#: rest filled in per VISUALIZATION_T0_MODE. E.g. mode="random" +
#: N_SAMPLES=4 + ANCHOR_LAST=True -> 1 "last" sample + 4 random ones
#: (5 total). No-op for mode="last" (redundant) and mode="manual" (add
#: the last index to VISUALIZATION_T0_INDICES directly instead).
VISUALIZATION_ANCHOR_LAST: bool = False

# =====================================================================
# DATASET / TARGET PARAMETERS
# =====================================================================

#: NetCDF variable(s) used as model input features. May include one or
#: more variables independently of TARGET_VARIABLE, allowing multivariate
#: input while still forecasting a single target.
INPUT_FEATURES: tuple[str, ...] = (
    "tbb_13",
)

#: NetCDF variable name to be forecasted. Only Band-13 brightness
#: temperature (tbb_13) is used in this project. Kept independent from
#: INPUT_FEATURES so the input feature set can grow without changing
#: the forecasting target.
TARGET_VARIABLE: str = "tbb_13"

# =====================================================================
# DATASET SPLIT (TIME-SERIES, MONTH-LEVEL)
# =====================================================================

#: Month folders (YYYYMM) used for training. A month-based split is used
#: instead of a random split to preserve temporal ordering and avoid
#: leakage between training and testing periods.
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
#: dataset_builder.split_dataset(). This is a sample-level split applied
#: after windowing (e.g. for a train/validation split within the data
#: already selected by TRAIN_MONTHS) -- distinct from the month-level
#: TRAIN_MONTHS/TEST_MONTHS split above.
TRAIN_RATIO: float = 0.8

#: Whether dataset_builder.split_dataset() shuffles samples before
#: splitting. Note: windowed samples from the same pixel overlap by
#: (WINDOW_SIZE - 1) timesteps, so shuffling can place near-duplicate,
#: overlapping windows on both sides of the train/test boundary.
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
#: Used as-is only when config.RUN_EXPERIMENT is False; otherwise
#: main.py overrides this per-run with the winning candidate from
#: experiment.run_window_size_search() (see WINDOW_CANDIDATES below).
WINDOW_SIZE: int = 10

#: Number of future timesteps to predict ahead of the input window.
FORECAST_HORIZON: int = 1

#: Step size (in timesteps) between consecutive windows when generating
#: samples from the time series. A stride of 1 produces maximally
#: overlapping windows; larger strides reduce sample count/overlap.
WINDOW_STRIDE: int = 1

#: Real-world minutes represented by one native data timestep (one
#: Himawari .nc file = 10 minutes). Used to convert a lead time in
#: minutes (see FORECAST_SCENARIOS below) into a number of timesteps
#: (horizon, in the sense windowing.create_windows/FORECAST_HORIZON
#: use it) -- never hardcode that conversion elsewhere.
NATIVE_INTERVAL_MINUTES: int = 10

#: Lead-time scenarios the multi-horizon window size search / training
#: / forecasting run for, as ``(lead_time_minutes, horizon_steps)``
#: pairs -- horizon_steps = lead_time_minutes / NATIVE_INTERVAL_MINUTES.
#: Each scenario gets its own window size search (see
#: experiment.run_multi_horizon_window_size_search), its own trained
#: model (window size + horizon specific to that scenario, saved as
#: f"{DEFAULT_MODEL}_{lead_time_minutes}min"), and its own
#: dashboard/GIF -- so "30 minutes" is a model actually trained to
#: predict 3 steps ahead, not a 1-step model rolled out 3 times.
FORECAST_SCENARIOS: tuple[tuple[int, int], ...] = tuple(
    (minutes, max(1, minutes // NATIVE_INTERVAL_MINUTES))
    for minutes in (10, 30, 60)
)

# =====================================================================
# EXPERIMENTATION (WINDOW SIZE SEARCH)
# =====================================================================

#: Window size candidates evaluated by
#: experiment.run_window_size_search() to find the best-performing
#: window size before final training. When config.RUN_EXPERIMENT is
#: True, main.py uses the search's winning candidate as the actual
#: window size for windowing/training/forecasting/visualization for
#: the rest of that run (WINDOW_SIZE above is only the fallback used
#: when RUN_EXPERIMENT is False). Checks every window size from 1 to
#: 24 timesteps.
WINDOW_CANDIDATES: tuple[int, ...] = tuple(range(1, 25))

# =====================================================================
# CROSS-VALIDATION
# =====================================================================

#: Number of folds experiment.run_cross_validation() uses to
#: cross-validate the (WINDOW_SIZE, DEFAULT_MODEL) combination the main
#: pipeline actually trains with -- a more robust performance estimate
#: (mean +/- std across folds) than the single train/test split
#: dataset_builder.build_dataset() reports. Uses blocked, per-pixel
#: time-series folds (dataset_builder.time_series_kfold_indices), not a
#: plain shuffled K-fold -- see that function's docstring for why a
#: plain shuffle would leak overlapping windows across folds here.
CV_N_SPLITS: int = 5

# =====================================================================
# FORECAST UPDATE STRATEGY
# =====================================================================

#: Strategy used to update/correct forecasts with newly observed data
#: during recursive forecasting. Replaces a simple on/off bias-correction
#: flag with a named, extensible strategy.
#: Supported values: "forecast_only", "teacher_forcing", "bias_correction".
FORECAST_UPDATE_METHOD: str = "bias_correction"

#: Forecast update methods supported by this project. Used to validate
#: FORECAST_UPDATE_METHOD without hardcoding logic in the pipeline.
SUPPORTED_UPDATE_METHODS: tuple[str, ...] = (
    "forecast_only",
    "teacher_forcing",
    "bias_correction",
)

# =====================================================================
# OUTPUT TOGGLES
# =====================================================================

#: Whether to save animated GIFs of forecast/observation sequences.
SAVE_GIF: bool = True

#: Whether to save JSON summaries of runs (metrics, metadata, configs).
SAVE_JSON: bool = True

#: Whether to save execution logs to disk.
SAVE_LOG: bool = True

#: Whether to persist trained model artifacts to disk.
SAVE_MODEL: bool = True

#: Whether to save static PNG plots (e.g. forecast vs. observation).
SAVE_PNG: bool = True

#: Whether to save tabular results (e.g. metrics, predictions) as CSV.
SAVE_CSV: bool = True

#: Whether to save raw array results (e.g. forecasts, class arrays,
#: confusion matrices) as NumPy .npy files.
SAVE_NUMPY: bool = True

# =====================================================================
# REPRODUCIBILITY
# =====================================================================

#: Global random seed used across NumPy, PyTorch, and other libraries
#: to ensure reproducible experiments.
RANDOM_SEED: int = 42

# =====================================================================
# MODEL REGISTRY
# =====================================================================

#: Identifiers of models supported by this project. Used later phases
#: (training/forecasting) to validate user-selected model choices.
SUPPORTED_MODELS: tuple[str, ...] = (
    "ann",
    "gaussian_process",
    "gradient_boosting",
)

#: Model used by default when no explicit model choice is provided.
DEFAULT_MODEL: str = "gradient_boosting"

#: Maximum training samples "gaussian_process" is fit on. Exact GP
#: regression is O(n^3) in time / O(n^2) in memory (see
#: models/gaussian_process.py's module docstring), so fitting it on
#: TRAIN_MONTHS's full sample count (which can run into the hundreds
#: of thousands here) is impractical. When DEFAULT_MODEL ==
#: "gaussian_process", main.py subsamples training data down to this
#: cap first (dataset_builder.coverage_then_random_subsample: one
#: sample per pixel, then filled randomly) -- other models still train
#: on the full set, so this doesn't affect their results or make
#: cross-model comparisons unfair. Increase this if you have the
#: memory/time budget for a larger GP fit.
GP_MAX_TRAIN_SAMPLES: int = 3000

#: Default Bandung region bounding box used by subset_bandung().

BANDUNG_LAT_MIN: float = -7.0
BANDUNG_LAT_MAX: float = -6.8
BANDUNG_LON_MIN: float = 107.5
BANDUNG_LON_MAX: float = 107.8

# =====================================================================
# CLASSIFICATION THRESHOLDS
# =====================================================================
# Threshold, label, and color scheme used by classification.py to turn
# continuous tbb_13 predictions into discrete classes. Bin edges are in
# ascending order; there is always one more label/color than there are
# bin edges (values below the first edge and above the last edge each
# get their own label/color).

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
# Controls which steps main.py's orchestration pipeline executes. Each
# toggle gates exactly one pipeline step; setting it to False skips
# that step entirely (see main.py for the exact step boundaries).

#: Whether to run experiment.run_window_size_search() before training.
RUN_EXPERIMENT: bool = True

#: Whether to run experiment.run_cross_validation() before training --
#: cross-validates DEFAULT_MODEL at WINDOW_SIZE (the combination the
#: main pipeline actually trains with) for a more robust performance
#: estimate than a single train/test split. Adds CV_N_SPLITS extra
#: model fits to the run.
RUN_CROSS_VALIDATION: bool = True

#: Whether to train a fresh model. If False, main.py loads a
#: previously saved model (via training.load_model()) instead,
#: provided RUN_FORECAST is True.
RUN_TRAINING: bool = True

#: Whether to run inference (forecasting.predict()) on the held-out
#: test set.
RUN_FORECAST: bool = True

#: Whether to classify forecast results into cloud/risk classes.
RUN_CLASSIFICATION: bool = True

#: Whether to compute regression/classification evaluation metrics.
RUN_EVALUATION: bool = True

#: Whether to export results (forecast, classification, evaluation,
#: metadata) to disk.
RUN_EXPORT: bool = True

#: Whether to render visualization artifacts (plots).
RUN_VISUALIZATION: bool = True

#: Whether to also calibrate a per-rollout-step bias (on top of the
#: constant bias from evaluation.estimate_bias) from TRAIN_MONTHS data,
#: for drift that grows/shrinks specifically with rollout step rather
#: than being a constant offset. The candidate is only kept if it
#: measurably reduces MAE on a held-out half of the calibration t0's
#: (main.py's `_calibrate_step_bias`) -- a pattern calibrated on
#: TRAIN_MONTHS isn't guaranteed to transfer to TEST_MONTHS, so it's
#: validated before being trusted, never applied blindly. If False, or
#: if it doesn't pass that check, only the constant bias is used (same
#: as before this option existed).
RUN_STEP_BIAS_CALIBRATION: bool = True

#: Total t0 starting points (within TRAIN_MONTHS) to use for step-bias
#: calibration, split roughly in half between fitting the candidate and
#: validating it on data that wasn't used to fit it. More samples ->
#: more stable estimate and a more trustworthy validation check, but
#: each one costs a full multi-step rollout (plus a second one for the
#: validation half, corrected vs. uncorrected).
STEP_BIAS_CALIBRATION_SAMPLES: int = 10

# =====================================================================
# VISUALIZATION PARAMETERS
# =====================================================================

#: Resolution (dots per inch) used when saving static plots/maps.
FIGURE_DPI: int = 150

#: Figure size (width, height) in inches used for static plots/maps.
FIGURE_SIZE: tuple[int, int] = (8, 6)