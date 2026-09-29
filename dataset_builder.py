"""Training-dataset assembly for AI Himawari Forecasting.

This module is responsible ONLY for turning the sliding-window arrays
produced by ``windowing.py`` into a ready-to-train dataset: an optional
flatten step, a train/test split, and a metadata dictionary describing
the dataset.

Pipeline position:

    windowing.py (X_window, y_window, pixel_index, time_index)
        -> dataset_builder.build_dataset
        -> {"X_train", "X_test", "y_train", "y_test", "metadata"}

This module never loads files, never preprocesses raw data, never
builds sliding windows itself, never trains models, never forecasts,
never evaluates, and never visualizes. It only assembles and describes
whatever windowed ``(X, y)`` arrays it is given.

``flatten_windows`` is intentionally re-exported from ``windowing.py``
rather than reimplemented here: the reshape logic already lives there,
and duplicating it would violate separation of concern between the two
modules. Every other public function in this module is new.

Train/test split parameters (``TRAIN_RATIO``, ``SHUFFLE_DATASET``,
``RANDOM_SEED``) and dataset-description parameters (``WINDOW_SIZE``,
``FORECAST_HORIZON``, ``WINDOW_STRIDE``, ``INPUT_FEATURES``,
``TARGET_VARIABLE``) are read from ``config.py`` and are never
hardcoded here.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from config import (
    FORECAST_HORIZON,
    INPUT_FEATURES,
    RANDOM_SEED,
    SHUFFLE_DATASET,
    TARGET_VARIABLE,
    TRAIN_RATIO,
    WINDOW_SIZE,
    WINDOW_STRIDE,
)
from windowing import flatten_windows

logger = logging.getLogger(__name__)

__all__ = [
    "flatten_windows",
    "split_dataset",
    "create_metadata",
    "build_dataset",
    "coverage_then_random_subsample",
    "time_series_kfold_indices",
]


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = TRAIN_RATIO,
    shuffle: bool = SHUFFLE_DATASET,
    random_seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a windowed dataset into train and test partitions.

    Thin wrapper around ``sklearn.model_selection.train_test_split``.

    Args:
        X: Feature array of shape ``(n_samples, ...)`` -- either the 3D
            output of ``windowing.create_windows`` or the 2D output of
            ``flatten_windows``.
        y: Target array of shape ``(n_samples,)``.
        train_ratio: Fraction of samples allocated to the training set.
            Defaults to ``config.TRAIN_RATIO``.
        shuffle: Whether to shuffle samples before splitting. Defaults
            to ``config.SHUFFLE_DATASET``.
        random_seed: Seed controlling the shuffle, for reproducibility.
            Defaults to ``config.RANDOM_SEED``.

    Returns:
        Tuple ``(X_train, X_test, y_train, y_test)``.

    Raises:
        ValueError: If ``X`` and ``y`` do not have the same number of
            samples, or if ``train_ratio`` is not strictly between 0
            and 1.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "X and y must have the same number of samples; got "
            f"X.shape[0]={X.shape[0]} vs y.shape[0]={y.shape[0]}."
        )
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be strictly between 0 and 1, got {train_ratio}.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_ratio,
        shuffle=shuffle,
        random_state=random_seed if shuffle else None,
    )

    logger.info(
        "Split dataset: %d train sample(s), %d test sample(s) "
        "(train_ratio=%.2f, shuffle=%s).",
        X_train.shape[0], X_test.shape[0], train_ratio, shuffle,
    )
    return X_train, X_test, y_train, y_test


def coverage_then_random_subsample(
    X: np.ndarray,
    y: np.ndarray,
    pixel_index: np.ndarray,
    max_samples: int,
    random_seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample down to ``max_samples``, covering every pixel first.

    For models where fitting on the full sample count is impractical
    (e.g. exact Gaussian Process regression's O(n^3)/O(n^2) cost -- see
    ``models/gaussian_process.py``'s module docstring), rather than a
    plain random subsample (which could easily miss some pixels
    entirely by chance, if a handful of pixels dominate the sample
    count), this does two passes:

    1. Coverage pass: one random sample from every distinct pixel in
       ``pixel_index``, so every pixel contributes at least one sample
       (unless the number of distinct pixels itself exceeds
       ``max_samples`` -- see the Returns section).
    2. Random pass: the remaining budget is filled with a uniform
       random sample of whatever's left.

    Deterministic for a given ``random_seed`` (does not touch or depend
    on any global random state).

    Args:
        X: Feature array of shape ``(n_samples, ...)``.
        y: Target array of shape ``(n_samples,)``.
        pixel_index: Source-pixel index per sample, shape
            ``(n_samples,)``, as returned by ``windowing.create_windows``
            (or recovered post-split the same way
            ``main.py``'s bias-estimation step does, by calling
            ``split_dataset`` again with ``pixel_index`` in place of ``y``).
        max_samples: Target sample count. If ``X`` already has this
            many samples or fewer, it's returned unchanged (no-op).
        random_seed: Seed controlling both passes, for reproducibility.
            Defaults to ``config.RANDOM_SEED``.

    Returns:
        Tuple ``(X_subset, y_subset)``, with ``X_subset.shape[0] <=
        max_samples``. If the number of distinct pixels in
        ``pixel_index`` exceeds ``max_samples``, full coverage isn't
        possible; a random subset of the coverage picks themselves is
        used instead (still one sample per pixel, just not every
        pixel).

    Raises:
        ValueError: If ``X``, ``y``, and ``pixel_index`` do not all
            have the same number of samples, or if ``max_samples < 1``.
    """
    if X.shape[0] != y.shape[0] or X.shape[0] != pixel_index.shape[0]:
        raise ValueError(
            "X, y, and pixel_index must have the same number of "
            f"samples; got X={X.shape[0]}, y={y.shape[0]}, "
            f"pixel_index={pixel_index.shape[0]}."
        )
    if max_samples < 1:
        raise ValueError(f"max_samples must be >= 1, got {max_samples}.")

    n_samples = X.shape[0]
    if n_samples <= max_samples:
        logger.info(
            "coverage_then_random_subsample: %d sample(s) already <= "
            "max_samples=%d; returning unchanged.", n_samples, max_samples,
        )
        return X, y

    rng = np.random.default_rng(random_seed)
    all_indices = np.arange(n_samples)

    coverage_indices = np.array([
        int(rng.choice(all_indices[pixel_index == pixel]))
        for pixel in np.unique(pixel_index)
    ], dtype=int)

    if coverage_indices.shape[0] >= max_samples:
        chosen = rng.choice(coverage_indices, size=max_samples, replace=False)
        logger.info(
            "coverage_then_random_subsample: %d distinct pixel(s) already "
            ">= max_samples=%d; using a random subset of the per-pixel "
            "coverage picks (full pixel coverage not possible).",
            coverage_indices.shape[0], max_samples,
        )
    else:
        remaining_pool = np.setdiff1d(all_indices, coverage_indices, assume_unique=True)
        remaining_budget = max_samples - coverage_indices.shape[0]
        extra = rng.choice(
            remaining_pool, size=min(remaining_budget, remaining_pool.shape[0]), replace=False,
        )
        chosen = np.concatenate([coverage_indices, extra])
        rng.shuffle(chosen)
        logger.info(
            "coverage_then_random_subsample: %d -> %d sample(s) (%d "
            "distinct pixel(s) covered, %d filled randomly).",
            n_samples, chosen.shape[0], coverage_indices.shape[0], extra.shape[0],
        )

    return X[chosen], y[chosen]


def create_metadata(
    X_window: np.ndarray,
    pixel_index: np.ndarray,
    time_index: np.ndarray,
    window_size: int = WINDOW_SIZE,
    forecast_horizon: int = FORECAST_HORIZON,
    stride: int = WINDOW_STRIDE,
    feature_names: tuple[str, ...] = INPUT_FEATURES,
    target_variable: str = TARGET_VARIABLE,
) -> dict[str, Any]:
    """Build a metadata dictionary describing a windowed dataset.

    Args:
        X_window: The original (pre-flatten) feature array of shape
            ``(n_samples, window_size, n_features)``, as returned by
            ``windowing.create_windows``.
        pixel_index: Source-pixel index per sample, shape
            ``(n_samples,)``, as returned by ``windowing.create_windows``.
        time_index: Target-timestep index per sample, shape
            ``(n_samples,)``, as returned by ``windowing.create_windows``.
        window_size: Number of past timesteps per window. Defaults to
            ``config.WINDOW_SIZE``.
        forecast_horizon: Number of timesteps ahead being forecast.
            Defaults to ``config.FORECAST_HORIZON``.
        stride: Step size (in timesteps) between window start indices.
            Defaults to ``config.WINDOW_STRIDE``. ``windowing.py``
            currently uses a fixed stride of 1; this value is recorded
            here purely for dataset documentation and must stay
            consistent with that behavior.
        feature_names: Names of the input feature variables, in the
            same order as the last axis of ``X_window``. Defaults to
            ``config.INPUT_FEATURES``.
        target_variable: Name of the forecasted variable. Defaults to
            ``config.TARGET_VARIABLE``.

    Returns:
        Dictionary with the following keys: ``window_size``,
        ``forecast_horizon``, ``stride``, ``feature_names``,
        ``target_variable``, ``pixel_index``, ``time_index``,
        ``n_samples``, ``n_features``.

    Raises:
        ValueError: If ``X_window`` does not have 3 dimensions, if
            ``pixel_index``/``time_index`` do not match the number of
            samples in ``X_window``, or if ``len(feature_names)`` does
            not match the number of features in ``X_window``.
    """
    if X_window.ndim != 3:
        raise ValueError(
            "Expected X_window with 3 dimensions (n_samples, "
            f"window_size, n_features), got shape {X_window.shape}. "
            "Pass the original windowing.create_windows() output, not "
            "a flattened array."
        )

    n_samples, _, n_features = X_window.shape

    if pixel_index.shape[0] != n_samples or time_index.shape[0] != n_samples:
        raise ValueError(
            "pixel_index and time_index must have length n_samples "
            f"({n_samples}); got pixel_index={pixel_index.shape[0]}, "
            f"time_index={time_index.shape[0]}."
        )
    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names has {len(feature_names)} entries but "
            f"X_window has {n_features} feature(s)."
        )

    metadata: dict[str, Any] = {
        "window_size": window_size,
        "forecast_horizon": forecast_horizon,
        "stride": stride,
        "feature_names": feature_names,
        "target_variable": target_variable,
        "pixel_index": pixel_index,
        "time_index": time_index,
        "n_samples": n_samples,
        "n_features": n_features,
    }

    logger.info(
        "Created dataset metadata: n_samples=%d, n_features=%d, "
        "window_size=%d, forecast_horizon=%d, stride=%d.",
        n_samples, n_features, window_size, forecast_horizon, stride,
    )
    return metadata


def build_dataset(
    X_window: np.ndarray,
    y_window: np.ndarray,
    pixel_index: np.ndarray,
    time_index: np.ndarray,
    flatten: bool = True,
    train_ratio: float = TRAIN_RATIO,
    shuffle: bool = SHUFFLE_DATASET,
    random_seed: int = RANDOM_SEED,
    horizon: int = FORECAST_HORIZON,
) -> dict[str, Any]:
    """Assemble a ready-to-train dataset from windowed arrays.

    Pipeline: (optional flatten) -> train/test split -> metadata ->
    return dictionary.

    Args:
        X_window: Feature array of shape
            ``(n_samples, window_size, n_features)``, as returned by
            ``windowing.create_windows``.
        y_window: Target array of shape ``(n_samples,)``, as returned
            by ``windowing.create_windows``.
        pixel_index: Source-pixel index per sample, shape
            ``(n_samples,)``, as returned by ``windowing.create_windows``.
        time_index: Target-timestep index per sample, shape
            ``(n_samples,)``, as returned by ``windowing.create_windows``.
        flatten: If ``True``, flatten ``X_window`` via
            :func:`flatten_windows` to ``(n_samples, window_size *
            n_features)`` before splitting -- for classical models
            (Gradient Boosting, Gaussian Process). If ``False``
            (default), the original 3D array is split and returned
            as-is -- for sequential models (ANN).
        train_ratio: Fraction of samples allocated to the training set.
            Defaults to ``config.TRAIN_RATIO``.
        shuffle: Whether to shuffle samples before splitting. Defaults
            to ``config.SHUFFLE_DATASET``.
        random_seed: Seed controlling the shuffle, for reproducibility.
            Defaults to ``config.RANDOM_SEED``.
        horizon: The forecast horizon ``X_window``/``y_window`` were
            actually built with (e.g. via
            ``windowing.create_windows(..., horizon=horizon)``) --
            recorded in the returned metadata. Defaults to
            ``config.FORECAST_HORIZON``; override this if the caller
            used a different horizon (e.g. ``experiment.py``'s window
            size search), so the metadata doesn't silently claim the
            global default instead of what was actually used.

    Returns:
        Dictionary with keys ``"X_train"``, ``"X_test"``, ``"y_train"``,
        ``"y_test"``, and ``"metadata"`` (see :func:`create_metadata`).

    Raises:
        ValueError: If the input arrays have inconsistent shapes (see
            :func:`create_metadata` and :func:`split_dataset` for the
            specific checks performed).
    """
    metadata = create_metadata(
        X_window=X_window,
        pixel_index=pixel_index,
        time_index=time_index,
        # window_size is derived from X_window's own shape (never a
        # separate parameter here) so it can never mismatch what was
        # actually used to build X_window -- unlike horizon, which
        # doesn't appear in any array's shape and so must be passed in.
        window_size=X_window.shape[1],
        forecast_horizon=horizon,
    )

    X_for_split = flatten_windows(X_window) if flatten else X_window
    metadata["flattened"] = flatten

    X_train, X_test, y_train, y_test = split_dataset(
        X_for_split,
        y_window,
        train_ratio=train_ratio,
        shuffle=shuffle,
        random_seed=random_seed,
    )

    logger.info(
        "Built dataset: X_train%s, X_test%s, y_train%s, y_test%s "
        "(flatten=%s).",
        X_train.shape, X_test.shape, y_train.shape, y_test.shape, flatten,
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "metadata": metadata,
    }


def time_series_kfold_indices(
    pixel_index: np.ndarray, n_splits: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Blocked, per-pixel time-series K-fold split.

    A plain shuffled K-fold is unsafe for this project's windowed
    samples: consecutive windows from the same pixel overlap by
    ``window_size - 1`` timesteps (same reason ``config.SHUFFLE_DATASET``
    is ``False`` for the ordinary train/test split -- see that setting's
    docstring), so randomly assigning individual samples to folds would
    put near-duplicate, overlapping windows on both sides of a
    train/test boundary within the same fold. A single pixel-range split
    (like :func:`split_dataset` under ``shuffle=False``) avoids that but
    can leave a fold with little or no coverage of some pixels (see
    :func:`coverage_then_random_subsample`'s docstring for why that
    happens with this project's pixel-major sample ordering).

    This does both correctly: for every distinct pixel in
    ``pixel_index``, that pixel's own samples -- in the order given,
    which ``windowing.create_windows`` produces already sorted
    chronologically within each pixel -- are divided into ``n_splits``
    contiguous, non-overlapping blocks. Fold ``k``'s test set is the
    union of block ``k`` across every pixel; its train set is every
    other block, still contiguous per pixel. So every fold has full
    pixel coverage, and no fold's train/test boundary falls inside a
    pixel's overlapping-window region.

    Args:
        pixel_index: Source-pixel index per sample, shape
            ``(n_samples,)``, in the order ``windowing.create_windows``
            produced them (chronological within each pixel) -- do not
            shuffle before calling this.
        n_splits: Number of folds. Must be >= 2.

    Returns:
        List of ``n_splits`` ``(train_indices, test_indices)`` tuples,
        each a 1D array of positions into the original ``(n_samples,)``
        arrays (sorted ascending).

    Raises:
        ValueError: If ``n_splits < 2``, or if any distinct pixel has
            fewer than ``n_splits`` samples (can't form that many
            contiguous blocks for it).
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}.")

    fold_test_positions: list[list[int]] = [[] for _ in range(n_splits)]
    for pixel in np.unique(pixel_index):
        pixel_positions = np.flatnonzero(pixel_index == pixel)
        if pixel_positions.shape[0] < n_splits:
            raise ValueError(
                f"Pixel {pixel!r} has only {pixel_positions.shape[0]} "
                f"sample(s), fewer than n_splits={n_splits}; reduce "
                "n_splits or provide more data."
            )
        for fold_i, block in enumerate(np.array_split(pixel_positions, n_splits)):
            fold_test_positions[fold_i].extend(block.tolist())

    all_positions = np.arange(pixel_index.shape[0])
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_i in range(n_splits):
        test_idx = np.array(sorted(fold_test_positions[fold_i]), dtype=int)
        train_idx = np.setdiff1d(all_positions, test_idx, assume_unique=True)
        splits.append((train_idx, test_idx))

    logger.info(
        "time_series_kfold_indices: %d fold(s) over %d distinct "
        "pixel(s), %d sample(s) total.",
        n_splits, len(np.unique(pixel_index)), pixel_index.shape[0],
    )
    return splits