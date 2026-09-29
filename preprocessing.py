"""Preprocessing utilities for AI Himawari Forecasting.

This module is responsible ONLY for turning raw Himawari-9 ``xarray.Dataset``
objects into clean, model-ready ``numpy.ndarray`` feature/target arrays. It
implements the following pipeline, applied per timestamp:

    xarray.Dataset
        -> validate_dataset
        -> subset_bandung
        -> extract input features / target variable
        -> handle_missing_values
        -> flatten_frame
        -> optional scaling (transform_features, if a fitted scaler is
           supplied by the caller)
        -> numpy.ndarray

Feature and target variable names, as well as the Bandung bounding box,
are read from ``config.py`` (``INPUT_FEATURES``, ``TARGET_VARIABLE``,
``SCALER``, ``BANDUNG_LAT_MIN``/``BANDUNG_LAT_MAX``/``BANDUNG_LON_MIN``/
``BANDUNG_LON_MAX``) and are never hardcoded here.

This module never fits scalers and never loads data from disk. It only
transforms whatever ``xarray.Dataset`` object(s) it is given:

- Scaler *fitting* is the responsibility of the training pipeline
  (e.g. ``training.py``); this module only exposes ``fit_scaler``,
  ``transform_features``, and ``inverse_transform`` as reusable
  utilities that operate on scalers the caller decides when to fit.
- Data loading is the responsibility of the caller (e.g. ``main.py`` /
  ``pipeline.py``), which is expected to obtain ``xarray.Dataset``
  objects via ``loader.py`` and pass them in.

No windowing, forecasting, model training, evaluation, visualization,
or data-loading logic belongs in this module.
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from config import (
    BANDUNG_LAT_MAX,
    BANDUNG_LAT_MIN,
    BANDUNG_LON_MAX,
    BANDUNG_LON_MIN,
    INPUT_FEATURES,
    SCALER,
    TARGET_VARIABLE,
)

logger = logging.getLogger(__name__)

#: Type alias for the scaler objects this module works with. ``None``
#: represents the "no scaling" case.
Scaler = StandardScaler | MinMaxScaler | None

#: Missing-value handling strategies supported by handle_missing_values().
SUPPORTED_MISSING_STRATEGIES: tuple[str, ...] = ("mean", "median", "zero", "drop")

#: Scaler identifiers supported by fit_scaler().
SUPPORTED_SCALERS: tuple[str, ...] = ("standard", "minmax", "none")

#: Spatial dimension names expected in raw Himawari datasets.
_SPATIAL_DIMS: tuple[str, ...] = ("latitude", "longitude")

#: Name of the flattened spatial dimension created by this module.
_PIXEL_DIM: str = "pixel"


def validate_dataset(dataset: xr.Dataset) -> None:
    """Validate that a raw dataset has the structure this pipeline expects.

    Checks that ``dataset`` is an ``xarray.Dataset``, that it exposes
    ``latitude``/``longitude`` coordinates, that all configured
    ``INPUT_FEATURES`` and ``TARGET_VARIABLE`` are present as data
    variables, and that the spatial grid is non-empty.

    Args:
        dataset: Raw dataset loaded via ``loader.load_single_nc`` (or
            equivalent).

    Raises:
        TypeError: If ``dataset`` is not an ``xarray.Dataset``.
        KeyError: If required coordinates or variables are missing.
        ValueError: If the spatial grid has zero size along latitude or
            longitude.
    """
    if not isinstance(dataset, xr.Dataset):
        raise TypeError(
            f"Expected an xarray.Dataset, got: {type(dataset).__name__}."
        )

    missing_coords = [c for c in _SPATIAL_DIMS if c not in dataset.coords]
    if missing_coords:
        raise KeyError(
            f"Dataset is missing required coordinate(s): {missing_coords}."
        )

    required_vars = set(INPUT_FEATURES) | {TARGET_VARIABLE}
    missing_vars = sorted(v for v in required_vars if v not in dataset.data_vars)
    if missing_vars:
        raise KeyError(
            f"Dataset is missing required data variable(s): {missing_vars}."
        )

    if dataset.sizes.get("latitude", 0) == 0 or dataset.sizes.get("longitude", 0) == 0:
        raise ValueError("Dataset has an empty spatial grid (latitude/longitude).")

    logger.debug("Dataset validated successfully.")


def subset_bandung(
    dataset: xr.Dataset,
    lat_min: float = BANDUNG_LAT_MIN,
    lat_max: float = BANDUNG_LAT_MAX,
    lon_min: float = BANDUNG_LON_MIN,
    lon_max: float = BANDUNG_LON_MAX,
) -> xr.Dataset:
    """Subset a dataset to the Bandung region bounding box.

    Uses a boolean mask instead of ``.sel(slice(...))`` so the subset is
    correct regardless of whether the latitude coordinate is stored in
    ascending or descending order.

    Args:
        dataset: Raw or already-validated dataset containing ``latitude``
            and ``longitude`` coordinates.
        lat_min: Minimum latitude (inclusive) of the target region.
        lat_max: Maximum latitude (inclusive) of the target region.
        lon_min: Minimum longitude (inclusive) of the target region.
        lon_max: Maximum longitude (inclusive) of the target region.

    Returns:
        Dataset cropped to the requested bounding box.

    Raises:
        ValueError: If the resulting subset is empty (no pixels fall
            within the requested bounding box).
    """
    lat_mask = (dataset["latitude"] >= lat_min) & (dataset["latitude"] <= lat_max)
    lon_mask = (dataset["longitude"] >= lon_min) & (dataset["longitude"] <= lon_max)

    subset = dataset.where(lat_mask & lon_mask, drop=True)

    if subset.sizes.get("latitude", 0) == 0 or subset.sizes.get("longitude", 0) == 0:
        raise ValueError(
            "Bandung subset is empty for bounding box "
            f"lat=[{lat_min}, {lat_max}], lon=[{lon_min}, {lon_max}]."
        )

    logger.debug(
        "Subset to Bandung region: %d latitude x %d longitude points.",
        subset.sizes["latitude"],
        subset.sizes["longitude"],
    )
    return subset


def _extract_input_features(
    dataset: xr.Dataset, input_features: tuple[str, ...] = INPUT_FEATURES
) -> xr.Dataset:
    """Select the configured input-feature variables from a dataset.

    Args:
        dataset: Dataset to select variables from.
        input_features: Names of variables to keep, in order.

    Returns:
        Dataset containing only ``input_features``, in the given order.

    Raises:
        KeyError: If any requested feature is not present in ``dataset``.
    """
    missing = [f for f in input_features if f not in dataset.data_vars]
    if missing:
        raise KeyError(f"Input feature(s) not found in dataset: {missing}.")
    return dataset[list(input_features)]


def extract_target_variable(
    dataset: xr.Dataset, target_variable: str = TARGET_VARIABLE
) -> xr.DataArray:
    """Extract the target variable from a dataset.

    Args:
        dataset: Dataset to extract the target from.
        target_variable: Name of the target data variable.

    Returns:
        The target variable as an ``xarray.DataArray``.

    Raises:
        KeyError: If ``target_variable`` is not present in ``dataset``.
    """
    if target_variable not in dataset.data_vars:
        raise KeyError(f"Target variable '{target_variable}' not found in dataset.")
    return dataset[target_variable]


def _fill_residual_nan(data: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """Safety net for the "mean"/"median" strategies in handle_missing_values.

    Both strategies fill NaN with a value computed *from the same
    timestamp's own spatial grid*. If an entire timestamp's grid is
    missing for some variable (uncommon but real over a large enough
    dataset -- e.g. one corrupted/blank source file among tens of
    thousands), that computed mean/median is itself NaN, so filling
    with it leaves the original NaN in place. Whatever's still NaN
    after the primary strategy is filled with ``0.0`` here instead, and
    logged, so it's a visible, explicit fallback rather than a NaN that
    silently survives all the way to a downstream consumer that rejects
    it outright (e.g. most scikit-learn estimators).

    Args:
        data: Result of applying the "mean"/"median" fill.

    Returns:
        ``data`` with any remaining NaN filled with ``0.0``.
    """
    if isinstance(data, xr.Dataset):
        remaining = int(sum(int(data[var].isnull().sum()) for var in data.data_vars))
    else:
        remaining = int(data.isnull().sum())

    if remaining:
        logger.warning(
            "handle_missing_values: %d value(s) still NaN after the "
            "requested mean/median strategy -- likely an entire "
            "timestamp's grid missing for some variable (so the "
            "per-timestamp mean/median itself was NaN). Filling the "
            "remainder with 0.0 as a last resort.",
            remaining,
        )
        data = data.fillna(0.0)
    return data


def handle_missing_values(
    data: xr.Dataset | xr.DataArray, strategy: str = "mean"
) -> xr.Dataset | xr.DataArray:
    """Handle missing values in gridded Himawari data.

    Supported strategies:

    - ``"mean"``: fill NaNs with the per-variable spatial mean.
    - ``"median"``: fill NaNs with the per-variable spatial median.
    - ``"zero"``: fill NaNs with ``0.0``.
    - ``"drop"``: stack the spatial dimensions into a single ``pixel``
      dimension and drop any pixel containing a NaN in any variable.

    Args:
        data: Gridded dataset or data array, typically the output of
            :func:`subset_bandung` or the combined feature/target
            dataset built in :func:`preprocess_single_dataset`.
        strategy: One of ``"mean"``, ``"median"``, ``"zero"``, ``"drop"``.

    Returns:
        Data with missing values handled. For ``"mean"``/``"median"``/
        ``"zero"`` the spatial grid shape is preserved. For ``"drop"``
        the spatial dimensions are collapsed into a single ``pixel``
        dimension with any incomplete pixels removed.

    Raises:
        ValueError: If ``strategy`` is not supported, or if the "drop"
            strategy removes every pixel.
    """
    if strategy not in SUPPORTED_MISSING_STRATEGIES:
        raise ValueError(
            f"Unsupported missing-value strategy: '{strategy}'. "
            f"Supported strategies: {SUPPORTED_MISSING_STRATEGIES}."
        )

    if strategy == "zero":
        return data.fillna(0.0)

    if strategy == "mean":
        reduce_dims = [d for d in data.dims if d in (*_SPATIAL_DIMS, _PIXEL_DIM)]
        return _fill_residual_nan(data.fillna(data.mean(dim=reduce_dims, skipna=True)))

    if strategy == "median":
        reduce_dims = [d for d in data.dims if d in (*_SPATIAL_DIMS, _PIXEL_DIM)]
        return _fill_residual_nan(data.fillna(data.median(dim=reduce_dims, skipna=True)))

    # strategy == "drop"
    spatial_dims = [d for d in data.dims if d in _SPATIAL_DIMS]
    stacked = data.stack({_PIXEL_DIM: spatial_dims}) if spatial_dims else data

    cleaned = stacked.dropna(dim=_PIXEL_DIM, how="any")
    if cleaned.sizes.get(_PIXEL_DIM, 0) == 0:
        raise ValueError("The 'drop' missing-value strategy removed every pixel.")

    n_removed = stacked.sizes[_PIXEL_DIM] - cleaned.sizes[_PIXEL_DIM]
    if n_removed:
        logger.info("Dropped %d pixel(s) containing missing values.", n_removed)

    return cleaned


def flatten_frame(data: xr.Dataset | xr.DataArray) -> np.ndarray:
    """Flatten a gridded (or pixel-stacked) frame into a 2D/1D array.

    Spatial dimensions (``latitude``/``longitude``) are stacked into a
    single ``pixel`` axis if not already stacked (e.g. by the "drop"
    missing-value strategy). ``xarray.Dataset`` inputs produce an array
    of shape ``(n_pixels, n_variables)`` with columns ordered by the
    dataset's variable order. ``xarray.DataArray`` inputs produce a 1D
    array of shape ``(n_pixels,)``.

    Args:
        data: Gridded or pixel-stacked dataset/data array to flatten.

    Returns:
        A ``numpy.ndarray`` with pixels as rows.

    Raises:
        ValueError: If no spatial or pixel dimension can be found.
    """
    spatial_dims = [d for d in data.dims if d in _SPATIAL_DIMS]
    if spatial_dims:
        data = data.stack({_PIXEL_DIM: spatial_dims})

    if _PIXEL_DIM not in data.dims:
        raise ValueError(
            f"Could not find spatial dims {_SPATIAL_DIMS} or a '{_PIXEL_DIM}' "
            f"dimension to flatten. Got dims: {tuple(data.dims)}."
        )

    if isinstance(data, xr.Dataset):
        columns = [data[var].transpose(_PIXEL_DIM).values for var in data.data_vars]
        return np.column_stack(columns)

    return np.asarray(data.transpose(_PIXEL_DIM).values)


def fit_scaler(data: np.ndarray, scaler_type: str = SCALER) -> Scaler:
    """Fit a scaler on a 2D feature array.

    Args:
        data: Array of shape ``(n_samples, n_features)`` to fit the
            scaler on. For a single target variable, reshape a 1D array
            to ``(-1, 1)`` before calling this function.
        scaler_type: One of ``"standard"``, ``"minmax"``, ``"none"``.

    Returns:
        A fitted ``StandardScaler``/``MinMaxScaler`` instance, or
        ``None`` if ``scaler_type`` is ``"none"``.

    Raises:
        ValueError: If ``scaler_type`` is not supported.
    """
    if scaler_type not in SUPPORTED_SCALERS:
        raise ValueError(
            f"Unsupported scaler type: '{scaler_type}'. "
            f"Supported scalers: {SUPPORTED_SCALERS}."
        )

    if scaler_type == "none":
        return None

    scaler: Scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
    scaler.fit(data)

    logger.debug(
        "Fitted '%s' scaler on data of shape %s.", scaler_type, data.shape
    )
    return scaler


def transform_features(data: np.ndarray, scaler: Scaler) -> np.ndarray:
    """Apply a fitted scaler to a feature array.

    Args:
        data: Array of shape ``(n_samples, n_features)`` to transform.
        scaler: A fitted scaler returned by :func:`fit_scaler`, or
            ``None`` for no scaling.

    Returns:
        The transformed array, or an unchanged copy of ``data`` if
        ``scaler`` is ``None``.
    """
    if scaler is None:
        return np.array(data, copy=True)
    return scaler.transform(data)


def inverse_transform(data: np.ndarray, scaler: Scaler) -> np.ndarray:
    """Invert a fitted scaler's transformation.

    Args:
        data: Array of shape ``(n_samples, n_features)`` in scaled
            space.
        scaler: A fitted scaler returned by :func:`fit_scaler`, or
            ``None`` for no scaling.

    Returns:
        The array in its original (unscaled) space, or an unchanged
        copy of ``data`` if ``scaler`` is ``None``.
    """
    if scaler is None:
        return np.array(data, copy=True)
    return scaler.inverse_transform(data)


def preprocess_single_dataset(
    dataset: xr.Dataset,
    missing_strategy: str = "mean",
    feature_scaler: Scaler = None,
    target_scaler: Scaler = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the full preprocessing pipeline on a single timestamp.

    Pipeline: validate -> subset to Bandung -> extract input features and
    target -> handle missing values (jointly, so features and target stay
    row-aligned) -> flatten -> optional scaling.

    Args:
        dataset: Raw dataset for a single timestamp.
        missing_strategy: One of ``"mean"``, ``"median"``, ``"zero"``,
            ``"drop"``. See :func:`handle_missing_values`.
        feature_scaler: Optional pre-fitted scaler applied to the input
            features. If ``None``, features are returned unscaled.
        target_scaler: Optional pre-fitted scaler applied to the target
            variable. If ``None``, the target is returned unscaled.

    Returns:
        Tuple ``(X, y)`` where ``X`` has shape ``(n_pixels, n_features)``
        and ``y`` has shape ``(n_pixels,)``.

    Raises:
        TypeError: If ``dataset`` is not an ``xarray.Dataset``.
        KeyError: If required coordinates/variables are missing.
        ValueError: If the Bandung subset or missing-value handling
            yields no usable pixels, or an unsupported strategy is given.
    """
    validate_dataset(dataset)
    subset = subset_bandung(dataset)

    features_ds = _extract_input_features(subset, INPUT_FEATURES)
    target_da = extract_target_variable(subset, TARGET_VARIABLE)

    # Combine features and target before handling missing values so that,
    # under the "drop" strategy, both stay aligned on the same set of
    # remaining pixels.
    combined = features_ds.assign(**{TARGET_VARIABLE: target_da})
    combined = handle_missing_values(combined, strategy=missing_strategy)

    X = flatten_frame(combined[list(INPUT_FEATURES)])
    y = flatten_frame(combined[TARGET_VARIABLE])

    if feature_scaler is not None:
        X = transform_features(X, feature_scaler)
    if target_scaler is not None:
        y = transform_features(y.reshape(-1, 1), target_scaler).ravel()

    return X, y


def preprocess_day(
    datasets: list[xr.Dataset],
    missing_strategy: str = "mean",
    feature_scaler: Scaler = None,
    target_scaler: Scaler = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess a chronologically ordered list of single-timestamp datasets.

    Each dataset (timestamp) is preprocessed independently via
    :func:`preprocess_single_dataset` and stacked along a new leading
    time axis. This module never loads data from disk itself; the caller
    (e.g. ``main.py`` / ``pipeline.py``) is responsible for obtaining
    ``datasets`` -- typically via ``loader.load_day`` -- in chronological
    order.

    This function never fits scalers. If ``feature_scaler``/
    ``target_scaler`` are provided, they are applied via
    :func:`transform_features`. If omitted, the raw (unscaled) features
    and target are returned. Deciding when/whether to fit a scaler is
    the responsibility of the training pipeline.

    Args:
        datasets: Chronologically ordered list of single-timestamp
            datasets, e.g. as returned by ``loader.load_day``.
        missing_strategy: One of ``"mean"``, ``"median"``, ``"zero"``,
            ``"drop"``. See :func:`handle_missing_values`.
        feature_scaler: Optional pre-fitted scaler for input features.
            If ``None``, features are returned unscaled.
        target_scaler: Optional pre-fitted scaler for the target. If
            ``None``, the target is returned unscaled.

    Returns:
        Tuple ``(X, y)`` where ``X`` has shape
        ``(n_timesteps, n_pixels, n_features)`` and ``y`` has shape
        ``(n_timesteps, n_pixels)``.

    Raises:
        TypeError: If any dataset is not an ``xarray.Dataset``.
        KeyError: If required coordinates/variables are missing from
            any dataset.
        ValueError: If per-timestamp pixel counts differ (e.g. the
            "drop" strategy removes different pixels on different
            timestamps), which prevents stacking into a single array.
    """
    raw_X: list[np.ndarray] = []
    raw_y: list[np.ndarray] = []
    for dataset in datasets:
        X_i, y_i = preprocess_single_dataset(
            dataset,
            missing_strategy=missing_strategy,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
        )
        raw_X.append(X_i)
        raw_y.append(y_i)

    try:
        X = np.stack(raw_X, axis=0)
        y = np.stack(raw_y, axis=0)
    except ValueError as exc:
        raise ValueError(
            "Inconsistent pixel counts across timestamps; this can happen "
            "with the 'drop' missing-value strategy. "
            f"Original error: {exc}"
        ) from exc

    logger.info(
        "Preprocessed %d timestamp(s) -> X%s, y%s.",
        len(datasets), X.shape, y.shape,
    )
    return X, y