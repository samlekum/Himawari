"""Dataset loading utilities for AI Himawari Forecasting.

This module is responsible ONLY for locating and loading raw Himawari-9
NetCDF files from disk. It provides reusable, side-effect-free functions to:

- discover available months and days in the dataset tree,
- discover and validate NetCDF files within a given day,
- open a single NetCDF file as an xarray.Dataset,
- open all files for a given day or month as lists of datasets.

No preprocessing, flattening, windowing, or modeling logic belongs here.

Performance notes (why this loads in parallel PROCESSES, not threads):
    The netCDF4/HDF5 C libraries backing xarray are not thread-safe, so
    xarray/netCDF4-python serialize concurrent reads behind a
    process-wide lock. A ``ThreadPoolExecutor`` therefore barely
    parallelizes the actual file-opening/decoding work no matter how
    many threads are used -- it mostly adds thread-scheduling overhead
    on top of effectively serial decoding. Separate worker *processes*
    each get their own copy of that lock, so they genuinely run in
    parallel; this is the standard fix for "loading many small NetCDF
    files is much slower than it should be". On top of that:

    - The engine (backend) used to open every file is detected ONCE
      (see ``_resolve_engine``) and then passed explicitly to every
      subsequent open, instead of letting xarray re-probe every
      installed backend on every single file.
    - ``load_month`` shares ONE worker pool across every day in the
      month (via the ``executor`` parameter threaded through
      ``load_day``), instead of starting and tearing down a new pool
      for each day -- pool startup is comparatively expensive, and a
      month can have dozens of days.
    - Files are dispatched to workers in small batches
      (``chunksize``) rather than one at a time, which matters when
      individual files are tiny (tens of KB): with thousands of such
      files, the fixed cost of shipping one task and its result across
      the process boundary can otherwise rival the actual decode time.
"""

from __future__ import annotations

import concurrent.futures
import functools
import logging
import os
import threading
from pathlib import Path

import xarray as xr

from config import DATASET_PATH

logger = logging.getLogger(__name__)

NC_EXTENSION: str = ".nc"

# Upper bound on the number of parallel worker processes used when loading
# NetCDF files.
_MAX_WORKERS: int = 16

# Candidate xarray backend engines tried, in order, exactly once (see
# _resolve_engine). Whichever one can actually open this dataset is then
# reused explicitly for every subsequent file, skipping xarray's
# per-open automatic backend detection.
_CANDIDATE_ENGINES: tuple[str, ...] = ("h5netcdf", "netcdf4", "scipy")

_engine_lock = threading.Lock()
_resolved_engine: str | None = None
_engine_resolution_attempted: bool = False

_process_pool_lock = threading.Lock()
_process_pool_ok: bool | None = None


def _default_max_workers() -> int:
    """Compute a sensible default worker-process pool size.

    Returns:
        ``min(_MAX_WORKERS, os.cpu_count())`` (falls back to a CPU
        count of 1 if ``os.cpu_count()`` is unavailable), never fewer
        than 1. Unlike an I/O-bound thread pool, worker *processes*
        aren't worth oversubscribing past the physical core count:
        once bytes are off disk, decoding a NetCDF file is real CPU
        work, and (per-process) no longer contends with any
        interpreter-wide or C-library-wide lock -- see the module
        docstring -- so more processes than cores just adds scheduling
        overhead.
    """
    cpu_count = os.cpu_count() or 1
    return max(1, min(_MAX_WORKERS, cpu_count))


def _resolve_engine(sample_path: Path) -> str | None:
    """Determine which xarray backend engine can open this dataset, once.

    Tries each of ``_CANDIDATE_ENGINES`` against ``sample_path`` and
    caches (module-wide) the first one that succeeds, so every other
    call site in this module can pass that engine explicitly instead
    of letting xarray re-probe every installed backend on every open.

    Args:
        sample_path: Any real ``.nc`` file from the dataset, used only
            to test which engine can read it.

    Returns:
        The name of a working engine, or ``None`` if none of the
        candidates could open the sample file (callers then fall back
        to xarray's own per-file auto-detection, which is slower but
        still correct).
    """
    global _resolved_engine, _engine_resolution_attempted

    with _engine_lock:
        if _engine_resolution_attempted:
            return _resolved_engine

        for engine in _CANDIDATE_ENGINES:
            try:
                with xr.open_dataset(sample_path, engine=engine):
                    pass
            except Exception:  # noqa: BLE001
                continue
            else:
                logger.info("Using xarray engine '%s' for all NetCDF reads.", engine)
                _resolved_engine = engine
                break
        else:
            logger.warning(
                "Could not confirm a working engine among %s for '%s'; "
                "falling back to per-file auto-detection (slower).",
                _CANDIDATE_ENGINES, sample_path,
            )

        _engine_resolution_attempted = True
        return _resolved_engine


def _noop_worker() -> int:
    """Trivial, picklable function used only to test process-pool startup."""
    return 1


def _process_pool_available() -> bool:
    """Best-effort, cached check for whether worker processes can start here.

    Some sandboxed/restricted environments (or certain interactive
    notebook setups) disallow spawning subprocesses even though
    ``ProcessPoolExecutor`` can be constructed without error -- the
    failure only shows up once a task is actually submitted. Running
    one trivial task, once, up front, and caching the answer avoids
    ever misattributing a genuine per-file error (e.g. one truly
    corrupt NetCDF file) to "can't use processes here" during the real
    load later on.

    Returns:
        ``True`` if a ``ProcessPoolExecutor`` could run one no-op task,
        ``False`` otherwise (callers should use a ``ThreadPoolExecutor``
        instead -- slower for this workload, but always available).
    """
    global _process_pool_ok

    with _process_pool_lock:
        if _process_pool_ok is not None:
            return _process_pool_ok
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=1) as pool:
                pool.submit(_noop_worker).result(timeout=30)
            _process_pool_ok = True
        except Exception:  # noqa: BLE001
            logger.warning(
                "Process-based parallel NetCDF loading isn't available in "
                "this environment; falling back to a thread pool (slower "
                "for this workload -- see this module's docstring).",
                exc_info=True,
            )
            _process_pool_ok = False
        return _process_pool_ok


def _make_executor(max_workers: int) -> concurrent.futures.Executor:
    """Create the best available parallel executor for loading files.

    Args:
        max_workers: Pool size (see ``_default_max_workers``).

    Returns:
        A ``ProcessPoolExecutor`` where available (the fast path --
        see the module docstring), otherwise a ``ThreadPoolExecutor``.
    """
    if _process_pool_available():
        return concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
    return concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)


def scan_available_months(dataset_path: Path = DATASET_PATH) -> list[str]:
    """Scan the dataset root and list all available month folders.

    Args:
        dataset_path: Root directory of the NetCDF dataset.

    Returns:
        Sorted list of month folder names (e.g. ["202601", "202602"]).

    Raises:
        FileNotFoundError: If dataset_path does not exist, or contains
            no month subdirectories.
        NotADirectoryError: If dataset_path is not a directory.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset root not found: '{dataset_path}'.")
    if not dataset_path.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: '{dataset_path}'.")

    months = sorted(entry.name for entry in dataset_path.iterdir() if entry.is_dir())

    if not months:
        raise FileNotFoundError(f"No month folders found under dataset root: '{dataset_path}'.")

    logger.info("Found %d month folder(s) under '%s'.", len(months), dataset_path)
    return months


def scan_available_days(month: str, dataset_path: Path = DATASET_PATH) -> list[str]:
    """Scan a month folder and list all available day folders.

    Args:
        month: Month folder name in YYYYMM format (e.g. "202601").
        dataset_path: Root directory of the NetCDF dataset.

    Returns:
        Sorted list of day folder names (e.g. ["01", "02", "03"]).

    Raises:
        FileNotFoundError: If the month folder does not exist, or
            contains no day subdirectories.
        NotADirectoryError: If the month path is not a directory.
    """
    month_path = dataset_path / month

    if not month_path.exists():
        raise FileNotFoundError(f"Month folder not found: '{month_path}'.")
    if not month_path.is_dir():
        raise NotADirectoryError(f"Month path is not a directory: '{month_path}'.")

    days = sorted(entry.name for entry in month_path.iterdir() if entry.is_dir())

    if not days:
        raise FileNotFoundError(f"No day folders found under month folder: '{month_path}'.")

    logger.info("Found %d day folder(s) under '%s'.", len(days), month_path)
    return days


def scan_nc_files(month: str, day: str, dataset_path: Path = DATASET_PATH) -> list[Path]:
    """Scan a day folder and list all NetCDF (.nc) file paths.

    Args:
        month: Month folder name in YYYYMM format (e.g. "202601").
        day: Day folder name in DD format (e.g. "01").
        dataset_path: Root directory of the NetCDF dataset.

    Returns:
        Sorted list of absolute paths to .nc files within the day folder.

    Raises:
        FileNotFoundError: If the day folder does not exist, or contains
            no .nc files.
        NotADirectoryError: If the day path is not a directory.
    """
    day_path = dataset_path / month / day

    if not day_path.exists():
        raise FileNotFoundError(f"Day folder not found: '{day_path}'.")
    if not day_path.is_dir():
        raise NotADirectoryError(f"Day path is not a directory: '{day_path}'.")

    nc_files = sorted(day_path.glob(f"*{NC_EXTENSION}"))

    if not nc_files:
        raise FileNotFoundError(f"No '{NC_EXTENSION}' files found under day folder: '{day_path}'.")

    logger.info("Found %d NetCDF file(s) under '%s'.", len(nc_files), day_path)
    return nc_files


def load_single_nc(file_path: Path, engine: str | None = None) -> xr.Dataset:
    """Load a single NetCDF file into an xarray.Dataset.

    Args:
        file_path: Path to a single .nc file.
        engine: Explicit xarray backend engine to use (see
            ``_resolve_engine``). Defaults to ``None``, which resolves
            (and caches) a working engine from this very file the
            first time it's called with no explicit engine.

    Returns:
        The loaded dataset, read into memory.

    Raises:
        FileNotFoundError: If file_path does not exist.
        ValueError: If file_path does not have a .nc extension.
        OSError: If the file exists but cannot be opened/parsed.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"NetCDF file not found: '{file_path}'.")
    if file_path.suffix.lower() != NC_EXTENSION:
        raise ValueError(f"Expected a '{NC_EXTENSION}' file, got: '{file_path}'.")

    if engine is None:
        engine = _resolve_engine(file_path)

    try:
        with xr.open_dataset(file_path, engine=engine) as handle:
            dataset = handle.load()
    except Exception as exc:  # noqa: BLE001
        raise OSError(f"Failed to open NetCDF file '{file_path}': {exc}") from exc

    logger.debug("Loaded NetCDF file: '%s'.", file_path)
    return dataset


def _load_many(
    file_paths: list[Path],
    engine: str | None,
    executor: concurrent.futures.Executor | None = None,
) -> list[xr.Dataset]:
    """Load many NetCDF files in parallel, preserving input order.

    Args:
        file_paths: Paths to load, in the order they should appear in
            the returned list.
        engine: Explicit xarray engine to use for every file (see
            ``_resolve_engine``), or ``None`` to let each file
            auto-detect its own engine.
        executor: An already-running ``Executor`` to submit work to,
            letting callers (e.g. ``load_month``) reuse one pool
            across many calls instead of paying pool-startup cost
            repeatedly. A local pool is created and torn down
            automatically when this is ``None``.

    Returns:
        Loaded datasets, in the same order as ``file_paths``.

    Raises:
        OSError: If any individual file fails to load (same error
            ``load_single_nc`` raises for that file). Other in-flight
            loads are still allowed to finish before the error
            propagates; no partial results are ever returned.
    """
    total = len(file_paths)
    max_workers = _default_max_workers()
    # Batch several files per worker round-trip: with thousands of small
    # (tens-of-KB) files, the fixed cost of shipping one task and its
    # result across the process boundary can otherwise rival the actual
    # decode time.
    chunksize = max(1, total // (max_workers * 4))
    worker = functools.partial(load_single_nc, engine=engine)

    def _run(pool: concurrent.futures.Executor) -> list[xr.Dataset]:
        results: list[xr.Dataset] = []
        progress_step = max(1, total // 10)
        for done, dataset in enumerate(
            pool.map(worker, file_paths, chunksize=chunksize), start=1,
        ):
            results.append(dataset)
            if done % progress_step == 0 or done == total:
                logger.info("%d/%d loaded", done, total)
        return results

    if executor is not None:
        return _run(executor)

    pool = _make_executor(max_workers)
    try:
        return _run(pool)
    finally:
        pool.shutdown(wait=True)


def load_day(
    month: str,
    day: str,
    dataset_path: Path = DATASET_PATH,
    executor: concurrent.futures.Executor | None = None,
) -> list[xr.Dataset]:
    """Load all NetCDF files for a given day into a list of datasets.

    Args:
        month: Month folder name in YYYYMM format (e.g. "202601").
        day: Day folder name in DD format (e.g. "01").
        dataset_path: Root directory of the NetCDF dataset.
        executor: An already-running pool to load files with (see
            ``load_month``, which shares one pool across every day of
            a month). A local pool is created and torn down
            automatically when this is ``None``.

    Returns:
        List of loaded xarray.Dataset objects, in sorted order.

    Raises:
        FileNotFoundError: If the day folder does not exist or is empty.
        NotADirectoryError: If the day path is not a directory.
        OSError: If any individual NetCDF file fails to load.

    Note:
        Files are loaded in parallel worker processes (see this
        module's docstring for why processes, not threads), but the
        returned list is always in the same order as ``scan_nc_files``
        (i.e. sorted order) regardless of the order in which
        individual loads finish. If one or more files fail to load,
        the resulting OSError is the same as ``load_single_nc`` would
        raise for that file; other in-flight loads are still allowed
        to finish before the error propagates, matching the fact that
        no partial results are ever returned.
    """
    file_paths = scan_nc_files(month=month, day=day, dataset_path=dataset_path)
    total = len(file_paths)

    logger.info("Loading %d NetCDF files...", total)

    engine = _resolve_engine(file_paths[0])
    datasets = _load_many(file_paths, engine=engine, executor=executor)

    logger.info("Loaded %d dataset(s) for %s/%s.", len(datasets), month, day)
    return datasets


def load_month(month: str, dataset_path: Path = DATASET_PATH) -> dict[str, list[xr.Dataset]]:
    """Load all NetCDF files for every day within a given month.

    Args:
        month: Month folder name in YYYYMM format (e.g. "202601").
        dataset_path: Root directory of the NetCDF dataset.

    Returns:
        Dictionary mapping each day folder name to its list of loaded
        xarray.Dataset objects, in sorted day order.

    Raises:
        FileNotFoundError: If the month folder does not exist or has
            no day subdirectories.
        NotADirectoryError: If the month path is not a directory.
        OSError: If any individual NetCDF file fails to load.

    Note:
        A single worker pool is created once and shared across every
        day in the month (rather than one pool per day), since pool
        startup is comparatively expensive and a month can have dozens
        of days -- see this module's docstring.
    """
    days = scan_available_days(month=month, dataset_path=dataset_path)

    first_day_files = scan_nc_files(month=month, day=days[0], dataset_path=dataset_path)
    _resolve_engine(first_day_files[0])

    max_workers = _default_max_workers()
    month_data: dict[str, list[xr.Dataset]] = {}

    pool = _make_executor(max_workers)
    try:
        for day in days:
            month_data[day] = load_day(
                month=month, day=day, dataset_path=dataset_path, executor=pool,
            )
    finally:
        pool.shutdown(wait=True)

    total_files = sum(len(v) for v in month_data.values())
    logger.info(
        "Loaded %d day(s) / %d total file(s) for month '%s'.",
        len(month_data), total_files, month,
    )
    return month_data