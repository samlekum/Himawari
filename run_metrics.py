"""Timing and resource-usage helpers for reporting pipeline run performance.

This module is responsible ONLY for measuring wall-clock durations and
process resource usage (peak memory, CPU count). It contains no
Himawari/forecasting business logic and does not know about datasets,
models, or the pipeline's structure -- ``main.py`` is the only caller,
and it decides what to log and what to persist with the numbers this
module returns.

Added to support logging, at the end of a run, how long data loading
and training took, the total run time, and how much memory/CPU was
used -- and to feed the same numbers into
``output/<experiment_id>/summary/summary.json``.
"""

from __future__ import annotations

import logging
import os
import platform
import time
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "now",
    "elapsed_seconds",
    "get_peak_memory_gb",
    "get_resource_snapshot",
]


def now() -> float:
    """Return a monotonic timestamp suitable for measuring durations.

    Uses ``time.perf_counter()`` rather than ``time.time()`` since it
    is unaffected by system clock adjustments and is the standard
    choice for measuring elapsed wall-clock time in a single process.
    """
    return time.perf_counter()


def elapsed_seconds(start: float, end: float | None = None) -> float:
    """Return the elapsed time (seconds) between ``start`` and ``end``.

    Args:
        start: A timestamp previously returned by :func:`now`.
        end: A later timestamp returned by :func:`now`. Defaults to
            the current time (i.e. "elapsed since ``start``").

    Returns:
        Elapsed time in seconds (float).
    """
    if end is None:
        end = now()
    return end - start


def get_peak_memory_gb() -> float | None:
    """Best-effort peak resident memory usage of this process, in GB.

    Tries ``psutil`` first (accurate and cross-platform, if
    installed), then falls back to the standard-library ``resource``
    module (POSIX only -- unavailable on Windows). Returns ``None`` if
    neither is available, so callers can still log/write partial
    resource info instead of failing an entire run over a metrics
    helper.

    Note:
        ``resource.getrusage(...).ru_maxrss`` reports kilobytes on
        Linux but bytes on macOS -- handled below. ``psutil`` doesn't
        expose a uniform "peak" RSS across platforms, so current RSS
        is used there as a reasonable proxy (it only under-reports if
        memory usage has since dropped from an earlier peak).
    """
    try:
        import psutil  # type: ignore

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 3)
    except Exception:  # noqa: BLE001
        pass

    try:
        import resource  # POSIX only; not importable on Windows

        max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            return max_rss / (1024 ** 3)  # bytes -> GB
        return max_rss / (1024 ** 2)  # KiB -> GB
    except Exception:  # noqa: BLE001
        logger.debug("Could not determine peak memory usage.", exc_info=True)
        return None


def get_resource_snapshot() -> dict[str, Any]:
    """Return a small dict summarizing process/host resource usage.

    Returns:
        Dictionary with ``"peak_memory_gb"`` (``float`` or ``None``,
        see :func:`get_peak_memory_gb`), ``"cpu_count"`` (``int`` or
        ``None``), and ``"platform"`` (``str``).
    """
    return {
        "peak_memory_gb": get_peak_memory_gb(),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
    }