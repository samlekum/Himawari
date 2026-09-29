"""Model layer for AI Himawari Forecasting.

Public entry points:

- ``models.BaseModel``: common model interface/contract that every
  concrete model (ANN, Gradient Boosting, Gaussian Process, ...) must
  implement.
- ``models.get_model``: factory to instantiate a registered model by
  name (see ``config.SUPPORTED_MODELS`` / ``config.DEFAULT_MODEL``).

Concrete implementations live in their own modules (``models.ann``,
``models.gradient_boosting``, ``models.gaussian_process``) and are not
imported directly by other project modules -- orchestration code
(``experiment.py``, ``training.py``, ``forecasting.py``) should always
go through ``models.get_model``.
"""

from __future__ import annotations

from models.base import BaseModel
from models.registry import get_model

__all__ = ["BaseModel", "get_model"]
