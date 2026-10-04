"""Model registry for AI Himawari Forecasting.

Maps a model name (string, matching ``config.SUPPORTED_MODELS``) to a
concrete ``BaseModel`` subclass. This is the ONLY place in the project
that knows about every concrete model class -- orchestration modules
(``experiment.py``, ``training.py``, ``forecasting.py``) never import
``models.ann``, ``models.gradient_boosting``, or
``models.gaussian_process`` directly; they always go through
:func:`get_model`.

Adding a new model to the project means: implement ``BaseModel`` in a
new module under ``models/``, then add one entry to ``_REGISTRY``
below. No other module needs to change.
"""

from __future__ import annotations

import logging

from models.ann import ANNModel
from models.base import BaseModel
from models.gaussian_process import GaussianProcessModel
from models.gradient_boosting import GradientBoostingModel
from models.xgboost_model import XGBoostModel
from models.lightgbm_model import LightGBMModel
from models.catboost_model import CatBoostModel

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, type[BaseModel]] = {
    "ann": ANNModel,
    "gradient_boosting": GradientBoostingModel,
    "gaussian_process": GaussianProcessModel,
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
    "catboost": CatBoostModel,
}


def get_model(name: str) -> BaseModel:
    """Instantiate a registered model by name.

    Args:
        name: Model identifier. Must be a key of the registry (see
            ``config.SUPPORTED_MODELS`` for the canonical list of
            supported names).

    Returns:
        A freshly constructed instance implementing ``BaseModel``.

    Raises:
        ValueError: If ``name`` is not a registered model.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Registered models: {sorted(_REGISTRY)}."
        )

    model_cls = _REGISTRY[name]
    logger.debug("Instantiating model '%s' (%s).", name, model_cls.__name__)
    return model_cls()
