"""MLflow tracking through Lightning's ``MLFlowLogger``.

Two metric namespaces, kept separate:

- ``quality/...``: model quality (losses, learning rate, gradient norm, mAP,
  per-class AP).
- ``system/...``: infrastructure (images/s, step time, data-loader wait, GPU
  memory, checkpoint save/upload time and size, upload failures, resume
  recovery time and lost steps).

Resuming reattaches to the same MLflow run (its id is stored in the
checkpoint), so one training job is one MLflow run however many times it was
killed. Metrics are logged at the global step, so the resumed curve continues
where the checkpoint left off; steps between the checkpoint and the kill are
logged twice, and that overlap is itself the "lost progress" Experiment B
measures.

This module only adds what the logger does not do: connecting up front (so an
unreachable server is reported at the start, not mid-run), tags and JSON
artifacts on the run, and registering the finished model.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

log = logging.getLogger(__name__)


def make_logger(experiment: str, run_name: str, run_id: str | None):
    """An ``MLFlowLogger`` already connected to its run, or None if MLflow is
    unreachable. Training is the point: tracking failing is loud, not fatal."""
    from lightning.pytorch.loggers import MLFlowLogger

    if not os.environ.get("MLFLOW_TRACKING_URI"):
        log.warning("MLFLOW_TRACKING_URI is not set; MLflow will write to ./mlruns")
    try:
        logger = MLFlowLogger(
            experiment_name=experiment,
            run_name=run_name,
            tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"),
            run_id=run_id,
        )
        logger.experiment  # noqa: B018 - connects and creates (or reopens) the run now
        if logger.run_id is None:
            raise RuntimeError("MLflow returned no run id")
        return logger
    except Exception as err:
        log.error("MLflow unavailable (%s); continuing without tracking", err)
        return None


def run_id(logger) -> str | None:
    return getattr(logger, "run_id", None) if logger is not None else None


def _is_mlflow(logger) -> bool:
    try:
        from lightning.pytorch.loggers import MLFlowLogger
    except ImportError:
        return False
    return isinstance(logger, MLFlowLogger)


def set_tags(logger, tags: dict[str, Any]) -> None:
    if _is_mlflow(logger):
        for key, value in tags.items():
            logger.experiment.set_tag(logger.run_id, key, str(value))


def log_dict(logger, data: dict[str, Any], name: str) -> None:
    if _is_mlflow(logger):
        logger.experiment.log_dict(logger.run_id, data, name)


def log_metrics(loggers, metrics: dict[str, Any], step: int) -> None:
    """Log to every logger, dropping values that are None or NaN."""
    clean = {
        k: float(v)
        for k, v in metrics.items()
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    }
    if not clean:
        return
    for logger in loggers or []:
        logger.log_metrics(clean, step=step)


def register_model(
    logger, model, name: str | None, metadata: dict[str, Any], model_config: dict | None = None
) -> str | None:
    """Log the fine-tuned model (state dict + config, see registry.py) and
    register it as a new version of ``name``. Returns the version.

    A new version gets no alias: it serves only after ``crosswalk promote``
    has compared it with the champion.
    """
    if not name or not _is_mlflow(logger):
        return None
    from crosswalk import registry

    version = registry.register_run_model(
        logger.run_id,
        name,
        model,
        metadata.get("classes", []),
        model_config or {},
        metadata,
    )
    log.info("registered %s version %s (no alias yet: run crosswalk promote)", name, version)
    return version
