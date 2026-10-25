"""MLflow tracking, kept behind a small interface.

Two metric namespaces, logged separately as the spec asks:

- ``quality/...``: model quality (losses, mAP, per-class AP).
- ``system/...``: infrastructure (images/s, step time, data-loader wait, GPU
  memory, checkpoint save/upload time and size, upload failures, resume
  recovery time and lost steps).

Resuming reattaches to the same MLflow run (its id is stored in the
checkpoint), so one training job is one MLflow run however many times it was
killed. Metrics are logged with the global step, so the resumed curve
continues where the checkpoint left off; steps between the checkpoint and the
kill are logged twice, and that overlap is itself the "lost progress" that
Experiment B measures.

With tracking disabled (or MLflow unreachable at start), :class:`NullTracker`
keeps the same interface and training proceeds.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from crosswalk.config import flatten

log = logging.getLogger(__name__)


class NullTracker:
    run_id: str | None = None

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        pass

    def log_dict(self, data: dict[str, Any], name: str) -> None:
        pass

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        pass

    def set_tags(self, tags: dict[str, Any]) -> None:
        pass

    def register_model(self, model, name: str | None, metadata: dict[str, Any]) -> None:
        pass

    def end(self, status: str = "FINISHED") -> None:
        pass


class MlflowTracker:
    def __init__(self, experiment: str, run_name: str, run_id: str | None = None):
        import mlflow

        self.mlflow = mlflow
        mlflow.set_experiment(experiment)
        if run_id:
            self.run = mlflow.start_run(run_id=run_id)
        else:
            self.run = mlflow.start_run(run_name=run_name)
        self.run_id = self.run.info.run_id

    def log_params(self, params: dict[str, Any]) -> None:
        # MLflow params are immutable: on resume the same values are logged
        # again (allowed); a changed value raises, which is the right outcome.
        flat = {k: ("" if v is None else str(v))[:6000] for k, v in flatten(params).items()}
        items = list(flat.items())
        for i in range(0, len(items), 100):  # MLflow's per-call limit
            self.mlflow.log_params(dict(items[i : i + 100]))

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        clean = {k: float(v) for k, v in metrics.items() if v is not None}
        if clean:
            self.mlflow.log_metrics(clean, step=step)

    def log_dict(self, data: dict[str, Any], name: str) -> None:
        self.mlflow.log_dict(data, name)

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        self.mlflow.log_artifact(path, artifact_path=artifact_path)

    def set_tags(self, tags: dict[str, Any]) -> None:
        self.mlflow.set_tags({k: str(v) for k, v in tags.items()})

    def register_model(self, model, name: str | None, metadata: dict[str, Any]) -> None:
        """Log the fine-tuned model and register a new version under ``name``."""
        import mlflow.pytorch

        info = mlflow.pytorch.log_model(
            model,
            name="model",
            registered_model_name=name,
            metadata=metadata,
        )
        log.info("logged model %s", getattr(info, "model_uri", "model"))

    def end(self, status: str = "FINISHED") -> None:
        self.mlflow.end_run(status=status)


def make_tracker(enabled: bool, experiment: str, run_name: str, run_id: str | None):
    if not enabled:
        return NullTracker()
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        log.warning("MLFLOW_TRACKING_URI is not set; MLflow will write to ./mlruns")
    try:
        return MlflowTracker(experiment, run_name, run_id)
    except Exception as err:
        # Training is the point; tracking failing should be loud, not fatal.
        log.error("MLflow unavailable (%s); continuing without tracking", err)
        return NullTracker()


def write_json_artifact(tracker, data: dict[str, Any], name: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        tracker.log_artifact(str(path))
