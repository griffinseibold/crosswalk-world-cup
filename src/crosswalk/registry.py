"""The MLflow model registry, as this project uses it.

**What a registered model version is.** Training registers each finished run
as a new version of ``crosswalk-detector``. The version's artifacts are a
directory ``model/`` holding

- ``crosswalk-model.pt``: ``{"format", "model" (state dict), "classes",
  "model_config"}``, loadable with ``torch.load(weights_only=True)``;
- ``model.json``: the same metadata (classes, model config, dataset hash,
  step, eval mAP) for people and tools without PyTorch.

Not a pickled ``mlflow.pytorch`` model, deliberately: unpickling a whole
model executes code from the artifact store, and ties loading to the exact
class definitions at training time. A state dict plus the config that built
the network loads safely and survives refactors.

**Aliases, not stages.** MLflow deprecated model stages (Staging/Production)
in favour of aliases. This project uses three:

- ``champion``: the version the inference service serves
  (``models:/crosswalk-detector@champion``);
- ``challenger``: the last candidate ``crosswalk promote`` evaluated;
- ``previous-champion``: the version that was champion before the latest
  promotion, so a rollback has an obvious target.

The full champion history is kept in the registered model's
``champion_history`` tag (newest last), so repeated rollbacks walk back in
time.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CHAMPION = "champion"
CHALLENGER = "challenger"
PREVIOUS = "previous-champion"
HISTORY_TAG = "champion_history"
HISTORY_LIMIT = 50
WEIGHTS_FILE = "crosswalk-model.pt"
INFO_FILE = "model.json"
FORMAT = 1

_MODELS_URI = re.compile(r"^models:/(?P<name>[^/@]+)(?:@(?P<alias>[^/]+)|/(?P<version>\d+))/?$")


@dataclass(frozen=True)
class ModelRef:
    name: str
    version: str
    alias: str | None = None

    @property
    def uri(self) -> str:
        return f"models:/{self.name}/{self.version}"

    def describe(self) -> str:
        return f"{self.uri} (@{self.alias})" if self.alias else self.uri


def is_registry_uri(uri: str | None) -> bool:
    return bool(uri) and (uri.startswith("models:/") or uri.startswith("runs:/"))


def parse_models_uri(uri: str) -> tuple[str, str | None, str | None]:
    """(name, alias, version) from ``models:/name@alias`` or ``models:/name/3``."""
    m = _MODELS_URI.match(uri)
    if not m:
        raise ValueError(f"not a models:/ URI with an alias or version: {uri!r}")
    return m.group("name"), m.group("alias"), m.group("version")


def client(tracking_uri: str | None = None):
    from mlflow import MlflowClient

    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


def resolve(uri: str, mlflow_client=None) -> ModelRef:
    """Pin an alias to the version it points at right now."""
    name, alias, version = parse_models_uri(uri)
    c = mlflow_client or client()
    if alias:
        mv = c.get_model_version_by_alias(name, alias)
        return ModelRef(name, str(mv.version), alias)
    return ModelRef(name, str(version))


def write_model_files(
    directory: str | Path, state_dict, classes: list[str], model_config: dict, info: dict
) -> Path:
    """Write the two files a registered version consists of."""
    import torch

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": FORMAT,
            "model": state_dict,
            "classes": list(classes),
            "model_config": model_config,
        },
        directory / WEIGHTS_FILE,
    )
    (directory / INFO_FILE).write_text(
        json.dumps(
            {"format": FORMAT, "classes": list(classes), "model_config": model_config, **info},
            indent=2,
            default=str,
        )
    )
    return directory


def register_run_model(
    run_id: str, name: str, model, classes, model_config: dict, info: dict
) -> str:
    """Log the model files to ``run_id`` and register them as a new version.

    Uses the client API with an explicit run id, so it works whether or not
    that run is the active one (Lightning's MLflow logger never makes it so).
    Returns the new version number.
    """
    c = client()
    with tempfile.TemporaryDirectory() as tmp:
        write_model_files(tmp, model.state_dict(), classes, model_config, info)
        c.log_artifacts(run_id, tmp, artifact_path="model")
    try:
        c.create_registered_model(name)
    except Exception as err:  # already exists
        if "already exists" not in str(err).lower() and "RESOURCE_ALREADY_EXISTS" not in str(err):
            raise
    mv = c.create_model_version(
        name, f"runs:/{run_id}/model", run_id=run_id, tags={"source": "crosswalk train"}
    )
    return str(mv.version)


def download(uri: str, cache_dir: str = "/tmp/crosswalk-models") -> Path:
    """Local directory of a registered version (``models:/`` or ``runs:/``)."""
    import mlflow

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return Path(
        mlflow.artifacts.download_artifacts(
            artifact_uri=uri, dst_path=tempfile.mkdtemp(dir=cache_dir)
        )
    )


def load_weights(uri: str, cache_dir: str = "/tmp/crosswalk-models") -> dict[str, Any]:
    """The weights file of a registered version, loaded without unpickling code."""
    import torch

    path = download(uri, cache_dir) / WEIGHTS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{uri} has no {WEIGHTS_FILE}; was it registered by crosswalk train?"
        )
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format") != FORMAT:
        raise ValueError(f"{uri}: unsupported model format {saved.get('format')}")
    return saved


# -- aliases and history ---------------------------------------------------------------


def _history(c, name: str) -> list[str]:
    raw = c.get_registered_model(name).tags.get(HISTORY_TAG, "")
    return [v for v in raw.split(",") if v]


def _set_history(c, name: str, versions: list[str]) -> None:
    c.set_registered_model_tag(name, HISTORY_TAG, ",".join(versions[-HISTORY_LIMIT:]))


def alias_version(c, name: str, alias: str) -> str | None:
    try:
        return str(c.get_model_version_by_alias(name, alias).version)
    except Exception:
        return None


def make_champion(c, name: str, version: str) -> str | None:
    """Point ``champion`` at ``version``; returns the version it replaced."""
    before = alias_version(c, name, CHAMPION)
    c.set_registered_model_alias(name, CHAMPION, version)
    if before and before != version:
        c.set_registered_model_alias(name, PREVIOUS, before)
    history = _history(c, name)
    if not history or history[-1] != version:
        history.append(version)
    _set_history(c, name, history)
    return before


def rollback(c, name: str, to_version: str | None = None) -> tuple[str, str]:
    """Move ``champion`` back. Returns (from_version, to_version).

    Without ``to_version``, goes to the champion before the current one in
    the history (falling back to the ``previous-champion`` alias).
    """
    current = alias_version(c, name, CHAMPION)
    if current is None:
        raise RuntimeError(f"{name} has no champion to roll back")
    history = _history(c, name)
    if to_version is None:
        if len(history) >= 2 and history[-1] == current:
            to_version = history[-2]
        else:
            to_version = alias_version(c, name, PREVIOUS)
        if to_version is None or to_version == current:
            raise RuntimeError(
                f"{name}: no earlier champion to roll back to (history: {history or 'empty'})"
            )
    c.get_model_version(name, to_version)  # raises if it does not exist
    c.set_registered_model_alias(name, CHAMPION, to_version)
    if history and history[-1] == current:
        history.pop()
    if not history or history[-1] != to_version:
        history.append(to_version)
    _set_history(c, name, history)
    older = history[-2] if len(history) >= 2 else None
    if older:
        c.set_registered_model_alias(name, PREVIOUS, older)
    else:
        with contextlib.suppress(Exception):  # the alias may not exist
            c.delete_registered_model_alias(name, PREVIOUS)
    return current, to_version


def load_any(uri: str, cache_dir: str = "/tmp/crosswalk-models") -> dict[str, Any]:
    """Weights and classes from a registered version or a training checkpoint.

    ``models:/`` and ``runs:/`` URIs go through the registry; anything else
    (local path, ``s3://``) is a training checkpoint. Returns at least
    ``{"model": state_dict, "classes": [...]}``.
    """
    if is_registry_uri(uri):
        return load_weights(uri, cache_dir)
    from crosswalk.storage import fetch
    from crosswalk.training import state

    return state.weights(state.load_checkpoint(fetch(uri, cache_dir)))
