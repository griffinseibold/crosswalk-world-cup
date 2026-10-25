from pathlib import Path

import pytest
from pydantic import ValidationError

from crosswalk.config import (
    BatchingConfig,
    BenchmarkConfig,
    InferenceConfig,
    TrainingConfig,
    apply_overrides,
    config_hash,
    flatten,
    load_config,
)
from tests.conftest import repo_root

CONFIGS = {
    "inference.yaml": InferenceConfig,
    "benchmark-a.yaml": BenchmarkConfig,
    "train-bdd100k.yaml": TrainingConfig,
    "train-synthetic.yaml": TrainingConfig,
}


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_shipped_configs_validate(name):
    load_config(CONFIGS[name], repo_root() / "configs" / name)


def test_every_config_file_is_covered():
    shipped = {p.name for p in (repo_root() / "configs").glob("*.yaml")}
    assert shipped == set(CONFIGS)


def test_unknown_keys_are_rejected(tmp_path: Path):
    path = tmp_path / "c.yaml"
    path.write_text("batching:\n  max_wait_sm: 5\n")
    with pytest.raises(ValidationError, match="max_wait_sm"):
        load_config(InferenceConfig, path)


def test_overrides_parse_yaml_scalars_and_create_sections():
    data = {"batching": {"max_batch_size": 8}}
    out = apply_overrides(
        data,
        [
            "batching.max_batch_size=4",
            "batching.strategy=fixed",
            "source.realtime=false",
            "x.y=null",
        ],
    )
    assert out["batching"] == {"max_batch_size": 4, "strategy": "fixed"}
    assert out["source"]["realtime"] is False
    assert out["x"]["y"] is None
    assert data == {"batching": {"max_batch_size": 8}}  # input untouched


def test_override_errors():
    with pytest.raises(ValueError, match="key=value"):
        apply_overrides({}, ["nonsense"])
    with pytest.raises(ValueError, match="not a mapping"):
        apply_overrides({"a": 1}, ["a.b=2"])


def test_override_applied_by_load_config():
    cfg = load_config(
        InferenceConfig, repo_root() / "configs/inference.yaml", ["batching.max_batch_size=2"]
    )
    assert cfg.batching.max_batch_size == 2


def test_sequential_batch_size_is_one():
    assert BatchingConfig(strategy="sequential", max_batch_size=8).effective_batch_size == 1
    assert BatchingConfig(strategy="fixed", max_batch_size=8).effective_batch_size == 8


def test_queue_must_hold_a_batch():
    with pytest.raises(ValidationError, match="queue_size"):
        BatchingConfig(strategy="fixed", max_batch_size=16, queue_size=8)
    BatchingConfig(strategy="sequential", max_batch_size=16, queue_size=1)


def test_training_requires_classes_and_valid_run_name():
    base = {"data": {"train": {"format": "coco", "labels": "a", "images": "b"}}}
    with pytest.raises(ValidationError, match="classes"):
        TrainingConfig.model_validate({"run_name": "ok", **base})
    with pytest.raises(ValidationError, match="run_name"):
        TrainingConfig.model_validate(
            {"run_name": "Bad Name", "model": {"classes": ["car"]}, **base}
        )


def test_config_hash_is_stable_and_sensitive():
    a = InferenceConfig()
    assert config_hash(a) == config_hash(InferenceConfig())
    b = InferenceConfig.model_validate({"batching": {"max_batch_size": 4}})
    assert config_hash(a) != config_hash(b)


def test_flatten():
    assert flatten({"a": {"b": 1, "c": {"d": 2}}, "e": 3}) == {"a.b": 1, "a.c.d": 2, "e": 3}
