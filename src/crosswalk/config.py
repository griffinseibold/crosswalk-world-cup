"""YAML configuration for every command, validated with pydantic.

Every model forbids unknown keys, so a typo such as ``max_wait_sm`` fails at
load time instead of silently falling back to a default and producing a
benchmark that measured something other than what the file says.

Endpoints (S3, MLflow) deliberately do not live here. They come from the
standard environment variables (``AWS_ENDPOINT_URL_S3``,
``MLFLOW_TRACKING_URI``) that the Helm chart sets, so the same config file
works on a laptop, in CI and on the homelab.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Device = Literal["auto", "cpu", "gpu", "rocm", "cuda"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(StrictModel):
    # fcos_tiny is a few-thousand-parameter FCOS used by tests and smoke runs on
    # CPU. It shares every code path with the real model except the backbone.
    name: Literal["fcos_resnet50_fpn", "fcos_tiny"] = "fcos_resnet50_fpn"
    # "COCO_V1" loads TorchVision's pretrained COCO detector; "none" starts
    # from random weights (the Milestone 0 validation uses this so it needs no
    # download).
    weights: Literal["COCO_V1", "none"] = "COCO_V1"
    # A fine-tuned checkpoint to load instead: a local path or s3:// URI of a
    # file written by the training loop. Its class names override `classes`.
    checkpoint: str | None = None
    # Project class names, index 0 excluded (TorchVision detection models keep
    # label 0 for background). None means the COCO categories of the weights.
    classes: list[str] | None = None
    # TorchVision detectors resize every input inside the model so the short
    # side is min_size (capped by max_size on the long side). The defaults
    # match 1280x720 footage so frames are not silently upscaled to 800 px.
    min_size: int = Field(720, gt=0)
    max_size: int = Field(1280, gt=0)
    score_threshold: float = Field(0.3, ge=0.0, le=1.0)
    trainable_backbone_layers: int = Field(3, ge=0, le=5)


class BatchingConfig(StrictModel):
    # sequential: one frame at a time, no waiting.
    # fixed: always wait for exactly max_batch_size frames (a partial batch is
    #        only flushed at end of stream).
    # dynamic: dispatch at max_batch_size frames or after max_wait_ms since the
    #          oldest queued frame arrived, whichever comes first.
    strategy: Literal["sequential", "fixed", "dynamic"] = "dynamic"
    max_batch_size: int = Field(8, ge=1)
    max_wait_ms: float = Field(10.0, ge=0.0)
    # Frames waiting for the GPU. In real-time mode a full queue drops the
    # newest frame (counted as dropped) instead of letting latency grow.
    queue_size: int = Field(64, ge=1)

    @property
    def effective_batch_size(self) -> int:
        return 1 if self.strategy == "sequential" else self.max_batch_size


class SourceConfig(StrictModel):
    # Recorded footage only: local files, directories, or s3://bucket/prefix/.
    videos: list[str] = Field(default_factory=list)
    # Real-time mode paces decoding at the file's frame rate, like a camera,
    # and drops frames the GPU cannot keep up with. Offline mode decodes as
    # fast as the batcher accepts frames and never drops.
    realtime: bool = True
    # Start the footage again from the top after the last file (the service).
    loop: bool = False
    # Resize decoded frames to width x height; None keeps the source size.
    resize: tuple[int, int] | None = (1280, 720)
    # Where s3:// footage is downloaded before decoding.
    cache_dir: str = "/cache/footage"
    # Stop after this many frames per file (None: whole file).
    max_frames: int | None = Field(None, gt=0)


class OutputConfig(StrictModel):
    # JSON-lines detection files, one per pass over each video.
    dir: str = "/data/detections"
    # Optional s3://bucket/prefix/ to upload each finished file to.
    s3_prefix: str | None = None


class ServerConfig(StrictModel):
    port: int = Field(8080, gt=0, lt=65536)


class InferenceConfig(StrictModel):
    device: Device = "auto"
    precision: Literal["fp32", "fp16"] = "fp32"
    model: ModelConfig = Field(default_factory=ModelConfig)
    batching: BatchingConfig = Field(default_factory=BatchingConfig)
    source: SourceConfig = Field(default_factory=SourceConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)


class DatasetConfig(StrictModel):
    format: Literal["bdd100k", "coco"]
    # BDD100K: a det_*.json list file or a directory of per-image JSON files.
    # COCO: an instances JSON file.
    labels: str
    images: str
    # Map dataset category names to project class names; categories mapped to
    # nothing (or missing from both the map and the class list) are skipped.
    category_map: dict[str, str] = Field(default_factory=dict)
    # Deterministic subset: images are sorted by name, then sampled with the
    # run seed, so the same config always selects the same images.
    max_images: int | None = Field(None, gt=0)
    # BDD100K frame attributes to keep, e.g. {"timeofday": ["daytime"]}.
    attribute_filter: dict[str, list[str]] = Field(default_factory=dict)


class DataConfig(StrictModel):
    train: DatasetConfig
    val: DatasetConfig | None = None
    # The hand-labeled fixed-camera set (COCO JSON) that the fine-tuned model
    # and the pretrained baseline are both scored on.
    eval: DatasetConfig | None = None
    num_workers: int = Field(2, ge=0)
    # Include every selected image's bytes in the dataset hash. Slower on a
    # full BDD100K subset (reads every image once) but catches re-encoded or
    # re-blurred images that label-only hashing would miss.
    hash_images: bool = True


class OptimizerConfig(StrictModel):
    name: Literal["sgd", "adamw"] = "sgd"
    lr: float = Field(0.005, gt=0)
    momentum: float = Field(0.9, ge=0, lt=1)
    weight_decay: float = Field(1e-4, ge=0)


class SchedulerConfig(StrictModel):
    # Linear warmup, then cosine decay to min_lr_ratio * lr over the run.
    warmup_steps: int = Field(100, ge=0)
    min_lr_ratio: float = Field(0.05, ge=0, le=1)


class CheckpointConfig(StrictModel):
    every_steps: int = Field(200, ge=1)
    keep_last: int = Field(3, ge=1)
    # s3://checkpoints/<run_name>/ by default (filled in from run_name).
    s3_uri: str | None = None
    # Always written when S3 is unavailable; also the scratch area for the
    # temporary file every checkpoint is serialized to first.
    local_dir: str = "/scratch/checkpoints"
    # auto: resume if a checkpoint exists. never: always start fresh (refuses
    # to overwrite an existing run). must: fail if there is nothing to resume.
    resume: Literal["auto", "never", "must"] = "auto"


class TrackingConfig(StrictModel):
    enabled: bool = True
    experiment: str = "crosswalk-detector"
    # Register the final model under this name in the MLflow model registry.
    registered_model_name: str | None = "crosswalk-detector"


class EvalConfig(StrictModel):
    every_epochs: int = Field(1, ge=1)
    # Keep low-confidence boxes for mAP; COCO AP integrates over scores.
    score_threshold: float = Field(0.001, ge=0.0, le=1.0)
    batch_size: int = Field(4, ge=1)
    # Score the pretrained COCO model on the eval set too, as the baseline.
    baseline: bool = True


class TrainingConfig(StrictModel):
    run_name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")
    seed: int = 1234
    # torch.use_deterministic_algorithms in warn-only mode: some detection ops
    # have no deterministic GPU kernel, so this narrows nondeterminism rather
    # than eliminating it. See docs/benchmarks.md.
    deterministic: bool = True
    device: Device = "auto"
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    epochs: int = Field(10, ge=1)
    batch_size: int = Field(4, ge=1)
    # Stop after this many optimizer steps in total (smoke tests, Experiment B).
    max_steps: int | None = Field(None, ge=1)
    grad_clip_norm: float | None = Field(10.0, gt=0)
    amp: bool = False
    log_every_steps: int = Field(20, ge=1)

    @model_validator(mode="after")
    def _classes_required(self) -> TrainingConfig:
        if not self.model.classes:
            raise ValueError("model.classes must list the project classes to fine-tune on")
        return self


class StrategySpec(StrictModel):
    name: str
    batching: BatchingConfig


class BenchmarkSourceConfig(StrictModel):
    # A recorded video decoded once into memory, or synthetic frames when
    # video is null (useful for isolating the GPU from the footage).
    video: str | None = None
    frames: int = Field(300, gt=0)
    # Frame rate each simulated stream submits at in realtime mode; null uses
    # the video's own rate (30 for synthetic frames).
    fps: float | None = Field(None, gt=0)


class BenchmarkConfig(StrictModel):
    experiment: str = "A"
    device: Device = "auto"
    precision: Literal["fp32", "fp16"] = "fp32"
    model: ModelConfig = Field(default_factory=ModelConfig)
    source: BenchmarkSourceConfig = Field(default_factory=BenchmarkSourceConfig)
    # Every combination of resolution x strategy x concurrency is run
    # `repeats` times.
    resolutions: list[tuple[int, int]] = Field(default_factory=lambda: [(1280, 720)])
    strategies: list[StrategySpec]
    concurrency: list[int] = Field(default_factory=lambda: [1])
    # offline: streams submit as fast as the queue accepts (throughput).
    # realtime: each stream submits at `fps` and drops when the queue is full.
    mode: Literal["offline", "realtime"] = "offline"
    frames_per_run: int = Field(600, gt=0)
    warmup_batches: int = Field(5, ge=0)
    repeats: int = Field(3, ge=1)
    output_dir: str = "results"
    notes: str = ""

    @field_validator("concurrency")
    @classmethod
    def _positive(cls, value: list[int]) -> list[int]:
        if not value or any(c < 1 for c in value):
            raise ValueError("concurrency values must be >= 1")
        return value


def _parse_scalar(text: str) -> Any:
    return yaml.safe_load(text) if text != "" else ""


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``a.b.c=value`` overrides; values are parsed as YAML scalars."""
    result = copy.deepcopy(data)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} is not key=value")
        dotted, raw = item.split("=", 1)
        keys = dotted.strip().split(".")
        node = result
        for key in keys[:-1]:
            child = node.get(key)
            if child is None:
                child = node[key] = {}
            if not isinstance(child, dict):
                raise ValueError(f"override {item!r}: {key!r} is not a mapping")
            node = child
        node[keys[-1]] = _parse_scalar(raw)
    return result


def read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def load_config[T: BaseModel](
    model: type[T], path: str | Path, overrides: list[str] | None = None
) -> T:
    return model.model_validate(apply_overrides(read_yaml(path), overrides or []))


def config_hash(config: BaseModel) -> str:
    """Stable short hash of a validated config, recorded with every result."""
    canonical = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested config into dotted keys (MLflow params are flat)."""
    flat: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat
