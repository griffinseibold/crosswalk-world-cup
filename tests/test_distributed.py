"""Data-parallel training: the sharding maths (no torch) and a real two-process
run on CPU with the gloo backend (torch; CI runs it).

The two-process test is the same code path as N GPUs under torchrun, except
for the backend (gloo instead of NCCL/RCCL) and the device binding.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from crosswalk.config import TrainingConfig
from crosswalk.training import distributed
from crosswalk.training.distributed import DistInfo, ShardedSampler
from crosswalk.training.loop import resume_hash
from crosswalk.training.sampling import ResumableSampler
from tests.conftest import subprocess_env

# -- environment ------------------------------------------------------------------


def test_from_env_is_single_process_without_torchrun_variables():
    assert distributed.from_env({}) == DistInfo()
    assert not DistInfo().enabled and DistInfo().is_main
    assert distributed.from_env({"WORLD_SIZE": "1", "RANK": "0"}) == DistInfo()
    info = distributed.from_env({"WORLD_SIZE": "4", "RANK": "3", "LOCAL_RANK": "1"})
    assert info == DistInfo(rank=3, local_rank=1, world_size=4)
    assert info.enabled and not info.is_main


def test_backend_follows_the_device_not_the_vendor():
    # torch's "nccl" is RCCL on ROCm builds, so GPUs of either vendor use it.
    assert distributed.backend_for("cuda") == "nccl"
    assert distributed.backend_for("cpu") == "gloo"


def test_helpers_are_no_ops_for_one_process():
    info = DistInfo()
    assert distributed.init(info, "cpu") is None
    sentinel = object()
    assert distributed.wrap(sentinel, info, "cpu") is sentinel
    assert distributed.unwrap(sentinel) is sentinel
    assert distributed.broadcast_object(7, info) == 7
    distributed.barrier(info)
    distributed.shutdown(info)


# -- sharding -----------------------------------------------------------------------


@pytest.mark.parametrize("world", [2, 3, 4])
def test_ranks_split_each_global_batch_without_overlap(world):
    per_rank, n = 2, 29
    base = ResumableSampler(n, per_rank * world, seed=5)
    shards = [list(ShardedSampler(base, DistInfo(r, r, world))) for r in range(world)]
    order = list(base)
    g = per_rank * world
    for b in range(len(order) // g):
        global_batch = order[b * g : (b + 1) * g]
        from_ranks = [i for s in shards for i in s[b * per_rank : (b + 1) * per_rank]]
        # Rank r's b-th batch is the r-th slice of global batch b.
        assert from_ranks == global_batch
    assert all(len(s) == len(order) // world for s in shards)
    assert len(ShardedSampler(base, DistInfo(0, 0, world))) == len(order) // world


def test_resume_position_means_the_same_global_batch_on_every_rank():
    world, per_rank = 2, 2
    base = ResumableSampler(16, per_rank * world, seed=1)
    full = [list(ShardedSampler(base, DistInfo(r, r, world))) for r in range(world)]
    resumed = []
    for r in range(world):
        sampler = ShardedSampler(
            ResumableSampler(16, per_rank * world, seed=1), DistInfo(r, r, world)
        )
        sampler.set_position(epoch=0, start_batch=2)
        resumed.append(list(sampler))
    for r in range(world):
        assert resumed[r] == full[r][2 * per_rank :]
    # A new epoch reshuffles: same position, different indices.
    base.set_position(epoch=1)
    assert list(ShardedSampler(base, DistInfo(0, 0, world))) != full[0]


def test_resume_hash_includes_the_world_size_only_when_distributed(synthetic_data):
    config = TrainingConfig.model_validate(
        {
            "run_name": "h",
            "model": {"name": "fcos_tiny", "weights": "none", "classes": ["car"]},
            "data": {"train": {"format": "bdd100k", "labels": synthetic_data["bdd100k"],
                               "images": synthetic_data["images"]}},
        }
    )  # fmt: skip
    # Single-process hashes are unchanged by DDP support (old checkpoints resume).
    assert resume_hash(config) == resume_hash(config, 1)
    assert resume_hash(config, 2) != resume_hash(config) != resume_hash(config, 4)


# -- a real two-process run (gloo on CPU) ----------------------------------------------


def ddp_config(data: dict, out: Path, **overrides) -> dict:
    base = {
        "run_name": "ddp-run",
        "seed": 7,
        "device": "cpu",
        "model": {"name": "fcos_tiny", "weights": "none", "classes": ["car", "pedestrian"],
                  "min_size": 64, "max_size": 96},
        "data": {"train": {"format": "bdd100k", "labels": data["bdd100k"], "images": data["images"]},
                 "num_workers": 0},
        "optimizer": {"name": "sgd", "lr": 0.01},
        "scheduler": {"warmup_steps": 1},
        "checkpoint": {"every_steps": 2, "keep_last": 2, "local_dir": str(out / "ckpt")},
        "tracking": {"enabled": False},
        "eval": {"baseline": False},
        "epochs": 2,
        "batch_size": 2,
        "log_every_steps": 1,
    }  # fmt: skip
    base.update(overrides)
    return base


def torchrun(config_path: Path, nproc: int) -> subprocess.CompletedProcess:
    # python -m torch.distributed.run is the torchrun entry point; --standalone
    # gives a local rendezvous on a free port.
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
           f"--nproc-per-node={nproc}", "-m", "crosswalk", "train", "--config", str(config_path)]  # fmt: skip
    return subprocess.run(
        cmd,
        env=subprocess_env(OMP_NUM_THREADS="1"),
        capture_output=True,
        text=True,
        timeout=600,
    )


@pytest.mark.slow
def test_two_process_gloo_training_shards_checkpoints_once_and_pins_the_world_size(
    synthetic_data, tmp_path
):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.fail("this torch build has no gloo backend; DDP on CPU cannot be tested")
    from crosswalk.training import state

    cfg = tmp_path / "ddp.yaml"
    cfg.write_text(yaml.safe_dump(ddp_config(synthetic_data, tmp_path)))
    result = torchrun(cfg, 2)
    assert result.returncode == 0, result.stderr[-4000:]
    assert "rank 1/2 (gloo)" in result.stderr

    ckpt = tmp_path / "ckpt" / "ddp-run"
    # 16 images, global batch 2 x 2 = 4: 4 steps per epoch, 2 epochs. A single
    # process with batch 2 would take 16 steps; the two ranks shared the data.
    summary = json.loads((ckpt / "summary.json").read_text())
    assert summary["final_step"] == summary["total_steps"] == 8
    # Exactly one writer: one set of checkpoints, no per-rank copies.
    names = sorted(p.name for p in ckpt.iterdir())
    assert names == [
        "heartbeat.json",
        "latest.json",
        "step-00000006.pt",
        "step-00000008.pt",
        "summary.json",
    ]
    saved = state.load_state(str(ckpt / "step-00000008.pt"))
    assert saved["progress"] == {"step": 8, "epoch": 2, "batch_in_epoch": 0}
    # Saved from the unwrapped model: no DDP "module." prefix, so the weights
    # load into a plain detector (inference, promotion, single-process resume).
    assert not any(k.startswith("module.") for k in saved["model"])

    # Re-running with the same world size resumes the finished run...
    again = torchrun(cfg, 2)
    assert again.returncode == 0, again.stderr[-4000:]
    assert json.loads((ckpt / "summary.json").read_text())["resumed"] is True

    # ...but one process would silently halve the global batch: refused.
    single = subprocess.run(
        [sys.executable, "-m", "crosswalk", "train", "--config", str(cfg)],
        env=subprocess_env(OMP_NUM_THREADS="1"),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert single.returncode != 0 and "different training config" in single.stderr
