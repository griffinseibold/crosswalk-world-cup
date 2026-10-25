"""Storage backends and the checkpoint manager (no PyTorch needed).

The same behavioural tests run against LocalStore and S3Store (moto), which is
what justifies training treating them interchangeably.
"""

import json
import os
from pathlib import Path

import pytest

from crosswalk import storage
from crosswalk.training.checkpoint import POINTER, CheckpointManager, make_manager, step_key


@pytest.fixture(params=["local", "s3"])
def store(request, tmp_path):
    if request.param == "local":
        return storage.LocalStore(str(tmp_path / "store"))
    client = request.getfixturevalue("s3_client")
    return storage.S3Store("checkpoints", "run-a", client=client)


def test_store_round_trip(store, tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"x" * 1000)
    store.put_file(str(src), "a/b.bin")
    store.put_bytes(b"hello", "c.txt")
    assert store.get_bytes("c.txt") == b"hello"
    assert store.get_bytes("missing") is None
    out = tmp_path / "out" / "b.bin"
    store.get_file("a/b.bin", str(out))
    assert out.read_bytes() == b"x" * 1000
    assert store.list() == ["a/b.bin", "c.txt"]
    assert store.list("a/") == ["a/b.bin"]
    store.delete("c.txt")
    store.delete("c.txt")  # idempotent
    assert store.list() == ["a/b.bin"]


def test_local_store_rejects_escaping_keys(tmp_path):
    with pytest.raises(ValueError):
        storage.LocalStore(str(tmp_path)).put_bytes(b"x", "../outside")


def test_atomic_write_leaves_old_file_on_failure(tmp_path, monkeypatch):
    target = tmp_path / "f.json"
    storage.atomic_write_bytes(target, b"old")

    def boom(*_args):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        storage.atomic_write_bytes(target, b"new")
    assert target.read_bytes() == b"old"
    assert [p.name for p in tmp_path.iterdir()] == ["f.json"]  # temp file cleaned up


def test_parse_s3_uri_and_open_store(tmp_path):
    assert storage.parse_s3_uri("s3://bucket/a/b") == ("bucket", "a/b")
    with pytest.raises(ValueError):
        storage.parse_s3_uri("http://x/y")
    assert isinstance(storage.open_store(str(tmp_path)), storage.LocalStore)
    assert isinstance(storage.open_store("s3://b/p"), storage.S3Store)


def test_s3_client_uses_path_style(aws_env, monkeypatch):
    pytest.importorskip("boto3")
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", "http://s3.ml-storage.svc.cluster.local:9000")
    client = storage.make_s3_client()
    assert client.meta.endpoint_url == "http://s3.ml-storage.svc.cluster.local:9000"
    assert client.meta.config.s3["addressing_style"] == "path"


# -- checkpoint manager -------------------------------------------------------------


def writer(content: str):
    def write(path: str) -> None:
        Path(path).write_text(content)

    return write


def test_save_prunes_and_points_at_latest(tmp_path):
    m = CheckpointManager(str(tmp_path / "ck"), keep_last=2)
    for step in (10, 20, 30):
        result = m.save(step, writer(f"state-{step}"), {"epoch": 0})
    assert result.remote_ok is None and result.bytes == len("state-30")
    assert sorted(p.name for p in Path(m.local_dir).iterdir()) == [
        POINTER,
        step_key(20),
        step_key(30),
    ]
    pointer = json.loads((Path(m.local_dir) / POINTER).read_text())
    assert pointer["step"] == 30 and pointer["epoch"] == 0 and pointer["key"] == step_key(30)
    found = m.find_latest()
    assert found.step == 30 and found.path.read_text() == "state-30"


def test_crash_during_serialization_keeps_previous_checkpoint(tmp_path):
    m = CheckpointManager(str(tmp_path / "ck"))
    m.save(10, writer("good"), {})

    def crash(path):
        Path(path).write_text("half")
        raise KeyboardInterrupt  # stands in for a kill mid-write

    with pytest.raises(KeyboardInterrupt):
        m.save(20, crash, {})
    assert not (Path(m.local_dir) / step_key(20)).exists()
    assert not [p for p in Path(m.local_dir).iterdir() if p.suffix == ".tmp"]
    assert m.find_latest().step == 10


def test_corrupt_latest_falls_back_to_previous(tmp_path):
    m = CheckpointManager(str(tmp_path / "ck"))
    m.save(10, writer("ten"), {})
    m.save(20, writer("twenty"), {})
    (Path(m.local_dir) / step_key(20)).write_text("corrupted")  # checksum mismatch
    found = m.find_latest()
    assert found.step == 10 and found.path.read_text() == "ten"


def test_unreadable_pointer_scans_files(tmp_path):
    m = CheckpointManager(str(tmp_path / "ck"))
    m.save(10, writer("ten"), {})
    (Path(m.local_dir) / POINTER).write_text("{not json")
    assert m.find_latest().step == 10


def test_nothing_to_resume(tmp_path):
    assert CheckpointManager(str(tmp_path / "ck")).find_latest() is None


def test_remote_checkpoints_and_resume_on_a_fresh_node(tmp_path, s3_client):
    remote = storage.S3Store("checkpoints", "run-a", client=s3_client)
    m = CheckpointManager(str(tmp_path / "node1"), remote=remote, keep_last=2)
    for step in (5, 10, 15):
        assert m.save(step, writer(f"s{step}"), {}).remote_ok is True
    assert remote.list() == [POINTER, step_key(10), step_key(15)]
    # A rescheduled pod has empty scratch: it must download from S3.
    fresh = CheckpointManager(str(tmp_path / "node2"), remote=remote)
    found = fresh.find_latest()
    assert (found.step, found.source, found.path.read_text()) == (15, "remote", "s15")


def test_upload_failure_keeps_training_with_local_copy(tmp_path):
    class Broken:
        uri = "s3://broken"

        def put_file(self, *a):
            raise ConnectionError("S3 down")

        put_bytes = put_file

        def list(self, prefix=""):
            raise ConnectionError("S3 down")

        def get_bytes(self, key):
            raise ConnectionError("S3 down")

    m = CheckpointManager(str(tmp_path / "ck"), remote=Broken())
    result = m.save(10, writer("local only"), {})
    assert result.remote_ok is False and m.upload_failures == 1
    found = m.find_latest()
    assert found.step == 10 and found.source == "local"


def test_newer_local_checkpoint_wins_over_stale_remote(tmp_path, s3_client):
    remote = storage.S3Store("checkpoints", "run-b", client=s3_client)
    m = CheckpointManager(str(tmp_path / "ck"), remote=remote)
    m.save(10, writer("ten"), {})
    m.remote = None  # S3 outage: step 20 only reaches local disk
    m.save(20, writer("twenty"), {})
    m.remote = remote
    assert m.find_latest().step == 20


def test_make_manager_falls_back_when_bucket_missing(tmp_path, s3_client):
    manager, desc = make_manager(
        "run-c", "s3://no-such-bucket/run-c", str(tmp_path), 3, s3_client=s3_client
    )
    assert manager.remote is None and desc.startswith("local:")
    manager, desc = make_manager("run-c", None, str(tmp_path), 3, s3_client=s3_client)
    assert manager.remote is None  # no endpoint configured, no explicit URI
    manager, desc = make_manager(
        "run-c", "s3://checkpoints/run-c", str(tmp_path), 3, s3_client=s3_client
    )
    assert manager.remote is not None and "s3://checkpoints/run-c" in desc


def test_sync_prefix_downloads_once(tmp_path, s3_client):
    for key, body in (
        ("bdd/labels/det.json", b"[]"),
        ("bdd/images/a.jpg", b"jpeg"),
        ("other/x", b"no"),
        ("bdd/empty-dir/", b""),
    ):
        s3_client.put_object(Bucket="datasets", Key=key, Body=body)
    first = storage.sync_prefix("s3://datasets/bdd", str(tmp_path / "d"), client=s3_client)
    assert first == {"downloaded": 2, "skipped": 0, "bytes": 6}
    assert (tmp_path / "d/images/a.jpg").read_bytes() == b"jpeg"
    assert not (tmp_path / "d/x").exists()
    second = storage.sync_prefix("s3://datasets/bdd/", str(tmp_path / "d"), client=s3_client)
    assert second == {"downloaded": 0, "skipped": 2, "bytes": 0}
