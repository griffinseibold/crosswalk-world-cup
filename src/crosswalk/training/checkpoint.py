"""Periodic checkpoints on S3 with a local fallback, and finding the latest one.

Layout under a run's prefix (``s3://checkpoints/<run>/`` and/or the local dir)::

    step-00000200.pt
    step-00000400.pt
    latest.json      {"step": 400, "key": "step-00000400.pt", "sha256": "...", ...}

Write order is what makes this crash-safe:

1. Serialize to a temporary file in the local dir, fsync, rename to
   ``step-N.pt`` (atomic on POSIX). A crash mid-write leaves only a ``.tmp``
   file, never a truncated ``step-N.pt``.
2. Upload ``step-N.pt`` to S3 as one object (S3 makes an object visible only
   once the upload completes).
3. Only then rewrite ``latest.json``, locally and on S3, each atomically.

So ``latest.json`` only ever names a complete checkpoint. On resume the file's
SHA-256 is checked against the pointer anyway; a mismatch (e.g. a hand-copied
file) falls back to the newest older checkpoint that verifies.

If S3 is unreachable at start, or an upload fails, the checkpoint is still on
local disk; the run keeps going and the failure is counted, logged and
reported to MLflow (``system/checkpoint_upload_failures``). Resume considers
both locations and takes the newest checkpoint that verifies.

Nothing here imports PyTorch: the caller supplies ``write(path)`` and reads the
returned path itself. That keeps this logic testable on any machine.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crosswalk.storage import LocalStore, Store, atomic_write_bytes
from crosswalk.training.hashing import file_sha256

log = logging.getLogger(__name__)

POINTER = "latest.json"
STEP_FILE = re.compile(r"^step-(\d{8,})\.pt$")


def step_key(step: int) -> str:
    return f"step-{step:08d}.pt"


@dataclass
class SaveResult:
    step: int
    local_path: Path
    sha256: str
    bytes: int
    serialize_s: float
    upload_s: float | None
    remote_ok: bool | None  # None: no remote configured


@dataclass
class Found:
    step: int
    path: Path  # local, verified copy
    pointer: dict[str, Any]
    source: str  # "remote" or "local"


@dataclass
class CheckpointManager:
    local_dir: str
    remote: Store | None = None
    keep_last: int = 3
    upload_failures: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        Path(self.local_dir).mkdir(parents=True, exist_ok=True)
        self.local = LocalStore(self.local_dir)

    # -- saving -------------------------------------------------------------

    def save(self, step: int, write: Callable[[str], None], meta: dict[str, Any]) -> SaveResult:
        local_path = Path(self.local_dir) / step_key(step)
        t0 = time.monotonic()
        fd, tmp = tempfile.mkstemp(prefix=f".{local_path.name}.", suffix=".tmp", dir=self.local_dir)
        os.close(fd)
        try:
            write(tmp)
            with open(tmp, "rb") as handle:
                os.fsync(handle.fileno())
            os.replace(tmp, local_path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        serialize_s = time.monotonic() - t0
        digest = file_sha256(local_path)
        size = local_path.stat().st_size
        pointer = {
            "step": step,
            "key": local_path.name,
            "sha256": digest,
            "bytes": size,
            "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            **meta,
        }
        pointer_bytes = json.dumps(pointer, indent=2).encode()

        upload_s = None
        remote_ok = None
        if self.remote is not None:
            t1 = time.monotonic()
            try:
                self.remote.put_file(str(local_path), local_path.name)
                self.remote.put_bytes(pointer_bytes, POINTER)
                remote_ok = True
            except Exception as err:  # network, auth, bucket missing...
                self.upload_failures += 1
                remote_ok = False
                log.warning("checkpoint upload failed (kept locally): %s", err)
            upload_s = time.monotonic() - t1

        # Local pointer last: it must never name a file that is not on disk.
        atomic_write_bytes(Path(self.local_dir) / POINTER, pointer_bytes)
        self._prune()
        result = SaveResult(step, local_path, digest, size, serialize_s, upload_s, remote_ok)
        self.events.append({"event": "checkpoint", **pointer, "remote_ok": remote_ok})
        return result

    def _prune(self) -> None:
        for store in (s for s in (self.remote, self.local) if s is not None):
            try:
                steps = sorted(
                    int(m.group(1)) for k in store.list() if (m := STEP_FILE.match(k))
                )
                for old in steps[: -self.keep_last]:
                    store.delete(step_key(old))
            except Exception as err:
                log.warning("pruning old checkpoints failed: %s", err)

    # -- finding ------------------------------------------------------------

    def _candidates(self, store: Store) -> list[tuple[int, dict[str, Any] | None]]:
        """(step, pointer-or-None) newest first: the pointer target, then other files."""
        out: list[tuple[int, dict[str, Any] | None]] = []
        raw = store.get_bytes(POINTER)
        pointer = None
        if raw:
            try:
                pointer = json.loads(raw)
                out.append((int(pointer["step"]), pointer))
            except (ValueError, KeyError, TypeError):
                log.warning("unreadable %s in %s; scanning files instead", POINTER, store.uri)
        for key in store.list():
            m = STEP_FILE.match(key)
            if m and (pointer is None or int(m.group(1)) != int(pointer["step"])):
                out.append((int(m.group(1)), None))
        return sorted(out, key=lambda item: item[0], reverse=True)

    def find_latest(self) -> Found | None:
        """Newest checkpoint across remote and local that passes verification."""
        options: list[tuple[int, str, Store, dict[str, Any] | None]] = []
        if self.remote is not None:
            try:
                options += [(s, "remote", self.remote, p) for s, p in self._candidates(self.remote)]
            except Exception as err:
                log.warning("cannot list remote checkpoints (%s); using local only", err)
        options += [(s, "local", self.local, p) for s, p in self._candidates(self.local)]
        # Prefer newer steps; at equal steps prefer the local copy (no download).
        options.sort(key=lambda o: (o[0], o[1] == "local"), reverse=True)
        tried: set[tuple[int, str]] = set()
        for step, source, store, pointer in options:
            if (step, source) in tried:
                continue
            tried.add((step, source))
            path = self._materialize(step, source, store)
            if path is None:
                continue
            expected = pointer.get("sha256") if pointer else None
            if expected and file_sha256(path) != expected:
                log.warning("checkpoint step %d from %s failed its checksum; skipping", step, source)
                continue
            return Found(step, path, pointer or {"step": step, "key": step_key(step)}, source)
        return None

    def _materialize(self, step: int, source: str, store: Store) -> Path | None:
        local_path = Path(self.local_dir) / step_key(step)
        if source == "local":
            return local_path if local_path.exists() else None
        try:
            store.get_file(step_key(step), str(local_path))
            return local_path
        except Exception as err:
            log.warning("downloading checkpoint step %d failed: %s", step, err)
            return None


def make_manager(
    run_name: str,
    s3_uri: str | None,
    local_dir: str,
    keep_last: int,
    s3_client=None,
) -> tuple[CheckpointManager, str]:
    """Build a manager; returns (manager, storage description for logs/MLflow)."""
    from crosswalk.storage import S3Store, parse_s3_uri

    local = str(Path(local_dir) / run_name)
    remote = None
    description = f"local:{local}"
    uri = s3_uri or (f"s3://checkpoints/{run_name}" if _s3_configured() else None)
    if uri:
        bucket, prefix = parse_s3_uri(uri)
        store = S3Store(bucket, prefix, client=s3_client)
        try:
            store.check()
            remote = store
            description = f"{store.uri} (local copy {local})"
        except Exception as err:
            log.warning("S3 unavailable at %s (%s): checkpoints stay local", uri, err)
    return CheckpointManager(local, remote, keep_last), description


def _s3_configured() -> bool:
    return bool(os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL"))
