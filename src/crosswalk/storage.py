"""Object storage: S3 (the homelab's in-cluster S3 service) or local directories.

Everything that persists state (checkpoints, detection files, footage) goes
through :class:`Store`, which has two implementations with the same
behaviour: :class:`S3Store` and :class:`LocalStore`. Tests run the same suite
against both (S3 via moto), which is what lets training fall back to
local disk without a second code path.

Atomicity:

- S3: a single PUT is atomic: readers see the old object or the new one,
  never a partial upload. Checkpoints are therefore uploaded as one object and
  a small ``latest.json`` pointer is written *after* the checkpoint upload
  succeeds, so the pointer never names a missing or partial file.
- Local: write to a temporary file in the same directory, fsync, then
  ``os.replace``, which is atomic on POSIX filesystems.

Endpoint and credentials come from the standard AWS environment variables
(``AWS_ENDPOINT_URL_S3``, ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``),
which the chart fills from the ``s3-credentials`` Secret.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse


class Store(Protocol):
    uri: str

    def put_file(self, local_path: str, key: str) -> None: ...
    def put_bytes(self, data: bytes, key: str) -> None: ...
    def get_file(self, key: str, local_path: str) -> None: ...
    def get_bytes(self, key: str) -> bytes | None: ...
    def list(self, prefix: str = "") -> list[str]: ...
    def delete(self, key: str) -> None: ...


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
    _fsync_dir(path.parent)


def atomic_copy(src: str | Path, dest: str | Path) -> None:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".tmp", dir=dest.parent)
    try:
        with os.fdopen(fd, "wb") as out, open(src, "rb") as inp:
            shutil.copyfileobj(inp, out, length=8 * 1024 * 1024)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, dest)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
    _fsync_dir(dest.parent)


def _fsync_dir(path: Path) -> None:
    # Makes the rename itself durable across a power loss, not just the data.
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


@dataclass
class LocalStore:
    root: str

    @property
    def uri(self) -> str:
        return str(Path(self.root).resolve())

    def _path(self, key: str) -> Path:
        path = (Path(self.root) / key).resolve()
        if Path(self.root).resolve() not in path.parents and path != Path(self.root).resolve():
            raise ValueError(f"key {key!r} escapes the store root")
        return path

    def put_file(self, local_path: str, key: str) -> None:
        atomic_copy(local_path, self._path(key))

    def put_bytes(self, data: bytes, key: str) -> None:
        atomic_write_bytes(self._path(key), data)

    def get_file(self, key: str, local_path: str) -> None:
        atomic_copy(self._path(key), local_path)

    def get_bytes(self, key: str) -> bytes | None:
        try:
            return self._path(key).read_bytes()
        except FileNotFoundError:
            return None

    def list(self, prefix: str = "") -> list[str]:
        root = Path(self.root)
        if not root.exists():
            return []
        keys = []
        for path in root.rglob("*"):
            if path.is_file() and not path.name.startswith("."):
                key = path.relative_to(root).as_posix()
                if key.startswith(prefix):
                    keys.append(key)
        return sorted(keys)

    def delete(self, key: str) -> None:
        with contextlib.suppress(FileNotFoundError):
            self._path(key).unlink()


class S3Store:
    def __init__(self, bucket: str, prefix: str = "", client=None):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._client = client

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}".rstrip("/")

    @property
    def client(self):
        if self._client is None:
            self._client = make_s3_client()
        return self._client

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def put_file(self, local_path: str, key: str) -> None:
        # upload_file switches to multipart above 8 MB; S3 only makes a
        # multipart object visible once CompleteMultipartUpload succeeds, so
        # it is still all-or-nothing for readers.
        self.client.upload_file(local_path, self.bucket, self._key(key))

    def put_bytes(self, data: bytes, key: str) -> None:
        self.client.put_object(Bucket=self.bucket, Key=self._key(key), Body=data)

    def get_file(self, key: str, local_path: str) -> None:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        tmp = f"{local_path}.part"
        self.client.download_file(self.bucket, self._key(key), tmp)
        os.replace(tmp, local_path)

    def get_bytes(self, key: str) -> bytes | None:
        from botocore.exceptions import ClientError

        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self._key(key))
        except ClientError as err:
            if err.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                return None
            raise
        return response["Body"].read()

    def list(self, prefix: str = "") -> list[str]:
        paginator = self.client.get_paginator("list_objects_v2")
        full = self._key(prefix)
        strip = f"{self.prefix}/" if self.prefix else ""
        keys = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full):
            for item in page.get("Contents", []):
                keys.append(item["Key"][len(strip) :])
        return sorted(keys)

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=self._key(key))

    def check(self) -> None:
        """Raise if the bucket is unreachable (used to decide on local fallback)."""
        self.client.head_bucket(Bucket=self.bucket)


def make_s3_client():
    import boto3
    from botocore.config import Config

    # Path-style addressing: the in-cluster S3 service is reached by its
    # Service name, so virtual-host style (bucket.s3...) would not resolve.
    config = Config(
        s3={"addressing_style": "path"},
        retries={"max_attempts": 5, "mode": "standard"},
        connect_timeout=5,
        read_timeout=60,
    )
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL"),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        config=config,
    )


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"not an s3:// URI: {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def open_store(uri: str, client=None) -> Store:
    if uri.startswith("s3://"):
        bucket, prefix = parse_s3_uri(uri)
        return S3Store(bucket, prefix, client=client)
    return LocalStore(uri.removeprefix("file://"))


def fetch(uri: str, cache_dir: str, client=None) -> str:
    """Return a local path for a file given as a path or an s3:// URI."""
    if not uri.startswith("s3://"):
        return uri
    bucket, key = parse_s3_uri(uri)
    local = Path(cache_dir) / bucket / key
    if not local.exists():
        S3Store(bucket, client=client).get_file(key, str(local))
    return str(local)


def expand_s3_videos(uri: str, cache_dir: str, suffixes: set[str], client=None) -> list[str]:
    """Download every video under an s3:// prefix (or a single object) and list them."""
    bucket, prefix = parse_s3_uri(uri)
    store = S3Store(bucket, client=client)
    keys = [k for k in store.list(prefix) if Path(k).suffix.lower() in suffixes] or [prefix]
    paths = []
    for key in keys:
        local = Path(cache_dir) / bucket / key
        if not local.exists():
            store.get_file(key, str(local))
        paths.append(str(local))
    return paths


def sync_prefix(uri: str, local_dir: str, client=None) -> dict[str, int]:
    """Download everything under an s3:// prefix into ``local_dir``.

    Files already present with the same size are skipped, so a training Job
    whose scratch volume survived (persistent scratch, or a pod restart on the
    same node) does not download the dataset again. Returns counts for logs.
    """
    bucket, prefix = parse_s3_uri(uri)
    prefix = prefix.rstrip("/") + "/" if prefix else ""
    store = S3Store(bucket, client=client)
    paginator = store.client.get_paginator("list_objects_v2")
    counts = {"downloaded": 0, "skipped": 0, "bytes": 0}
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith("/"):
                continue  # "directory" placeholder objects
            local = Path(local_dir) / key[len(prefix) :]
            if local.exists() and local.stat().st_size == item["Size"]:
                counts["skipped"] += 1
                continue
            store.get_file(key, str(local))
            counts["downloaded"] += 1
            counts["bytes"] += int(item["Size"])
    return counts
