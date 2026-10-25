"""Experiment T: the detector's dense network served by Triton vs in-process PyTorch.

What is compared, and why only part of the detector:

TorchVision's FCOS ``forward`` has three stages: a transform (normalize,
resize, pad to a multiple of 32), the dense network (backbone + FPN + the
classification and regression heads), and postprocessing (per-level top-k,
box decoding against the anchors, batched NMS). Postprocessing has
data-dependent shapes (how many candidates pass the score threshold, how
many boxes survive NMS) and per-image Python control flow, which
``torch.jit.trace`` would freeze to whatever the example input happened to
produce, so it does not export cleanly. The transform's resize depends on the
input shape too. So the exported model is the dense part only: it takes an
already-normalized, fixed-size float batch ``[N, 3, H, W]`` and returns the
three head tensors (``cls_logits``, ``bbox_regression``, ``bbox_ctrness``).
Normalization and postprocessing stay in Python on the client, are the same
whichever backend produced the head outputs, and are excluded from the
timings. These numbers therefore measure what the serving layer changes
(Triton's cross-request batching, HTTP and serialization, libtorch instead
of Python dispatch), not end-to-end detection latency, which Experiment A
measures.

Methodology:

1. ``crosswalk triton-export`` builds the detector exactly as the inference
   service does (same ModelConfig and checkpoint loading, plus a fixed seed
   so randomly initialised test models are the same model in every process),
   wraps the dense part, traces it to TorchScript on the CPU, checks the
   traced module against eager at several batch sizes (the batch dimension
   must stay dynamic for Triton's batcher) and writes a model repository for
   Triton's ``pytorch_libtorch`` backend. Triton moves the module to its
   instance's device when it loads it, so a CPU trace serves on a GPU too.
2. ``crosswalk triton-benchmark`` rebuilds the same module in-process, waits
   for Triton to report the model ready, checks that Triton's input shape and
   its outputs on one batch match the local module (so both sides provably
   run the same model), warms both backends up at every batch size Triton's
   batcher can form, then runs every (batch size, concurrency) cell
   ``repeats`` times on each backend, alternating backends cell by cell so
   slow drift (clocks, thermals) affects both alike.
3. In a cell, ``concurrency`` client threads each send an equal share of
   ``requests_per_run`` requests back to back (closed loop). In-process,
   that is N threads calling one module, as a naive threaded service would;
   with Triton, concurrent requests can be merged server-side by its dynamic
   batcher, which is much of what is being measured.
4. Both backends are timed from a float32 numpy batch in host memory to the
   three outputs as numpy arrays in host memory, so the in-process side pays
   its host-to-device and device-to-host copies just as Triton does.
5. Triton is spoken to over the KServe v2 HTTP/REST protocol with the binary
   tensor data extension (a JSON header, then raw little-endian bytes), using
   only the standard library, one persistent connection per client thread.
   No ``tritonclient`` dependency. Run the client on the same host as Triton
   to keep the network out of the numbers, or record that it was not.
6. Result files follow Experiment A's conventions (schema version, config
   hash, environment, method, runs, summary).

A possible follow-up: export a module that takes uint8 ``[N, H, W, 3]`` and
normalizes in the graph, which cuts the bytes sent per request by four.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import http.client
import itertools
import json
import logging
import math
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import numpy as np

from crosswalk import hardware, stats
from crosswalk.benchmark import harness
from crosswalk.config import TritonBenchmarkConfig, TritonServerConfig, config_hash

log = logging.getLogger(__name__)

RESULT_SCHEMA_VERSION = 1

# Triton's PyTorch backend binds tensors by position; the NAME__index
# convention makes the position explicit in config.pbtxt.
INPUT_NAME = "INPUT__0"
OUTPUT_NAMES = ("OUTPUT__0", "OUTPUT__1", "OUTPUT__2")
# What each output holds: FCOSHead's dict keys, in output order.
HEAD_KEYS = ("cls_logits", "bbox_regression", "bbox_ctrness")
BACKENDS = ("eager", "triton")

# Traced and eager run the same kernels on the same device, so they should
# agree to float rounding; anything bigger means the trace changed the graph.
TRACE_RTOL = 1e-4
TRACE_ATOL = 1e-5

# Result file layout, mirroring Experiment A (crosswalk.benchmark.harness).
RESULT_KEYS = harness.RESULT_KEYS
ENVIRONMENT_KEYS = {"hardware", "runtime", "model", "triton"}
RUN_KEYS = {
    "backend",
    "batch_size",
    "concurrency",
    "repeat",
    "requests_submitted",
    "requests_completed",
    "images_completed",
    "errors",
    "wall_s",
    "throughput_images_per_s",
    "requests_per_s",
    "latency_ms",
}
LATENCY_KEYS = harness.LATENCY_KEYS

METHOD = {
    "scope": "dense network only (backbone + FPN + heads); normalization and postprocessing excluded",
    "timer": "time.perf_counter around each request, from a host numpy batch to host numpy outputs",
    "eager": "N threads calling one in-process module under torch.inference_mode",
    "triton": "KServe v2 HTTP with binary tensor data, one keep-alive connection per thread",
    "load": "closed loop: each of `concurrency` threads sends its requests back to back",
    "warmup": "every batch size 1..max_batch_size run warmup_requests times per backend, discarded",
    "parity": "one batch through both backends must match within parity_rtol/parity_atol",
    "percentiles": "linear interpolation over all per-request samples",
    "throughput": "completed images / wall time from the first request start to the last request end",
}


class TritonError(RuntimeError):
    """Triton was unreachable, not ready, or answered with an error."""


class ModelMismatch(RuntimeError):
    """Two things that must be the same model are not (trace vs eager, Triton vs local)."""


# -- the dense module ---------------------------------------------------------------------


def dense_module(model):
    """Wrap a TorchVision FCOS so only its dense part runs.

    Input: an already-normalized float batch ``[N, 3, H, W]`` with H and W
    multiples of 32 (what FCOS's transform hands the backbone). Output: the
    head tensors as a tuple, ``cls_logits [N, A, classes]``,
    ``bbox_regression [N, A, 4]`` and ``bbox_ctrness [N, A, 1]``, where A is
    the anchor count summed over FPN levels.

    The backbone and head are the model's own modules, not copies.
    """
    import torch
    from torch import nn

    class DenseFCOS(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, images: torch.Tensor):
            features = self.backbone(images)
            # Same handling as FCOS.forward: a backbone returns one tensor or
            # an OrderedDict of FPN levels, and the head takes a list in level
            # order. Under tracing the dict never reaches the graph's
            # interface; only the three tensors returned below do.
            levels = [features] if isinstance(features, torch.Tensor) else list(features.values())
            out = self.head(levels)
            return out["cls_logits"], out["bbox_regression"], out["bbox_ctrness"]

    return DenseFCOS(model.backbone, model.head).eval()


@dataclass
class DenseModel:
    module: Any  # DenseFCOS on `device`, eval mode
    # The full TorchVision FCOS the module shares its backbone and head with;
    # its transform and postprocessing are what the client side would run.
    detector: Any
    device: Any
    # FCOS's normalization constants (GeneralizedRCNNTransform), applied on
    # the client before either backend sees the batch.
    mean: list[float]
    std: list[float]
    version: str


def build_dense(config: TritonBenchmarkConfig, device_name: str) -> DenseModel:
    """The detector's dense part, built exactly as the inference service builds the detector."""
    import torch

    from crosswalk.inference.predictor import TorchPredictor

    # Random initialisation happens on the CPU before the move to the device,
    # so this seed gives identical weights on every device and in every
    # process. Pretrained and checkpointed weights overwrite it anyway.
    torch.manual_seed(config.seed)
    predictor = TorchPredictor(config.model, device_name)
    transform = predictor.model.transform
    return DenseModel(
        module=dense_module(predictor.model),
        detector=predictor.model,
        device=predictor.device,
        mean=[float(v) for v in transform.image_mean],
        std=[float(v) for v in transform.image_std],
        version=predictor.model_version,
    )


def normalize(images: Sequence[np.ndarray], mean: Sequence[float], std: Sequence[float]):
    """uint8 H x W x 3 RGB frames -> float32 ``[N, 3, H, W]``, as FCOS's transform normalizes."""
    batch = np.stack(images).astype(np.float32) / np.float32(255.0)
    batch = (batch - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(batch.transpose(0, 3, 1, 2))


def example_batch(dense: DenseModel, width: int, height: int, batch_size: int, seed: int = 0):
    """A normalized batch of synthetic frames (same tensors as footage; see harness)."""
    frames = harness.synthetic_frames(batch_size, width, height, seed=seed)
    return normalize([frames[i % len(frames)] for i in range(batch_size)], dense.mean, dense.std)


# -- export ---------------------------------------------------------------------------------


def trace(module, example):
    import torch

    with torch.no_grad():
        # check_trace=False: verify_trace does a stricter check, at other
        # batch sizes too, and fails with a clearer message.
        return torch.jit.trace(module, example, check_trace=False)


def verify_trace(eager, traced, batches, rtol: float = TRACE_RTOL, atol: float = TRACE_ATOL):
    """Compare traced and eager outputs on each batch; return the largest absolute difference.

    Batches of sizes other than the one traced catch a trace that baked the
    batch size into a reshape, which would break Triton's batcher.
    """
    import torch

    largest = 0.0
    with torch.no_grad():
        for x in batches:
            size = int(x.shape[0])
            for name, want, got in zip(HEAD_KEYS, eager(x), traced(x), strict=True):
                if want.shape != got.shape:
                    raise ModelMismatch(
                        f"batch {size}: traced {name} has shape {tuple(got.shape)}, eager "
                        f"{tuple(want.shape)}; the trace froze a shape"
                    )
                diff = float((want - got).abs().max()) if want.numel() else 0.0
                largest = max(largest, diff)
                if not torch.allclose(got, want, rtol=rtol, atol=atol):
                    raise ModelMismatch(
                        f"batch {size}: traced {name} differs from eager by up to {diff:.3g} "
                        f"(rtol={rtol}, atol={atol}); refusing to export"
                    )
    return largest


def _dims(values: Sequence[int]) -> str:
    return "[ " + ", ".join(str(int(v)) for v in values) + " ]"


def render_model_config(
    triton: TritonServerConfig,
    input_dims: Sequence[int],
    output_dims: Sequence[Sequence[int]],
    comments: Sequence[str] = (),
) -> str:
    """config.pbtxt for the pytorch_libtorch backend.

    Dims leave out the batch dimension: with max_batch_size > 0 Triton adds
    it, which is what lets the dynamic batcher stack requests.
    """
    lines = [f"# {c}" for c in comments]
    lines += [
        f'name: "{triton.model_name}"',
        'platform: "pytorch_libtorch"',
        f"max_batch_size: {triton.max_batch_size}",
        "input [",
        "  {",
        f'    name: "{INPUT_NAME}"',
        "    data_type: TYPE_FP32",
        f"    dims: {_dims(input_dims)}",
        "  }",
        "]",
        "output [",
    ]
    for i, (name, dims) in enumerate(zip(OUTPUT_NAMES, output_dims, strict=True)):
        lines += [
            "  {",
            f'    name: "{name}"',
            "    data_type: TYPE_FP32",
            f"    dims: {_dims(dims)}",
            "  }" + ("," if i < len(OUTPUT_NAMES) - 1 else ""),
        ]
    lines.append("]")
    batching = triton.dynamic_batching
    if batching.enabled:
        lines.append("dynamic_batching {")
        if batching.preferred_batch_sizes:
            lines.append(f"  preferred_batch_size: {_dims(batching.preferred_batch_sizes)}")
        lines.append(f"  max_queue_delay_microseconds: {batching.max_queue_delay_microseconds}")
        lines.append("}")
    if triton.instance_kind != "auto":
        kind = "KIND_GPU" if triton.instance_kind == "gpu" else "KIND_CPU"
        lines += [
            "instance_group [",
            "  {",
            f"    count: {triton.instance_count}",
            f"    kind: {kind}",
            "  }",
            "]",
        ]
    # Run under torch.inference_mode like the in-process side, whatever the
    # backend version's default is.
    lines += [
        "parameters: {",
        '  key: "INFERENCE_MODE"',
        '  value: { string_value: "true" }',
        "}",
    ]
    return "\n".join(lines) + "\n"


@dataclass
class ExportResult:
    model_dir: Path
    model_path: Path
    config_path: Path
    input_dims: list[int]
    output_dims: list[list[int]]
    trace_max_abs_diff: float
    model_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            k: str(v) if isinstance(v, Path) else v for k, v in dataclasses.asdict(self).items()
        }


def export_repository(config: TritonBenchmarkConfig, out_dir: str | Path) -> ExportResult:
    """Trace the dense module and write ``<out_dir>/<model_name>/{config.pbtxt,1/model.pt}``."""
    import torch

    from crosswalk.storage import atomic_write_bytes

    dense = build_dense(config, "cpu")
    width, height = config.resolution
    max_batch = config.triton.max_batch_size
    # Trace at batch 2 when allowed, so batch 1 (a common special case) is
    # one of the sizes verified rather than the one traced.
    trace_batch = min(2, max_batch)
    example = torch.from_numpy(example_batch(dense, width, height, trace_batch))
    traced = trace(dense.module, example)
    checks = [
        torch.from_numpy(example_batch(dense, width, height, n, seed=n))
        for n in sorted({1, trace_batch, max_batch})
    ]
    diff = verify_trace(dense.module, traced, checks)
    with torch.no_grad():
        output_dims = [list(t.shape[1:]) for t in dense.module(example)]
    input_dims = [3, height, width]

    model_dir = Path(out_dir) / config.triton.model_name
    model_path = model_dir / "1" / "model.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    # Write then rename, so a Triton polling the repository never loads a
    # half-written file.
    partial = model_path.with_name(".model.pt.partial")
    traced.save(str(partial))
    os.replace(partial, model_path)
    config_path = model_dir / "config.pbtxt"
    comments = [
        "Generated by `crosswalk triton-export`; re-export instead of editing.",
        f"model {dense.version}, config hash {config_hash(config)}, seed {config.seed}",
        "outputs: " + ", ".join(f"{o}={k}" for o, k in zip(OUTPUT_NAMES, HEAD_KEYS, strict=True)),
    ]
    text = render_model_config(config.triton, input_dims, output_dims, comments)
    atomic_write_bytes(config_path, text.encode())
    log.info("exported %s (trace max |diff| %.3g)", model_dir, diff)
    return ExportResult(
        model_dir=model_dir,
        model_path=model_path,
        config_path=config_path,
        input_dims=input_dims,
        output_dims=output_dims,
        trace_max_abs_diff=diff,
        model_version=dense.version,
    )


# -- KServe v2 binary tensor protocol -------------------------------------------------------

# Triton datatype <-> numpy dtype, keyed by (kind, itemsize) so byte order
# does not matter on the way in. Tensors always go over the wire little-endian.
_DTYPES = {
    "BOOL": np.dtype("|b1"),
    "UINT8": np.dtype("|u1"),
    "INT32": np.dtype("<i4"),
    "INT64": np.dtype("<i8"),
    "FP16": np.dtype("<f2"),
    "FP32": np.dtype("<f4"),
    "FP64": np.dtype("<f8"),
}
_BY_KIND = {(d.kind, d.itemsize): name for name, d in _DTYPES.items()}


def triton_datatype(dtype) -> str:
    dtype = np.dtype(dtype)
    try:
        return _BY_KIND[(dtype.kind, dtype.itemsize)]
    except KeyError:
        raise TritonError(f"no Triton datatype for numpy {dtype}") from None


def _numpy_dtype(datatype: str) -> np.dtype:
    try:
        return _DTYPES[datatype]
    except KeyError:
        raise TritonError(f"unsupported Triton datatype {datatype!r}") from None


def encode_infer_request(
    inputs: Mapping[str, np.ndarray], outputs: Sequence[str]
) -> tuple[bytes, int]:
    """Body of ``POST /v2/models/<name>/infer`` using the binary tensor extension.

    Layout: a JSON header describing every tensor, then each input's raw
    bytes (C order, little-endian) in the order listed. ``binary_data_size``
    tells the server where each input ends; ``"binary_data": true`` asks for
    the outputs the same way rather than as JSON number lists, which would be
    several times larger and slow to parse. Returns the body and the header
    length, which goes in the ``Inference-Header-Content-Length`` header.
    """
    tensors, blobs = [], []
    for name, array in inputs.items():
        array = np.asarray(array)
        datatype = triton_datatype(array.dtype)
        data = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<")).tobytes()
        tensors.append(
            {
                "name": name,
                "shape": [int(d) for d in array.shape],
                "datatype": datatype,
                "parameters": {"binary_data_size": len(data)},
            }
        )
        blobs.append(data)
    header = json.dumps(
        {
            "inputs": tensors,
            "outputs": [{"name": n, "parameters": {"binary_data": True}} for n in outputs],
        },
        separators=(",", ":"),
    ).encode()
    return b"".join([header, *blobs]), len(header)


def decode_infer_response(body: bytes, header_length: int | None) -> dict[str, np.ndarray]:
    """Output tensors from an inference response.

    ``header_length`` is the response's ``Inference-Header-Content-Length``;
    None means the whole body is JSON (no binary section). Binary outputs'
    bytes follow the JSON header in the order the outputs are listed; any
    output without ``binary_data_size`` carries its values in ``data``.
    """
    if header_length is None:
        header_length = len(body)
    if not 0 < header_length <= len(body):
        raise TritonError(f"bad header length {header_length} for a {len(body)}-byte response")
    try:
        header = json.loads(body[:header_length])
    except ValueError as err:
        raise TritonError(f"unparseable response header: {err}") from err
    arrays: dict[str, np.ndarray] = {}
    offset = header_length
    for out in header.get("outputs", []):
        name = out["name"]
        shape = [int(d) for d in out["shape"]]
        dtype = _numpy_dtype(out["datatype"])
        count = math.prod(shape)
        params = out.get("parameters") or {}
        if "binary_data_size" in params:
            size = int(params["binary_data_size"])
            if size != count * dtype.itemsize:
                raise TritonError(
                    f"{name}: {size} bytes cannot hold {out['datatype']} of shape {shape}"
                )
            if offset + size > len(body):
                raise TritonError(
                    f"{name}: response truncated ({len(body) - offset} of {size} bytes)"
                )
            arrays[name] = np.frombuffer(body, dtype=dtype, count=count, offset=offset).reshape(
                shape
            )
            offset += size
        else:
            arrays[name] = np.asarray(out["data"], dtype=dtype).reshape(shape)
    return arrays


def _error_text(data: bytes) -> str:
    # Triton reports failures as {"error": "..."}.
    try:
        return str(json.loads(data)["error"])
    except (ValueError, KeyError, TypeError):
        return data[:200].decode("utf-8", errors="replace")


class TritonClient:
    """Minimal Triton HTTP client: readiness, metadata and binary inference.

    Thread-safe: each thread gets its own persistent (keep-alive) connection,
    so concurrent clients do not serialize on one socket and no request pays
    for a TCP handshake. (http.client sets TCP_NODELAY when it connects, so
    the separate header and body writes are not held back by Nagle's
    algorithm waiting for a delayed ACK.)
    """

    def __init__(self, url: str, model_name: str, timeout_s: float = 30.0):
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"expected an http(s):// URL, got {url!r}")
        self.url = url
        self.model_name = model_name
        self.timeout_s = timeout_s
        self._https = parts.scheme == "https"
        self._host = parts.hostname
        self._port = parts.port
        # A path prefix, for a Triton behind a reverse proxy at /something.
        self._prefix = parts.path.rstrip("/")
        self._model_path = f"{self._prefix}/v2/models/{quote(model_name, safe='')}"
        self._local = threading.local()

    def _new_connection(self) -> http.client.HTTPConnection:
        cls = http.client.HTTPSConnection if self._https else http.client.HTTPConnection
        conn = cls(self._host, self._port, timeout=self.timeout_s)
        self._local.conn = conn
        return conn

    def close(self) -> None:
        """Close the calling thread's connection."""
        conn = getattr(self._local, "conn", None)
        self._local.conn = None
        if conn is not None:
            conn.close()

    def _exchange(self, conn, method, path, body, headers):
        try:
            conn.request(method, path, body=body, headers=headers or {})
            response = conn.getresponse()
            data = response.read()
        except BaseException:
            self.close()
            raise
        return response.status, response.headers, data

    def _send(self, method: str, path: str, body: bytes | None = None, headers=None):
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                return self._exchange(conn, method, path, body, headers)
            except (http.client.RemoteDisconnected, ConnectionResetError, BrokenPipeError):
                # A keep-alive connection the server closed while idle fails
                # on first reuse; retry once on a fresh one. A fresh
                # connection failing is a real error and propagates.
                pass
        return self._exchange(self._new_connection(), method, path, body, headers)

    def _get_json(self, path: str) -> dict[str, Any]:
        status, _, data = self._send("GET", path)
        if status != 200:
            raise TritonError(f"GET {path}: HTTP {status}: {_error_text(data)}")
        return json.loads(data)

    def server_ready(self) -> bool:
        return self._send("GET", f"{self._prefix}/v2/health/ready")[0] == 200

    def model_ready(self) -> bool:
        return self._send("GET", f"{self._model_path}/ready")[0] == 200

    def wait_until_ready(self, timeout_s: float, poll_s: float = 0.5) -> None:
        """Poll server and model readiness; raise TritonError after ``timeout_s``."""
        deadline = time.monotonic() + timeout_s
        last = "no answer yet"
        while True:
            try:
                if not self.server_ready():
                    last = "server not ready"
                elif not self.model_ready():
                    last = f"model {self.model_name!r} not ready (is it in the model repository?)"
                else:
                    return
            except (OSError, http.client.HTTPException) as err:
                # Connection refused while the server starts, for example.
                last = f"{type(err).__name__}: {err}"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TritonError(f"Triton at {self.url} not ready after {timeout_s:g}s: {last}")
            time.sleep(min(poll_s, remaining))

    def server_metadata(self) -> dict[str, Any]:
        return self._get_json(f"{self._prefix}/v2")

    def model_metadata(self) -> dict[str, Any]:
        return self._get_json(self._model_path)

    def infer(self, batch: np.ndarray) -> list[np.ndarray]:
        """Run one batch; returns the outputs in OUTPUT_NAMES order."""
        body, header_length = encode_infer_request({INPUT_NAME: batch}, OUTPUT_NAMES)
        headers = {
            "Content-Type": "application/octet-stream",
            "Inference-Header-Content-Length": str(header_length),
        }
        status, response_headers, data = self._send(
            "POST", f"{self._model_path}/infer", body, headers
        )
        if status != 200:
            raise TritonError(f"inference failed: HTTP {status}: {_error_text(data)}")
        raw = response_headers.get("Inference-Header-Content-Length")
        outputs = decode_infer_response(data, int(raw) if raw is not None else None)
        missing = [n for n in OUTPUT_NAMES if n not in outputs]
        if missing:
            raise TritonError(f"response is missing outputs {missing}")
        return [outputs[n] for n in OUTPUT_NAMES]


# -- benchmark ------------------------------------------------------------------------------


def eager_call(dense: DenseModel) -> Callable[[np.ndarray], list[np.ndarray]]:
    import torch

    module, device = dense.module, dense.device

    def call(batch: np.ndarray) -> list[np.ndarray]:
        with torch.inference_mode():
            x = torch.from_numpy(batch).to(device)
            # .cpu() waits for the GPU to finish, so the timing covers the
            # work and the copy back rather than only queueing kernels.
            return [t.cpu().numpy() for t in module(x)]

    return call


def check_input_shape(metadata: dict[str, Any], input_dims: Sequence[int]) -> None:
    """Fail early, with a clear message, when Triton serves a different input size."""
    inputs = {t.get("name"): t for t in metadata.get("inputs", [])}
    shape = inputs.get(INPUT_NAME, {}).get("shape")
    # Triton reports the batch dimension as -1 when max_batch_size > 0.
    if shape is None or [int(d) for d in shape[-3:]] != list(input_dims):
        raise ModelMismatch(
            f"Triton's {metadata.get('name')!r} takes {INPUT_NAME} {shape}, this config "
            f"{list(input_dims)}; re-export with `crosswalk triton-export` from this config"
        )


def check_parity(
    eager: Callable[[np.ndarray], list[np.ndarray]],
    triton: Callable[[np.ndarray], list[np.ndarray]],
    batch: np.ndarray,
    rtol: float,
    atol: float,
) -> float:
    """Largest |eager - Triton| on one batch; raises if they disagree beyond tolerance."""
    largest = 0.0
    for name, want, got in zip(HEAD_KEYS, eager(batch), triton(batch), strict=True):
        if want.shape != got.shape:
            raise ModelMismatch(f"{name}: Triton returned {got.shape}, eager {want.shape}")
        diff = float(np.max(np.abs(want - got))) if want.size else 0.0
        largest = max(largest, diff)
        if not np.allclose(got, want, rtol=rtol, atol=atol):
            raise ModelMismatch(
                f"{name}: Triton and eager differ by up to {diff:.3g} (rtol={rtol}, atol={atol}). "
                "Triton is serving a different model than this config builds; re-export with "
                "`crosswalk triton-export` from this config (same model, checkpoint and seed)"
            )
    return largest


def run_cell(
    call: Callable[[np.ndarray], Any],
    batch: np.ndarray,
    concurrency: int,
    total_requests: int,
    thread_done: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """``concurrency`` closed-loop clients sharing ``total_requests`` requests of ``batch``.

    ``thread_done`` runs at the end of each client thread (the Triton client
    closes that thread's connection with it).
    """
    per_client = [total_requests // concurrency] * concurrency
    for i in range(total_requests % concurrency):
        per_client[i] += 1
    latencies: list[list[float]] = [[] for _ in range(concurrency)]
    errors = [0] * concurrency
    spans: list[tuple[float, float] | None] = [None] * concurrency
    first_error: list[str] = []
    # Clients start together, so the first requests overlap as the
    # concurrency level says instead of trickling in as threads are created.
    barrier = threading.Barrier(concurrency)

    def client(i: int) -> None:
        barrier.wait()
        start = time.perf_counter()
        for _ in range(per_client[i]):
            t0 = time.perf_counter()
            try:
                call(batch)
            except Exception as err:
                errors[i] += 1
                if not first_error:
                    first_error.append(f"{type(err).__name__}: {err}")
                continue
            latencies[i].append(time.perf_counter() - t0)
        spans[i] = (start, time.perf_counter())
        if thread_done is not None:
            thread_done()

    threads = [threading.Thread(target=client, args=(i,), daemon=True) for i in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if first_error:
        log.warning("%d failed requests; first: %s", sum(errors), first_error[0])

    samples = [s for client_samples in latencies for s in client_samples]
    finished = [s for s in spans if s is not None]
    wall = (max(e for _, e in finished) - min(s for s, _ in finished)) if finished else 0.0
    images = len(samples) * int(batch.shape[0])
    return {
        "requests_submitted": total_requests,
        "requests_completed": len(samples),
        "images_completed": images,
        "errors": sum(errors),
        "first_error": first_error[0] if first_error else None,
        "wall_s": wall,
        "throughput_images_per_s": images / wall if wall > 0 else None,
        "requests_per_s": len(samples) / wall if wall > 0 else None,
        "latency_ms": stats.summarize([s * 1000.0 for s in samples]),
    }


def _median(group: list[dict[str, Any]], *path: str) -> float | None:
    values = []
    for run in group:
        v: Any = run
        for p in path:
            v = v.get(p) if isinstance(v, dict) else None
        if v is not None:
            values.append(v)
    return stats.percentile(values, 50)


def summarize_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Median across repeats per (backend, batch size, concurrency), plus Triton vs eager."""
    cells: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for run in runs:
        cells.setdefault((run["backend"], run["batch_size"], run["concurrency"]), []).append(run)
    rows = [
        {
            "backend": backend,
            "batch_size": batch_size,
            "concurrency": concurrency,
            "repeats": len(group),
            "throughput_images_per_s_median": _median(group, "throughput_images_per_s"),
            "requests_per_s_median": _median(group, "requests_per_s"),
            "latency_p50_ms_median": _median(group, "latency_ms", "p50"),
            "latency_p95_ms_median": _median(group, "latency_ms", "p95"),
            "latency_p99_ms_median": _median(group, "latency_ms", "p99"),
            "errors_total": sum(r["errors"] for r in group),
        }
        for (backend, batch_size, concurrency), group in cells.items()
    ]
    # Each row's throughput relative to eager in the same cell: the number
    # the experiment is about (eager rows are 1.0 by definition).
    eager = {
        (r["batch_size"], r["concurrency"]): r["throughput_images_per_s_median"]
        for r in rows
        if r["backend"] == "eager"
    }
    for row in rows:
        base = eager.get((row["batch_size"], row["concurrency"]))
        value = row["throughput_images_per_s_median"]
        row["throughput_vs_eager"] = value / base if base and value is not None else None
    return rows


def run_benchmark(
    config: TritonBenchmarkConfig,
    dense: DenseModel | None = None,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if dense is None:
        dense = build_dense(config, config.device)
    if runtime is None:
        from crosswalk import device as devices

        runtime = devices.describe(dense.device)
    width, height = config.resolution
    input_dims = [3, height, width]
    server = config.triton

    client = TritonClient(server.url, server.model_name, server.request_timeout_s)
    log.info(
        "waiting up to %gs for %s at %s", server.ready_timeout_s, server.model_name, server.url
    )
    client.wait_until_ready(server.ready_timeout_s)
    server_metadata = client.server_metadata()
    model_metadata = client.model_metadata()
    check_input_shape(model_metadata, input_dims)

    calls = {"eager": eager_call(dense), "triton": client.infer}
    thread_done = {"eager": None, "triton": client.close}
    batches = {n: example_batch(dense, width, height, n) for n in sorted(set(config.batch_sizes))}
    parity = check_parity(
        calls["eager"],
        calls["triton"],
        batches[max(batches)],
        config.parity_rtol,
        config.parity_atol,
    )
    output_dims = [list(o.shape[1:]) for o in calls["eager"](batches[min(batches)])]
    log.info("Triton matches eager (max |diff| %.3g)", parity)

    # Triton's batcher can hand the backend any size up to max_batch_size,
    # and both PyTorch and libtorch do one-off work (kernel selection,
    # TorchScript's profiling executor) the first times they see a shape.
    for backend, call in calls.items():
        log.info("warming up %s", backend)
        for n in range(1, server.max_batch_size + 1):
            warm = batches.get(n)
            if warm is None:
                warm = example_batch(dense, width, height, n)
            for _ in range(config.warmup_requests):
                call(warm)

    runs: list[dict[str, Any]] = []
    for batch_size, concurrency, repeat in itertools.product(
        config.batch_sizes, config.concurrency, range(config.repeats)
    ):
        for backend in BACKENDS:
            log.info(
                "%s batch=%d concurrency=%d repeat=%d", backend, batch_size, concurrency, repeat
            )
            cell = run_cell(
                calls[backend],
                batches[batch_size],
                concurrency,
                config.requests_per_run,
                thread_done[backend],
            )
            runs.append(
                {
                    "backend": backend,
                    "batch_size": batch_size,
                    "concurrency": concurrency,
                    "repeat": repeat,
                    **cell,
                }
            )
    client.close()

    return {
        "schema": RESULT_SCHEMA_VERSION,
        "experiment": config.experiment,
        "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "config": config.model_dump(mode="json"),
        "config_hash": config_hash(config),
        "environment": {
            "hardware": hardware.collect(),
            "runtime": runtime,
            "model": {
                "name": config.model.name,
                "weights": config.model.weights,
                "checkpoint": config.model.checkpoint,
                "version": dense.version,
                "seed": config.seed,
                "precision": "fp32",
                "scope": "dense (backbone + FPN + heads)",
                "input_dims": input_dims,
                "output_dims": dict(zip(HEAD_KEYS, output_dims, strict=True)),
                "source": "synthetic",
            },
            "triton": {
                "url": server.url,
                "model_name": server.model_name,
                "server": server_metadata,
                "model": model_metadata,
                "parity_max_abs_diff": parity,
            },
        },
        "method": METHOD,
        "runs": runs,
        "summary": summarize_runs(runs),
    }


def validate_result(result: dict[str, Any]) -> list[str]:
    """Return a list of schema problems (empty means valid)."""
    problems = [f"missing key {k!r}" for k in RESULT_KEYS - result.keys()]
    env = result.get("environment", {})
    problems += [f"environment missing {k!r}" for k in ENVIRONMENT_KEYS - env.keys()]
    for i, run in enumerate(result.get("runs", [])):
        problems += [f"run {i} missing {k!r}" for k in RUN_KEYS - run.keys()]
        missing = LATENCY_KEYS - run.get("latency_ms", {}).keys()
        problems += [f"run {i} latency_ms missing {k!r}" for k in missing]
    return problems


# Same naming as Experiment A: experiment-T-<timestamp>-<config hash>.json.
write_result = harness.write_result


def markdown_table(result: dict[str, Any]) -> str:
    """Summary rows ready to paste into docs/benchmarks.md."""

    def fmt(v, digits=1):
        return "" if v is None else f"{v:.{digits}f}"

    lines = [
        "| Backend | Batch | Concurrency | Throughput (img/s) | vs eager | p50 (ms) | p95 (ms) | p99 (ms) | Errors |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in result["summary"]:
        lines.append(
            f"| {row['backend']} | {row['batch_size']} | {row['concurrency']} "
            f"| {fmt(row['throughput_images_per_s_median'])} | {fmt(row['throughput_vs_eager'], 2)} "
            f"| {fmt(row['latency_p50_ms_median'])} | {fmt(row['latency_p95_ms_median'])} "
            f"| {fmt(row['latency_p99_ms_median'])} | {row['errors_total']} |"
        )
    return "\n".join(lines)
