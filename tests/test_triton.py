"""Experiment T: TorchScript export, Triton model repository, binary protocol and harness.

No real Triton here: the end-to-end tests run against ``FakeTriton``, a
standard-library HTTP server that speaks just enough of the KServe v2
protocol and runs the exported TorchScript file itself. Its request parsing
and response encoding are written independently of the client under test,
from the binary tensor extension's spec, so a bug in the client's encoding
cannot cancel out against the same bug on the server side.
"""

from __future__ import annotations

import json
import re
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from crosswalk.benchmark import triton
from crosswalk.config import TritonBenchmarkConfig, TritonServerConfig

MODEL_NAME = "fcos-tiny-dense"
WIDTH, HEIGHT = 96, 64
# fcos_tiny at 64x96: FPN levels at strides 4 and 8 (16x24 + 8x12 locations,
# one anchor each) and two project classes plus background.
EXPECTED_OUTPUT_DIMS = [[480, 3], [480, 4], [480, 1]]


def config_dict(**overrides) -> dict:
    data = {
        "device": "cpu",
        "seed": 7,
        "model": {
            "name": "fcos_tiny",
            "weights": "none",
            "classes": ["car", "pedestrian"],
            "min_size": HEIGHT,
            "max_size": WIDTH,
        },
        "resolution": [WIDTH, HEIGHT],
        "triton": {
            "model_name": MODEL_NAME,
            "max_batch_size": 4,
            "dynamic_batching": {
                "max_queue_delay_microseconds": 500,
                "preferred_batch_sizes": [2, 4],
            },
            "ready_timeout_s": 5,
            "request_timeout_s": 10,
        },
        "batch_sizes": [1, 2],
        "concurrency": [1, 2],
        "requests_per_run": 6,
        "warmup_requests": 1,
        "repeats": 1,
    }
    data.update(overrides)
    return data


def tiny_config(**overrides) -> TritonBenchmarkConfig:
    return TritonBenchmarkConfig.model_validate(config_dict(**overrides))


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    config = tiny_config()
    result = triton.export_repository(config, tmp_path_factory.mktemp("repo"))
    return config, result


# -- a fake Triton ------------------------------------------------------------------------


class FakeTriton:
    """Health, metadata and binary inference over HTTP/1.1 keep-alive, like Triton."""

    def __init__(self, model_path: Path, input_dims: list[int], ready: bool = True):
        import torch

        self.module = torch.jit.load(str(model_path)).eval()
        self.input_dims = input_dims
        self.ready = threading.Event()
        if ready:
            self.ready.set()
        self.lock = threading.Lock()
        self.batch_sizes: list[int] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"  # keep-alive, which the client relies on
            # Headers and body go out in two writes; with Nagle on, the second
            # waits ~40 ms for a delayed ACK on every response.
            disable_nagle_algorithm = True

            def log_message(self, *args):
                pass

            def _send(self, code: int, body: bytes = b"", headers: dict | None = None):
                self.send_response(code)
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _json(self, code: int, obj: dict):
                self._send(code, json.dumps(obj).encode(), {"Content-Type": "application/json"})

            def do_GET(self):
                model = f"/v2/models/{MODEL_NAME}"
                if self.path == "/v2/health/ready":
                    self._send(200 if fake.ready.is_set() else 503)
                elif self.path == f"{model}/ready":
                    self._send(200 if fake.ready.is_set() else 400)
                elif self.path == "/v2":
                    self._json(200, {"name": "fake-triton", "version": "0.0.0", "extensions": []})
                elif self.path == model:
                    self._json(
                        200,
                        {
                            "name": MODEL_NAME,
                            "versions": ["1"],
                            "platform": "pytorch_libtorch",
                            "inputs": [
                                {
                                    "name": "INPUT__0",
                                    "datatype": "FP32",
                                    "shape": [-1, *fake.input_dims],
                                }
                            ],
                            "outputs": [
                                {"name": f"OUTPUT__{i}", "datatype": "FP32", "shape": [-1, *dims]}
                                for i, dims in enumerate(EXPECTED_OUTPUT_DIMS)
                            ],
                        },
                    )
                else:
                    self._json(404, {"error": f"unknown path {self.path}"})

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                try:
                    if self.path != f"/v2/models/{MODEL_NAME}/infer":
                        raise LookupError(f"unknown path {self.path}")
                    payload, headers = fake.infer(body, self.headers)
                except Exception as err:
                    self._json(400, {"error": f"{type(err).__name__}: {err}"})
                    return
                self._send(200, payload, headers)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def infer(self, body: bytes, headers) -> tuple[bytes, dict[str, str]]:
        import torch

        header_length = int(headers["Inference-Header-Content-Length"])
        request = json.loads(body[:header_length])
        (tensor,) = request["inputs"]
        assert tensor["name"] == "INPUT__0" and tensor["datatype"] == "FP32"
        size = tensor["parameters"]["binary_data_size"]
        assert len(body) == header_length + size, "body is not header + input bytes"
        assert tensor["shape"][1:] == self.input_dims
        x = np.frombuffer(body, dtype="<f4", offset=header_length).reshape(tensor["shape"])
        with self.lock, torch.no_grad():
            outputs = self.module(torch.from_numpy(x.copy()))
            self.batch_sizes.append(int(x.shape[0]))
        binary = {
            o["name"]: o.get("parameters", {}).get("binary_data", False) for o in request["outputs"]
        }
        entries, blobs = [], []
        for i, out in enumerate(outputs):
            name = f"OUTPUT__{i}"
            if name not in binary:
                continue
            array = out.numpy().astype("<f4")
            entry = {"name": name, "datatype": "FP32", "shape": list(array.shape)}
            if binary[name]:
                data = array.tobytes()
                entry["parameters"] = {"binary_data_size": len(data)}
                blobs.append(data)
            else:
                entry["data"] = array.ravel().tolist()
            entries.append(entry)
        header = json.dumps({"model_name": MODEL_NAME, "model_version": "1", "outputs": entries})
        response_headers = {
            "Content-Type": "application/octet-stream",
            "Inference-Header-Content-Length": str(len(header)),
        }
        return header.encode() + b"".join(blobs), response_headers

    def __enter__(self) -> FakeTriton:
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()


def with_url(config: TritonBenchmarkConfig, url: str, **updates) -> TritonBenchmarkConfig:
    return config.model_copy(
        update={"triton": config.triton.model_copy(update={"url": url}), **updates}
    )


# -- export ---------------------------------------------------------------------------------


def test_traced_module_matches_eager_at_other_batch_sizes(exported):
    import torch

    config, result = exported
    # Same seed, so the same weights as the exporting call built.
    dense = triton.build_dense(config, "cpu")
    traced = torch.jit.load(str(result.model_path))
    for n in (1, 3, 4):  # traced at 2
        x = torch.from_numpy(triton.example_batch(dense, WIDTH, HEIGHT, n, seed=n))
        with torch.no_grad():
            expected, got = dense.module(x), traced(x)
        assert len(got) == 3
        for e, g in zip(expected, got, strict=True):
            assert g.shape == e.shape and g.shape[0] == n
            assert torch.allclose(g, e, rtol=1e-4, atol=1e-5)
        assert [list(t.shape[1:]) for t in got] == EXPECTED_OUTPUT_DIMS
    assert result.trace_max_abs_diff <= 1e-4


def test_dense_module_is_what_fcos_runs_internally(exported):
    """Client-side normalization + the dense module reproduce FCOS's own head outputs."""
    import torch

    config, _ = exported
    dense = triton.build_dense(config, "cpu")
    frames = [np.full((HEIGHT, WIDTH, 3), v, dtype=np.uint8) for v in (40, 200)]
    frames[1][10:30, 20:50] = 90  # some structure, not just flat colour

    captured = {}
    dense.detector.head.register_forward_hook(lambda _m, _i, out: captured.update(out))
    with torch.no_grad():
        # 64x96 frames with min_size 64 / max_size 96: FCOS's transform resizes
        # by exactly 1.0 and pads nothing, so its head sees the same tensor.
        detections = dense.detector(
            [torch.from_numpy(f).permute(2, 0, 1).float() / 255 for f in frames]
        )
        captured_full = dict(captured)
        mine = dense.module(torch.from_numpy(triton.normalize(frames, dense.mean, dense.std)))
    assert len(detections) == 2 and "boxes" in detections[0]
    for key, tensor in zip(triton.HEAD_KEYS, mine, strict=True):
        assert torch.allclose(tensor, captured_full[key], rtol=1e-4, atol=1e-4), key


def test_repository_layout_and_model_config(exported):
    _, result = exported
    root = result.model_dir.parent
    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    assert files == [f"{MODEL_NAME}/1/model.pt", f"{MODEL_NAME}/config.pbtxt"]
    assert result.input_dims == [3, HEIGHT, WIDTH]
    assert result.output_dims == EXPECTED_OUTPUT_DIMS

    text = result.config_path.read_text()
    assert f'name: "{MODEL_NAME}"' in text
    assert 'platform: "pytorch_libtorch"' in text
    assert re.search(r"^max_batch_size: 4$", text, re.M)
    assert re.search(r'name: "INPUT__0"\s+data_type: TYPE_FP32\s+dims: \[ 3, 64, 96 \]', text)
    for i, dims in enumerate(EXPECTED_OUTPUT_DIMS):
        dims_text = ", ".join(map(str, dims))
        assert re.search(
            rf'name: "OUTPUT__{i}"\s+data_type: TYPE_FP32\s+dims: \[ {dims_text} \]', text
        )
    assert re.search(
        r"dynamic_batching \{\s+preferred_batch_size: \[ 2, 4 \]\s+"
        r"max_queue_delay_microseconds: 500\s+\}",
        text,
    )
    assert "instance_group" not in text  # instance_kind: auto
    assert text.count("{") == text.count("}") and text.count("[") == text.count("]")


def test_model_config_options_without_torch():
    server = TritonServerConfig(
        model_name="m",
        max_batch_size=2,
        dynamic_batching={"enabled": False},
        instance_kind="cpu",
        instance_count=3,
    )
    text = triton.render_model_config(server, [3, 32, 32], [[8, 2], [8, 4], [8, 1]], ["hello"])
    assert text.startswith("# hello\n")
    assert "dynamic_batching" not in text
    assert re.search(r"instance_group \[\s+\{\s+count: 3\s+kind: KIND_CPU\s+\}\s+\]", text)
    # Repeated messages in list syntax are comma-separated: two commas for three outputs.
    assert len(re.findall(r"^  \},$", text, re.M)) == 2


def test_config_validation():
    with pytest.raises(ValidationError, match="multiples of 32"):
        tiny_config(resolution=[1280, 720])
    with pytest.raises(ValidationError, match="max_batch_size"):
        tiny_config(batch_sizes=[1, 8])
    with pytest.raises(ValidationError, match="preferred_batch_sizes"):
        TritonServerConfig(max_batch_size=4, dynamic_batching={"preferred_batch_sizes": [8]})
    with pytest.raises(ValidationError, match="requests_per_run"):
        tiny_config(requests_per_run=1, concurrency=[2])


# -- binary tensor protocol -----------------------------------------------------------------


def test_request_encoding_follows_the_binary_extension():
    array = np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2)
    body, header_length = triton.encode_infer_request({"INPUT__0": array}, triton.OUTPUT_NAMES)
    header = json.loads(body[:header_length])
    assert header == {
        "inputs": [
            {
                "name": "INPUT__0",
                "shape": [2, 3, 2, 2],
                "datatype": "FP32",
                "parameters": {"binary_data_size": 96},
            }
        ],
        "outputs": [{"name": n, "parameters": {"binary_data": True}} for n in triton.OUTPUT_NAMES],
    }
    assert body[header_length:] == array.astype("<f4").tobytes()
    # Big-endian or non-contiguous input still goes out little-endian, C order.
    swapped = np.asfortranarray(array.astype(">f4"))
    body2, length2 = triton.encode_infer_request({"INPUT__0": swapped}, triton.OUTPUT_NAMES)
    assert body2[length2:] == body[header_length:]


def test_response_decoding_binary_and_json_outputs():
    logits = np.random.default_rng(0).normal(size=(2, 5, 3)).astype("<f4")
    labels = np.array([[1, 2], [3, 4]], dtype="<i8")
    header = json.dumps(
        {
            "outputs": [
                {
                    "name": "A",
                    "datatype": "FP32",
                    "shape": [2, 5, 3],
                    "parameters": {"binary_data_size": logits.nbytes},
                },
                {"name": "B", "datatype": "INT64", "shape": [2, 2], "data": [1, 2, 3, 4]},
                {
                    "name": "C",
                    "datatype": "INT64",
                    "shape": [2, 2],
                    "parameters": {"binary_data_size": labels.nbytes},
                },
            ]
        }
    ).encode()
    body = header + logits.tobytes() + labels.tobytes()
    out = triton.decode_infer_response(body, len(header))
    np.testing.assert_array_equal(out["A"], logits)
    np.testing.assert_array_equal(out["B"], labels)
    np.testing.assert_array_equal(out["C"], labels)
    assert out["A"].dtype == np.float32 and out["C"].dtype == np.int64

    # A plain JSON response (no Inference-Header-Content-Length) decodes too.
    plain = json.dumps(
        {"outputs": [{"name": "B", "datatype": "INT64", "shape": [4], "data": [1, 2, 3, 4]}]}
    )
    assert triton.decode_infer_response(plain.encode(), None)["B"].tolist() == [1, 2, 3, 4]

    with pytest.raises(triton.TritonError, match="truncated"):
        triton.decode_infer_response(body[:-8], len(header))
    bad = header.replace(b'"shape": [2, 5, 3]', b'"shape": [2, 5, 4]')
    with pytest.raises(triton.TritonError, match="cannot hold"):
        triton.decode_infer_response(bad + logits.tobytes() + labels.tobytes(), len(bad))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_readiness_times_out_when_nothing_listens():
    client = triton.TritonClient(f"http://127.0.0.1:{free_port()}", MODEL_NAME, timeout_s=1)
    with pytest.raises(triton.TritonError, match="not ready after"):
        client.wait_until_ready(0.3, poll_s=0.05)


def test_readiness_waits_for_the_server(exported):
    _, result = exported
    with FakeTriton(result.model_path, result.input_dims, ready=False) as server:
        client = triton.TritonClient(server.url, MODEL_NAME, timeout_s=2)
        with pytest.raises(triton.TritonError, match="server not ready"):
            client.wait_until_ready(0.3, poll_s=0.05)
        threading.Timer(0.2, server.ready.set).start()
        client.wait_until_ready(5, poll_s=0.05)
        assert client.model_ready()
        missing = triton.TritonClient(server.url, "other-model", timeout_s=2)
        with pytest.raises(triton.TritonError, match="'other-model' not ready"):
            missing.wait_until_ready(0.3, poll_s=0.05)


def test_client_round_trip_matches_the_traced_module(exported):
    import torch

    config, result = exported
    dense = triton.build_dense(config, "cpu")
    batch = triton.example_batch(dense, WIDTH, HEIGHT, 3)
    with FakeTriton(result.model_path, result.input_dims) as server:
        client = triton.TritonClient(server.url, MODEL_NAME)
        outputs = client.infer(batch)
        outputs_again = client.infer(batch)  # reuses the keep-alive connection
        with pytest.raises(triton.TritonError, match="HTTP 400"):
            client.infer(batch[:, :, :32])  # wrong input size: Triton-style error
    traced = torch.jit.load(str(result.model_path))
    with torch.no_grad():
        expected = traced(torch.from_numpy(batch))
    for got, again, want in zip(outputs, outputs_again, expected, strict=True):
        np.testing.assert_allclose(got, want.numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(again, got, rtol=1e-5, atol=1e-6)
    assert server.batch_sizes == [3, 3]


def test_parity_check_catches_a_different_model(exported):
    config, _ = exported
    dense = triton.build_dense(config, "cpu")
    other = triton.build_dense(config.model_copy(update={"seed": config.seed + 1}), "cpu")
    batch = triton.example_batch(dense, WIDTH, HEIGHT, 2)
    with pytest.raises(triton.ModelMismatch, match="different model"):
        triton.check_parity(triton.eager_call(dense), triton.eager_call(other), batch, 1e-2, 1e-2)
    with pytest.raises(triton.ModelMismatch, match="re-export"):
        triton.check_input_shape(
            {"name": "m", "inputs": [{"name": "INPUT__0", "shape": [-1, 3, 32, 32]}]},
            [3, HEIGHT, WIDTH],
        )


# -- the harness, end to end ----------------------------------------------------------------


def test_benchmark_against_fake_triton(exported, tmp_path):
    config, result = exported
    with FakeTriton(result.model_path, result.input_dims) as server:
        out = triton.run_benchmark(with_url(config, server.url, output_dir=str(tmp_path)))
    assert triton.validate_result(out) == []
    assert out["experiment"] == "T"
    assert len(out["runs"]) == 2 * 2 * 2 * 1  # backends x batch sizes x concurrency x repeats
    for run in out["runs"]:
        assert run["errors"] == 0, run["first_error"]
        assert run["requests_completed"] == run["requests_submitted"] == 6
        assert run["images_completed"] == 6 * run["batch_size"]
        assert run["throughput_images_per_s"] > 0 and run["requests_per_s"] > 0
        assert run["latency_ms"]["count"] == 6
        assert run["latency_ms"]["p50"] <= run["latency_ms"]["p95"] <= run["latency_ms"]["p99"]
    assert {r["backend"] for r in out["runs"]} == {"eager", "triton"}
    # Warmup covers every batch size Triton's batcher could form (1..4).
    assert set(server.batch_sizes) >= {1, 2, 3, 4}

    env = out["environment"]
    assert env["triton"]["server"]["name"] == "fake-triton"
    assert env["triton"]["parity_max_abs_diff"] <= 1e-4
    assert env["model"]["input_dims"] == [3, HEIGHT, WIDTH]
    assert list(env["model"]["output_dims"].values()) == EXPECTED_OUTPUT_DIMS
    assert "cpu_count" in env["hardware"] and env["runtime"]["device"] == "cpu"

    assert len(out["summary"]) == 8
    for row in out["summary"]:
        assert row["throughput_vs_eager"] > 0
        if row["backend"] == "eager":
            assert row["throughput_vs_eager"] == 1.0

    path = triton.write_result(out, str(tmp_path))
    assert re.fullmatch(
        rf"experiment-T-\d{{8}}T\d{{6}}\+0000-{out['config_hash']}\.json", path.name
    )
    assert triton.validate_result(json.loads(path.read_text())) == []
    table = triton.markdown_table(out)
    assert table.count("\n") == 2 + len(out["summary"]) - 1


def test_validate_result_reports_problems():
    problems = triton.validate_result({"runs": [{"latency_ms": {}}], "environment": {"model": {}}})
    assert any("missing key 'config_hash'" in p for p in problems)
    assert any("environment missing 'triton'" in p for p in problems)
    assert any("run 0 missing 'throughput_images_per_s'" in p for p in problems)
    assert any("latency_ms missing 'p99'" in p for p in problems)


def test_cli_export_then_benchmark(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from crosswalk import cli

    config_path = tmp_path / "triton.yaml"
    config_path.write_text(yaml.safe_dump(config_dict()))
    repo = tmp_path / "repo"
    assert cli.main(["triton-export", "--config", str(config_path), "--out", str(repo)]) == 0
    model_path = repo / MODEL_NAME / "1" / "model.pt"
    assert model_path.is_file() and (repo / MODEL_NAME / "config.pbtxt").is_file()

    results = tmp_path / "results"
    with FakeTriton(model_path, [3, HEIGHT, WIDTH]) as server:
        code = cli.main(
            [
                "triton-benchmark", "--config", str(config_path), "--url", server.url,
                "--output-dir", str(results), "--set", "concurrency=[1]", "--set", "batch_sizes=[2]",
            ]
        )  # fmt: skip
    assert code == 0
    (result_file,) = results.glob("experiment-T-*.json")
    result = json.loads(result_file.read_text())
    assert triton.validate_result(result) == []
    assert result["config"]["triton"]["url"] == server.url  # --url is what gets recorded
    assert len(result["runs"]) == 2
