# Crosswalk World Cup

A GPU computer-vision platform that runs on one AMD graphics card in a
homelab. It finds cars and pedestrians in recorded road footage and will one
day rank intersections against each other in a crosswalk leaderboard.

> The computer vision is the workload. The GPU platform is the accomplishment.
> The leaderboard is the hook.

The detector is an off-the-shelf TorchVision model. What this repository is
really about is everything around it: serving it on a GPU inside Kubernetes
with dynamic batching and real metrics, fine-tuning it with a training loop
that survives being killed, tracking every run in MLflow, and measuring all of
it in a way someone else could reproduce.

## Scope and safety limits

- **Recorded footage only.** No live camera feeds, RTSP or webcams, by design.
- **Faces and licence plates are blurred before footage is stored.** Blurring
  does not grant permission to use footage; see [data handling](docs/data.md).
- **BDD100K is not downloaded by this code.** Its licence is non-commercial
  and needs registration; you point the config at your own copy.
- **No invented numbers.** [Benchmarks](docs/benchmarks.md) has the method
  and empty results tables. They get filled in from real runs on the homelab.

## Architecture

```text
Recorded footage (blurred) ──► s3://datasets/footage/
                                     │
               ┌─────────────────────┴───────────────────────────┐
               │  crosswalk namespace (Argo CD, chart/crosswalk)  │
               │                                                  │
               │  inference Deployment (1 GPU)                    │
               │    PyAV decode ─► dynamic batcher ─► FCOS ─► JSONL ──► s3://datasets/detections/
               │        │                                        │
               │        └─ /metrics ─► Prometheus ─► Grafana dashboard
               │           /status  ─► http://crosswalk.localhost:8080
               │                                                  │
               │  one-off Jobs (scripts/submit-training.py)       │
               │    validate   Milestone 0 GPU check              │
               │    benchmark  Experiment A ──► s3://datasets/results/
               │    train      custom loop ──► s3://checkpoints/<run>/
               │                    └──────────► MLflow (params, metrics,
               │                                  artifacts, model registry)
               └──────────────────────────────────────────────────┘
```

Everything runs on ROCm. PyTorch's ROCm build exposes AMD GPUs through the
`torch.cuda` API, so the code asks for a device (`auto`, `gpu`, `rocm`,
`cuda`, `cpu`) and never branches on the vendor. The vendor shows up in only
two places: the container image (`docker/Dockerfile.rocm`) and the facts
recorded with each result (`src/crosswalk/device.py`, `hardware.py`). A
CUDA image would be one more Dockerfile; no code changes.

| Path | What it is |
| --- | --- |
| `src/crosswalk/inference/` | Decoding, batcher, predictor, metrics, status server, service loop |
| `src/crosswalk/benchmark/` | Experiment A harness and result schema |
| `src/crosswalk/training/` | Datasets, hashing, checkpoints, the training loop, evaluation, MLflow |
| `src/crosswalk/validate.py` | Milestone 0 GPU validation |
| `configs/` | Inference, Experiment A, BDD100K training, synthetic smoke run |
| `chart/crosswalk/` | Helm chart: Deployment, Jobs, ServiceMonitor, route, dashboard |
| `docker/` | ROCm image (built on the homelab) and CPU image (built by CI) |
| `scripts/` | Build the ROCm image into the cluster; submit Jobs |
| `tests/` | pytest suite, CPU-only, including a kill-and-resume training test |

## What it demonstrates

| Skill | Where |
| --- | --- |
| GPU inference serving on Kubernetes | Inference Deployment: GPU request, startup/readiness probes, Recreate rollout for one GPU, ServiceMonitor |
| Batching and the latency/throughput trade-off | `inference/batcher.py`: sequential, fixed and dynamic (max batch + max wait) strategies; Experiment A measures them |
| Observability | Prometheus histograms for p50/p95/p99 latency, queue wait, batch size, drops, errors, model version; Grafana dashboard |
| Reproducible benchmarking | Every result records hardware, kernel, ROCm, PyTorch and HIP versions, model, config hash and method |
| Writing a training loop, not configuring one | `training/loop.py`: optimizer, LR schedule, AMP, clipping, checkpointing and resume written out |
| Fault tolerance | Atomic checkpoints to S3 with local fallback, exact mid-epoch resume, Experiment B measures recovery |
| ML lifecycle tooling | MLflow params, metrics, artifacts and model registry; dataset content hashes |
| Model evaluation | COCO mAP with pycocotools, fine-tuned model vs. pretrained baseline on a hand-labeled set |
| AMD/ROCm platform work | ROCm image, device-plugin GPU scheduling, MIOpen cache, vendor-agnostic code |

## Quickstart (CPU, no cluster)

Python 3.12 or newer.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install --requirement requirements/cpu.txt --requirement requirements/dev.txt
pip install --no-deps --editable .
pytest
```

Try the platform end to end on synthetic data in a couple of minutes:

```bash
crosswalk make-synthetic /tmp/cw --video
crosswalk validate --device cpu --model fcos_tiny --height 64 --width 96 --steps 3

# Train (tracking off: no MLflow server needed)
crosswalk train --config configs/train-synthetic.yaml \
  --set data.train.labels=/tmp/cw/det_train.json --set data.train.images=/tmp/cw/images \
  --set data.eval.labels=/tmp/cw/instances.json --set data.eval.images=/tmp/cw/images \
  --set checkpoint.local_dir=/tmp/cw/ckpt --set tracking.enabled=false

# Kill it mid-run and run the same command again: it resumes from the last
# checkpoint and reports lost steps and recovery time.
```

Detection on the test clip (downloads the pretrained COCO weights once):

```bash
crosswalk infer --config configs/inference.yaml \
  --set 'source.videos=[/tmp/cw/clip.mp4]' --set output.dir=/tmp/cw/out --set source.resize=null
```

## On the homelab

[Running on the homelab](docs/running-on-the-homelab.md) has the full steps.
In short:

1. In the homelab repo: `./scripts/ml-credentials.sh crosswalk`.
2. `kubectl --context kind-homelab-dev apply -f deploy/argocd-application.yaml`.
   Argo CD deploys the chart. It runs on CPU until the GPU is set up.
3. Build the ROCm image on the host and load it into the cluster:
   `scripts/build-rocm-image.sh`.
4. Once the homelab runs AMD's GPU device plugin, run Milestone 0 first:
   `scripts/submit-training.py validate --image-tag <tag>`.
5. Then set `gpu.enabled=true` and `image.gpu.tag=<tag>` on the Argo CD
   application.

There is one GPU, and the homelab's on-demand language model uses it too. A
GPU pod here waits Pending while the model runs. The homelab's
`./scripts/lab-mode.py status` shows who holds it.

## Roadmap

| Milestone | Status | What |
| --- | --- | --- |
| 0. Validate the stack | Built, not yet run on the card | `crosswalk validate`: a few detector training steps on gfx1030, printing device, HIP version and throughput. The RX 6900 XT shares gfx1030 with AMD's supported Radeon PRO W6800, but is not on AMD's official list itself. |
| 1. GPU inference on Kubernetes | Built, not yet run on the card | FCOS on recorded footage; dynamic batching; Prometheus metrics and dashboard; JSON-lines detections; Experiment A harness |
| 2. PyTorch training platform | Built, not yet run on the card | Custom loop; S3 checkpoints with resume; MLflow tracking and registry; dataset hashing; mAP vs. baseline; Experiment B |
| 3. Streaming and the leaderboard | Not started | Kafka topics for frames and detections, a Rust ingestion service, multi-object tracking to count crossings, the crosswalk leaderboard |
| 4. NVIDIA (optional) | Not started | A CUDA image as a drop-in; the same benchmarks on an NVIDIA card for a cross-vendor comparison |

Milestone 3 is where this becomes a streaming system: detections go out
as events, tracks turn into "a pedestrian crossed here" facts, and the
leaderboard ranks intersections. The JSON-lines records written today
(`docs/benchmarks.md#detection-output`) are shaped to become those events.

## Learn more

- [Running on the homelab](docs/running-on-the-homelab.md): images, GPU
  scheduling, credentials, deploying, submitting Jobs, troubleshooting.
- [Benchmarks](docs/benchmarks.md): methodology and result templates for
  Experiments A and B; output schemas.
- [Data](docs/data.md): datasets, licences, privacy, labeling the eval set.
