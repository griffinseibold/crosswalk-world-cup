# Crosswalk World Cup

**Which crosswalk in the world is the safest to cross?**

Crosswalk World Cup tries to answer that from video. It watches recorded
footage of pedestrian crossings, finds every pedestrian and vehicle, follows
them through the scene, and measures how often people on foot and vehicles
come uncomfortably close. Crossings are then ranked against each other on a
public leaderboard, with the uncertainty of every number shown next to it.

The rules are the same for every crossing: same detector, same thresholds,
same minimum amount of footage, same statistics. A crossing tops the table
only when the data shows few near misses *and* there is enough footage to
believe it.

> The leaderboard compares observable indicators from recorded footage. It is
> an experiment, not a risk model and not safety advice. A crossing near the
> top is one where little conflict was *seen*, not a guarantee of anything.
> [What the numbers do not tell you](docs/indicators.md#what-the-numbers-do-not-tell-you).

## How a crossing is scored

```text
recorded clip of a crossing (faces and plates blurred)
   │
   ▼
frame sampler ─► detector ─► tracker ─► conflict indicators ─► ranking ─► leaderboard
(Rust, 5 fps)    (FCOS, GPU)  (Kalman +   (per 60 s window)      (per crossing,
                              Hungarian)                          with 95% intervals)
```

1. **Sample.** The clip is decoded with FFmpeg and sampled a few times a
   second. Each frame becomes an event on Kafka; the image goes to object
   storage.
2. **Detect.** A TorchVision FCOS detector finds pedestrians, cars, trucks,
   buses, bicycles and motorcycles in each frame, on the GPU, with dynamic
   batching.
3. **Track.** Detections are linked into tracks across frames (constant-velocity
   Kalman filter, Hungarian assignment), so one person crossing the road is
   one pedestrian, not a hundred detections.
4. **Measure.** For each minute of video, the indicators stage counts how
   people and vehicles interacted inside the crosswalk's region of interest:
   close approaches, minimum separation, crossing paths, time to collision
   (TTC) and post-encroachment time (PET). With a calibrated camera these are
   in metres and seconds on the road plane.
5. **Rank.** Windows are summed per crossing and normalised by how much foot
   traffic was actually observed. Crossings are ordered by the upper end of a
   95% confidence interval, so a short, lucky clip cannot win.

The whole pipeline runs end to end on a laptop (generated footage, no GPU or
Kafka needed) and on a Kubernetes cluster with one GPU for real footage.

## What "safest" means here

The headline number is **close approaches per 100 pedestrian-minutes**: how
many times a pedestrian and a vehicle came within 2 m of each other (on a
calibrated camera), for every 100 minutes that people spent at the crossing.

| Rule | Why |
| --- | --- |
| Normalise by pedestrian exposure | Raw counts would just rank crossings by how busy they are |
| Exact 95% Poisson interval on every rate | Small numbers are noisy; the interval says how noisy |
| Rank by the interval's upper bound, lowest first | The highest rate the data cannot rule out. Zero events in five minutes is not a win |
| At least 10 minutes of video, 5 windows and 30 pedestrian-minutes | Below that, a crossing is listed as "not ranked yet", with the reason |
| Calibrated (metres) and uncalibrated (pixels) crossings in separate tables | Pixel distances depend on camera height and angle and cannot be compared across cameras |
| Each pair of tracks counts at most once per window | A pedestrian waiting next to a parked car is one close approach, not 150 |

TTC and PET are long-established surrogate safety measures in traffic
engineering (FHWA's SSAM uses TTC ≤ 1.5 s). They are reported for every
calibrated crossing alongside the headline rate.

Full definitions: [conflict indicators](docs/indicators.md) and
[ranking rules](docs/ranking.md).

## The leaderboard

The ranking service serves one self-contained page (no external requests,
light and dark, works on a phone) and a JSON API:

| Path | What |
| --- | --- |
| `/` | The leaderboard: ranked crossings, rates with intervals, rank changes over the last day, and crossings not ranked yet |
| `/api/leaderboard` | The same data as JSON, with the footage source and licence for each crossing |
| `/api/crossings?camera=<id>` | Every stored window for one crossing |
| `/api/history` | Recent leaderboard snapshots |

Every crossing carries its footage source, licence and attribution, and the
time its numbers were computed.

## Try it locally

Python 3.12 or newer. No GPU, Kafka or object storage needed.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install --requirement requirements/cpu.txt --requirement requirements/dev.txt
pip install --no-deps --editable .

crosswalk demo demo-out --serve
```

`crosswalk demo` generates two crossings, one where drivers yield and one
where they do not, and runs them through the real tracking, indicators and
ranking code. It prints the resulting table and serves the leaderboard at
<http://localhost:8000>. (The demo finds the generated shapes with a simple
colour detector, because a model trained on real photos sees nothing in
coloured rectangles. Everything after detection is production code.)

Add `--video` to encode the scene as a real clip and decode it with PyAV, or
`--ingest-bin ingest/target/release/crosswalk-ingest` to sample it with the
Rust frame sampler (`cargo build --release` in `ingest/`).

Run the tests:

```bash
pytest
```

## Adding a crossing

1. **Footage.** A fixed camera overlooking one crosswalk, at least 10
   minutes of it. Only recorded, openly licensed or self-recorded footage,
   with faces and licence plates blurred before it is stored. See
   [data handling](docs/data.md) and the researched list of
   [candidate sources](docs/data-sources.md).
2. **Camera config.** `configs/cameras/<camera>.yaml` names the crossing,
   draws the region of interest around the crosswalk, and records the
   footage's source, licence and attribution.
3. **Calibration (recommended).** Four image points with known positions on
   the road, usually the crosswalk's corners, turn pixels into metres. This
   moves the crossing into the metric table and unlocks TTC and PET.
4. Upload the clip and let the pipeline run. The crossing appears under
   "not ranked yet" until it has enough footage.

## Accuracy of the detector

Every count on the leaderboard is only as good as the detections behind it.
The pretrained detector was trained on everyday photos, not on cameras
looking down at a crosswalk, so the project:

- fine-tunes it on road scenes with a resumable training loop that logs
  every run (parameters, metrics, dataset hash, model) to MLflow;
- scores both the fine-tuned and the pretrained model on a hand-labelled set
  of frames from fixed crossing cameras (COCO mAP, per-class AP for
  pedestrians and cars);
- only promotes a new model to the leaderboard if it does not get worse at
  finding pedestrians (`crosswalk promote`, with `crosswalk rollback` to undo).

## Running it on a cluster

The full pipeline runs on Kubernetes: the Rust sampler, the GPU inference
service, the tracking, indicators and ranking stages, Kafka between them,
Prometheus metrics and alerts, and a Grafana dashboard. One Helm chart
deploys everything (`chart/crosswalk`), and Argo CD keeps it in sync.

The reference deployment is a single machine with one AMD Radeon RX 6900 XT
(ROCm 7.2). The same code runs on NVIDIA GPUs with the CUDA image and
`values-nvidia.yaml`; device selection never branches on the vendor.

- [Deployment](docs/deployment.md): images, GPU scheduling, credentials,
  the streaming pipeline, training and benchmark Jobs, troubleshooting.
- [Streaming](docs/streaming.md): topics, event formats, delivery
  guarantees, dead letters, scaling.
- [Benchmarks](docs/benchmarks.md): how throughput, latency, failure
  recovery and model quality are measured. Results are added only from real
  runs.
- [NVIDIA](deploy/nvidia/README.md): running the same pipeline and
  benchmarks on a rented NVIDIA GPU.

## Repository layout

| Path | What it is |
| --- | --- |
| `ingest/` | Rust frame sampler: FFmpeg decode, sampling, JPEGs to S3, frame events to Kafka |
| `src/crosswalk/inference/` | GPU detection service: decoding, dynamic batcher, predictor, metrics |
| `src/crosswalk/tracking/` | Multi-object tracking within one clip |
| `src/crosswalk/indicators/` | Conflict indicators, camera configs and calibration |
| `src/crosswalk/ranking/` | Ranking rules, statistics, SQLite store and the leaderboard page |
| `src/crosswalk/stream/` | Kafka and in-memory event bus, stage runner, replay load tester |
| `src/crosswalk/training/` | Datasets, training loop, checkpoints, evaluation, MLflow tracking |
| `src/crosswalk/benchmark/` | Throughput and latency benchmarks |
| `configs/` | Inference, stages, training, benchmark and camera configs |
| `chart/crosswalk/` | Helm chart, Grafana dashboard and alert rules |
| `deploy/` | Argo CD application; Ansible for an NVIDIA GPU machine |
| `docker/` | CPU, ROCm and CUDA images |
| `scripts/` | Build the GPU image, submit Jobs, queueing experiment |
| `tests/` | pytest suite (CPU only), including an end-to-end pipeline test |

## Status

| | Status |
| --- | --- |
| Pipeline (sampler → detector → tracker → indicators → ranking → leaderboard) | Built and tested end to end on generated footage |
| Cluster deployment, metrics, alerts | Built; first GPU run pending |
| Detector fine-tuning and evaluation | Built; waiting for the hand-labelled fixed-camera set |
| Crossings on the leaderboard | None yet. First candidate: NGSIM Lankershim Boulevard, Los Angeles, once its video licence is confirmed |

Next: the first real crossings, calibrated, with enough footage to rank.

## Scope and limits

- **Recorded footage only.** No live camera feeds, RTSP or webcams, by design.
- **Faces and licence plates are blurred before footage is stored.** Blurring
  does not grant permission to use footage; see [data handling](docs/data.md).
- **No re-identification.** Tracks never leave the clip they came from.
- **BDD100K is not downloaded by this code.** Its licence is non-commercial
  and needs registration; you point the config at your own copy.
- **No invented numbers.** Every number on the leaderboard and in the
  benchmarks comes from a real run with its result file.
