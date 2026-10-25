# Benchmarks

How the two experiments are run and recorded, and where their results go.
**No results are filled in yet.** Every table below stays empty until a real
run on the homelab produces its numbers. Copy rows from the tools' output;
do not type them by hand.

## Ground rules

1. **Every number has a result file.** Experiment A writes a JSON file with the
   full environment; Experiment B's numbers are logged in MLflow and in
   `summary.json` next to the checkpoints. A table row links to its file or run.
2. **The environment is part of the result.** Each file records the GPU name
   and architecture, kernel, amdgpu driver, ROCm userspace, PyTorch, HIP and
   TorchVision versions, CPU, memory limit, container image, git commit and a
   hash of the exact config. Results with different environments go in
   different rows.
3. **Medians of repeats.** Every configuration runs `repeats` times (3 by
   default); tables report the median, which one noisy run cannot move.
4. **Nothing else on the GPU.** Before a run, check that the homelab LLM and
   the inference service are not holding the card (see
   [running on the homelab](running-on-the-homelab.md#one-card-shared)).
5. **Say what was not controlled.** GPU clocks and temperatures are not
   pinned on a consumer card; note anything unusual in the config's `notes`.

## Experiment A: batching strategies

*Question:* how do sequential, fixed and dynamic batching trade throughput
against latency on one RX 6900 XT, across resolutions and numbers of
simultaneous streams?

### Method

`crosswalk benchmark --config configs/benchmark-a.yaml` (or
`scripts/submit-training.py benchmark …` on the cluster):

1. **Frames.** A recorded clip is decoded once per resolution and held in
   memory, so disk and decoding do not leak into the GPU numbers. (The
   service measures decode cost separately: `crosswalk_decode_seconds`.)
   With `source.video: null`, random frames of the same shape are used
   instead. Synthetic results are a GPU reference only, because detection
   post-processing depends on what is in the image.
2. **Warmup.** Before each configuration, every batch size it can produce runs
   `warmup_batches` times and is discarded. MIOpen chooses (and may compile)
   convolution kernels the first time it sees a shape; that one-off cost is
   not what is being measured.
3. **Load.** `concurrency` producer threads, one per simulated stream, share
   `frames_per_run` frames and submit them into one batcher. One worker
   thread feeds the GPU.
   - `offline` mode: producers submit as fast as the queue accepts. This
     measures the throughput ceiling.
   - `realtime` mode: each producer submits at the clip's frame rate, like a
     camera, and drops frames when the queue is full. This measures whether
     N cameras can be kept up with, and at what latency.
4. **Timing.** Inference time is wall-clock time around one batched model
   call. The call ends with a GPU synchronize, so queued GPU work is
   included. Per-frame latency is queue wait plus inference. Percentiles use
   linear interpolation over every per-frame sample, not histogram buckets.
5. **Throughput** is completed frames divided by the time from the first
   batch's start to the last batch's end.

### Strategies compared

| Name | Strategy | Max batch | Max wait |
| --- | --- | --- | --- |
| sequential | sequential | 1 | — |
| fixed-4 | fixed | 4 | — |
| fixed-8 | fixed | 8 | — |
| dynamic-8-10ms | dynamic | 8 | 10 ms |
| dynamic-8-25ms | dynamic | 8 | 25 ms |

Resolutions 1280x720 and 960x540; concurrency 1, 2 and 4; offline mode;
600 frames per run; 3 repeats.

### Results

*Environment:* fill in from the result file's `environment` (GPU and
architecture, ROCm userspace, kernel and amdgpu driver, PyTorch and HIP,
image tag, git commit). *Result file:* link it here.

`crosswalk benchmark` prints this table (`markdown_table` in
`benchmark/harness.py`); paste it as printed.

| Resolution | Strategy | Concurrency | Throughput (fps) | p50 (ms) | p95 (ms) | p99 (ms) | Mean batch | Dropped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | | |

*Findings:* write them only after the table has numbers.

### Result file schema

`experiment-A-<timestamp>-<config hash>.json`, validated by
`harness.validate_result` before it is written.

| Field | Meaning |
| --- | --- |
| `schema` | Result format version (1) |
| `config`, `config_hash` | The exact validated config and a hash of it |
| `environment.hardware` | Host, kernel, CPU, memory and cgroup limit, ROCm userspace and amdgpu driver versions, image, git commit |
| `environment.runtime` | PyTorch, TorchVision, HIP/CUDA versions, GPU name, architecture (`gfx1030`), VRAM, MIOpen version |
| `environment.model` | Model, weights or checkpoint, precision, frame source |
| `method` | How timing, warmup, percentiles and throughput were computed |
| `runs[]` | One per (resolution, strategy, concurrency, repeat): frames submitted/completed/dropped, errors, wall time, throughput, `latency_ms`/`queue_wait_ms`/`inference_ms` (count, mean, p50, p95, p99, max), batch-size mean and histogram, peak GPU memory |
| `summary[]` | Median across repeats for each cell |

## Experiment B: failure recovery

*Question:* when a training pod dies mid-run with no warning, how much
progress is lost and how long until training is moving again?

### Method

The training loop records what it needs:

- a **heartbeat** (`heartbeat.json`: last completed step and wall time),
  written next to the checkpoints every `log_every_steps` steps;
- **checkpoints** every `checkpoint.every_steps` steps, written atomically
  (local temporary file, fsync and rename, upload to S3, then the
  `latest.json` pointer).

On resume it computes:

| Metric | Meaning |
| --- | --- |
| `lost_steps` | Heartbeat step minus checkpoint step: work done again. Expected to average about half of `every_steps` |
| `downtime_s` | Time from the last heartbeat to the new process starting: detection of the failure, rescheduling, container start |
| `restore_s` | Finding, downloading, verifying (SHA-256) and loading the checkpoint |
| `recovery_s` | New process start to the first completed step after resuming |

These go to MLflow as `system/resume_*` on the same run, and into
`summary.json` under `resume`.

Two ways to cause the failure:

1. **From inside (repeatable):** set `CROSSWALK_FAULT_AT_STEP=N` on a fresh
   run. The process SIGKILLs itself right after step N. The Job controller
   replaces the pod, and the replacement resumes and ignores the variable.
   On the cluster:

   ```bash
   scripts/submit-training.py train --config configs/train-bdd100k.yaml \
     --set run_name=expb-1 --env CROSSWALK_FAULT_AT_STEP=500 --image-tag rocm-abc1234
   kubectl --context kind-homelab-dev -n crosswalk get pods -l job-name=crosswalk-train-expb-1 -w
   ```

   The first pod ends with exit code 137 (SIGKILL) and the second carries on.
   Read the recovery numbers from the run's `summary.json` or MLflow.
2. **From outside (realistic):** while the Job runs,
   `kubectl -n crosswalk delete pod <pod> --grace-period=0 --force`. The
   heartbeat records where it was.

Test both checkpoint locations: S3 (normal), and S3 unreachable (scale the
S3 service to zero), where the run falls back to its scratch volume and
survives a process kill but not a lost pod.

`tests/test_training.py::test_resume_after_kill_matches_uninterrupted_run`
runs the inside method on CPU in CI. It kills at step 7 with checkpoints
every 3 steps, then asserts `lost_steps == 1` and that the final weights match
an uninterrupted run with the same seed. It proves the mechanism; the numbers
below are about the homelab.

### Results

*Environment:* GPU, image tag, git commit, checkpoint size and interval,
where checkpoints were stored. *MLflow runs:* link each.

| Failure | Checkpoint every | Checkpoint size | Lost steps | Downtime (s) | Restore (s) | Recovery (s) | Checkpoint save (s) | Upload (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | | |

Checkpoint save and upload times come from `system/checkpoint_serialize_s`
and `system/checkpoint_upload_s`. Together with `lost_steps` they show the
trade-off in choosing `every_steps`.

*Findings:* only after the table has numbers.

## Model quality

Logged under `quality/` in MLflow, separately from the `system/`
infrastructure metrics. Both models are scored on the same hand-labeled
fixed-camera set with pycocotools.

| Model | Eval set (sha256) | mAP@[.5:.95] | mAP@.5 | AP car | AP pedestrian | MLflow run |
| --- | --- | --- | --- | --- | --- | --- |
| Pretrained FCOS (COCO), labels mapped | | | | | | |
| Fine-tuned on BDD100K subset | | | | | | |

## Detection output

The inference service writes one JSON object per frame, one file per pass
over each clip (`<clip>-<timestamp>.jsonl`). A file appears under its final
name only when complete.

```json
{"schema": 1, "video": "intersection-01.mp4", "stream": "intersection-01", "frame": 412,
 "pts_s": 13.7333, "width": 1280, "height": 720, "decoded_at": 1793000000.123,
 "model_version": "fcos_resnet50_fpn:COCO_V1", "batch_size": 6,
 "latency_ms": {"queue_wait": 8.1, "inference": 41.7, "end_to_end": 49.8},
 "detections": [{"label": "person", "label_id": 1, "score": 0.91, "box": [604.2, 311.0, 640.8, 402.5]}]}
```

Boxes are `[x1, y1, x2, y2]` in pixels of the frame as decoded (after
`source.resize`). The example values above show the shape of a record and
are not measurements.
