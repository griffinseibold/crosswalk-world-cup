# Deployment

How Crosswalk World Cup runs on Kubernetes: what the cluster has to provide,
how the GPU is scheduled, the images, deploying the inference service and the
streaming pipeline, submitting training and benchmark Jobs, promoting a
model, and what to check when something goes wrong.

The reference deployment is a single machine running a Kind cluster (kubectl
context `kind-homelab-dev`) with one AMD Radeon RX 6900 XT. The scripts
default to that context; set `CROSSWALK_KUBE_CONTEXT` (or pass `--context`)
for any other cluster, for example `homelab-dev` on k3s.

## What the cluster provides

| Need | Expected at | Notes |
| --- | --- | --- |
| S3-compatible storage | `http://s3.ml-storage.svc.cluster.local:9000`, path-style | Buckets `datasets`, `checkpoints`, `mlflow-artifacts` |
| S3 credentials | Secret `s3-credentials` in `crosswalk` | Keys `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` |
| Experiment tracking | MLflow at `http://mlflow.mlflow.svc.cluster.local:5000` | The server stores artifacts, so pods need no extra credentials |
| Event streaming | Kafka at `kafka.kafka.svc.cluster.local:9092`, PLAINTEXT | Only for the streaming pipeline (`pipeline.enabled`) |
| Batch queueing | Kueue with a ClusterQueue `gpu` | Optional (`kueue.enabled`); Jobs wait in a queue for the GPU |
| Metrics and dashboards | kube-prometheus-stack | ServiceMonitors in any namespace; ConfigMaps labelled `grafana_dashboard: "1"`; PrometheusRules labelled `release: kube-prometheus-stack` |
| Web access | A Gateway API Gateway (`homelab` in `gateway-system`) | `http://crosswalk.localhost:8080/` |
| GPU | AMD device plugin (`amd.com/gpu`) or NVIDIA device plugin (`nvidia.com/gpu`) | See below |
| GitOps | Argo CD | `deploy/argocd-application.yaml` |

Every address is a chart value (`s3.*`, `mlflow.*`, `kafka.*`,
`httpRoute.*`, `kueue.*` in `chart/crosswalk/values.yaml`), so a cluster
with different names only needs overrides.

## The GPU

### One card, shared

The reference machine has one 16 GB GPU (gfx1030). Kubernetes treats it as a
single `amd.com/gpu`, so only one pod can hold it at a time: the inference
service, a training Job, a benchmark, or any other GPU workload on the
cluster.

- **A GPU pod waits Pending while something else holds the card.** Its
  events say `Insufficient amd.com/gpu`. It starts by itself once the card is
  free. With Kueue on, batch Jobs wait suspended in the queue instead of
  piling up Pending (`kubectl -n crosswalk get workloads`).
- **The inference service holds the card too.** Before a training run or a
  benchmark, scale it down (`inference.replicaCount=0` on the Argo CD
  application) and back up afterwards. Training and inference on one 16 GB
  card at the same time would fight over memory anyway.

### Without a device plugin: CPU fallback

`amd.com/gpu` is advertised by
[AMD's Kubernetes device plugin](https://github.com/ROCm/k8s-device-plugin).
On a cluster without it, a pod that requests `amd.com/gpu` stays Pending
forever.

So `gpu.enabled` defaults to `false`. The chart then runs the CPU image with
no GPU request. Everything works, slowly: use it to check the wiring (S3,
MLflow, Kafka, the route, the dashboard) and to run the synthetic training
smoke test. Turn the GPU on once the plugin is installed.

What the plugin needs on a Kind cluster:

- `/dev/kfd` as well as `/dev/dri` mounted into the worker nodes. ROCm
  compute needs `/dev/kfd`. Kind fixes mounts when a cluster is created, so
  adding it means recreating the cluster.
- The plugin DaemonSet. It advertises `amd.com/gpu` on each node where it
  finds the GPU and puts the device nodes into pods that request it.
- If several Kind workers see the same physical card, run the plugin on one
  worker only (a node label and the DaemonSet's `nodeSelector`), or the
  scheduler could place two GPU pods on what is really one GPU. Set
  `gpu.nodeSelector` in this chart to match.

On k3s the plugin runs directly on the host and none of this applies.

### What the pods are allowed to do

Every pod runs as non-root user 10001 with seccomp `RuntimeDefault`, no
privilege escalation, all Linux capabilities dropped, a read-only root
filesystem and no hostPath. ROCm needs exactly one relaxation:

- **Supplemental groups `video` and `render`.** The device plugin puts
  `/dev/kfd` and `/dev/dri/renderD*` into the pod with the host's ownership:
  root, group `render` (and `video` on some hosts). A non-root process can
  only open them as a member of those groups. The chart adds them by numeric
  GID (`gpu.videoGid`, `gpu.renderGid`), because group names inside the image
  mean nothing on the host. Check the real values on the host:

  ```bash
  getent group video render
  # video:x:44:...
  # render:x:992:...     <- Ubuntu allocates this one dynamically
  ```

Things ROCm guides often suggest that the chart does **not** do:

- `privileged: true`. The device plugin injects the devices.
- `--security-opt seccomp=unconfined`. AMD lists it as optional, for HPC
  memory-affinity tools (`numactl`) this project does not use. If MIOpen or
  the runtime ever fails with a permission error from a blocked syscall, this
  is the setting to try, scoped to the one Job.
- `--ipc=host`. DataLoader workers share memory through a memory-backed
  `/dev/shm` volume in the Job (`job.shmSize`) instead.

Writable paths are explicit volumes: `/tmp`, `/cache` (PyTorch weights,
footage, and MIOpen's kernel database, so tuned kernels survive restarts),
`/data` and `/scratch`.

## Images

| Image | Built by | Used for |
| --- | --- | --- |
| `ghcr.io/griffinseibold/crosswalk-world-cup:<version>` | CI, on version tags (CPU PyTorch) | `gpu.enabled=false`, and always for tracking, indicators and ranking |
| `ghcr.io/griffinseibold/crosswalk-ingest:<version>` | CI, on version tags | The Rust frame sampler |
| `crosswalk-world-cup:rocm-<commit>` | `scripts/build-rocm-image.sh` on the GPU host | `gpu.enabled=true` on AMD |
| `crosswalk-world-cup:cuda-<commit>` | `docker build -f docker/Dockerfile.cuda` on the GPU host | NVIDIA ([deploy/nvidia](../deploy/nvidia/README.md)) |

The ROCm image is built on
`rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.9.1`, the image
AMD validates for ROCm 7.2.4 with PyTorch 2.9.1. It is tens of gigabytes, too
large for GitHub's hosted runners, so it is built where it runs and never
goes through a registry.

```bash
scripts/build-rocm-image.sh                    # Kind: kind load docker-image into every node
RUNTIME=k3s scripts/build-rocm-image.sh        # k3s: docker save | k3s ctr images import
```

The script prints the tag to use. The chart uses `imagePullPolicy:
IfNotPresent` and a tag that never moves, so Kubernetes never tries to pull
an image that exists only on the nodes. Build a new tag for every change.

## Deploying

Once:

```bash
# Register with Argo CD; it creates the namespace and the chart
kubectl --context kind-homelab-dev apply -f deploy/argocd-application.yaml

# S3 credentials for the namespace (any tool that creates the Secret works)
kubectl --context kind-homelab-dev -n crosswalk create secret generic s3-credentials \
  --from-literal=AWS_ACCESS_KEY_ID=... --from-literal=AWS_SECRET_ACCESS_KEY=...
```

Argo CD creates the `crosswalk` namespace (labelled for the Gateway and
Kueue) and keeps the chart in step with `main`. Until the Secret exists, pods that
need S3 report `CreateContainerConfigError`; they start once it is created.

### The inference service on its own

With `pipeline.enabled=false` (the default) the inference service reads
clips straight from `s3://datasets/footage/`, plays them in a loop at their
own frame rate like a camera, and writes one JSON-lines file of detections
per pass to `s3://datasets/detections/`. Its status page is
<http://crosswalk.localhost:8080/status>. With no footage uploaded yet it
says `waiting for footage` and checks again every minute.

### The streaming pipeline and the leaderboard

`pipeline.enabled=true` deploys every stage between a clip and the
leaderboard ([streaming](streaming.md)):

| Component | Image | Replicas |
| --- | --- | --- |
| `crosswalk-ingest` | Rust sampler | 1 per footage set; syncs `pipeline.ingest.footage` into an emptyDir and samples it at `pipeline.ingest.sampling` (default `fps:5`) |
| `crosswalk-inference` | GPU (or CPU) | 1 (one GPU); switches to Kafka mode |
| `crosswalk-tracking`, `crosswalk-indicators` | CPU | Up to the topic's partitions (3) |
| `crosswalk-ranking` | CPU | Exactly 1 (SQLite on a PersistentVolumeClaim, `Recreate`) |

Before turning it on:

1. Kafka is reachable at `kafka.bootstrapServers`. Topics are created by the
   components themselves at startup.
2. Each crossing has a camera config. The chart ships the examples from
   `chart/crosswalk/files/cameras/`; add your own with
   `--set-file pipeline.cameras.<name>=configs/cameras/<name>.yaml` (or in
   the Argo CD application's values). A clip belongs to the camera whose
   `camera_id` matches its file name up to the first `__`: both
   `lankershim-01.mp4` and `lankershim-01__2005-06-16.mp4` belong to camera
   `lankershim-01`.
3. Blurred clips are in `s3://datasets/footage/` ([data](data.md)).

The leaderboard is then at <http://crosswalk.localhost:8080/>, and the
inference status at `/status` on the same host.

**From other devices on the local network:** `httpRoute.lan.enabled=true`
adds `crosswalk.lab.internal` on the Gateway's `lan` listener and labels the
namespace for it. It is off by default because anyone on the network could
then open the page. Point the name at the Gateway in the router's DNS.

**Alerts.** The chart installs PrometheusRules: inference p99 over budget,
a stage down or not consuming, consumer lag that keeps growing, dead-lettered
events, and rankings that go stale while frames are flowing. Each alert's
description says where to look next. `crosswalk replay --speed N` replays
recorded frame events at N times real speed to find where the pipeline
saturates ([benchmarks](benchmarks.md#experiment-d-streaming-pipeline-under-load)).

### Switching to the GPU

As Argo CD parameters or in `deploy/argocd-application.yaml`:

| Parameter | Value |
| --- | --- |
| `gpu.enabled` | `true` |
| `image.gpu.tag` | the tag `build-rocm-image.sh` printed |
| `gpu.renderGid` | the host's render GID, if not 992 |

The first start takes a few minutes: it downloads the COCO weights to the
cache volume and runs every batch size once so MIOpen picks its kernels. The
startup probe allows ten minutes.

The [Grafana dashboard](http://grafana.localhost:8080/d/crosswalk-inference)
shows frames per second, latency percentiles, queue wait, batch sizes, drops,
errors, the serving model and, with the pipeline on, consumer lag and
throughput per stage. Gaps while the service is scaled to zero or waiting for
the GPU are normal.

## Submitting Jobs

Training, validation, benchmarks and promotion are one-off Jobs, not part of
what Argo CD syncs (Argo CD would prune them or keep re-creating them).
`scripts/submit-training.py` renders them from the same chart (`helm template
--show-only templates/job.yaml`), so they get the same security settings, GPU
request and S3/MLflow wiring, then creates them with `kubectl create`.

```bash
# Run this first on a new GPU: does PyTorch train on it?
scripts/submit-training.py validate --image-tag rocm-abc1234

# Training smoke test on synthetic data (works on CPU)
scripts/submit-training.py train --config configs/train-synthetic.yaml --synthetic --cpu --image-tag 0.1.0

# Fine-tuning: BDD100K subset, evaluated on the fixed-camera set
scripts/submit-training.py train --config configs/train-bdd100k.yaml --image-tag rocm-abc1234

# Batching benchmark (Experiment A)
scripts/submit-training.py benchmark --config configs/benchmark-a.yaml \
  --sync s3://datasets/footage=/scratch/data/footage --image-tag rocm-abc1234
```

- **Queueing.** With `kueue.enabled` (the default) every Job is created
  suspended with `kueue.x-k8s.io/queue-name: crosswalk`, and Kueue admits it
  when the `gpu` ClusterQueue has room. `kubectl -n crosswalk get workloads`
  shows the queue. `--no-kueue` submits a plain Job on a cluster without
  Kueue.
- **Data.** Training configs read from `/scratch/data/<name>/…`. An init
  container first downloads each `<name>` from `s3://datasets/<name>/`.
  `--sync` overrides this; `--no-sync` skips it.
- **One run, one Job.** A training Job is named after the config's
  `run_name`, so submitting the same run twice fails with "already exists"
  instead of changing a running Job.
- **Kills and evictions.** A killed training pod is replaced, up to
  `job.backoffLimit` times, and resumes from its newest checkpoint in
  `s3://checkpoints/<run_name>/`. If the Job has given up, delete it and
  submit again; training resumes from the checkpoint. Use a new `run_name` to
  start fresh.
- **Several GPUs.** `--gpus N` (train only) launches `torchrun` with one
  DistributedDataParallel process per GPU in one pod. `batch_size` in the
  config is per GPU, and the learning rate is not rescaled. The resume hash
  includes the world size, so a run cannot silently resume with a different
  global batch. This is for multi-GPU machines (for example a rented NVIDIA
  VM with `kueue.enabled=false`); on a one-GPU cluster Kueue would never
  admit it.
- **Per-run changes.** `--set epochs=1` passes a config override to the Job,
  `--env NAME=VALUE` an environment variable. `--dry-run` prints the manifest.
- **Cleanup.** Finished Jobs and their config are deleted after a day
  (`job.ttlSecondsAfterFinished`). Results live in S3 and MLflow.

## Promoting a model

The leaderboard should only change because the crossings changed, not
because a worse detector started serving. Fine-tuned models therefore go
through a gate before they serve:

```bash
scripts/submit-training.py promote --config configs/train-bdd100k.yaml --image-tag rocm-abc1234
```

The promote Job scores the newest registered version and the current
`@champion` on the same hand-labelled fixed-camera set, times both, and moves
the `champion` alias in the MLflow registry only if mAP did not drop,
pedestrian AP did not drop by more than a point, and p95 latency did not grow
by more than 20%. The decision and its reasons are logged to MLflow
(experiment `crosswalk-promotion`) and as tags on the version.

With `inference.servePromoted=true` the inference service serves
`models:/crosswalk-detector@champion`. After a promotion, restart it to load
the new version:

```bash
kubectl -n crosswalk rollout restart deployment/crosswalk-inference
```

`crosswalk rollback` (in a pod, or locally with `MLFLOW_TRACKING_URI` set)
moves `champion` back to the previous version and records why; restart the
Deployment again afterwards.

## Troubleshooting

| Symptom | Next check |
| --- | --- |
| GPU pod Pending, `Insufficient amd.com/gpu` | Something else holds the card (`kubectl get pods -A` with a GPU request), or the device plugin is not installed; use `gpu.enabled=false` until it is |
| Job stays suspended | Kueue has not admitted it: `kubectl -n crosswalk get workloads` and `kubectl describe clusterqueue gpu` |
| `ErrImageNeverPull` / `ErrImagePull` for `crosswalk-world-cup:rocm-…` | The image is not on that node: rerun `scripts/build-rocm-image.sh` and check `image.gpu.tag` |
| Helm error "gpu.enabled=true needs image.gpu.tag" | Build the GPU image and set the tag it printed |
| `validate` says `PyTorch sees no GPU` | `/dev/kfd` or `/dev/dri` missing in the pod (`kubectl exec … -- ls -l /dev/kfd /dev/dri`), or the render GID is wrong (permission denied in the logs) |
| `hipErrorNoBinaryForGPU` | The PyTorch build has no kernels for this GPU. Check `rocminfo` on the host. As a last resort, set `gpu.hsaOverrideGfxVersion=10.3.0` and rerun validate |
| First batch takes many seconds | MIOpen kernel selection, once per shape. The `/cache` volume keeps the result; warmup runs it before the pod reports ready |
| Status says `waiting for footage` | Upload clips to `s3://datasets/footage/`, or check the error shown with it (credentials, endpoint) |
| A stage's status page shows a Kafka error | `kafka.bootstrapServers` unreachable; the stage keeps retrying and answers `/metrics` meanwhile (`CrosswalkStageNotConsuming` fires after 10 minutes) |
| A crossing never leaves "not ranked yet" | The page lists the reason: not enough video, windows or pedestrian-minutes, or a mix of calibrated and uncalibrated windows |
| Events in `crosswalk.<stage>.dlq` | Each record carries the error and the original event ([streaming](streaming.md)) |
| Training logs `S3 unavailable … checkpoints stay local` | Checkpoints are kept only in the pod's scratch volume. Check the `s3-credentials` Secret and the endpoint; a lost pod would lose progress |
| Training refuses to resume: "different training config" | The config changed under the same `run_name`. Use a new `run_name`, or restore the original config |
| MLflow run missing | `MLFLOW_TRACKING_URI` unreachable: training continues untracked and logs an error. Check the `mlflow` namespace |
