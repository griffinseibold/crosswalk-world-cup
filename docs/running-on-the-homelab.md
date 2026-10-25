# Running on the homelab

How Crosswalk World Cup runs on the [homelab](https://github.com/griffinseibold/Homelab):
the images, how the GPU gets scheduled, the credentials it needs, and how
to deploy the service and submit Jobs. Commands target the homelab's Kind
cluster, `kind-homelab-dev`.

## What the homelab provides

| Need | Provided by | Address |
| --- | --- | --- |
| S3 storage | `ml-storage` namespace | `http://s3.ml-storage.svc.cluster.local:9000`, path-style; buckets `datasets`, `checkpoints`, `mlflow-artifacts` |
| S3 credentials | `scripts/ml-credentials.sh crosswalk` (homelab repo) | Secret `s3-credentials` in `crosswalk`, keys `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` |
| Experiment tracking | MLflow in `mlflow` namespace | `http://mlflow.mlflow.svc.cluster.local:5000`; the server stores artifacts, so pods need no extra credentials |
| Metrics and dashboards | kube-prometheus-stack | ServiceMonitors in any namespace; ConfigMaps labelled `grafana_dashboard: "1"` |
| Web access | Envoy Gateway | `http://crosswalk.localhost:8080/status` |
| GPU | AMD device plugin (to be installed; see below) | `amd.com/gpu: 1` |

The chart sets all of these. If an address changes, override it under
`s3.*` and `mlflow.*` in `chart/crosswalk/values.yaml` or on the Argo CD
application.

## The GPU

### One card, shared

The host has one Radeon RX 6900 XT (16 GB, gfx1030). Kubernetes treats it as
a single `amd.com/gpu`, so only one pod can hold it at a time: the inference
service, a training Job, a benchmark, or the homelab's on-demand language
model.

- **A GPU pod waits Pending while something else holds the card.** Its
  events say `Insufficient amd.com/gpu`. It starts by itself once the card is
  free.
- **The homelab LLM is the usual holder.** It runs in LLM mode, or while an
  application labelled `homelab.local/llm: requested` is running (Flip Finder
  is). This app never requests it. To see who holds the card and free it:

  ```bash
  # in the homelab repo
  ./scripts/lab-mode.py status    # mode, whether the LLM runs, who requests it
  ./scripts/lab-mode.py apps      # leave LLM mode; the LLM stops 5 minutes after the last request
  ```

  If an application keeps the LLM running, scale that application to zero
  in Argo CD for the duration of a training run.
- **The inference service holds the card too.** Before a training run or a
  benchmark, scale the service down (`inference.replicaCount=0` on the Argo
  CD application), and scale it back up afterwards. Training and inference on
  one 16 GB card at the same time would fight over memory anyway.

### Until the device plugin is installed: CPU fallback

`amd.com/gpu` is a resource advertised by
[AMD's Kubernetes device plugin](https://github.com/ROCm/k8s-device-plugin).
The homelab does not run it yet. Today its LLM reaches the GPU with
privileged `/dev/dri` hostPath mounts instead. Until the plugin is installed,
a pod that requests `amd.com/gpu` stays Pending forever.

So `gpu.enabled` defaults to `false`. The chart then runs the CPU image with
no GPU request. Everything works, slowly: use it to check wiring (S3,
MLflow, the route, the dashboard) and to run the synthetic training smoke
test. Turn the GPU on once the plugin is in.

What the homelab side needs for the plugin (for whoever does that work):

- `/dev/kfd` mounted into the Kind worker nodes as well as `/dev/dri`. ROCm
  compute needs `/dev/kfd`; the current Kind config mounts only `/dev/dri`
  (enough for the LLM's Vulkan backend). Kind fixes mounts when a cluster is
  created, so this means a cluster rebuild ([recovery guide](https://github.com/griffinseibold/Homelab/blob/main/docs/recovery.md)).
- The plugin DaemonSet. It advertises `amd.com/gpu` on each node where it
  finds the GPU, and puts the device nodes into pods that request it.
- Both Kind workers see the same physical card. If the plugin advertises it
  on both, the scheduler could place two GPU pods at once on what is really
  one GPU. Run the plugin on one worker only, using a node label and the
  DaemonSet's `nodeSelector`. Then set `gpu.nodeSelector` in this chart to
  match.

### What the pods are allowed to do

The pods use the same locked-down security settings as hello-crud: non-root
user 10001, seccomp `RuntimeDefault`, no privilege escalation, all Linux
capabilities dropped, read-only root filesystem, no hostPath. ROCm needed
exactly one relaxation:

- **Supplemental groups `video` and `render`.** The device plugin puts
  `/dev/kfd` and `/dev/dri/renderD*` into the pod with the host's ownership:
  root, group `render` (and `video` for some nodes). A non-root process can
  only open them as a member of those groups. The chart adds them by numeric
  GID (`gpu.videoGid`, `gpu.renderGid`), because group names inside the image
  mean nothing on the host. Check the real values on the host and set them
  if they differ:

  ```bash
  getent group video render
  # video:x:44:...
  # render:x:992:...     <- Ubuntu allocates this one dynamically
  ```

Things ROCm guides often suggest that the chart does **not** do:

- `privileged: true`. The device plugin injects the devices, so it is not needed.
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

| Image | Built by | Used when |
| --- | --- | --- |
| `ghcr.io/griffinseibold/crosswalk-world-cup:<version>` | CI, on version tags (CPU PyTorch) | `gpu.enabled=false` |
| `crosswalk-world-cup:rocm-<commit>` | `scripts/build-rocm-image.sh` on the homelab host | `gpu.enabled=true` |

The ROCm image is built on
`rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.9.1`, the image
AMD validates for ROCm 7.2.4 with PyTorch 2.9.1. gfx1030 is a listed target
of ROCm 7.2.4. The image is tens of gigabytes, too large for GitHub's hosted
runners, so it is built where it runs and never goes through a registry.

### Building and loading it

```bash
scripts/build-rocm-image.sh
```

The script builds `crosswalk-world-cup:rocm-<git commit>` and loads it into
the cluster's nodes. It prints the tag to use.

- **Kind (today's homelab):** the default. It runs
  `kind load docker-image crosswalk-world-cup:<tag> --name homelab-dev`,
  which copies the image into every node. Expect one copy per node on disk.
- **k3s (a possible future homelab):** `RUNTIME=k3s scripts/build-rocm-image.sh`
  streams the image into k3s's own containerd with
  `docker save … | sudo k3s ctr images import -`.

The chart uses `imagePullPolicy: IfNotPresent` and a tag that never moves,
so Kubernetes never tries to pull an image that exists only on the nodes.
Build a new tag for every change rather than reusing one.

## Deploying the service

Once:

```bash
# in the homelab repo: S3 credentials for this namespace
./scripts/ml-credentials.sh crosswalk

# in this repo: register with Argo CD
kubectl --context kind-homelab-dev apply -f deploy/argocd-application.yaml
```

Argo CD creates the `crosswalk` namespace (labelled for the Gateway) and the
service, and keeps it in step with `main`. Then open
<http://crosswalk.localhost:8080/status>.

With no footage uploaded yet, the status page says `waiting for footage` and
checks again every minute. Upload blurred clips ([data](data.md)) to
`s3://datasets/footage/`. The service plays them in a loop at their own frame
rate, like a camera. It writes one JSON-lines file per pass over each clip to
`s3://datasets/detections/`.

Switching to the GPU, as Argo CD parameters or in
`deploy/argocd-application.yaml`:

| Parameter | Value |
| --- | --- |
| `gpu.enabled` | `true` |
| `image.gpu.tag` | the tag `build-rocm-image.sh` printed |
| `gpu.renderGid` | the host's render GID, if not 992 |

The first start after a switch takes a few minutes. It downloads the COCO
weights to the cache volume and runs every batch size once so MIOpen picks
its kernels. The startup probe allows ten minutes.

The [Grafana dashboard](http://grafana.localhost:8080/d/crosswalk-inference)
shows frames per second, latency percentiles, queue wait, batch sizes, drops,
errors and the serving model. Gaps while the service is scaled to zero or
waiting for the GPU are normal.

## Submitting Jobs

Training, validation and benchmarks are one-off Jobs, not part of what Argo CD
syncs. Argo CD would prune them or keep re-creating them.
`scripts/submit-training.py` renders them from the same chart
(`helm template --show-only templates/job.yaml`), so they get the same
security settings, GPU request and S3/MLflow wiring. It then creates them
with `kubectl create`.

```bash
# Milestone 0: does PyTorch train on this GPU? Run this first.
scripts/submit-training.py validate --image-tag rocm-abc1234
kubectl --context kind-homelab-dev -n crosswalk logs -f job/crosswalk-validate-<timestamp>

# Training platform smoke test on synthetic data (works on CPU today)
scripts/submit-training.py train --config configs/train-synthetic.yaml --synthetic --cpu --image-tag 0.1.0

# Real training: BDD100K subset, evaluated against the fixed-camera set
scripts/submit-training.py train --config configs/train-bdd100k.yaml --image-tag rocm-abc1234

# Experiment A
scripts/submit-training.py benchmark --config configs/benchmark-a.yaml \
  --sync s3://datasets/footage=/scratch/data/footage --image-tag rocm-abc1234
```

- **Data.** Training configs read from `/scratch/data/<name>/…`. Before
  training, an init container downloads each `<name>` from
  `s3://datasets/<name>/`. `--sync` overrides this; `--no-sync` skips it.
- **One run, one Job.** A training Job is named after the config's
  `run_name`, so submitting the same run twice fails with "already exists"
  instead of changing a running Job.
- **Kills and evictions.** A killed training pod is replaced, up to
  `job.backoffLimit` times, and resumes from its newest checkpoint in
  `s3://checkpoints/<run_name>/`. If the Job has given up, delete it and
  submit again; training resumes from the checkpoint. Use a new `run_name` to
  start fresh.
- **Per-run changes.** `--set epochs=1` passes a config override to the Job.
  `--dry-run` prints the manifest instead of creating it.
- **Cleanup.** Finished Jobs and their config are deleted after a day
  (`job.ttlSecondsAfterFinished`). Results live in S3 and MLflow.

## Troubleshooting

| Symptom | Next check |
| --- | --- |
| GPU pod Pending, `Insufficient amd.com/gpu` | Something else holds the card: `./scripts/lab-mode.py status` (homelab) and `kubectl -n crosswalk get pods`. If it never starts, the device plugin is not installed; use `gpu.enabled=false` |
| `ErrImageNeverPull` / `ErrImagePull` for `crosswalk-world-cup:rocm-…` | The image is not on that node: rerun `scripts/build-rocm-image.sh` and check `image.gpu.tag` |
| Helm error "gpu.enabled=true needs image.gpu.tag" | Build the ROCm image and set the tag it printed |
| `validate` says `PyTorch sees no GPU` | `/dev/kfd` or `/dev/dri` missing in the pod (`kubectl exec … -- ls -l /dev/kfd /dev/dri`), or the render GID is wrong (permission denied in the logs) |
| `hipErrorNoBinaryForGPU` | The PyTorch build has no gfx1030 kernels. Check `rocminfo` on the host. As a last resort, set `gpu.hsaOverrideGfxVersion=10.3.0` and re-run validate |
| First batch takes many seconds | MIOpen kernel selection, once per shape. The `/cache` volume keeps the result; warmup runs it before the pod reports ready |
| Status says `waiting for footage` | Upload clips to `s3://datasets/footage/`, or check the error shown with it (credentials, endpoint) |
| Training logs `S3 unavailable … checkpoints stay local` | Checkpoints are kept only in the pod's scratch volume. Check the `s3-credentials` Secret and the endpoint; a pod restart would lose progress |
| Training refuses to resume: "different training config" | The config changed under the same `run_name`. Use a new `run_name`, or restore the original config |
| MLflow run missing | `MLFLOW_TRACKING_URI` unreachable: training continues untracked and logs an error. Check the `mlflow` namespace |
