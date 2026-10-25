# Handoff — 2026-10-24

State of the Crosswalk World Cup project and the homelab work around it, written
at the end of a working session so the next one can pick up without rereading
the conversation. Delete this file once its next steps are done.

## 1. Done and live on GitHub

| Repo | Branch | What |
| --- | --- | --- |
| Homelab | `main` (54ad68d) | LLM on demand: a demand controller scales llama-server when a running workload carries `homelab.local/llm: requested`, stops it 5 min after the last request. `scripts/lab-mode.py llm\|apps\|status` switches the lab between apps mode and LLM-only mode (chat UI runs only in llm mode). Alert, bootstrap and Flux drift fixes. 90 tests pass. **Not yet verified on the real cluster.** |
| Flip-Finder | `master` (8d3edcd) | Chart requests the LLM (`llm.enabled`, default true); backend waits for a starting model and tells the user; frontend shows the waiting note. CI passed. |
| hello-crud | `main` (c2e959a) | Optional LLM example, off by default: `GET /llm`, `POST /items/<id>/describe`, 503 + `Retry-After` while the model starts. 12 tests pass. |
| crosswalk-world-cup | `main` (a086a85) | WIP Python package core (see section 3). Not runnable yet. |

## 2. Waiting on Griffin

1. **Publish two releases** (this session can push branches but not tags):
   in Flip-Finder and in hello-crud, GitHub → Releases → Draft a new release →
   tag `v0.4.0`, target branch **`release-0.4.0`** → Publish. CI then builds
   the 0.4.0 images.
2. After the images exist, push each repo's `release-0.4.0` commit to its
   default branch (Flip-Finder 9727a30 → `master`, hello-crud f73c0b4 →
   `main`) so Argo CD rolls onto 0.4.0. Never before: the chart would point at
   an image that doesn't exist.
3. **Check the homelab rollout:** `flux get kustomizations`, then
   `./scripts/lab-mode.py status` — Flip Finder should appear as requesting
   the LLM; chat should be stopped in apps mode.
4. Optional cleanup (this session cannot delete branches): `llm-on-demand` in
   Flip-Finder and hello-crud, and `release-0.4.0` in both after step 2.

## 3. Crosswalk repo: what exists and what's next

Built under `src/crosswalk/`, smoke-tested only where torch isn't needed
(the sandbox blocked PyPI and download.pytorch.org):

- Shared: `config.py` (strict YAML configs, overrides, hash), `device.py`
  (vendor-agnostic device choice via `torch.cuda`), `hardware.py`,
  `stats.py`, `storage.py` (S3 + local, atomic writes), `synthetic.py`.
- Milestone 1: `models.py` (FCOS-ResNet50-FPN, chosen over Faster R-CNN:
  single-stage, anchor-free, cheaper, fewer exotic ROCm kernels; the relative
  compute claim still needs checking against the torchvision docs),
  `inference/` (sequential/fixed/dynamic batcher, PyAV decode, Prometheus
  metrics, JSON-lines output, status server), `benchmark/harness.py`
  (Experiment A).
- Milestone 2 pieces: BDD100K + COCO parsing, dataset hashing, checkpoints
  (local → S3 → `latest.json` pointer, SHA-256 verified resume), resumable
  sampler, warmup-cosine schedule, MLflow tracking that reattaches on resume,
  pycocotools mAP. Fix: `ruff` flags two intentional open handles in
  `inference/pipeline.py` → add `# noqa: SIM115`.

Next, in order:

1. `training/loop.py` — the custom training loop (seed, FCOS for training,
   optimizer + `LambdaLR(WarmupCosine)`, resume via
   `make_manager(...).find_latest()` + `state.restore`, checkpoint every N
   steps with the MLflow run id, Experiment B recovery metrics, evaluate
   model vs. baseline, register model).
2. `cli.py` (`serve`, `infer`, `benchmark`, `train`, `validate`,
   `make-synthetic`) and `validate.py` (Milestone 0 GPU check).
3. pytest suite incl. a subprocess resume-after-kill test.
4. `requirements/` (CPU and ROCm lists) and `configs/*.yaml`.
5. Containers: `docker/Dockerfile.cpu`, `docker/Dockerfile.rocm`
   (`FROM rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.9.1`),
   `scripts/build-rocm-image.sh`, `scripts/submit-training.py`.
6. Helm chart `chart/crosswalk` + `deploy/argocd-application.yaml`
   (namespace `crosswalk`, `amd.com/gpu: 1`, no LLM label).
7. CI (ruff, pytest on CPU, helm lint, CPU image to GHCR on tags) and docs
   (README, running-on-the-homelab, benchmarks with empty tables, data).

Version pins found (not yet written into files): ROCm 7.2.4 image above
(ships torchvision 0.24.0); CPU `torch==2.9.1`, `torchvision==0.24.1`;
av 19.0.2, prometheus-client 0.26.0, mlflow-skinny 3.17.2, boto3 1.43.118,
moto 5.2.4, pycocotools 2.0.11, pydantic 2.14.1, numpy 2.5.4 (Python ≥ 3.12).

## 4. Homelab Milestone 0: researched, not built

No code was written; branch `ml-platform` exists only locally and equals
`main`. Findings to build from:

- **Object storage: Garage, not MinIO.** MinIO's repo was archived on
  2026-04-25 ("no longer maintained"). Garage `dxflrs/garage:v2.4.1`
  (v2.5.0 came out today), `garage server --single-node`,
  `replication_factor = 1`, `db_engine = "sqlite"`, S3 on 3900 (expose as
  Service `s3` port 9000), admin API on 3903. A small Job (stdlib Python, like
  the demand controller) creates the buckets `datasets`, `checkpoints`,
  `mlflow-artifacts` through admin API v2 (`CreateBucket`, `ImportKey`,
  `AllowBucketKey`). Unconfirmed: the key-ID format and the
  `UpdateClusterLayout` body.
- **MLflow:** `ghcr.io/mlflow/mlflow:v3.17.2-full` (includes boto3; set a
  non-root uid). `--serve-artifacts --artifacts-destination
  s3://mlflow-artifacts`, SQLite on a small PVC, `MLFLOW_S3_ENDPOINT_URL`,
  `MLFLOW_BOTO_CLIENT_ADDRESSING_STYLE=path`, `MLFLOW_DISABLE_TELEMETRY=true`,
  and `--allowed-hosts` must list `mlflow.localhost:8080` and
  `mlflow.mlflow.svc.cluster.local:5000` (Host checks since 3.5.0).
- **Credentials:** bootstrap generates Garage secrets once into
  `~/.config/homelab/ml-storage/` (like the lab CA); `scripts/ml-credentials.py
  <namespace>` creates Secret `s3-credentials` for an app. Add `ml-storage`
  and `mlflow` to `lab-mode.py`'s platform namespaces. Backups already cover
  new local-path PVCs.
- **k3s (separate branch, not for `main` until cutover):** k3s v1.36.5+k3s1.
  AMD device plugin `rocm/k8s-device-plugin:1.31.0.11` mounts `/dev/kfd` and
  the render node, so llama-server can drop privileged mode and request
  `amd.com/gpu`. Exposure: hostPort on `127.0.0.1:8080` and
  `LAN_ADDRESS:443` via an Envoy deployment patch — first confirm Envoy
  Gateway v1.9.1's container ports for listeners 80/443. Kind alternative if
  k3s waits: recreate Kind with `/dev/kfd` mounted.
- **GPU metrics:** AMD's device-metrics-exporter doesn't support the RX 6900
  XT. Use node-exporter's `--collector.drm` (busy %, VRAM) plus hwmon
  (temperature, power) in kube-prometheus-stack, keeping the chart's existing
  extraArgs.

## 5. Decisions for Griffin (defaults chosen; say if you disagree)

1. Chat UI runs only in llm mode.
2. LLM idle grace 5 min; new chat install not stopped for 3 min.
3. llm mode stops every Deployment/StatefulSet in an Argo app's namespace.
4. Garage v2.4.1 instead of MinIO.
5. GPU metrics from node-exporter instead of AMD's exporter.
6. Detector: FCOS. ROCm image: 7.2.4 / PyTorch 2.9.1 (AMD-validated), not the
   newest.
7. Risk to know: the RX 6900 XT (gfx1030) is not on ROCm 7.2.4's official
   GPU list (only the W6800/V620 with the same chip). The Milestone 0
   validation job exists to prove it works; fallback is
   `HSA_OVERRIDE_GFX_VERSION`.
