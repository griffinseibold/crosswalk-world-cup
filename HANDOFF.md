# Handoff — updated 2026-10-25

Where the Crosswalk World Cup project and its homelab platform stand, so the
next session can pick up without rereading the conversation. Delete this file
once its open steps are done.

## 1. Live on GitHub

| Repo | Branch | What |
| --- | --- | --- |
| Homelab | `main` (58d7095) | LLM on demand (controller + `scripts/lab-mode.py`) and ML platform: Garage S3 (`dxflrs/garage:v2.4.1`, Service `s3:9000`, buckets `datasets`, `checkpoints`, `mlflow-artifacts`), MLflow `v3.17.2-full` at `mlflow.localhost:8080`, `scripts/ml-credentials.py <namespace>`. 116 tests pass. **Not yet run on the real cluster.** |
| Homelab | `k3s` (cb83d2d) | Kind → single-node k3s v1.36.5+k3s1, AMD device plugin `rocm/k8s-device-plugin:1.31.0.11`, llama-server via `amd.com/gpu` (no privileged), node-exporter drm/hwmon GPU dashboard, hostPort exposure on 127.0.0.1:8080 and LAN_ADDRESS:443, nftables guard, rewritten bootstrap and backups. 145 tests pass. **Not for `main` until the cutover in `docs/migrate-to-k3s.md`.** |
| Flip-Finder | `master` | Requests the LLM; waits for a starting model. CI green. |
| hello-crud | `main` | Optional LLM example, off by default. CI green. |
| crosswalk-world-cup | `main` | Milestones 1–2 code: FCOS inference service with dynamic batching, Experiment A harness, custom training loop with checkpoint/resume and Experiment B fault injection, MLflow, `crosswalk validate` (Milestone 0 GPU check), Helm chart, Argo app, CPU/ROCm Dockerfiles, docs. CI green, CPU image published as `:main`. |

## 2. Waiting on Griffin

1. **Publish releases** (this session can't push tags): GitHub → Releases →
   Draft a new release.
   - Flip-Finder: tag `v0.4.0`, target branch `release-0.4.0`.
   - hello-crud: tag `v0.4.0`, target branch `release-0.4.0`.
   - crosswalk-world-cup: tag `v0.1.0`, target `main` (the chart's CPU image
     tag is `0.1.0`).
   After the Flip-Finder and hello-crud images build, push each
   `release-0.4.0` commit to its default branch (Flip-Finder 9727a30 →
   `master`, hello-crud f73c0b4 → `main`), never before.
2. **On the homelab machine:** `git pull`, then rerun
   `./scripts/bootstrap-dev.sh`. It generates the Garage secrets (kept in
   `~/.config/homelab/ml-storage/`), gives MLflow its S3 key and waits for
   both. Until then the `ml-storage` and `mlflow` Kustomizations stay not
   ready; nothing else depends on them.
3. Check: `./scripts/lab-mode.py status` (Flip Finder should request the
   LLM), `http://mlflow.localhost:8080`.
4. Crosswalk on Kind (CPU only, before any GPU work):
   `./scripts/ml-credentials.py crosswalk`, apply
   `deploy/argocd-application.yaml`, run the synthetic smoke training Job
   with `scripts/submit-training.py ... --cpu`.
5. **k3s cutover (planned for winter break):** follow
   `docs/migrate-to-k3s.md` on the `k3s` branch. Then
   `scripts/build-rocm-image.sh RUNTIME=k3s`, `submit-training.py validate`,
   and turn on `gpu.enabled`.
6. Optional cleanup (this session can't delete branches): `llm-on-demand` in
   Flip-Finder and hello-crud; `release-0.4.0` in both after step 1.

## 3. Only verifiable on the real machine

- Garage admin API responses (scripts stop with the response text if they
  differ) and that `--single-node` sets up the layout itself; MLflow's first
  start and artifact proxying.
- k3s: hostPorts binding only to 127.0.0.1 and LAN_ADDRESS, the guard blocking
  6443/10250 from another device, Envoy listener ports 10080/10443 (inferred
  from `wellKnownPortShift = 10000`), cgroup freezer paths for backups, the
  claim names during restore.
- GPU: the device plugin advertising the RX 6900 XT, Vulkan llama-server as
  non-root through the plugin, drm/hwmon metric names, and ROCm 7.2.4 on
  gfx1030 (the RX 6900 XT isn't on AMD's official list; fallback
  `HSA_OVERRIDE_GFX_VERSION`). Crosswalk chart assumes video gid 44 and
  render gid 992 — check `getent group video render` and set `gpu.renderGid`.

## 4. Defaults chosen (say if you disagree)

1. Chat UI runs only in llm mode; LLM idle grace 5 min.
2. llm mode stops every workload in an Argo app's namespace.
3. Garage instead of MinIO (MinIO archived April 2026); node-exporter
   drm/hwmon instead of AMD's exporter (doesn't support RDNA2 consumer cards).
4. S3 keys per namespace (`k8s-<ns>`), not one shared key.
5. On k3s, llama-server runs as the model directory's owner, so a root-owned
   model directory is refused; Docker stays installed after migration.
6. Detector FCOS; ROCm image 7.2.4 / PyTorch 2.9.1 (AMD-validated, not the
   newest); Crosswalk service cache on a persistent 20 Gi claim.

## 5. Next build work (Milestone 3, after the above)

Rust frame sampler (FFmpeg decodes), Kafka between stages, tracking and
rule-based conflict indicators, two recorded fixed-camera videos, ranking API,
static leaderboard page. Hand-label ~200 fixed-camera frames (COCO) for the
promotion gate.
