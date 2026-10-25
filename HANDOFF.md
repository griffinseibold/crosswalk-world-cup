# Handoff — updated 2026-10-25 (midday)

Where Crosswalk World Cup and its homelab platform stand, so the next session
can resume without the conversation. Delete this file once its open steps are
done.

## 1. On GitHub now

| Repo | Branch | State |
| --- | --- | --- |
| crosswalk-world-cup | `main` | Milestones 0–4 complete, Milestone 5 partly done (`bbbc6d7`, marked WIP). See section 3. |
| Homelab | `main` (a93b0aa) | On-demand LLM + `lab-mode.py`, Garage S3, MLflow, Kafka (`apache/kafka:4.3.1`), Kueue 0.20.3. Live via Flux, **not yet checked on the real cluster.** |
| Homelab | `k3s` (9c793b4) | Kind → k3s migration with AMD device plugin, plus the same Kafka/Kueue. **Not for `main` until the cutover** in `docs/migrate-to-k3s.md`. |
| Flip-Finder | `master` (9727a30) | **0.4.0 released and deployed**: chat requests the on-demand LLM and waits while it starts. |
| hello-crud | `main` | Optional LLM example (off by default). Griffin chose not to release it; the cluster keeps 0.3.0, which lacks the new endpoints. |

## 2. CI status: one failure to fix first

The last pushed CI run (`6448675`, run 40207870593) passed tests, lint and
the Rust `ingest` job but **failed `lint-chart` at `promtool test rules
tests/promtool/alerts-test.yaml`** — an alert-rule unit test assertion. The
job log could not be read from this session (GitHub log storage blocked), so
the failing alert is unknown. Start by running `promtool test rules` locally
or reading the job log on GitHub.

The WIP Milestone 5 commit has never run in CI: it adds the gloo DDP test,
multi-GPU chart test and an `ansible` CI job.

## 3. Crosswalk build: done and remaining

Done (commit):

1. Rust frame sampler `ingest/` (ae264db): rskafka (pure Rust), hand-written
   SigV4 S3 client, FFmpeg as a subprocess, backpressure, metrics.
2. Kafka mode in the inference service (f5f71c2): at-least-once with dedupe,
   retries, dead-letter topic.
3. Tracking (9d459f9). 4. Conflict indicators with optional homography
   (5a12be7). 5. Ranking service, SQLite WAL, and leaderboard page (10715ed).
6. `crosswalk demo` end to end, chart for every stage, dashboard (010fdfe).
7. `docs/data-sources.md` and camera configs (8412a99) — NGSIM and Xiph
   licences **not verified**; check before using.
8. `crosswalk promote` / `rollback` on MLflow aliases (758e1ec).
9. Kueue integration and Experiment C harness (60a1bd6).
10. Alert rules and replay load tester (6448675).
11. WIP (bbbc6d7): CUDA image, `values-nvidia.yaml`, `deploy/nvidia/`
    Ansible (provider-agnostic, never calls a cloud API), DDP via torchrun
    (`job.gpus`, `submit-training.py --gpus`).

Remaining, in order:

1. Fix the promtool failure (section 2); confirm the WIP commit passes CI.
2. Triton comparison harness (`src/crosswalk/benchmark/triton.py`): FCOS's
   dynamic postprocessing likely blocks a clean export, so export and
   benchmark backbone + FPN + heads and document why.
3. `docs/benchmarks.md`: Experiments A–D plus AMD vs NVIDIA, time-slicing p99
   and DDP scaling, all with **empty** results tables, and the
   `#nvidia-experiments` anchor the NVIDIA README already links to.
4. README (roadmap, architecture, pipeline, demo) and
   `docs/running-on-the-homelab.md` (pipeline.enabled, Kafka, Kueue,
   promote/rollback, LAN route, `--gpus`, context `homelab-dev` after k3s —
   scripts still default to `kind-homelab-dev`).
5. Optional: motion-based adaptive sampling in the Rust sampler.
6. Website write-up: **Griffin reviews everything first.**

## 4. Needs Griffin

1. **Link the homelab computer** to this chat (Claude desktop app on that
   machine), or run these yourself: `git pull` in Homelab, then
   `./scripts/bootstrap-dev.sh` to bring up Garage, MLflow, Kafka and Kueue;
   `./scripts/lab-mode.py status` to check Flip Finder requests the LLM.
2. **BDD100K** "100K Images" and detection labels are in Griffin's Downloads
   folder. Next: check which label format it is, upload a training subset and
   labels to Garage's `datasets` bucket, run a CPU smoke training on it.
3. **Release for crosswalk-world-cup** once sections 2–3 land and CI is
   green: one release covering the CPU image and the ingest image (the chart
   currently expects CPU image `0.1.0` and ingest `0.2.0`; align them to one
   version before tagging). This session cannot push tags.
4. **Label ~200 fixed-camera frames** (COCO, e.g. with CVAT) for the
   promotion gate.
5. **k3s cutover** (planned for winter break), then
   `scripts/build-rocm-image.sh`, `submit-training.py validate`,
   `gpu.enabled=true`, and the benchmarks — real numbers only.
6. NVIDIA milestone is optional and costs money (rented GPU); the code is
   ready, nothing has been rented.
7. Optional cleanup: branches `llm-on-demand` (Flip-Finder, hello-crud),
   `release-0.4.0` (both).

## 5. Defaults chosen (say if you disagree)

- Homelab: chat UI only in llm mode; LLM idle grace 5 min; llm mode stops
  every workload in an Argo app's namespace; Garage over MinIO (archived);
  node-exporter drm/hwmon for GPU metrics; per-namespace S3 keys; Kafka 4.3.1
  over 4.4.0; Kueue quota 16 CPU / 32 Gi.
- Crosswalk: FCOS detector; ROCm 7.2.4 / PyTorch 2.9.1; rskafka and a
  hand-written SigV4 client in Rust; models registered as state dict +
  `model.json` (not pickled); SQLite for rankings; `pipeline.enabled` off by
  default; demo uses a simple colour detector, not the trained model; DDP
  keeps per-process batch size and does not rescale the learning rate.
- Risk: the RX 6900 XT isn't on ROCm 7.2.4's official list (same chip as
  the W6800/V620); `crosswalk validate` proves it, with
  `HSA_OVERRIDE_GFX_VERSION` as fallback. Check `getent group video render`
  and set the chart's `gpu.renderGid`.
