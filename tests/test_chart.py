"""Render the Helm chart and check what it produces.

Needs the helm binary (CI has it; skipped otherwise). The checks cover the
namespace labels, the route, the GPU request, the security context and the
S3/MLflow/Kafka wiring.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest
import yaml

from tests.conftest import repo_root

CHART = repo_root() / "chart" / "crosswalk"
helm = shutil.which("helm")
pytestmark = pytest.mark.skipif(helm is None, reason="helm not installed")


def render(*args: str) -> list[dict]:
    out = subprocess.run(
        [helm, "template", "crosswalk", str(CHART), "--namespace", "crosswalk", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [d for d in yaml.safe_load_all(out) if d]


def by_kind(docs: list[dict], kind: str) -> list[dict]:
    return [d for d in docs if d["kind"] == kind]


def pod_spec(doc: dict) -> dict:
    return doc["spec"]["template"]["spec"]


def test_lint():
    subprocess.run([helm, "lint", str(CHART)], check=True, capture_output=True)


def test_default_render_follows_the_homelab_contract():
    docs = render()
    kinds = sorted(d["kind"] for d in docs)
    assert kinds == sorted(
        [
            "Namespace",
            "ConfigMap",
            "ConfigMap",
            "Deployment",
            "LocalQueue",
            "PersistentVolumeClaim",
            "PrometheusRule",
            "Service",
            "ServiceMonitor",
            "HTTPRoute",
        ]
    )
    [ns] = by_kind(docs, "Namespace")
    assert ns["metadata"]["name"] == "crosswalk"
    assert ns["metadata"]["labels"]["gateway.homelab/access"] == "public"
    [route] = by_kind(docs, "HTTPRoute")
    assert route["spec"]["hostnames"] == ["crosswalk.localhost"]
    assert route["spec"]["parentRefs"] == [
        {"name": "homelab", "namespace": "gateway-system", "sectionName": "http"}
    ]
    dashboards = [
        d
        for d in by_kind(docs, "ConfigMap")
        if d["metadata"].get("labels", {}).get("grafana_dashboard") == "1"
    ]
    assert len(dashboards) == 1


def test_cpu_fallback_has_no_gpu_request():
    [dep] = by_kind(render(), "Deployment")
    spec = pod_spec(dep)
    container = spec["containers"][0]
    assert container["image"].startswith("ghcr.io/griffinseibold/crosswalk-world-cup:")
    assert "amd.com/gpu" not in container["resources"]["limits"]
    assert "supplementalGroups" not in spec["securityContext"]


def test_gpu_mode_requests_one_amd_gpu_with_minimal_relaxation():
    [dep] = by_kind(
        render("--set", "gpu.enabled=true", "--set", "image.gpu.tag=rocm-abc"), "Deployment"
    )
    assert dep["spec"]["strategy"]["type"] == "Recreate"
    assert dep["spec"]["replicas"] == 1
    spec = pod_spec(dep)
    container = spec["containers"][0]
    assert container["image"] == "crosswalk-world-cup:rocm-abc"
    assert container["imagePullPolicy"] == "IfNotPresent"
    assert container["resources"]["limits"]["amd.com/gpu"] == 1
    sec = spec["securityContext"]
    assert sec["runAsNonRoot"] and sec["supplementalGroups"] == [44, 992]
    assert sec["seccompProfile"]["type"] == "RuntimeDefault"
    csec = container["securityContext"]
    assert csec["allowPrivilegeEscalation"] is False and csec["readOnlyRootFilesystem"] is True
    assert csec["capabilities"]["drop"] == ["ALL"]
    assert "privileged" not in csec
    assert not [v for v in spec["volumes"] if "hostPath" in v]


def test_gpu_mode_without_a_built_image_fails_loudly():
    result = subprocess.run(
        [helm, "template", "crosswalk", str(CHART), "--set", "gpu.enabled=true"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "build-rocm-image.sh" in result.stderr


def test_s3_and_mlflow_wiring():
    [dep] = by_kind(render(), "Deployment")
    env = {e["name"]: e for e in pod_spec(dep)["containers"][0]["env"]}
    assert env["AWS_ENDPOINT_URL_S3"]["value"] == "http://s3.ml-storage.svc.cluster.local:9000"
    assert env["MLFLOW_TRACKING_URI"]["value"] == "http://mlflow.mlflow.svc.cluster.local:5000"
    for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        ref = env[key]["valueFrom"]["secretKeyRef"]
        assert ref == {"name": "s3-credentials", "key": key}


def test_inference_config_map_is_valid_config():
    from crosswalk.config import InferenceConfig

    [cm] = [
        d
        for d in by_kind(render(), "ConfigMap")
        if d["metadata"]["name"] == "crosswalk-inference-config"
    ]
    InferenceConfig.model_validate(yaml.safe_load(cm["data"]["inference.yaml"]))


def test_training_job_render(tmp_path):
    config = repo_root() / "configs" / "train-bdd100k.yaml"
    docs = render(
        "--show-only", "templates/job.yaml", "--show-only", "templates/job-config.yaml",
        "--set", "job.enabled=true", "--set-string", "job.kind=train", "--set-string", "job.name=crosswalk-train-x",
        "--set", "gpu.enabled=true", "--set-string", "image.gpu.tag=rocm-abc",
        "--set-file", f"job.config={config}",
        "--set-string", "job.sync[0]=s3://datasets/bdd100k=/scratch/data/bdd100k",
        "--set-string", "job.args[0]=--set", "--set-string", "job.args[1]=epochs=1",
    )  # fmt: skip
    [job] = by_kind(docs, "Job")
    [cm] = by_kind(docs, "ConfigMap")
    assert yaml.safe_load(cm["data"]["config.yaml"]) == yaml.safe_load(config.read_text())
    spec = pod_spec(job)
    assert spec["restartPolicy"] == "Never" and job["spec"]["backoffLimit"] == 4
    init = spec["initContainers"][0]
    assert init["args"] == ["sync", "s3://datasets/bdd100k=/scratch/data/bdd100k"]
    main = spec["containers"][0]
    assert main["args"] == ["train", "--config", "/config/config.yaml", "--set", "epochs=1"]
    assert main["resources"]["limits"]["amd.com/gpu"] == 1
    assert "amd.com/gpu" not in init["resources"].get("limits", {})
    mounts = {m["mountPath"] for m in main["volumeMounts"]}
    assert {"/config", "/scratch", "/cache", "/tmp", "/dev/shm"} <= mounts


def test_validation_job_on_cpu():
    docs = render(
        "--show-only", "templates/job.yaml", "--set", "job.enabled=true",
        "--set-string", "job.name=crosswalk-validate-1",
    )  # fmt: skip
    [job] = docs
    args = pod_spec(job)["containers"][0]["args"]
    assert args[:3] == ["validate", "--device", "cpu"]
    assert "initContainers" not in pod_spec(job)


def test_job_requires_config_for_training():
    result = subprocess.run(
        [helm, "template", "crosswalk", str(CHART), "--show-only", "templates/job.yaml",
         "--set", "job.enabled=true", "--set-string", "job.kind=train", "--set-string", "job.name=x"],
        capture_output=True,
        text=True,
    )  # fmt: skip
    assert result.returncode != 0 and "job.config" in result.stderr


def test_argocd_application_points_at_this_chart():
    app = yaml.safe_load((repo_root() / "deploy" / "argocd-application.yaml").read_text())
    assert app["spec"]["source"]["path"] == "chart/crosswalk"
    assert app["spec"]["destination"]["namespace"] == "crosswalk"
    assert app["spec"]["source"]["repoURL"].endswith("griffinseibold/crosswalk-world-cup")


# -- The streaming pipeline -------------------------------------------


def pipeline_docs(*args: str) -> list[dict]:
    return render("--set", "pipeline.enabled=true", *args)


def test_pipeline_is_off_by_default_and_inference_stays_in_file_mode():
    docs = render()
    assert not [d for d in docs if d["metadata"]["name"].startswith("crosswalk-ingest")]
    [cm] = [
        d
        for d in by_kind(docs, "ConfigMap")
        if d["metadata"]["name"] == "crosswalk-inference-config"
    ]
    assert yaml.safe_load(cm["data"]["inference.yaml"])["mode"] == "file"


def test_pipeline_renders_every_stage_with_kafka_and_s3_wiring():
    docs = pipeline_docs()
    deployments = {d["metadata"]["name"]: d for d in by_kind(docs, "Deployment")}
    assert set(deployments) == {
        "crosswalk-inference", "crosswalk-ingest", "crosswalk-tracking", "crosswalk-indicators", "crosswalk-ranking",
    }  # fmt: skip
    for name, dep in deployments.items():
        spec = pod_spec(dep)
        container = spec["containers"][0]
        env = {e["name"]: e for e in container["env"]}
        assert env["KAFKA_BOOTSTRAP_SERVERS"]["value"] == "kafka.kafka.svc.cluster.local:9092", name
        assert env["AWS_ENDPOINT_URL_S3"]["value"] == "http://s3.ml-storage.svc.cluster.local:9000"
        assert spec["securityContext"]["runAsNonRoot"] is True
        assert container["securityContext"]["readOnlyRootFilesystem"] is True
        if name != "crosswalk-inference":
            assert "amd.com/gpu" not in container.get("resources", {}).get("limits", {}), name
    ingest = pod_spec(deployments["crosswalk-ingest"])
    assert ingest["containers"][0]["image"].startswith("ghcr.io/griffinseibold/crosswalk-ingest:")
    assert ingest["initContainers"][0]["args"] == ["sync", "s3://datasets/footage/=/footage"]
    ranking = deployments["crosswalk-ranking"]
    assert ranking["spec"]["replicas"] == 1 and ranking["spec"]["strategy"]["type"] == "Recreate"
    claims = [
        v["persistentVolumeClaim"]["claimName"]
        for v in pod_spec(ranking)["volumes"]
        if "persistentVolumeClaim" in v
    ]
    assert claims == ["crosswalk-ranking"]
    [cm] = [
        d
        for d in by_kind(docs, "ConfigMap")
        if d["metadata"]["name"] == "crosswalk-inference-config"
    ]
    assert yaml.safe_load(cm["data"]["inference.yaml"])["mode"] == "kafka"


def test_pipeline_stage_config_is_valid_and_points_at_the_mounts():
    from crosswalk.stages import StagesConfig

    docs = pipeline_docs()
    [cm] = [d for d in by_kind(docs, "ConfigMap") if d["metadata"]["name"] == "crosswalk-stages"]
    config = StagesConfig.model_validate(yaml.safe_load(cm["data"]["stages.yaml"]))
    assert config.indicators.cameras_dir == "/config/cameras"
    assert config.ranking.database == "/data/ranking.sqlite3"
    [cams] = [d for d in by_kind(docs, "ConfigMap") if d["metadata"]["name"] == "crosswalk-cameras"]
    from crosswalk.indicators.camera import CameraConfig

    for text in cams["data"].values():
        CameraConfig.model_validate(yaml.safe_load(text))


def test_leaderboard_route_and_opt_in_lan_access():
    docs = pipeline_docs()
    [route] = by_kind(docs, "HTTPRoute")
    assert route["spec"]["hostnames"] == ["crosswalk.localhost"]
    backends = [r["backendRefs"][0]["name"] for r in route["spec"]["rules"]]
    assert backends == ["crosswalk", "crosswalk-ranking"]  # /status, then everything else
    [ns] = by_kind(docs, "Namespace")
    assert "gateway.homelab/lan" not in ns["metadata"]["labels"]

    docs = pipeline_docs("--set", "httpRoute.lan.enabled=true")
    lan = [r for r in by_kind(docs, "HTTPRoute") if r["metadata"]["name"] == "crosswalk-lan"]
    assert lan and lan[0]["spec"]["hostnames"] == ["crosswalk.lab.internal"]
    assert lan[0]["spec"]["parentRefs"][0]["sectionName"] == "lan"
    [ns] = by_kind(docs, "Namespace")
    assert ns["metadata"]["labels"]["gateway.homelab/lan"] == "true"


def test_pipeline_metrics_are_scraped():
    docs = pipeline_docs()
    monitors = {d["metadata"]["name"]: d for d in by_kind(docs, "ServiceMonitor")}
    selector = monitors["crosswalk-pipeline"]["spec"]["selector"]["matchLabels"]
    services = [
        s
        for s in by_kind(docs, "Service")
        if all(s["metadata"]["labels"].get(k) == v for k, v in selector.items())
    ]
    assert sorted(s["metadata"]["name"] for s in services) == [
        "crosswalk-indicators", "crosswalk-ingest", "crosswalk-ranking", "crosswalk-tracking",
    ]  # fmt: skip


def test_inference_can_serve_the_promoted_champion():
    docs = render("--set", "inference.servePromoted=true")
    [cm] = [
        d
        for d in by_kind(docs, "ConfigMap")
        if d["metadata"]["name"] == "crosswalk-inference-config"
    ]
    config = yaml.safe_load(cm["data"]["inference.yaml"])
    assert config["model"]["checkpoint"] == "models:/crosswalk-detector@champion"


def test_promote_job_never_retries_a_rejection():
    config = repo_root() / "configs" / "train-bdd100k.yaml"
    docs = render(
        "--show-only", "templates/job.yaml", "--set", "job.enabled=true", "--set-string", "job.kind=promote",
        "--set-string", "job.name=crosswalk-promote-1", "--set-file", f"job.config={config}",
    )  # fmt: skip
    [job] = docs
    assert job["spec"]["backoffLimit"] == 0
    assert pod_spec(job)["containers"][0]["args"][:3] == [
        "promote",
        "--config",
        "/config/config.yaml",
    ]


def test_kueue_queue_and_jobs_by_default():
    docs = render()
    [queue] = by_kind(docs, "LocalQueue")
    assert queue["apiVersion"] == "kueue.x-k8s.io/v1beta2"
    assert queue["metadata"]["name"] == "crosswalk" and queue["spec"]["clusterQueue"] == "gpu"
    [ns] = by_kind(docs, "Namespace")
    assert ns["metadata"]["labels"]["kueue.homelab/queues"] == "true"
    [job] = render(
        "--show-only", "templates/job.yaml", "--set", "job.enabled=true", "--set-string", "job.name=crosswalk-validate-1",
    )  # fmt: skip
    assert job["metadata"]["labels"]["kueue.x-k8s.io/queue-name"] == "crosswalk"
    assert job["spec"]["suspend"] is True


def test_kueue_can_be_turned_off_for_plain_clusters():
    docs = render("--set", "kueue.enabled=false")
    assert not by_kind(docs, "LocalQueue")
    [ns] = by_kind(docs, "Namespace")
    assert "kueue.homelab/queues" not in ns["metadata"]["labels"]
    [job] = render(
        "--show-only", "templates/job.yaml", "--set", "job.enabled=true", "--set-string", "job.name=x",
        "--set", "kueue.enabled=false",
    )  # fmt: skip
    assert "kueue.x-k8s.io/queue-name" not in job["metadata"]["labels"]
    assert "suspend" not in job["spec"]


def test_alert_rules_cover_the_pipeline_and_query_real_metrics():
    from tests.test_metrics_dashboard import exported_metric_names

    [rule] = by_kind(render("--set", "pipeline.enabled=true"), "PrometheusRule")
    assert rule["metadata"]["labels"]["release"] == "kube-prometheus-stack"
    alerts = {r["alert"]: r for g in rule["spec"]["groups"] for r in g["rules"]}
    assert set(alerts) == {
        "CrosswalkInferenceLatencyHigh", "CrosswalkStageDown", "CrosswalkStageNotConsuming",
        "CrosswalkConsumerLagGrowing", "CrosswalkDeadLetters", "CrosswalkRankingsStale",
    }  # fmt: skip
    import re

    used = {n for r in alerts.values() for n in re.findall(r"\bcrosswalk_[a-z_]+", r["expr"])}
    assert used <= exported_metric_names(), used - exported_metric_names()
    # Without the pipeline only the inference alert remains.
    [rule] = by_kind(render(), "PrometheusRule")
    assert [r["alert"] for g in rule["spec"]["groups"] for r in g["rules"]] == [
        "CrosswalkInferenceLatencyHigh"
    ]


def test_nvidia_values_switch_the_gpu_resource_and_drop_amd_settings():
    values = str(CHART / "values-nvidia.yaml")
    [dep] = [
        d
        for d in render("--values", values, "--set", "image.gpu.tag=cuda-abc")
        if d["kind"] == "Deployment"
    ]
    spec = pod_spec(dep)
    limits = spec["containers"][0]["resources"]["limits"]
    assert limits["nvidia.com/gpu"] == 1 and "amd.com/gpu" not in limits
    assert "supplementalGroups" not in spec["securityContext"]
    assert spec["containers"][0]["image"] == "crosswalk-world-cup:cuda-abc"
    env = {e["name"] for e in spec["containers"][0]["env"]}
    assert "HSA_OVERRIDE_GFX_VERSION" not in env
    result = subprocess.run(
        [helm, "template", "crosswalk", str(CHART), "--values", values],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0  # no image tag set yet


def test_multi_gpu_train_job_runs_torchrun_and_requests_every_gpu(tmp_path):
    config = repo_root() / "configs" / "train-bdd100k.yaml"
    common = (
        "--show-only", "templates/job.yaml", "--set", "job.enabled=true",
        "--set-string", "job.name=crosswalk-ddp", "--set", "gpu.enabled=true",
        "--set-string", "image.gpu.tag=cuda-abc", "--set-file", f"job.config={config}",
    )  # fmt: skip
    [job] = render(*common, "--set-string", "job.kind=train", "--set", "job.gpus=4",
                   "--values", str(CHART / "values-nvidia.yaml"))  # fmt: skip
    main = pod_spec(job)["containers"][0]
    assert main["command"] == ["torchrun"]
    assert main["args"][:4] == ["--standalone", "--nproc-per-node=4", "-m", "crosswalk"]
    assert main["args"][4:] == ["train", "--config", "/config/config.yaml"]
    assert main["resources"]["limits"]["nvidia.com/gpu"] == 4
    # One GPU (the default) is the unchanged single-process path.
    [single] = render(*common, "--set-string", "job.kind=train")
    assert "command" not in pod_spec(single)["containers"][0]
    assert pod_spec(single)["containers"][0]["resources"]["limits"]["amd.com/gpu"] == 1
    # DDP is for training only.
    bad = subprocess.run(
        [helm, "template", "crosswalk", str(CHART), *common, "--set-string", "job.kind=benchmark",
         "--set", "job.gpus=2"],
        capture_output=True, text=True,
    )  # fmt: skip
    assert bad.returncode != 0 and "job.gpus" in bad.stderr
