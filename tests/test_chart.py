"""Render the Helm chart and check the platform contract.

Needs the helm binary (CI has it; skipped otherwise). The structural checks
below are the contract with the homelab: namespace label, route parent, GPU
request, security context, S3/MLflow wiring and the absence of the LLM label.
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
            "PersistentVolumeClaim",
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


def test_nothing_requests_the_homelab_llm():
    for args in ([], ["--set", "gpu.enabled=true", "--set", "image.gpu.tag=rocm-x"]):
        for doc in render(*args):
            labels = doc["metadata"].get("labels", {})
            assert "homelab.local/llm" not in labels, doc["kind"]


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
