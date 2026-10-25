# Crosswalk World Cup on a rented NVIDIA GPU

Run the same benchmarks on an NVIDIA card for the AMD vs NVIDIA comparison
([benchmarks](../../docs/benchmarks.md#nvidia-experiments)). An Ansible
playbook turns a fresh Ubuntu 24.04 VM with one NVIDIA GPU into a
single-node k3s that runs the project's CUDA image.

**Nothing here talks to a cloud provider.** You create the VM by hand in your
provider's console, give the playbook its IP address and SSH key, and delete
the VM by hand when you are done. That keeps the repository provider-agnostic,
and it means no script can leave a GPU running and billing because it failed
halfway through.

## Billing safety: read before renting

A GPU VM bills by the hour (sometimes by the second) **from the moment it is
created until it is deleted**. Stopping it often still bills for the disk and
any reserved IP address. Before creating anything:

1. **Set a budget alert or spending cap** in the provider's billing settings,
   at a number you would not mind losing. Many providers support a hard cap on
   prepaid credit. Prefer those.
2. **Pick the smallest GPU that answers the question.** The comparison needs
   one GPU. A spot or interruptible instance is fine: every Job here resumes
   from its checkpoint (Experiment B is exactly that).
3. **Write down the delete time before you start**, and set a phone alarm.
   Plan a single session: provision, build, run the experiments, copy the
   results, tear down.
4. **Never open the Kubernetes API (6443), the kubelet (10250) or any
   NodePort to the internet.** The playbook binds nothing publicly. Use the
   SSH tunnel below. Allow only SSH (22) from your own IP in the provider's
   firewall.
5. **Do not put cloud credentials on the VM.** It needs none. Results come
   back over SSH (`scp`).

## Provision

On your machine, with Ansible 2.21 or newer:

```bash
cd deploy/nvidia
cp inventory.example.ini inventory.ini   # fill in the VM's IP, user and SSH key
ansible-playbook -i inventory.ini site.yml
```

The playbook, in order (each step is idempotent, so re-running is safe):

| Step | Version | Source |
| --- | --- | --- |
| NVIDIA driver from Ubuntu (`nvidia-driver-580-server`), reboot if new | branch 580 | Ubuntu archive |
| NVIDIA Container Toolkit, apt repository from NVIDIA's install guide | 1.20.0 | [install guide](https://github.com/NVIDIA/cloud-native-docs/blob/main/container-toolkit/install-guide.md), [releases](https://github.com/NVIDIA/nvidia-container-toolkit/releases) |
| k3s with `default-runtime: nvidia` (k3s detects the NVIDIA runtime at start) | v1.36.5+k3s1, as on the homelab | [k3s: NVIDIA Container Runtime Support](https://docs.k3s.io/advanced) |
| NVIDIA device plugin with **time-slicing** (`gpu_time_slicing_replicas`, default 4) | chart 0.20.0 | [k8s-device-plugin](https://github.com/NVIDIA/k8s-device-plugin) |
| DCGM exporter (GPU utilisation, memory, power, temperature on :9400) | chart 4.8.4 | [dcgm-exporter](https://github.com/NVIDIA/dcgm-exporter) |
| A smoke pod: PyTorch in the CUDA base image sees the GPU through Kubernetes | `pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime` | |

It finishes by copying the kubeconfig to `deploy/nvidia/kubeconfig-<host>.yaml`
(git-ignored). Reach the API through SSH:

```bash
ssh -N -L 6443:127.0.0.1:6443 ubuntu@<vm-ip> &
export KUBECONFIG=$PWD/kubeconfig-gpu-vm.yaml
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu
```

Time-slicing makes one physical GPU appear as N `nvidia.com/gpu`. Pods share
it in turns, with **no memory isolation**: two pods can still run each other
out of GPU memory. Set `gpu_time_slicing_replicas: 1` for exclusive use, which
the AMD vs NVIDIA comparison needs, and re-run the playbook.

## Build the image and run the experiments

Build on the VM itself, where the CUDA base layers are already cached by the
smoke pod. Import the image into k3s's containerd:

```bash
ssh ubuntu@<vm-ip>
git clone https://github.com/griffinseibold/crosswalk-world-cup && cd crosswalk-world-cup
sudo apt-get install -y docker.io   # build tool only; k3s does not use Docker
sudo docker build -f docker/Dockerfile.cuda -t crosswalk-world-cup:cuda-$(git rev-parse --short HEAD) \
  --build-arg GIT_COMMIT=$(git rev-parse --short HEAD) .
sudo docker save crosswalk-world-cup:cuda-$(git rev-parse --short HEAD) | sudo k3s ctr images import -
```

Then, from your machine through the tunnel, run Jobs with the NVIDIA values
(GPU resource `nvidia.com/gpu`, no AMD groups, no Kueue, no Gateway):

```bash
scripts/submit-training.py validate --values chart/crosswalk/values-nvidia.yaml \
  --image-tag cuda-abc1234 --context '' --no-kueue
scripts/submit-training.py benchmark --config configs/benchmark-a.yaml \
  --values chart/crosswalk/values-nvidia.yaml --image-tag cuda-abc1234 --context '' --no-kueue \
  --sync s3://datasets/footage=/scratch/data/footage
```

The benchmark Job needs footage and S3. Either run a temporary S3 on the VM,
or copy the clip in and run `crosswalk benchmark` directly in a pod. The
simplest path is `kubectl cp` and `kubectl exec` into a long-running pod made
from the same image. Copy every result file back with `scp` **before** tearing
down.

## Tear down: one command, then the checklist

```bash
ansible-playbook -i inventory.ini teardown.yml   # removes k3s and everything it ran
```

That cleans the VM but **does not stop billing**. Then, in the provider's
console:

- [ ] Results copied off the VM (`scp ubuntu@<vm-ip>:'~/crosswalk-world-cup/results/*' results/`)
- [ ] **VM deleted** (terminated or destroyed, not just stopped)
- [ ] Its **boot and data disks deleted** (some providers keep them by default)
- [ ] **Snapshots and images** made during the session deleted
- [ ] **Reserved or static IP** released
- [ ] Billing page shows no running resources, and the budget alert can stay
- [ ] Local `kubeconfig-*.yaml` and `inventory.ini` deleted

## What was and was not tested

The playbook and these instructions were written without access to an
NVIDIA machine. CI checks them with `ansible-lint` and `ansible-playbook
--syntax-check`. Versions and repository URLs were taken from each project's
own releases and documentation. The first real run is the real test:
expect to adjust the driver branch for very new GPUs (`ubuntu-drivers list
--gpgpu` on the VM shows what Ubuntu recommends).
