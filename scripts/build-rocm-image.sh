#!/usr/bin/env bash
# Build the ROCm image on the homelab host and load it into the cluster.
#
# The image is too large for GitHub-hosted runners, so it is built where it
# runs. The cluster then uses it without a registry (imagePullPolicy:
# IfNotPresent and a tag that never moves).
#
#   scripts/build-rocm-image.sh                 # build, load into Kind (today's homelab)
#   RUNTIME=k3s scripts/build-rocm-image.sh     # build, import into k3s containerd
#   RUNTIME=none scripts/build-rocm-image.sh    # build only
#
# Environment:
#   RUNTIME       kind (default) | k3s | none
#   KIND_CLUSTER  Kind cluster name (default homelab-dev)
#   TAG           image tag (default: rocm-<short git commit>[-dirty])
#   BASE          override the rocm/pytorch base image
#
# Disk: expect the base image to need roughly 30-60 GB under
# /var/lib/docker, plus a second copy inside each Kind node it is loaded into
# (both workers on the homelab), or inside k3s's containerd store.
set -euo pipefail

cd "$(dirname "$0")/.."

RUNTIME="${RUNTIME:-kind}"
KIND_CLUSTER="${KIND_CLUSTER:-homelab-dev}"
REPOSITORY="crosswalk-world-cup"
commit="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    commit="${commit}-dirty"
fi
TAG="${TAG:-rocm-${commit}}"
IMAGE="${REPOSITORY}:${TAG}"

build_args=(--build-arg "GIT_COMMIT=${commit}" --build-arg "IMAGE=${IMAGE}")
if [ -n "${BASE:-}" ]; then
    build_args+=(--build-arg "BASE=${BASE}")
fi

echo "==> building ${IMAGE}"
docker build --file docker/Dockerfile.rocm --tag "${IMAGE}" "${build_args[@]}" .

case "${RUNTIME}" in
    kind)
        # Copies the image into every node of the cluster. Both homelab
        # workers can run GPU pods (they share the one physical GPU), so both
        # need it.
        echo "==> loading into Kind cluster ${KIND_CLUSTER}"
        kind load docker-image "${IMAGE}" --name "${KIND_CLUSTER}"
        ;;
    k3s)
        # k3s runs its own containerd, separate from Docker's image store.
        # Streams the image without a temporary tarball; needs sudo because
        # the containerd socket is root-owned.
        echo "==> importing into k3s containerd"
        docker save "${IMAGE}" | sudo k3s ctr images import -
        ;;
    none)
        ;;
    *)
        echo "unknown RUNTIME=${RUNTIME} (expected kind, k3s or none)" >&2
        exit 2
        ;;
esac

cat <<EOF

Built ${IMAGE}. Point the chart at it, for example as Argo CD parameters:
  image.repository=${REPOSITORY}
  image.tag=${TAG}
  image.pullPolicy=IfNotPresent
or one-off:
  helm template crosswalk chart/crosswalk --namespace crosswalk \\
    --set image.repository=${REPOSITORY} --set image.tag=${TAG} ...
EOF
