#!/usr/bin/env bash
set -euo pipefail

# Build the Apptainer/Singularity images used by the SLURM deployment.
#
# vLLM is pulled straight from Docker Hub (no Docker daemon needed).
# The gateway image is built from a Docker archive (llm-gateway.tar) so no
# root/buildah is required on the cluster. If Docker is available here, the
# archive is built locally; otherwise build it on a machine with Docker:
#
#   docker build -t llm-gateway:latest -f gateway/Dockerfile .
#   docker save llm-gateway:latest -o images/llm-gateway.tar
#   scp images/llm-gateway.tar datalab:~/thesis-llm-serving/images/
#
# Run once, on a node with apptainer installed.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF_DIR="${SIF_DIR:-$PROJECT_ROOT/images}"

VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"   # CUDA image for GPU nodes
VLLM_SIF="$SIF_DIR/vllm-openai.sif"
GATEWAY_SIF="$SIF_DIR/llm-gateway.sif"
GATEWAY_TAR="$SIF_DIR/llm-gateway.tar"

mkdir -p "$SIF_DIR"

if [ -f "$VLLM_SIF" ]; then
  echo "==> vLLM image already present: $VLLM_SIF"
else
  echo "==> Pulling vLLM image: $VLLM_IMAGE"
  apptainer pull "$VLLM_SIF" "docker://$VLLM_IMAGE"
fi

echo "==> Building gateway image"
if [ -f "$GATEWAY_TAR" ]; then
  echo "    (using existing docker archive: $GATEWAY_TAR)"
  apptainer build --force "$GATEWAY_SIF" docker-archive://"$GATEWAY_TAR"
elif command -v docker >/dev/null 2>&1; then
  echo "    (building docker archive with Docker)"
  docker build -t llm-gateway:latest -f "$PROJECT_ROOT/gateway/Dockerfile" "$PROJECT_ROOT"
  docker save llm-gateway:latest -o "$GATEWAY_TAR"
  apptainer build --force "$GATEWAY_SIF" docker-archive://"$GATEWAY_TAR"
  rm -f "$GATEWAY_TAR"
else
  echo "ERROR: no llm-gateway.tar and no Docker. On a machine with Docker run:" >&2
  echo "       docker build -t llm-gateway:latest -f gateway/Dockerfile ." >&2
  echo "       docker save llm-gateway:latest -o images/llm-gateway.tar" >&2
  echo "       scp images/llm-gateway.tar datalab:~/thesis-llm-serving/images/" >&2
  echo "       then rerun this script." >&2
  exit 1
fi

echo "==> Built:"
echo "    $VLLM_SIF"
echo "    $GATEWAY_SIF"
