#!/usr/bin/env bash
set -euo pipefail

# Build the Apptainer/Singularity images used by the SLURM deployment.
#
# vLLM is pulled straight from Docker Hub (no Docker daemon needed).
# The gateway is built from gateway/Dockerfile; this step needs Docker.
# If you don't have Docker on the cluster, build llm-gateway.sif on any
# machine that does and copy it into images/.
#
# Run once, on a node with apptainer installed.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF_DIR="${SIF_DIR:-$PROJECT_ROOT/images}"

VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"   # CUDA image for GPU nodes
VLLM_SIF="$SIF_DIR/vllm-openai.sif"
GATEWAY_SIF="$SIF_DIR/llm-gateway.sif"

mkdir -p "$SIF_DIR"

echo "==> Pulling vLLM image: $VLLM_IMAGE"
apptainer pull --force "$VLLM_SIF" "docker://$VLLM_IMAGE"

echo "==> Building gateway image"
if command -v docker >/dev/null 2>&1; then
  docker build -t llm-gateway:latest -f "$PROJECT_ROOT/gateway/Dockerfile" "$PROJECT_ROOT"
  docker save llm-gateway:latest -o "$SIF_DIR/llm-gateway.tar"
  apptainer build --force "$GATEWAY_SIF" docker-archive://"$SIF_DIR/llm-gateway.tar"
  rm -f "$SIF_DIR/llm-gateway.tar"
else
  echo "ERROR: docker not found. Build llm-gateway.sif elsewhere and place it at:" >&2
  echo "       $GATEWAY_SIF" >&2
  exit 1
fi

echo "==> Built:"
echo "    $VLLM_SIF"
echo "    $GATEWAY_SIF"
