#!/usr/bin/env bash
#SBATCH --job-name=llm-gateway
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --partition=CPU-amd
##SBATCH --account=<your-account>
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$REPO_ROOT/scripts/slurm/common.sh"

SIF="$SIF_DIR/llm-gateway.sif"
[ -f "$SIF" ] || { echo "ERROR: missing $SIF — run scripts/build-apptainer.sh first" >&2; exit 1; }

VLLM_URL="${VLLM_URL:-$(cat "$RUN_DIR/vllm.url" 2>/dev/null || true)}"
if [ -z "$VLLM_URL" ]; then
  echo "ERROR: no vLLM endpoint — start scripts/slurm/serve-vllm.sbatch first" >&2
  exit 1
fi

wait_for_http "http://$VLLM_URL/health" 300

export BACKEND_MODE=vllm
export VLLM_BASE_URL="http://$VLLM_URL"
export DEFAULT_MODEL="base-model"
export SCHEDULER_MODE="${SCHEDULER_MODE:-fifo}"
export REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-60}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"
export TENANT_WEIGHTS="${TENANT_WEIGHTS:-}"
export TENANT_QUANTA="${TENANT_QUANTA:-}"
export LORA_TENANTS="${LORA_TENANTS:-tenant-a,tenant-b,tenant-c}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/models/base/tokenizer.json}"

NODE="$(hostname)"
write_state gateway.url "$NODE:$GATEWAY_PORT"
echo "Gateway: node=$NODE port=$GATEWAY_PORT scheduler=$SCHEDULER_MODE backend=$VLLM_URL"

apptainer exec --pwd /workspace \
  --bind "$MODEL_ROOT:/models,$LOG_DIR:/workspace/logs,$PROJECT_ROOT/configs:/configs" \
  --env BACKEND_MODE=vllm \
  --env VLLM_BASE_URL="http://$VLLM_URL" \
  --env DEFAULT_MODEL=base-model \
  --env SCHEDULER_MODE="$SCHEDULER_MODE" \
  --env REQUEST_TIMEOUT_S="$REQUEST_TIMEOUT_S" \
  --env MAX_CONCURRENCY="$MAX_CONCURRENCY" \
  --env TENANT_WEIGHTS="$TENANT_WEIGHTS" \
  --env TENANT_QUANTA="$TENANT_QUANTA" \
  --env LORA_TENANTS="$LORA_TENANTS" \
  --env TOKENIZER_PATH="$TOKENIZER_PATH" \
  "$SIF" python -m uvicorn gateway.proxy:app --host 0.0.0.0 --port "$GATEWAY_PORT"
