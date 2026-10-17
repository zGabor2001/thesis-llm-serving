#!/usr/bin/env bash
#SBATCH --job-name=llm-gateway
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --partition=cpu            # gateway is CPU-only; adjust to your site
##SBATCH --account=<your-account>
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

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

NODE="$(hostname)"
write_state gateway.url "$NODE:$GATEWAY_PORT"
echo "Gateway: node=$NODE port=$GATEWAY_PORT scheduler=$SCHEDULER_MODE backend=$VLLM_URL"

apptainer exec --pwd /workspace \
  --bind "$LOG_DIR:/workspace/logs,$PROJECT_ROOT/configs:/configs" \
  --env BACKEND_MODE,VLLM_BASE_URL,DEFAULT_MODEL,SCHEDULER_MODE,REQUEST_TIMEOUT_S \
  "$SIF" python -m uvicorn gateway.proxy:app --host 0.0.0.0 --port "$GATEWAY_PORT"
