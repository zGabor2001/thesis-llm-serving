#!/usr/bin/env bash
#SBATCH --job-name=llm-vllm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=GPU-a40
##SBATCH --account=<your-account>  # uncomment if your site requires it
##SBATCH --qos=<your-qos>
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$REPO_ROOT/scripts/slurm/common.sh"

SIF="$SIF_DIR/vllm-openai.sif"
[ -f "$SIF" ] || { echo "ERROR: missing $SIF — run scripts/build-apptainer.sh first" >&2; exit 1; }

TP_SIZE="${TP_SIZE:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"

# Remove any stale state from a previous run.
cleanup() { rm -f "$RUN_DIR/vllm.url" "$RUN_DIR/vllm.jobid"; }
trap cleanup EXIT

echo "Starting vLLM: tensor-parallel=$TP_SIZE gpu-mem-util=$GPU_MEM_UTIL"

START_TS="$(date +%s.%N)"
apptainer exec --nv \
  --bind "$MODEL_ROOT:/models,$LOG_DIR:/logs,$PROJECT_ROOT/traces:/traces,$PROJECT_ROOT/configs:/configs" \
  "$SIF" vllm serve /models/base \
    --served-model-name base-model \
    --enable-lora \
    --max-loras 4 \
    --max-lora-rank 64 \
    --lora-modules tenant-a=/models/loras/tenant-a tenant-b=/models/loras/tenant-b tenant-c=/models/loras/tenant-c \
    --max-model-len 4096 \
    --tensor-parallel-size "$TP_SIZE" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" &
VLLM_PID=$!

NODE="$(hostname)"
if wait_for_http "http://$NODE:$VLLM_PORT/health" 900; then
  READY_TS="$(date +%s.%N)"
  LOAD_TIME_S="$(awk "BEGIN { printf \"%.3f\", $READY_TS - $START_TS }")"
  write_state vllm.url "$NODE:$VLLM_PORT"
  write_state vllm.jobid "$SLURM_JOB_ID"
  write_state vllm.load_time_s "$LOAD_TIME_S"
  echo "vLLM ready at $NODE:$VLLM_PORT (job=$SLURM_JOB_ID load=${LOAD_TIME_S}s pid=$VLLM_PID)"
else
  echo "ERROR: vLLM did not become ready" >&2
  kill "$VLLM_PID" || true
  exit 1
fi

wait "$VLLM_PID"
