#!/usr/bin/env bash
# Shared helpers for the LLM serving SLURM jobs.
# Source from sbatch scripts or the experiment driver.

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$_SCRIPT_DIR/../.." && pwd)}"
MODEL_ROOT="${MODEL_ROOT:-$PROJECT_ROOT/models}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/run}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
SIF_DIR="${SIF_DIR:-$PROJECT_ROOT/images}"

VLLM_PORT="${VLLM_PORT:-8000}"
GATEWAY_PORT="${GATEWAY_PORT:-8080}"

# Scheduler / serving knobs (exported into the gateway container by
# run-gateway.sbatch). Set these when submitting, e.g.:
#   SCHEDULER_MODE=drr TENANT_QUANTA=tenant-a:8,tenant-b:8 sbatch ...
SCHEDULER_MODE="${SCHEDULER_MODE:-fifo}"
TENANT_WEIGHTS="${TENANT_WEIGHTS:-}"
TENANT_QUANTA="${TENANT_QUANTA:-}"
LORA_TENANTS="${LORA_TENANTS:-tenant-a,tenant-b,tenant-c}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/models/base/tokenizer.json}"

mkdir -p "$RUN_DIR" "$LOG_DIR" "$SIF_DIR"

wait_for_http() {
  # $1 = url, $2 = timeout seconds
  local url="$1" timeout="${2:-300}"
  local deadline=$((SECONDS + timeout))
  until curl -fsS "$url" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      echo "ERROR: timeout waiting for $url" >&2
      return 1
    fi
    sleep 5
  done
  return 0
}

write_state() {
  # $1 = filename, $2 = value
  echo "$2" > "$RUN_DIR/$1"
}
