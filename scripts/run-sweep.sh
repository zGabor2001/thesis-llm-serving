#!/usr/bin/env bash
set -euo pipefail

# Run a scheduler × concurrency sweep against a (possibly running) vLLM.
#
# Usage:
#   SCHEDULERS="fifo,wrr,drr" CONCURRENCIES="1,4,8,16" REQUESTS=200 scripts/run-sweep.sh
#
# Reuses the running vLLM if present, otherwise submits it. Each iteration
# submits a fresh gateway with the chosen scheduler, runs the load generator,
# waits for it to finish, then moves on. Only submission/polling happens here
# (no heavy compute on the login node).

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/slurm/common.sh"

SCHEDULERS="${SCHEDULERS:-fifo,wrr,drr}"
CONCURRENCIES="${CONCURRENCIES:-1,4,8,16}"
REQUESTS="${REQUESTS:-200}"
TENANTS="${TENANTS:-tenant-a:2,tenant-b:1,tenant-c:1}"

wait_for_job() {
  local jobid="$1"
  while squeue -h -j "$jobid" 2>/dev/null | grep -q .; do
    sleep 5
  done
}

GATEWAY_JOB=""
stop_gateway() {
  if [ -n "$GATEWAY_JOB" ]; then
    scancel "$GATEWAY_JOB" 2>/dev/null || true
    GATEWAY_JOB=""
  fi
  rm -f "$RUN_DIR/gateway.url"
}
trap stop_gateway EXIT

# 1) Ensure vLLM is up.
if [ -f "$RUN_DIR/vllm.url" ] && wait_for_http "http://$(cat "$RUN_DIR/vllm.url")/health" 5; then
  echo "vLLM already running: $(cat "$RUN_DIR/vllm.url")"
else
  echo "==> Submitting vLLM"
  VLLM_JOB=$(sbatch --parsable "$HERE/slurm/serve-vllm.sbatch")
  echo "    job id: $VLLM_JOB"
  for _ in $(seq 1 180); do
    [ -f "$RUN_DIR/vllm.url" ] && break
    sleep 5
  done
  [ -f "$RUN_DIR/vllm.url" ] || { echo "vLLM never became ready" >&2; exit 1; }
fi

for scheduler in ${SCHEDULERS//,/ }; do
  for concurrency in ${CONCURRENCIES//,/ }; do
    echo ""
    echo "==> scheduler=$scheduler concurrency=$concurrency requests=$REQUESTS"

    stop_gateway

    GATEWAY_JOB=$(SCHEDULER_MODE="$scheduler" sbatch --parsable --export=ALL \
      "$HERE/slurm/run-gateway.sbatch")
    echo "    gateway job: $GATEWAY_JOB"

    for _ in $(seq 1 60); do
      [ -f "$RUN_DIR/gateway.url" ] && break
      sleep 2
    done
    [ -f "$RUN_DIR/gateway.url" ] || { echo "gateway never started" >&2; exit 1; }

    wait_for_http "http://$(cat "$RUN_DIR/gateway.url")/health" 300

    LOADGEN_JOB=$(CONCURRENCY="$concurrency" REQUESTS="$REQUESTS" TENANTS="$TENANTS" \
      sbatch --parsable --export=ALL "$HERE/slurm/run-loadgen.sbatch")
    echo "    loadgen job: $LOADGEN_JOB"

    wait_for_job "$LOADGEN_JOB"
    echo "    loadgen finished"
  done
done

stop_gateway

echo ""
echo "==> Sweep done."
echo "    Results:  $LOG_DIR/loadgen.jsonl"
echo "    Requests: $LOG_DIR/requests.jsonl"
echo "    Analyze:  python scripts/analyze.py --log $LOG_DIR/requests.jsonl"
