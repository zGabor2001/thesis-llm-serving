#!/usr/bin/env bash
set -euo pipefail

# End-to-end experiment: ensure vLLM is up, start the gateway with a chosen
# scheduler, run the load generator, and leave the services running.
#
# Usage:
#   SCHEDULER_MODE=drr CONCURRENCY=16 REQUESTS=200 scripts/run-experiment.sh
#
# Endpoints are discovered via state files in run/ (vllm.url, gateway.url).

source "$(dirname "${BASH_SOURCE[0]}")/slurm/common.sh"

SCHEDULER_MODE="${SCHEDULER_MODE:-fifo}"
CONCURRENCY="${CONCURRENCY:-8}"
REQUESTS="${REQUESTS:-200}"
TENANT="${TENANT:-tenant-a}"

# 1) vLLM backend
if [ -f "$RUN_DIR/vllm.url" ] && wait_for_http "http://$(cat "$RUN_DIR/vllm.url")/health" 5; then
  echo "vLLM already running: $(cat "$RUN_DIR/vllm.url")"
else
  echo "==> Submitting vLLM job"
  VLLM_JOB=$(sbatch --parsable "$(dirname "${BASH_SOURCE[0]}")/slurm/serve-vllm.sbatch")
  echo "    job id: $VLLM_JOB"
  for _ in $(seq 1 180); do
    [ -f "$RUN_DIR/vllm.url" ] && break
    sleep 5
  done
  [ -f "$RUN_DIR/vllm.url" ] || { echo "vLLM never became ready" >&2; exit 1; }
fi
VLLM_URL="$(cat "$RUN_DIR/vllm.url")"
echo "==> vLLM ready at $VLLM_URL"

# 2) Gateway
rm -f "$RUN_DIR/gateway.url"
echo "==> Submitting gateway (scheduler=$SCHEDULER_MODE)"
GATEWAY_JOB=$(sbatch --parsable \
  --export=ALL,SCHEDULER_MODE="$SCHEDULER_MODE" \
  "$(dirname "${BASH_SOURCE[0]}")/slurm/run-gateway.sbatch")
echo "    job id: $GATEWAY_JOB"

for _ in $(seq 1 60); do
  [ -f "$RUN_DIR/gateway.url" ] && break
  sleep 2
done
[ -f "$RUN_DIR/gateway.url" ] || { echo "gateway never started" >&2; exit 1; }
GATEWAY_URL="$(cat "$RUN_DIR/gateway.url")"
wait_for_http "http://$GATEWAY_URL/health" 300
echo "==> Gateway ready at $GATEWAY_URL"

# 3) Load test
echo "==> Running load generator ($REQUESTS requests @ $CONCURRENCY concurrency)"
python3 "$(dirname "${BASH_SOURCE[0]}")/loadgen.py" \
  --url "http://$GATEWAY_URL" \
  --requests "$REQUESTS" \
  --concurrency "$CONCURRENCY" \
  --tenant "$TENANT"

echo ""
echo "==> Done."
echo "    Request log:  $LOG_DIR/requests.jsonl"
echo "    vLLM job:     $(cat "$RUN_DIR/vllm.jobid")"
echo "    Gateway job:  $GATEWAY_JOB"
echo "    Cleanup:      scancel $GATEWAY_JOB $(cat "$RUN_DIR/vllm.jobid")"
