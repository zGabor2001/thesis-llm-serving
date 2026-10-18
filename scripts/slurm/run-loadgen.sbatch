#!/usr/bin/env bash
#SBATCH --job-name=llm-loadgen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --partition=GPU-a40      # CPU-only (no --gres); adjust if a CPU partition exists
##SBATCH --account=<your-account>
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SIF="$SIF_DIR/llm-gateway.sif"
[ -f "$SIF" ] || { echo "ERROR: missing $SIF — run scripts/build-apptainer.sh first" >&2; exit 1; }

GATEWAY_URL="${GATEWAY_URL:-$(cat "$RUN_DIR/gateway.url" 2>/dev/null || true)}"
if [ -z "$GATEWAY_URL" ]; then
  echo "ERROR: no gateway endpoint — start run-gateway.sbatch first" >&2
  exit 1
fi
wait_for_http "http://$GATEWAY_URL/health" 300

CONCURRENCY="${CONCURRENCY:-8}"
REQUESTS="${REQUESTS:-200}"
TENANTS="${TENANTS:-tenant-a:2,tenant-b:1,tenant-c:1}"

echo "Load test: $REQUESTS requests @ $CONCURRENCY concurrency, tenants=$TENANTS"

# Reuse the gateway image (it has python + requests); bind the scripts dir.
apptainer exec \
  --bind "$PROJECT_ROOT/scripts:/scripts,$LOG_DIR:/logs" \
  "$SIF" python /scripts/loadgen.py \
    --url "http://$GATEWAY_URL" \
    --requests "$REQUESTS" \
    --concurrency "$CONCURRENCY" \
    --tenants "$TENANTS" \
    --variable-prompt \
    --out /logs/loadgen.jsonl
